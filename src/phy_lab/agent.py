from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

sys.dont_write_bytecode = True

from config import (  # noqa: E402
    ConfigError,
    TargetConfig,
    config_dir,
    format_targets,
    init_config_interactive,
    load_config,
    pick_cache_dir,
    pick_state_path,
    resolve_path,
)
from ollama import OllamaClient  # noqa: E402
from phy_engine import (  # noqa: E402
    PhyEngineError,
    ensure_phyengine_lib,
    ensure_verilog2plsav,
    prebuild_phy_engine,
)
from plar import (  # noqa: E402
    PLARError,
    email_login,
    get_comments,
    get_experiment_context,
    get_messages,
    post_comment,
)
from state import (  # noqa: E402
    AgentState,
    StateError,
    append_conversation_turn,
    count_recent_requests,
    get_target_state,
    get_conversation_history,
    load_state,
    prune_processed_keys,
    record_request_timestamp,
    save_state,
)
from text import (  # noqa: E402
    contains_mention,
    parse_command,
    safe_mention_prefix,
    strip_leading_mention,
    truncate,
)
from tools import (  # noqa: E402
    build_and_maybe_publish_circuit,
    format_experiment_hits,
    llm_chat,
    llm_summarize,
    render_help,
    safe_reply,
    search_recent_experiments,
    simulate_series_vdc_two_resistors,
    web_search,
)


def _safe_at_mention(handle: str) -> str | None:
    handle = (handle or "").strip()
    if not handle:
        return None
    if any(ch in handle for ch in (":", " ")):
        return None
    return f"@{handle}"


_AT_HANDLE_RE = re.compile(r"(?:@|＠)\s*([^\s:：]{1,64})")


def _extract_safe_mentions(text: str) -> list[str]:
    text = text or ""
    out: list[str] = []
    seen: set[str] = set()
    for m in _AT_HANDLE_RE.finditer(text):
        h = (m.group(1) or "").strip()
        if not h:
            continue
        at = _safe_at_mention(h)
        if not at:
            continue
        if at in seen:
            continue
        seen.add(at)
        out.append(at)
        if len(out) >= 5:
            break
    return out


def _infer_publish_category(user_text: str, *, default_category: str) -> str:
    t = (user_text or "").lower()
    if any(x in t for x in ("discussion", "discuss", "black hole", "讨论", "黑洞", "交流")):
        return "Discussion"
    if any(x in t for x in ("experiment", "lab", "实验", "实验区")):
        return "Experiment"
    if default_category in ("Experiment", "Discussion"):
        return default_category
    return "Discussion"


_POLITICAL_RE = re.compile(
    r"(?i)\\b("
    r"politic|politics|election|vote|campaign|parliament|congress|senate|president|prime\\s+minister|government|regime|party|"
    r"propaganda|geopolit|sanction|war|invasion|conflict|military|terroris|"
    r"ukraine|russia|israel|palestin|gaza|taiwan|hong\\s*kong|xinjiang|tibet"
    r")\\b"
)


def _looks_political_sensitive(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _POLITICAL_RE.search(t) is not None:
        return True
    # CJK keywords (broad, intentionally conservative).
    cjk = (
        "政治",
        "选举",
        "投票",
        "总统",
        "政府",
        "政党",
        "宣传",
        "意识形态",
        "战争",
        "冲突",
        "军事",
        "制裁",
        "台海",
        "台湾",
        "香港",
        "新疆",
        "西藏",
        "以色列",
        "巴勒斯坦",
        "加沙",
        "乌克兰",
        "俄罗斯",
    )
    return any(k in t for k in cjk)


def _political_refusal_message(user_text: str) -> str:
    t = user_text or ""
    is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in t)
    if is_cjk:
        return "抱歉，我不能处理或搜索任何政治相关内容。我可以帮助你解决物理实验室社区相关问题。"
    return "Sorry, I can't help with political content or political web searches. I can help with Physics Lab AR community questions."


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _llm_decide_web_search(
    *,
    ollama: OllamaClient,
    system_prompt: str,
    user_text: str,
    context_json: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Return (use_web, query)."""
    router_prompt = (
        "You are a tool router for a Physics Lab AR community agent.\n"
        "Decide whether the user needs an external web lookup.\n\n"
        "Rules:\n"
        "- Use web search ONLY if the user explicitly asks to look things up online OR the question clearly requires up-to-date or external facts.\n"
        "- Do NOT use web search for pure physics reasoning or for questions answerable from the provided context.\n"
        "- If you choose web search, output a concise query suitable for Google.\n"
        "- Output STRICT JSON only, no prose. Schema:\n"
        "  {\"use_web\": true|false, \"query\": \"...\"}\n"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": router_prompt},
    ]
    if context_json is not None:
        messages.append(
            {
                "role": "system",
                "content": "Context JSON (current page):\n"
                + json.dumps(context_json, ensure_ascii=False, indent=2),
            }
        )
    messages.append({"role": "user", "content": user_text})
    raw = ollama.chat(messages=messages)
    obj = _try_parse_json_object(raw)
    if not obj:
        return False, ""
    use_web = bool(obj.get("use_web"))
    query = obj.get("query")
    if not isinstance(query, str):
        query = ""
    query = query.strip()
    if use_web and not query:
        query = user_text.strip()
    return use_web, query


def _llm_route_tool(
    *,
    ollama: OllamaClient,
    system_prompt: str,
    user_text: str,
    context_json: dict[str, Any] | None,
) -> dict[str, Any] | None:
    router_prompt = (
        "You are a tool router for a Physics Lab AR community agent.\n"
        "Select the best action for the user request.\n\n"
        "Allowed actions:\n"
        "- chat\n"
        "- summarize\n"
        "- search_plar  (search within Physics Lab community)\n"
        "- google  (web search)\n"
        "- circuit  (generate Verilog and compile to .sav)\n"
        "- simulate  (run a local circuit simulation demo)\n\n"
        "Publishing policy:\n"
        "- Only set publish=true if the user explicitly asks to publish/share/post the experiment.\n"
        "- Otherwise publish=false.\n\n"
        "Examples:\n"
        "- User: 'Please implement a 4-bit adder and publish it as an experiment' -> action=circuit, publish=true\n"
        "- User: 'Summarize this' (on an experiment page) -> action=summarize\n"
        "- User: 'Search for op amp experiments' -> action=search_plar\n"
        "- User: 'Look this up online' -> action=google\n\n"
        "Output STRICT JSON only, no prose. Schema:\n"
        "{\"action\":\"chat|summarize|search_plar|google|circuit|simulate\",\"arg\":\"...\",\"publish\":true|false}\n"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": router_prompt},
    ]
    if context_json is not None:
        messages.append(
            {
                "role": "system",
                "content": "Context JSON (current page):\n"
                + json.dumps(context_json, ensure_ascii=False, indent=2),
            }
        )
    messages.append({"role": "user", "content": user_text})
    raw = ollama.chat(messages=messages)
    # Keep logs privacy-friendly: only log the first short chunk.
    obj = _try_parse_json_object(raw)
    if not obj:
        return None
    action = obj.get("action")
    arg = obj.get("arg")
    publish = obj.get("publish")
    if not isinstance(action, str):
        return None
    action = action.strip()
    if action not in ("chat", "summarize", "search_plar", "google", "circuit", "simulate"):
        return None
    if not isinstance(arg, str):
        arg = ""
    arg = arg.strip()
    obj2: dict[str, Any] = {"action": action, "arg": arg, "publish": bool(publish)}
    return obj2


def _fallback_route_for_circuit(user_text: str) -> tuple[bool, bool]:
    """Return (looks_like_circuit_request, explicit_publish_intent)."""
    t = (user_text or "").casefold()
    if not t:
        return False, False

    publish_intent = any(x in t for x in ("publish", "post", "share", "发布", "投稿", "发布到"))
    looks_like_build = any(x in t for x in ("implement", "build", "make", "generate", "实现", "制作", "生成"))
    looks_like_circuit = any(x in t for x in ("verilog", "plsav", "sav", "circuit", "电路", "加法器", "adder"))
    wants_experiment = any(x in t for x in ("experiment", "实验区", "实验"))
    if looks_like_circuit and (looks_like_build or wants_experiment):
        return True, publish_intent
    return False, publish_intent


def _setup_logging(
    *,
    cache_dir: str,
    level: str,
) -> logging.Logger:
    log_dir = os.path.join(cache_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "phy_lab.log")

    logger = logging.getLogger("phy_lab")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_level = getattr(logging, level.upper(), logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.debug("Logging initialized (console=%s, file=%s)", level.upper(), log_path)
    return logger


def _comment_timestamp_ms(comment: dict[str, Any]) -> int | None:
    ts = comment.get("Timestamp")
    return ts if isinstance(ts, int) else None


def _comment_content(comment: dict[str, Any]) -> str | None:
    content = comment.get("Content")
    if isinstance(content, str):
        content = content.strip()
        return content if content else None
    return None


def _comment_author(comment: dict[str, Any]) -> tuple[str | None, str | None]:
    author_id: str | None = None
    nickname: str | None = None

    for key in ("UserID", "UserId", "AuthorID", "AuthorId"):
        value = comment.get(key)
        if isinstance(value, str) and value.strip():
            author_id = value.strip()
            break

    user_value = comment.get("User")
    if isinstance(user_value, dict):
        for key in ("ID", "Id"):
            value = user_value.get(key)
            if isinstance(value, str) and value.strip():
                author_id = author_id or value.strip()
                break
        for key in ("Nickname", "Name"):
            value = user_value.get(key)
            if isinstance(value, str) and value.strip():
                nickname = value.strip()
                break
    elif isinstance(user_value, str) and user_value.strip():
        author_id = author_id or user_value.strip()

    for key in ("Nickname", "UserNickname", "AuthorNickname"):
        value = comment.get(key)
        if isinstance(value, str) and value.strip():
            nickname = nickname or value.strip()
            break

    return author_id, nickname


def _comment_key(comment: dict[str, Any]) -> str:
    for key in ("ID", "Id", "CommentID", "CommentId"):
        value = comment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    author_id, nickname = _comment_author(comment)
    ts = _comment_timestamp_ms(comment)
    content = _comment_content(comment)
    fallback = {
        "author_id": author_id,
        "nickname": nickname,
        "timestamp_ms": ts,
        "content": content,
    }
    digest = hashlib.sha256(
        json.dumps(fallback, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"hash:{digest}"


def _recent_comments_context(comments: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for c in comments:
        ts = _comment_timestamp_ms(c)
        content = _comment_content(c)
        if ts is None or not content:
            continue
        author_id, nickname = _comment_author(c)
        items.append(
            {
                "timestamp_ms": ts,
                "author_id": author_id,
                "author_nickname": nickname,
                "content": truncate(content, max_chars=280),
            }
        )
    items.sort(key=lambda x: int(x.get("timestamp_ms") or 0), reverse=True)
    return items[: max(0, limit)]


def _should_trigger(
    *,
    text: str,
    require_mention: bool,
    mention_tag: str,
    command_prefix: str,
    commands_enabled: bool,
    reply_to_self: bool,
) -> bool:
    text = (text or "").strip()
    if reply_to_self:
        return True
    if not text:
        return False
    if commands_enabled and command_prefix and text.startswith(command_prefix):
        return True
    if not require_mention:
        return True
    return contains_mention(text, mention_tag=mention_tag)


def _infer_nl_tool(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "chat", ""

    low = text.casefold()

    if low in ("help", "h", "?") or text in ("帮助", "帮助一下", "指令", "命令"):
        return "help", ""

    for key, aliases in (
        ("summarize", ("summarize", "summary")),
        ("search", ("search", "find")),
        ("circuit", ("circuit", "verilog")),
        ("simulate", ("simulate", "simulation")),
        ("google", ("google", "web", "websearch")),
    ):
        for a in aliases:
            if low.startswith(a + " "):
                return key, text.split(None, 1)[1].strip()
            if low == a:
                return key, ""

    if low.startswith("generate circuit "):
        return "circuit", text[len("generate circuit ") :].strip()
    if low == "generate circuit":
        return "circuit", ""

    if text.startswith(("总结", "概括")):
        return "summarize", text[2:].strip()
    if text.startswith(("搜索", "查找")):
        return "search", text[2:].strip()
    if text.startswith("生成电路"):
        return "circuit", text[len("生成电路") :].strip()
    if "仿真" in text or "模拟" in text:
        return "simulate", text
    if text.startswith("谷歌"):
        return "google", text[len("谷歌") :].strip()

    return "chat", text


def _handle_comment(
    *,
    comment: dict[str, Any],
    user: Any,
    ollama: OllamaClient,
    cache_dir: str,
    config_base_dir: str,
    cfg: Any,
    dry_run: bool,
    logger: logging.Logger,
    conversation_key: str | None,
    experiment_context: dict[str, Any] | None,
    history: list[dict[str, str]],
) -> str | None:
    content = _comment_content(comment)
    if not content:
        logger.debug("Skip comment: empty content")
        return None
    author_id, nickname = _comment_author(comment)
    if getattr(cfg.agent, "log_include_comment_content", False):
        logger.debug("Incoming comment content: %r", truncate(content, max_chars=400))

    text = strip_leading_mention(content, mention_tag=cfg.agent.mention_tag)
    if logger.isEnabledFor(logging.DEBUG) and not getattr(
        cfg.agent, "log_include_comment_content", False
    ):
        changed = (content or "").lstrip() != (text or "").lstrip()
        logger.debug(
            "strip_leading_mention: changed=%s mention_tag=%r text_len=%d",
            changed,
            cfg.agent.mention_tag,
            len(text),
        )
    if cfg.agent.commands_enabled:
        cmd, arg = parse_command(text, prefix=cfg.agent.command_prefix)
        logger.debug("Parsed command=%r arg_len=%d (commands_enabled=true)", cmd, len(arg))
    else:
        cmd, arg = _infer_nl_tool(text)
        logger.debug("Parsed nl_tool=%r arg_len=%d (commands_enabled=false)", cmd, len(arg))

    if cmd is None:
        cmd = "chat"
        arg = text

    if cmd in ("", "help", "h", "?"):
        return render_help(command_prefix=cfg.agent.command_prefix)

    if cmd == "chat":
        if not arg:
            if experiment_context is not None:
                logger.debug("Tool chat invoked (empty input, summarizing context)")
                messages: list[dict[str, str]] = [{"role": "system", "content": cfg.agent.system_prompt}]
                messages.append(
                    {
                        "role": "system",
                        "content": "Context JSON (current content page):\n"
                        + json.dumps(experiment_context, ensure_ascii=False, indent=2),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Give a brief summary of the current content in 3 bullets and suggest one useful next question the user could ask.",
                    }
                )
                reply = ollama.chat(messages=messages)
                return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)
            return render_help(command_prefix=cfg.agent.command_prefix)
        logger.debug("Tool chat invoked (len=%d)", len(arg))

        # LLM-driven tool routing for natural language requests.
        if (not cfg.agent.commands_enabled) and bool(getattr(cfg.agent, "auto_tool_routing", True)):
            try:
                route = _llm_route_tool(
                    ollama=ollama,
                    system_prompt=cfg.agent.system_prompt,
                    user_text=arg,
                    context_json=experiment_context,
                )
            except Exception as e:
                logger.debug("Tool router failed: %s", e)
                route = None

            if route is not None:
                action = route.get("action")
                routed_arg = str(route.get("arg") or "").strip()
                publish_intent = bool(route.get("publish"))
                logger.info(
                    "Auto tool routing: action=%s publish=%s arg_len=%d",
                    action,
                    publish_intent,
                    len(routed_arg),
                )

                if action == "summarize":
                    if not routed_arg and experiment_context is not None:
                        messages: list[dict[str, str]] = [{"role": "system", "content": cfg.agent.system_prompt}]
                        messages.append(
                            {
                                "role": "system",
                                "content": "Context JSON (current content page):\n"
                                + json.dumps(experiment_context, ensure_ascii=False, indent=2),
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": "Summarize the current content. Keep it under 6 bullets.",
                            }
                        )
                        reply = ollama.chat(messages=messages)
                        return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)
                    if not routed_arg:
                        return "Provide text to summarize."
                    reply = llm_summarize(
                        ollama=ollama,
                        system_prompt=cfg.agent.system_prompt,
                        text=routed_arg,
                    )
                    return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

                if action == "search_plar":
                    if not routed_arg:
                        return "Provide a query string."
                    try:
                        hits = search_recent_experiments(
                            user=user, query=routed_arg, max_scan=200, max_results=5
                        )
                    except Exception as e:
                        return f"Search failed: {e}"
                    return safe_reply(format_experiment_hits(hits), max_chars=cfg.agent.max_reply_chars)

                if action == "google":
                    if not routed_arg:
                        return "Provide a query string."
                    if _looks_political_sensitive(routed_arg):
                        return _political_refusal_message(routed_arg)
                    if not bool(getattr(cfg.agent, "web_search_enabled", False)):
                        return "Web search is disabled. Set agent.web_search_enabled=true in config."
                    try:
                        reply = web_search(
                            query=routed_arg,
                            cache_dir=cache_dir,
                            proxy=str(getattr(cfg.agent, "web_search_proxy", "") or ""),
                            timeout_sec=int(getattr(cfg.agent, "web_search_timeout_sec", 20) or 20),
                            ttl_sec=int(getattr(cfg.agent, "web_search_cache_ttl_sec", 3600) or 3600),
                            max_results=int(getattr(cfg.agent, "web_search_max_results", 5) or 5),
                            fallback_to_ddg=bool(getattr(cfg.agent, "web_search_fallback_to_ddg", True)),
                        )
                    except Exception as e:
                        return f"Web search failed: {e}"
                    return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

                if action == "simulate":
                    try:
                        reply = simulate_series_vdc_two_resistors(
                            text=routed_arg or arg,
                            phy_engine_cfg=cfg.phy_engine,
                            config_base_dir=config_base_dir,
                        )
                    except Exception as e:
                        return f"Simulation failed: {e}"
                    return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

                if action == "circuit":
                    if not routed_arg:
                        routed_arg = arg
                    if not routed_arg:
                        return "Provide a circuit specification."
                    publish_wanted = bool(publish_intent) or bool(
                        getattr(cfg.agent, "publish_by_default", False)
                    )
                    enable_publish = (
                        bool(cfg.agent.enable_publish)
                        and bool(getattr(cfg.agent, "auto_publish", False))
                        and publish_wanted
                    )
                    publish_category_value = _infer_publish_category(
                        routed_arg, default_category=getattr(cfg.agent, "publish_category", "Discussion")
                    )
                    title = truncate(f"Auto Circuit: {routed_arg}", max_chars=60)
                    requested_by = nickname or author_id or "unknown"
                    mentions = []
                    by_at = _safe_at_mention(requested_by)
                    if by_at:
                        mentions.append(by_at)
                    mentions.extend(_extract_safe_mentions(routed_arg))
                    mentions = list(dict.fromkeys(mentions))[:5]
                    mention_line = f"Mentions: {' '.join(mentions)}\n\n" if mentions else ""
                    introduction = truncate(
                        f"{mention_line}Requested by: {requested_by}\n\nSpec:\n{routed_arg}",
                        max_chars=600,
                    )
                    try:
                        res = build_and_maybe_publish_circuit(
                            ollama=ollama,
                            user=user,
                            spec=routed_arg,
                            cache_dir=cache_dir,
                            phy_engine_cfg=cfg.phy_engine,
                            config_base_dir=config_base_dir,
                            keep_temp=cfg.storage.keep_temp,
                            enable_publish=enable_publish,
                            dry_run=dry_run,
                            max_attempts=int(getattr(cfg.agent, "circuit_max_attempts", 3) or 3),
                            publish_max_elements=int(getattr(cfg.agent, "publish_max_elements", 5000) or 5000),
                            title=title,
                            introduction=introduction,
                            publish_category_value=publish_category_value,
                            publish_tags=list(getattr(cfg.agent, "publish_tags", []) or []),
                        )
                    except Exception as e:
                        return safe_reply(
                            f"Sorry, I couldn't compile the circuit after multiple attempts. Error: {e}",
                            max_chars=cfg.agent.max_reply_chars,
                        )

                    if enable_publish and not res.published and res.publish_block_reason == "too_large":
                        limit = int(getattr(cfg.agent, "publish_max_elements", 5000) or 5000)
                        return safe_reply(
                            f"I generated the circuit, but I cannot publish it because the .sav is too large for Physics Lab (elements={res.plsav_elements}, limit={limit}). "
                            "Please simplify the design and try again.",
                            max_chars=cfg.agent.max_reply_chars,
                        )

                    if enable_publish and res.published:
                        return safe_reply(
                            "Done. I generated Verilog, compiled to .sav with -O4 and '--layout hier', and published it.\n"
                            f"Category: {publish_category_value}\nSummaryID: {res.summary_id}",
                            max_chars=cfg.agent.max_reply_chars,
                        )
                    return safe_reply(
                        "Done. I generated Verilog and compiled a .sav with -O4 and '--layout hier'.\n"
                        (
                            "Publishing is disabled (or dry-run). Enable publishing in config."
                            if not bool(cfg.agent.enable_publish)
                            else (
                                "Publishing was not requested. Ask to publish explicitly, or set agent.publish_by_default=true."
                                if not publish_wanted
                                else "Publishing did not run due to configuration."
                            )
                        ),
                        max_chars=cfg.agent.max_reply_chars,
                    )
            else:
                looks_like_circuit, explicit_publish = _fallback_route_for_circuit(arg)
                if looks_like_circuit:
                    logger.info(
                        "Fallback routing to circuit (explicit_publish=%s)",
                        explicit_publish,
                    )
                    publish_intent = explicit_publish
                    publish_wanted = bool(publish_intent) or bool(
                        getattr(cfg.agent, "publish_by_default", False)
                    )
                    enable_publish = (
                        bool(cfg.agent.enable_publish)
                        and bool(getattr(cfg.agent, "auto_publish", False))
                        and publish_wanted
                    )
                    publish_category_value = _infer_publish_category(
                        arg, default_category=getattr(cfg.agent, "publish_category", "Discussion")
                    )
                    title = truncate(f"Auto Circuit: {arg}", max_chars=60)
                    requested_by = nickname or author_id or "unknown"
                    mentions = []
                    by_at = _safe_at_mention(requested_by)
                    if by_at:
                        mentions.append(by_at)
                    mentions.extend(_extract_safe_mentions(arg))
                    mentions = list(dict.fromkeys(mentions))[:5]
                    mention_line = f"Mentions: {' '.join(mentions)}\n\n" if mentions else ""
                    introduction = truncate(
                        f"{mention_line}Requested by: {requested_by}\n\nSpec:\n{arg}",
                        max_chars=600,
                    )
                    try:
                        res = build_and_maybe_publish_circuit(
                            ollama=ollama,
                            user=user,
                            spec=arg,
                            cache_dir=cache_dir,
                            phy_engine_cfg=cfg.phy_engine,
                            config_base_dir=config_base_dir,
                            keep_temp=cfg.storage.keep_temp,
                            enable_publish=enable_publish,
                            dry_run=dry_run,
                            max_attempts=int(getattr(cfg.agent, "circuit_max_attempts", 3) or 3),
                            publish_max_elements=int(getattr(cfg.agent, "publish_max_elements", 5000) or 5000),
                            title=title,
                            introduction=introduction,
                            publish_category_value=publish_category_value,
                            publish_tags=list(getattr(cfg.agent, "publish_tags", []) or []),
                        )
                    except Exception as e:
                        return safe_reply(
                            f"Sorry, I couldn't compile the circuit after multiple attempts. Error: {e}",
                            max_chars=cfg.agent.max_reply_chars,
                        )

                    if enable_publish and not res.published and res.publish_block_reason == "too_large":
                        limit = int(getattr(cfg.agent, "publish_max_elements", 5000) or 5000)
                        return safe_reply(
                            f"I generated the circuit, but I cannot publish it because the .sav is too large for Physics Lab (elements={res.plsav_elements}, limit={limit}). "
                            "Please simplify the design and try again.",
                            max_chars=cfg.agent.max_reply_chars,
                        )

                    if enable_publish and res.published:
                        return safe_reply(
                            "Done. I generated Verilog, compiled to .sav with -O4 and '--layout hier', and published it.\n"
                            f"Category: {publish_category_value}\nSummaryID: {res.summary_id}",
                            max_chars=cfg.agent.max_reply_chars,
                        )
                    return safe_reply(
                        "Done. I generated Verilog and compiled a .sav with -O4 and '--layout hier'.\n"
                        (
                            "Publishing is disabled (or dry-run). Enable publishing in config."
                            if not bool(cfg.agent.enable_publish)
                            else (
                                "Publishing was not requested. Ask to publish explicitly, or set agent.publish_by_default=true."
                                if not publish_wanted
                                else "Publishing did not run due to configuration."
                            )
                        ),
                        max_chars=cfg.agent.max_reply_chars,
                    )

        context_blob = ""
        if experiment_context is not None:
            context_blob = (
                "You are replying to a comment on the current content page. "
                "Use the following context to answer accurately.\n"
                + json.dumps(experiment_context, ensure_ascii=False, indent=2)
                + "\n\n"
            )
        messages: list[dict[str, str]] = [{"role": "system", "content": cfg.agent.system_prompt}]
        if context_blob:
            messages.append({"role": "system", "content": context_blob})
        messages.extend(history)

        # LLM-driven automatic web search (optional).
        if bool(getattr(cfg.agent, "web_search_enabled", False)) and bool(
            getattr(cfg.agent, "auto_web_search", True)
        ):
            try:
                use_web, query = _llm_decide_web_search(
                    ollama=ollama,
                    system_prompt=cfg.agent.system_prompt,
                    user_text=arg,
                    context_json=experiment_context,
                )
            except Exception as e:
                logger.debug("Web router failed: %s", e)
                use_web, query = False, ""

            if use_web:
                if _looks_political_sensitive(query or arg):
                    return _political_refusal_message(query or arg)
                logger.info("Auto web search triggered (query=%r)", truncate(query, max_chars=200))
                try:
                    search_txt = web_search(
                        query=query,
                        cache_dir=cache_dir,
                        proxy=str(getattr(cfg.agent, "web_search_proxy", "") or ""),
                        timeout_sec=int(getattr(cfg.agent, "web_search_timeout_sec", 20) or 20),
                        ttl_sec=int(getattr(cfg.agent, "web_search_cache_ttl_sec", 3600) or 3600),
                        max_results=int(getattr(cfg.agent, "web_search_max_results", 5) or 5),
                        fallback_to_ddg=bool(getattr(cfg.agent, "web_search_fallback_to_ddg", True)),
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": "Web search results (use as external references; do not mention tools):\n"
                            + search_txt,
                        }
                    )
                except Exception as e:
                    logger.warning("Web search failed: %s", e)

        messages.append({"role": "user", "content": arg})
        reply = ollama.chat(messages=messages)
        return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

    if cmd in ("summarize", "sum"):
        if not arg and experiment_context is None:
            return "Provide text to summarize."
        if not arg and experiment_context is not None:
            logger.debug("Tool summarize invoked (using experiment context)")
            messages: list[dict[str, str]] = [{"role": "system", "content": cfg.agent.system_prompt}]
            messages.append(
                {
                    "role": "system",
                    "content": "Context JSON (current content page):\n"
                    + json.dumps(experiment_context, ensure_ascii=False, indent=2),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Summarize the current content. Include: title, key purpose, and what the circuit/experiment contains (use plsav_summary if present). Keep it under 6 bullets.",
                }
            )
            reply = ollama.chat(messages=messages)
            return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)
        logger.debug("Tool summarize invoked (len=%d)", len(arg))
        reply = llm_summarize(ollama=ollama, system_prompt=cfg.agent.system_prompt, text=arg)
        return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

    if cmd in ("search", "find"):
        if not arg:
            return "Provide a query string."
        logger.debug("Tool search invoked (query=%r)", arg[:200])
        try:
            hits = search_recent_experiments(user=user, query=arg, max_scan=200, max_results=5)
        except Exception as e:
            return f"Search failed: {e}"
        return safe_reply(format_experiment_hits(hits), max_chars=cfg.agent.max_reply_chars)

    if cmd in ("google", "web", "websearch"):
        if not arg:
            return "Provide a query string."
        if _looks_political_sensitive(arg):
            return _political_refusal_message(arg)
        if not bool(getattr(cfg.agent, "web_search_enabled", False)):
            return "Web search is disabled. Set agent.web_search_enabled=true in config."
        logger.debug("Tool google invoked (query=%r)", arg[:200])
        try:
            reply = web_search(
                query=arg,
                cache_dir=cache_dir,
                proxy=str(getattr(cfg.agent, "web_search_proxy", "") or ""),
                timeout_sec=int(getattr(cfg.agent, "web_search_timeout_sec", 20) or 20),
                ttl_sec=int(getattr(cfg.agent, "web_search_cache_ttl_sec", 3600) or 3600),
                max_results=int(getattr(cfg.agent, "web_search_max_results", 5) or 5),
                fallback_to_ddg=bool(getattr(cfg.agent, "web_search_fallback_to_ddg", True)),
            )
        except Exception as e:
            return f"Web search failed: {e}"
        return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

    if cmd in ("circuit", "verilog"):
        if not arg:
            return "Provide a circuit specification after the command."

        title = truncate(f"Auto Circuit: {arg}", max_chars=60)
        requested_by = nickname or author_id or "unknown"
        mentions = []
        by_at = _safe_at_mention(requested_by)
        if by_at:
            mentions.append(by_at)
        mentions.extend(_extract_safe_mentions(arg))
        mentions = list(dict.fromkeys(mentions))[:5]
        mention_line = f"Mentions: {' '.join(mentions)}\n\n" if mentions else ""
        introduction = truncate(
            f"{mention_line}Requested by: {requested_by}\n\nSpec:\n{arg}",
            max_chars=600,
        )
        logger.debug(
            "Tool circuit invoked (publish=%s, dry_run=%s, spec_len=%d)",
            bool(cfg.agent.enable_publish),
            dry_run,
            len(arg),
        )
        try:
            res = build_and_maybe_publish_circuit(
                ollama=ollama,
                user=user,
                spec=arg,
                cache_dir=cache_dir,
                phy_engine_cfg=cfg.phy_engine,
                config_base_dir=config_base_dir,
                keep_temp=cfg.storage.keep_temp,
                enable_publish=bool(cfg.agent.enable_publish),
                dry_run=dry_run,
                max_attempts=int(getattr(cfg.agent, "circuit_max_attempts", 3) or 3),
                publish_max_elements=int(getattr(cfg.agent, "publish_max_elements", 5000) or 5000),
                title=title,
                introduction=introduction,
            )
        except Exception as e:
            return f"Circuit generation failed: {e}"

        if bool(cfg.agent.enable_publish) and not res.published and res.publish_block_reason == "too_large":
            limit = int(getattr(cfg.agent, "publish_max_elements", 5000) or 5000)
            return safe_reply(
                f"I generated the circuit, but I cannot publish it because the .sav is too large for Physics Lab (elements={res.plsav_elements}, limit={limit}). "
                "Please simplify the design and try again.",
                max_chars=cfg.agent.max_reply_chars,
            )

        if res.published:
            return safe_reply(
                f"Your circuit has been generated and published. SummaryID: {res.summary_id}\n"
                f"Introduction: {introduction}",
                max_chars=cfg.agent.max_reply_chars,
            )

        return safe_reply(
            "Circuit generated, but not published (publish disabled or dry-run). "
            "Ask an admin to enable publishing, or re-run the agent with publish enabled.",
            max_chars=cfg.agent.max_reply_chars,
        )

    if cmd in ("simulate", "sim"):
        logger.debug("Tool simulate invoked (len=%d)", len(arg))
        try:
            reply = simulate_series_vdc_two_resistors(
                text=arg or text,
                phy_engine_cfg=cfg.phy_engine,
                config_base_dir=config_base_dir,
            )
        except Exception as e:
            return f"Simulation failed: {e}"
        return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

    return safe_reply(
        f"Unknown command: {cmd}. Use {cfg.agent.command_prefix}help",
        max_chars=cfg.agent.max_reply_chars,
    )


def _process_target(
    *,
    user: Any,
    target: TargetConfig,
    state: AgentState,
    ollama: OllamaClient,
    cfg: Any,
    cache_dir: str,
    config_base_dir: str,
    dry_run: bool,
    logger: logging.Logger,
) -> None:
    target_key = f"{target.type}:{target.id}"
    target_state = get_target_state(state, target_key)
    if target_state.last_seen_timestamp_ms == 0:
        lookback_sec = float(getattr(cfg.agent, "bootstrap_lookback_sec", 0.0) or 0.0)
        if lookback_sec > 0:
            target_state.last_seen_timestamp_ms = max(
                0, int(time.time() * 1000) - int(lookback_sec * 1000)
            )
            logger.info(
                "[%s] Bootstrapping last_seen to %d (lookback_sec=%.1f)",
                target_key,
                target_state.last_seen_timestamp_ms,
                lookback_sec,
            )

    try:
        logger.debug(
            "[%s] Fetching comments (last_seen=%d)...",
            target_key,
            target_state.last_seen_timestamp_ms,
        )
        comments = get_comments(
            user,
            target_id=target.id,
            target_type=target.type,
            take=cfg.agent.take,
        )
    except Exception as e:
        logger.warning("[%s] Failed to fetch comments: %s", target_key, e)
        return
    logger.debug("[%s] Fetched %d comments (take=%d)", target_key, len(comments), cfg.agent.take)

    new_comments: list[dict[str, Any]] = []
    for c in comments:
        ts = _comment_timestamp_ms(c)
        if ts is None:
            continue
        # Use >= to handle cases where the backend timestamp granularity is coarse (e.g. seconds),
        # which can cause multiple distinct comments to share the same timestamp.
        if ts >= target_state.last_seen_timestamp_ms:
            new_comments.append(c)

    new_comments.sort(key=lambda c: _comment_timestamp_ms(c) or 0)
    if not new_comments:
        latest_ts = max(
            (_comment_timestamp_ms(c) or 0 for c in comments),
            default=0,
        )
        logger.debug(
            "[%s] No new comments (last_seen=%d, latest=%d)",
            target_key,
            target_state.last_seen_timestamp_ms,
            latest_ts,
        )
        return

    processed = set(target_state.processed_comment_keys)

    for comment in new_comments:
        ts = _comment_timestamp_ms(comment)
        if ts is None:
            continue

        key = _comment_key(comment)
        if key in processed:
            target_state.last_seen_timestamp_ms = max(target_state.last_seen_timestamp_ms, ts)
            continue

        author_id, nickname = _comment_author(comment)
        logger.debug(
            "[%s] Considering comment key=%s ts=%d author=%s nickname=%s",
            target_key,
            key,
            ts,
            author_id,
            nickname,
        )
        if author_id is not None and getattr(user, "user_id", None) == author_id:
            logger.debug("[%s] Skip comment: self authored", target_key)
            target_state.processed_comment_keys.append(key)
            processed.add(key)
            target_state.last_seen_timestamp_ms = max(target_state.last_seen_timestamp_ms, ts)
            continue

        content = _comment_content(comment) or ""
        reply_id = None
        for rk in ("ReplyID", "ReplyId", "ReplyUserID", "ReplyUserId"):
            rv = comment.get(rk)
            if isinstance(rv, str) and rv.strip():
                reply_id = rv.strip()
                break
        reply_to_self = bool(
            reply_id
            and isinstance(getattr(user, "user_id", None), str)
            and reply_id == getattr(user, "user_id", None)
        )

        effective_require_mention = cfg.agent.require_mention
        if target.type == "User":
            effective_require_mention = bool(cfg.agent.user_targets_require_mention)

        triggered = _should_trigger(
            text=content,
            require_mention=effective_require_mention,
            mention_tag=cfg.agent.mention_tag,
            command_prefix=cfg.agent.command_prefix,
            commands_enabled=cfg.agent.commands_enabled,
            reply_to_self=reply_to_self,
        )
        if logger.isEnabledFor(logging.DEBUG):
            startswith_prefix = bool(
                cfg.agent.commands_enabled
                and cfg.agent.command_prefix
                and content.strip().startswith(cfg.agent.command_prefix)
            )
            has_mention = contains_mention(content, mention_tag=cfg.agent.mention_tag)
            logger.debug(
                "[%s] Trigger check: triggered=%s startswith_prefix=%s has_mention=%s reply_to_self=%s require_mention=%s commands_enabled=%s",
                target_key,
                triggered,
                startswith_prefix,
                has_mention,
                reply_to_self,
                effective_require_mention,
                cfg.agent.commands_enabled,
            )
        if not triggered:
            logger.debug("[%s] Skip comment: not triggered by prefix/mention", target_key)
            target_state.processed_comment_keys.append(key)
            processed.add(key)
            target_state.last_seen_timestamp_ms = max(target_state.last_seen_timestamp_ms, ts)
            continue

        now_ms = int(time.time() * 1000)
        if bool(getattr(cfg.agent, "overload_protection_enabled", True)):
            window_ms = int(getattr(cfg.agent, "overload_window_sec", 600) or 600) * 1000
            max_req = int(getattr(cfg.agent, "overload_max_requests", 40) or 40)
            recent_n = count_recent_requests(state, now_ms=now_ms, window_ms=window_ms)
            if max_req > 0 and recent_n >= max_req:
                busy_msg = str(getattr(cfg.agent, "overload_message_en", "") or "").strip()
                if not busy_msg:
                    busy_msg = "Too many requests at the moment, please try again later."
                logger.warning(
                    "[%s] Overloaded: recent_requests=%d window_sec=%d max=%d; replying busy",
                    target_key,
                    recent_n,
                    int(window_ms / 1000),
                    max_req,
                )
                prefix = safe_mention_prefix(nickname) if nickname else None
                post_body = f"{prefix or ''}{busy_msg}".strip()
                try:
                    if dry_run:
                        logger.info("[%s] DRY RUN busy reply: %s", target_key, post_body)
                    else:
                        post_comment(
                            user,
                            target_id=target.id,
                            target_type=target.type,
                            content=post_body,
                            reply_id=author_id,
                        )
                        logger.info("[%s] Busy reply posted", target_key)
                except Exception as e:
                    logger.warning("[%s] Failed to post busy reply: %s", target_key, e)

                record_request_timestamp(state, ts_ms=now_ms)
                target_state.processed_comment_keys.append(key)
                processed.add(key)
                target_state.last_seen_timestamp_ms = max(target_state.last_seen_timestamp_ms, ts)
                prune_processed_keys(target_state, keep_last=500)
                continue

        conversation_key = None
        if isinstance(author_id, str) and author_id.strip():
            conversation_key = f"{target_key}|{author_id.strip()}"
        history = (
            get_conversation_history(state, key=conversation_key, max_turns=12)
            if conversation_key
            else []
        )

        experiment_context: dict[str, Any] | None = None
        if target.type in ("Experiment", "Discussion"):
            try:
                experiment_context = get_experiment_context(
                    user,
                    summary_id=target.id,
                    category_value="Experiment" if target.type == "Experiment" else "Discussion",
                    cache_dir=cache_dir,
                    ttl_sec=300,
                    max_json_chars=20_000,
                )
                experiment_context = dict(experiment_context)
                experiment_context["recent_comments"] = _recent_comments_context(comments, limit=8)
                logger.debug("[%s] Loaded experiment context (cached)", target_key)
            except Exception as e:
                logger.debug("[%s] Failed to load experiment context: %s", target_key, e)

        reply = _handle_comment(
            comment=comment,
            user=user,
            ollama=ollama,
            cache_dir=cache_dir,
            config_base_dir=config_base_dir,
            cfg=cfg,
            dry_run=dry_run,
            logger=logger,
            conversation_key=conversation_key,
            experiment_context=experiment_context,
            history=history,
        )

        record_request_timestamp(state, ts_ms=now_ms)

        if reply:
            prefix = safe_mention_prefix(nickname) if nickname else None
            post_body = f"{prefix or ''}{reply}".strip()
            try:
                if dry_run:
                    logger.info("[%s] DRY RUN reply: %s", target_key, post_body)
                else:
                    post_comment(
                        user,
                        target_id=target.id,
                        target_type=target.type,
                        content=post_body,
                        reply_id=author_id,
                    )
                    logger.info(
                        "[%s] Replied to %s",
                        target_key,
                        nickname or author_id or "unknown",
                    )
            except Exception as e:
                logger.warning("[%s] Failed to post reply: %s", target_key, e)
        else:
            logger.debug("[%s] No reply generated for this comment", target_key)

        if conversation_key:
            user_text = strip_leading_mention(content, mention_tag=cfg.agent.mention_tag)
            normalized_user_turn = user_text
            if cfg.agent.commands_enabled:
                cmd_name, cmd_arg = parse_command(
                    user_text, prefix=cfg.agent.command_prefix
                )
                normalized_user_turn = cmd_arg if cmd_name is not None else user_text
            append_conversation_turn(
                state,
                key=conversation_key,
                role="user",
                content=normalized_user_turn,
                ts_ms=ts,
                keep_last=20,
            )
            if reply:
                append_conversation_turn(
                    state,
                    key=conversation_key,
                    role="assistant",
                    content=reply,
                    ts_ms=int(time.time() * 1000),
                    keep_last=20,
                )

        target_state.processed_comment_keys.append(key)
        processed.add(key)
        target_state.last_seen_timestamp_ms = max(target_state.last_seen_timestamp_ms, ts)
    prune_processed_keys(target_state, keep_last=500)


def _message_timestamp_ms(message: dict[str, Any]) -> int | None:
    ts = message.get("Timestamp")
    return ts if isinstance(ts, int) else None


def _message_key(message: dict[str, Any]) -> str:
    for key in ("ID", "Id", "MessageID", "MessageId"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    digest = hashlib.sha256(
        json.dumps(message, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    return f"hash:{digest}"


def _notification_target_from_message(message: dict[str, Any]) -> TargetConfig | None:
    allowed = {"User", "Experiment", "Discussion"}

    def iter_strings(root: Any) -> list[str]:
        out: list[str] = []
        stack2: list[Any] = [root]
        while stack2:
            x = stack2.pop()
            if isinstance(x, str):
                if x.strip():
                    out.append(x.strip())
            elif isinstance(x, dict):
                for v in x.values():
                    if isinstance(v, (dict, list, str)):
                        stack2.append(v)
            elif isinstance(x, list):
                for v in x:
                    if isinstance(v, (dict, list, str)):
                        stack2.append(v)
        return out

    def target_from_urlish(text: str) -> TargetConfig | None:
        s = (text or "").strip()
        if not s:
            return None
        m = re.search(r"\b(Experiment|Discussion|User)\b.*?\b([0-9a-fA-F]{24})\b", s)
        if m:
            return TargetConfig(type=m.group(1), id=m.group(2))
        m = re.search(r"\b([0-9a-fA-F]{24})\b.*?\b(Experiment|Discussion|User)\b", s)
        if m:
            return TargetConfig(type=m.group(2), id=m.group(1))
        return None

    def collect_hints(root: Any) -> tuple[set[str], set[str], set[str]]:
        # Returns: (hex_ids, user_ids, type_hints)
        hex_ids: set[str] = set()
        user_ids: set[str] = set()
        type_hints: set[str] = set()

        stack3: list[Any] = [root]
        while stack3:
            x = stack3.pop()
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    continue
                if "experiment" in s.casefold():
                    type_hints.add("Experiment")
                if "discussion" in s.casefold():
                    type_hints.add("Discussion")
                if "user" in s.casefold():
                    type_hints.add("User")
                for mid in re.findall(r"\b[0-9a-fA-F]{24}\b", s):
                    hex_ids.add(mid)
                continue

            if isinstance(x, list):
                for v in x:
                    if isinstance(v, (dict, list, str)):
                        stack3.append(v)
                continue

            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(k, str):
                        lk = k.lower()
                        if lk in ("users",):
                            if isinstance(v, list):
                                for uid in v:
                                    if isinstance(uid, str) and uid.strip():
                                        user_ids.add(uid.strip())
                        if lk in ("userid", "user_id") or lk.endswith("userid") or lk.endswith("_user_id"):
                            if isinstance(v, str) and v.strip():
                                user_ids.add(v.strip())
                    if isinstance(v, (dict, list, str)):
                        stack3.append(v)
        return hex_ids, user_ids, type_hints

    stack: list[Any] = [message]
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            tt: str | None = None
            tid: str | None = None

            category_hint: str | None = None
            content_id_hint: str | None = None
            for k, v in obj.items():
                if isinstance(k, str):
                    lk = k.lower()
                    if isinstance(v, str):
                        urlish = target_from_urlish(v)
                        if urlish is not None:
                            return urlish
                    if lk in ("targettype", "target_type"):
                        if isinstance(v, str) and v.strip() in allowed:
                            tt = v.strip()
                    if lk in ("targetid", "target_id"):
                        if isinstance(v, str) and v.strip():
                            tid = v.strip()
                    if lk in ("category", "categoryvalue"):
                        if isinstance(v, str) and v.strip() in ("Experiment", "Discussion"):
                            category_hint = v.strip()
                    if lk in ("summaryid", "summary_id", "contentid", "content_id"):
                        if isinstance(v, str) and v.strip():
                            content_id_hint = v.strip()
                    if lk.endswith("experimentid") or lk == "experiment_id":
                        if isinstance(v, str) and v.strip():
                            return TargetConfig(type="Experiment", id=v.strip())
                    if lk.endswith("discussionid") or lk == "discussion_id":
                        if isinstance(v, str) and v.strip():
                            return TargetConfig(type="Discussion", id=v.strip())
                    if lk.endswith("userid") or lk == "user_id":
                        if isinstance(v, str) and v.strip():
                            return TargetConfig(type="User", id=v.strip())
                if isinstance(v, (dict, list)):
                    stack.append(v)
            if tt and tid:
                return TargetConfig(type=tt, id=tid)
            if category_hint and content_id_hint:
                return TargetConfig(type=category_hint, id=content_id_hint)
        elif isinstance(obj, list):
            for v in obj:
                if isinstance(v, (dict, list)):
                    stack.append(v)

    # Fallback: scan all strings for "Experiment/Discussion/User" + 24-hex.
    for s in iter_strings(message):
        urlish = target_from_urlish(s)
        if urlish is not None:
            return urlish

    # Heuristic: if the message contains multiple 24-hex IDs, try to remove user IDs and keep content IDs.
    hex_ids, user_ids, type_hints = collect_hints(message)
    candidate_ids = [i for i in hex_ids if i not in user_ids]
    if len(candidate_ids) == 1:
        if "Discussion" in type_hints:
            return TargetConfig(type="Discussion", id=candidate_ids[0])
        if "Experiment" in type_hints:
            return TargetConfig(type="Experiment", id=candidate_ids[0])

    return None


def _notification_fallback_targets(message: dict[str, Any]) -> list[TargetConfig]:
    # Best-effort: if we can find exactly one 24-hex content-like ID, try both Experiment and Discussion.
    hex_ids: set[str] = set()
    user_ids: set[str] = set()
    type_hints: set[str] = set()

    stack: list[Any] = [message]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            s = x.strip()
            if s:
                if "experiment" in s.casefold():
                    type_hints.add("Experiment")
                if "discussion" in s.casefold():
                    type_hints.add("Discussion")
                for mid in re.findall(r"\b[0-9a-fA-F]{24}\b", s):
                    hex_ids.add(mid)
        elif isinstance(x, list):
            for v in x:
                if isinstance(v, (dict, list, str)):
                    stack.append(v)
        elif isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str):
                    lk = k.lower()
                    if lk == "users" and isinstance(v, list):
                        for uid in v:
                            if isinstance(uid, str) and uid.strip():
                                user_ids.add(uid.strip())
                    if lk in ("userid", "user_id") or lk.endswith("userid") or lk.endswith("_user_id"):
                        if isinstance(v, str) and v.strip():
                            user_ids.add(v.strip())
                if isinstance(v, (dict, list, str)):
                    stack.append(v)

    candidate_ids = [i for i in hex_ids if i not in user_ids]
    if len(candidate_ids) != 1:
        return []

    cid = candidate_ids[0]
    if "Experiment" in type_hints and "Discussion" not in type_hints:
        return [TargetConfig(type="Experiment", id=cid)]
    if "Discussion" in type_hints and "Experiment" not in type_hints:
        return [TargetConfig(type="Discussion", id=cid)]
    return [TargetConfig(type="Experiment", id=cid), TargetConfig(type="Discussion", id=cid)]


def _process_notifications(
    *,
    user: Any,
    state: AgentState,
    cfg: Any,
    cache_dir: str,
    config_base_dir: str,
    dry_run: bool,
    logger: logging.Logger,
    ollama: OllamaClient,
) -> None:
    if not getattr(cfg.agent, "notifications_enabled", True):
        return
    category_ids = list(getattr(cfg.agent, "notification_category_ids", [3]) or [3])
    take = int(getattr(cfg.agent, "notification_take", 20) or 20)

    discovered: dict[str, TargetConfig] = {}

    for cat in category_ids:
        msg_target_key = f"Messages:{int(cat)}"
        msg_state = get_target_state(state, msg_target_key)
        if msg_state.last_seen_timestamp_ms == 0:
            lookback_sec = float(getattr(cfg.agent, "bootstrap_lookback_sec", 0.0) or 0.0)
            if lookback_sec > 0:
                msg_state.last_seen_timestamp_ms = max(
                    0, int(time.time() * 1000) - int(lookback_sec * 1000)
                )
                logger.info(
                    "[%s] Bootstrapping last_seen to %d (lookback_sec=%.1f)",
                    msg_target_key,
                    msg_state.last_seen_timestamp_ms,
                    lookback_sec,
                )

        try:
            messages, templates = get_messages(
                user,
                category_id=int(cat),
                skip=0,
                take=take,
                no_templates=False,
            )
        except Exception as e:
            logger.warning("[%s] Failed to fetch messages: %s", msg_target_key, e)
            continue

        logger.debug(
            "[%s] Fetched %d messages (take=%d) templates=%d",
            msg_target_key,
            len(messages),
            take,
            len(templates),
        )

        tmpl_by_id: dict[str, dict[str, Any]] = {}
        for t in templates:
            tid = t.get("ID") or t.get("Id")
            if isinstance(tid, (str, int)):
                tmpl_by_id[str(tid)] = t

        new_msgs: list[dict[str, Any]] = []
        for m in messages:
            ts = _message_timestamp_ms(m)
            if ts is None:
                continue
            if ts >= msg_state.last_seen_timestamp_ms:
                new_msgs.append(m)
        new_msgs.sort(key=lambda m: _message_timestamp_ms(m) or 0)

        processed = set(msg_state.processed_comment_keys)
        for m in new_msgs:
            ts = _message_timestamp_ms(m)
            if ts is None:
                continue
            key = _message_key(m)
            if key in processed:
                msg_state.last_seen_timestamp_ms = max(msg_state.last_seen_timestamp_ms, ts)
                continue

            tmpl_id = m.get("TemplateID") or m.get("TemplateId")
            tmpl = tmpl_by_id.get(str(tmpl_id)) if isinstance(tmpl_id, (str, int)) else None
            combined: dict[str, Any] = {"Message": m}
            if tmpl is not None:
                combined["Template"] = tmpl
            t = _notification_target_from_message(combined)
            candidates: list[TargetConfig] = []
            if t is not None:
                candidates = [t]
            else:
                candidates = _notification_fallback_targets(combined)
            if not candidates and logger.isEnabledFor(logging.DEBUG):
                tmpl_identifier = None
                if isinstance(tmpl, dict):
                    ident = tmpl.get("Identifier")
                    if isinstance(ident, str) and ident.strip():
                        tmpl_identifier = ident.strip()
                logger.debug(
                    "[%s] Unparsed message: template_id=%r template_identifier=%r message_keys=%s",
                    msg_target_key,
                    tmpl_id,
                    tmpl_identifier,
                    sorted([k for k in m.keys() if isinstance(k, str)])[:40],
                )
            logger.debug(
                "[%s] Message key=%s ts=%d target=%s",
                msg_target_key,
                key,
                ts,
                f"{candidates[0].type}:{candidates[0].id}" if candidates else None,
            )
            for t0 in candidates:
                discovered[f"{t0.type}:{t0.id}"] = t0

            msg_state.processed_comment_keys.append(key)
            processed.add(key)
            msg_state.last_seen_timestamp_ms = max(msg_state.last_seen_timestamp_ms, ts)

        prune_processed_keys(msg_state, keep_last=500)

    if not discovered:
        return

    logger.info("Discovered %d targets from notifications", len(discovered))
    for t in discovered.values():
        _process_target(
            user=user,
            target=t,
            state=state,
            ollama=ollama,
            cfg=cfg,
            cache_dir=cache_dir,
            config_base_dir=config_base_dir,
            dry_run=dry_run,
            logger=logger,
        )


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"Failed to load config: {e}")
        return 2

    base_dir = config_dir(args.config)
    cache_dir = pick_cache_dir(args.config, config=cfg)
    os.makedirs(cache_dir, exist_ok=True)
    logger = _setup_logging(cache_dir=cache_dir, level=(args.log_level or cfg.agent.log_level))
    logger.info("phy_lab starting (config=%s, cache_dir=%s)", os.path.abspath(args.config), cache_dir)

    state_path = args.state or pick_state_path(args.config)
    try:
        state = load_state(state_path)
    except (OSError, json.JSONDecodeError, StateError) as e:
        logger.error("Failed to load state: %s", e)
        return 2
    logger.debug("State loaded (path=%s, targets=%d)", state_path, len(state.targets))

    # Optional prebuild for Phy-Engine (reduces first-request latency).
    if bool(getattr(cfg.phy_engine, "prebuild_on_start", False)) and bool(
        getattr(cfg.phy_engine, "auto_build", False)
    ):
        cmake_source_dir = os.path.abspath(resolve_path(cfg.phy_engine.cmake_source_dir, base_dir=base_dir))
        cmake_build_dir = os.path.abspath(resolve_path(cfg.phy_engine.cmake_build_dir, base_dir=base_dir))
        try:
            logger.info("Prebuilding Phy-Engine via CMake (this may take a while)...")
            prebuild_phy_engine(
                cmake_source_dir=cmake_source_dir,
                cmake_build_dir=cmake_build_dir,
                cmake_build_type=str(getattr(cfg.phy_engine, "cmake_build_type", "Release")),
                build_timeout_sec=int(getattr(cfg.phy_engine, "build_timeout_sec", 900)),
                targets=["verilog2plsav", "phyengine"],
            )
            # Ensure artifacts are discoverable (and log paths).
            v2p = ensure_verilog2plsav(
                verilog2plsav_path=str(getattr(cfg.phy_engine, "verilog2plsav_path", "") or ""),
                auto_build=True,
                cmake_source_dir=cmake_source_dir,
                cmake_build_dir=cmake_build_dir,
                cmake_build_type=str(getattr(cfg.phy_engine, "cmake_build_type", "Release")),
                build_timeout_sec=int(getattr(cfg.phy_engine, "build_timeout_sec", 900)),
            )
            lib = ensure_phyengine_lib(
                phyengine_lib_path=str(getattr(cfg.phy_engine, "phyengine_lib_path", "") or ""),
                auto_build=True,
                cmake_source_dir=cmake_source_dir,
                cmake_build_dir=cmake_build_dir,
                cmake_build_type=str(getattr(cfg.phy_engine, "cmake_build_type", "Release")),
                build_timeout_sec=int(getattr(cfg.phy_engine, "build_timeout_sec", 900)),
            )
            logger.info("Phy-Engine prebuild complete (verilog2plsav=%s, phyengine=%s)", v2p, lib)
        except (PhyEngineError, OSError) as e:
            logger.warning("Phy-Engine prebuild failed (will try on-demand later): %s", e)

    try:
        password = getpass.getpass(f"Password for {cfg.account.email}: ")
    except KeyboardInterrupt:
        logger.info("Cancelled.")
        return 130
    logger.info("Logging in...")
    try:
        user = email_login(
            email=cfg.account.email,
            password=password,
            cache_dir=cache_dir,
            http_timeout_sec=60.0,
        )
    except KeyboardInterrupt:
        logger.info("Login cancelled.")
        return 130
    except (PLARError, Exception) as e:
        logger.error("Login failed: %s", e)
        return 3
    nickname = getattr(user, "nickname", None)
    user_id = getattr(user, "user_id", None)
    display_name = (
        nickname.strip()
        if isinstance(nickname, str) and nickname.strip()
        else (user_id.strip() if isinstance(user_id, str) and user_id.strip() else cfg.account.email)
    )
    logger.info("Login successful: %s", display_name)
    if isinstance(user_id, str) and user_id.strip():
        logger.info("User ID: %s", user_id.strip())

    targets: list[TargetConfig] = []
    if cfg.agent.include_self_wall:
        user_id = getattr(user, "user_id", None)
        if isinstance(user_id, str) and user_id.strip():
            targets.append(TargetConfig(type="User", id=user_id.strip()))
        else:
            print("Warning: could not determine user_id; self wall target disabled")
    targets.extend(cfg.agent.targets)

    if not targets:
        logger.info("No targets configured. Nothing to do.")
        return 0

    dry_run = bool(args.dry_run or cfg.agent.dry_run)
    ollama = OllamaClient(
        base_url=cfg.ollama.base_url,
        model=cfg.ollama.model,
        timeout_sec=cfg.ollama.request_timeout_sec,
        temperature=cfg.ollama.temperature,
    )

    logger.info("Targets: %s", format_targets(targets))
    logger.info("Ollama: %s (model=%s)", cfg.ollama.base_url, cfg.ollama.model)
    logger.info("Mode: %s", "DRY RUN" if dry_run else "LIVE")
    logger.info(
        "Trigger: mention=%r require_mention=%s commands_enabled=%s prefix=%r",
        cfg.agent.mention_tag,
        cfg.agent.require_mention,
        cfg.agent.commands_enabled,
        cfg.agent.command_prefix,
    )
    logger.info("Agent is running. Press Ctrl-C to stop.")

    try:
        while True:
            logger.debug("Polling cycle start")
            for target in targets:
                _process_target(
                    user=user,
                    target=target,
                    state=state,
                    ollama=ollama,
                    cfg=cfg,
                    cache_dir=cache_dir,
                    config_base_dir=base_dir,
                    dry_run=dry_run,
                    logger=logger,
                )
            _process_notifications(
                user=user,
                state=state,
                ollama=ollama,
                cfg=cfg,
                cache_dir=cache_dir,
                config_base_dir=base_dir,
                dry_run=dry_run,
                logger=logger,
            )
            logger.debug("Polling cycle end")

            try:
                save_state(state_path, state)
            except OSError as e:
                logger.warning("Failed to save state to %s: %s", state_path, e)

            if args.once:
                return 0
            time.sleep(cfg.agent.poll_interval_sec)
    except KeyboardInterrupt:
        logger.info("Stopped.")
        return 0


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        init_config_interactive(args.config, overwrite=args.overwrite)
    except ConfigError as e:
        print(f"Failed to create config: {e}")
        return 2
    print(f"Wrote config to {args.config}")
    return 0


def _cmd_cleanup(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"Failed to load config: {e}")
        return 2

    base_dir = config_dir(args.config)
    cache_dir = pick_cache_dir(args.config, config=cfg)
    cache_dir = os.path.abspath(cache_dir)
    if os.path.commonpath([cache_dir, base_dir]) != os.path.abspath(base_dir):
        print(f"Refusing to delete cache outside config directory: {cache_dir}")
        return 2

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Deleted cache directory: {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")
    return 0


def _cmd_reset_state(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"Failed to load config: {e}")
        return 2

    state_path = args.state or pick_state_path(args.config)
    try:
        state = load_state(state_path)
    except (OSError, json.JSONDecodeError, StateError) as e:
        print(f"Failed to load state: {e}")
        return 2

    scope = (args.scope or "all").strip().lower()
    if scope not in ("all", "notifications", "targets", "conversations"):
        print("Invalid --scope. Use: all, notifications, targets, conversations")
        return 2

    removed: list[str] = []
    if scope in ("all", "notifications"):
        for k in list(state.targets.keys()):
            if k.startswith("Messages:"):
                removed.append(k)
                del state.targets[k]
    if scope in ("all", "targets"):
        for k in list(state.targets.keys()):
            if not k.startswith("Messages:"):
                removed.append(k)
                del state.targets[k]
    if scope in ("all", "conversations"):
        state.conversations = {}

    try:
        save_state(state_path, state)
    except OSError as e:
        print(f"Failed to save state: {e}")
        return 2

    print(f"Reset complete. Removed {len(removed)} target states. State path: {state_path}")
    if removed:
        print("Removed keys:")
        for k in removed:
            print(f"- {k}")
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"Failed to load config: {e}")
        return 2

    base_dir = config_dir(args.config)
    cache_dir = pick_cache_dir(args.config, config=cfg)
    os.makedirs(cache_dir, exist_ok=True)
    logger = _setup_logging(cache_dir=cache_dir, level=(args.log_level or "DEBUG"))
    logger.info("phy_lab diagnose starting (config=%s, cache_dir=%s)", os.path.abspath(args.config), cache_dir)

    try:
        password = getpass.getpass(f"Password for {cfg.account.email}: ")
    except KeyboardInterrupt:
        logger.info("Cancelled.")
        return 130

    logger.info("Logging in...")
    try:
        user = email_login(
            email=cfg.account.email,
            password=password,
            cache_dir=cache_dir,
            http_timeout_sec=60.0,
        )
    except KeyboardInterrupt:
        logger.info("Login cancelled.")
        return 130
    except (PLARError, Exception) as e:
        logger.error("Login failed: %s", e)
        return 3

    nickname = getattr(user, "nickname", None)
    user_id = getattr(user, "user_id", None)
    display_name = (
        nickname.strip()
        if isinstance(nickname, str) and nickname.strip()
        else (user_id.strip() if isinstance(user_id, str) and user_id.strip() else cfg.account.email)
    )
    logger.info("Login successful: %s", display_name)

    targets: list[TargetConfig] = []
    if cfg.agent.include_self_wall and isinstance(user_id, str) and user_id.strip():
        targets.append(TargetConfig(type="User", id=user_id.strip()))
    targets.extend(cfg.agent.targets)

    if not targets:
        logger.info("No targets configured.")
        return 0

    logger.info("Targets: %s", format_targets(targets))
    logger.info(
        "Trigger config: mention=%r require_mention=%s commands_enabled=%s prefix=%r",
        cfg.agent.mention_tag,
        cfg.agent.require_mention,
        cfg.agent.commands_enabled,
        cfg.agent.command_prefix,
    )

    for target in targets:
        target_key = f"{target.type}:{target.id}"
        logger.info("[%s] Fetching last %d comments...", target_key, int(args.take))
        try:
            comments = get_comments(
                user,
                target_id=target.id,
                target_type=target.type,
                take=int(args.take),
            )
        except Exception as e:
            logger.error("[%s] Fetch failed: %s", target_key, e)
            continue

        logger.info("[%s] Received %d comments", target_key, len(comments))
        for idx, c in enumerate(comments[: int(args.take)]):
            ts = _comment_timestamp_ms(c)
            key = _comment_key(c)
            author_id, nickname = _comment_author(c)
            content = _comment_content(c) or ""
            has_mention = contains_mention(content, mention_tag=cfg.agent.mention_tag)
            startswith_prefix = bool(
                cfg.agent.commands_enabled
                and cfg.agent.command_prefix
                and content.strip().startswith(cfg.agent.command_prefix)
            )
            triggered = _should_trigger(
                text=content,
                require_mention=cfg.agent.require_mention,
                mention_tag=cfg.agent.mention_tag,
                command_prefix=cfg.agent.command_prefix,
                commands_enabled=cfg.agent.commands_enabled,
                reply_to_self=False,
            )

            summary = f"#{idx+1} ts={ts} key={key} author={nickname or author_id} mention={has_mention} prefix={startswith_prefix} triggered={triggered}"
            logger.info("[%s] %s", target_key, summary)
            if args.include_content:
                logger.info("[%s] content=%r", target_key, truncate(content, max_chars=400))

    if getattr(cfg.agent, "notifications_enabled", True):
        cat_ids = list(getattr(cfg.agent, "notification_category_ids", [3]) or [3])
        take = int(getattr(cfg.agent, "notification_take", 20) or 20)
        for cat in cat_ids:
            msg_key = f"Messages:{int(cat)}"
            logger.info("[%s] Fetching last %d messages...", msg_key, take)
            try:
                messages, templates = get_messages(
                    user,
                    category_id=int(cat),
                    skip=0,
                    take=take,
                    no_templates=False,
                )
            except Exception as e:
                logger.error("[%s] Fetch failed: %s", msg_key, e)
                continue
            logger.info("[%s] Received %d messages (templates=%d)", msg_key, len(messages), len(templates))
            for idx, m in enumerate(messages[:take]):
                ts = _message_timestamp_ms(m)
                key = _message_key(m)
                tmpl_id = m.get("TemplateID") or m.get("TemplateId")
                tmpl = None
                if isinstance(tmpl_id, (str, int)):
                    for t0 in templates:
                        tid0 = t0.get("ID") or t0.get("Id")
                        if isinstance(tid0, (str, int)) and str(tid0) == str(tmpl_id):
                            tmpl = t0
                            break
                combined: dict[str, Any] = {"Message": m}
                if tmpl is not None:
                    combined["Template"] = tmpl
                t = _notification_target_from_message(combined)
                logger.info(
                    "[%s] #%d ts=%s key=%s target=%s",
                    msg_key,
                    idx + 1,
                    ts,
                    key,
                    f"{t.type}:{t.id}" if t else None,
                )
                if args.include_content:
                    logger.info("[%s] raw=%s", msg_key, truncate(_safe_json(m), max_chars=500))

    logger.info("Diagnose complete.")
    return 0


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=False, default=str)
    except Exception:
        return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phy_lab-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create a config file (interactive)")
    p_init.add_argument("--config", required=True, help="Path to write config JSON")
    p_init.add_argument(
        "--overwrite", action="store_true", help="Overwrite config if it exists"
    )
    p_init.set_defaults(func=_cmd_init)

    p_run = sub.add_parser("run", help="Run the auto-reply agent")
    p_run.add_argument("--config", required=True, help="Path to config JSON")
    p_run.add_argument(
        "--state", default=None, help="Path to state JSON (default: alongside config)"
    )
    p_run.add_argument("--once", action="store_true", help="Process once and exit")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not post replies (overrides config)",
    )
    p_run.add_argument(
        "--log-level",
        default=None,
        help="Console log level (DEBUG, INFO, WARNING, ERROR). Defaults to config agent.log_level.",
    )
    p_run.set_defaults(func=_cmd_run)

    p_cleanup = sub.add_parser("cleanup", help="Delete cache directory")
    p_cleanup.add_argument("--config", required=True, help="Path to config JSON")
    p_cleanup.set_defaults(func=_cmd_cleanup)

    p_reset = sub.add_parser("reset-state", help="Reset agent state (safe, local)")
    p_reset.add_argument("--config", required=True, help="Path to config JSON")
    p_reset.add_argument(
        "--state", default=None, help="Path to state JSON (default: alongside config)"
    )
    p_reset.add_argument(
        "--scope",
        default="all",
        help="What to reset: all, notifications, targets, conversations",
    )
    p_reset.set_defaults(func=_cmd_reset_state)

    p_diag = sub.add_parser("diagnose", help="Fetch comments and print trigger decisions")
    p_diag.add_argument("--config", required=True, help="Path to config JSON")
    p_diag.add_argument(
        "--take",
        default=10,
        type=int,
        help="How many latest comments to fetch per target",
    )
    p_diag.add_argument(
        "--include-content",
        action="store_true",
        help="Include comment content in logs (privacy sensitive)",
    )
    p_diag.add_argument(
        "--log-level",
        default=None,
        help="Console log level (DEBUG, INFO, WARNING, ERROR). Defaults to DEBUG for diagnose.",
    )
    p_diag.set_defaults(func=_cmd_diagnose)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
