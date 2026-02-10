from __future__ import annotations

import argparse
import dataclasses
import getpass
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from queue import Empty, Queue
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
from ollama import OllamaClient, OllamaError, OllamaPool  # noqa: E402
from phy_engine import (  # noqa: E402
    PhyEngineError,
    ensure_phyengine_lib,
    ensure_verilog2plsav,
    prebuild_phy_engine,
)
from plar import (  # noqa: E402
    PLARError,
    best_effort_extract_text,
    email_login,
    get_comments,
    get_experiment_context,
    get_messages,
    get_relations,
    get_user_by_id,
    get_user_by_name,
    get_status_save,
    post_comment,
    query_experiments,
    upload_sav_as_experiment,
)
from plsav import load_plsav_counts  # noqa: E402
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
    simulate_ai_verilog_with_phyengine,
    simulate_ai_circuit_with_phyengine,
    simulate_ai_script_circuit_with_phyengine,
    simulate_series_vdc_resistors,
    simulate_series_vdc_two_resistors,
    simulate_status_save_with_phyengine,
    web_search,
)


@dataclasses.dataclass(frozen=True)
class _WorkItem:
    target: TargetConfig
    comment: dict[str, Any]
    comment_key: str
    comment_ts_ms: int


class _PendingTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_target: dict[str, dict[str, int]] = {}

    def has(self, target_key: str, comment_key: str) -> bool:
        with self._lock:
            return comment_key in self._by_target.get(target_key, {})

    def add(self, target_key: str, comment_key: str, ts_ms: int) -> None:
        with self._lock:
            if target_key not in self._by_target:
                self._by_target[target_key] = {}
            self._by_target[target_key][comment_key] = int(ts_ms)

    def remove(self, target_key: str, comment_key: str) -> None:
        with self._lock:
            m = self._by_target.get(target_key)
            if not m:
                return
            m.pop(comment_key, None)
            if not m:
                self._by_target.pop(target_key, None)

    def min_ts(self, target_key: str) -> int | None:
        with self._lock:
            m = self._by_target.get(target_key)
            if not m:
                return None
            return min(m.values()) if m else None


class _LockedUser:
    """Serialize access to the PhysicsLab user/session object across threads."""

    def __init__(self, user: Any, lock: threading.RLock):
        self._user = user
        self._lock = lock

    def __getattr__(self, name: str) -> Any:  # pragma: no cover (thin proxy)
        attr = getattr(self._user, name)
        if not callable(attr):
            return attr

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return wrapped


def _safe_at_mention(handle: str) -> str | None:
    handle = (handle or "").strip()
    if not handle:
        return None
    if any(ch in handle for ch in (":", " ")):
        return None
    return f"@{handle}"


_AT_HANDLE_RE = re.compile(r"(?:@|＠)\s*([^\s:：]{1,64})")
_HEX24_RE = re.compile(r"^[0-9a-fA-F]{24}$")
_PUBLIC_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])(?:@|＠)\s*([A-Za-z0-9_\u4e00-\u9fff-]{1,32})")


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


def _strip_public_mentions(text: str) -> str:
    """Remove @mentions from model output to avoid pinging non-askers.

    This deliberately avoids matching email addresses or Verilog '@(' syntax.
    """
    s = (text or "").strip()
    if not s:
        return s
    s2 = _PUBLIC_MENTION_RE.sub("", s)
    # Clean up repeated whitespace created by removals.
    s2 = re.sub(r"[ \t]{2,}", " ", s2).strip()
    return s2


def _infer_publish_category(user_text: str, *, default_category: str) -> str:
    t = (user_text or "").lower()
    if any(x in t for x in ("discussion", "discuss", "black hole", "讨论", "黑洞", "交流")):
        return "Discussion"
    if any(x in t for x in ("experiment", "lab", "实验", "实验区")):
        return "Experiment"
    if default_category in ("Experiment", "Discussion"):
        return default_category
    return "Discussion"


def _format_required_by_prefix(*, nickname: str | None, user_id: str | None) -> str:
    nick = (nickname or "").strip()
    uid = (user_id or "").strip() or "unknown"
    if nick:
        # Force a stable @mention form without spaces/newlines.
        nick2 = re.sub(r"\s+", "", nick.lstrip("@＠"))
        who = f"@{nick2}" if nick2 else f"@{uid}"
    else:
        who = f"@{uid}"
    # Required format (no space between 'by' and '@').
    return f"required by {who} ({uid})\n\n"


def _llm_generate_publish_title_intro(
    *,
    ollama: OllamaClient,
    spec: str,
    max_title_chars: int = 60,
    max_intro_chars: int = 520,
) -> tuple[str, str] | None:
    """Ask the LLM for a human-friendly title and intro body (without the required-by line)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    prompt = (
        "Generate a good title and a short introduction for a Physics Lab AR discussion post.\n"
        "Rules:\n"
        "- Output STRICT JSON only: {\"title\":\"...\",\"introduction\":\"...\"}\n"
        "- Do NOT include any @mentions.\n"
        "- Do NOT include the requester line; it will be added by the system.\n"
        "- Keep it concise and readable.\n"
    )
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Spec:\n{spec}"},
    ]
    raw = ollama.chat(messages=msgs)
    obj = _try_parse_json_object(raw)
    if not obj:
        return None
    title = obj.get("title")
    intro = obj.get("introduction")
    if not isinstance(title, str) or not isinstance(intro, str):
        return None
    title = re.sub(r"[@＠][^\s]+", "", title).strip()
    intro = re.sub(r"[@＠][^\s]+", "", intro).strip()
    title = truncate(title, max_chars=max_title_chars) if title else ""
    intro = truncate(intro, max_chars=max_intro_chars) if intro else ""
    if not title or not intro:
        return None
    return title, intro


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

def _looks_like_content_intro_request(text: str) -> bool:
    """Heuristic: user wants an intro/summary of a specific work/content."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.casefold()
    asks_intro = any(
        x in low
        for x in (
            "introduce",
            "introduction",
            "介绍",
            "简介",
            "讲讲",
            "看看",
            "内容",
            "是什么",
        )
    )
    mentions_work = any(x in low for x in ("work", "作品", "实验", "讨论", "experiment", "discussion"))
    refers_specific = any(
        x in low for x in ("发布", "uid:", "user:", "experiment:", "discussion:", "@", "＠")
    ) or bool(_HEX24_RE.search(t.strip()))
    return bool(asks_intro and mentions_work and refers_specific)

def _looks_like_pe_script_parse_failure(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("I couldn't parse the command script") or t.startswith(
        "我没能把命令脚本解析成可仿真的电路"
    )

def _looks_like_series_demo_prompt(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("To simulate a series circuit") or t.startswith(
        "I can simulate it, but I need the VDC voltage value."
    ) or t.startswith("要进行串联直流仿真") or t.startswith("我可以仿真，但需要你给出 VDC")

def _looks_like_series_demo_request(text: str) -> bool:
    t = (text or "").casefold()
    return any(x in t for x in ("串联", "series", "resistor", "电阻", "ohm", "ω", "v="))

def _looks_like_plar_search_request(text: str) -> bool:
    """Heuristic: user explicitly wants an in-app/community search.

    Physics Lab internal search is best-effort and can return no results; keep this conservative.
    """
    t = (text or "").strip()
    if not t:
        return False

    low = t.casefold()

    # Explicit English intent.
    if any(
        low.startswith(x)
        for x in (
            "search ",
            "find ",
            "look for ",
            "looking for ",
            "lookup ",
        )
    ):
        return True
    if any(x in low for x in (" search ", " find ", " look for ", " looking for ")):
        return True

    # Explicit Chinese intent.
    if any(x in t for x in ("搜索", "查找", "搜一下", "搜下", "查下", "找一下", "帮我找", "想找")):
        return True
    if any(x in t for x in ("有没有", "推荐", "求", "哪里有")) and any(
        y in t for y in ("实验", "作品", "电路", "工程", "讨论")
    ):
        return True

    # Explicit user lookup patterns.
    if low.startswith(("user:", "user ", "用户:", "用户 ")):
        return True
    if low.startswith(("@", "＠")):
        return True
    if ("@" in t or "＠" in t) and any(x in low for x in ("who is", "id", "uid")):
        return True
    if ("@" in t or "＠" in t) and any(x in t for x in ("是谁", "谁是", "ID", "id", "用户")):
        return True

    return False

def _read_password_from_env(email: str) -> str | None:
    # Do not store passwords in config files. Use an env var if you need non-interactive runs.
    for key in ("PHY_LAB_PASSWORD", "PHYSICSLAB_PASSWORD"):
        v = os.environ.get(key)
        if isinstance(v, str) and v.strip():
            return v
    # Optional per-email override (avoid collisions when running multiple accounts).
    if isinstance(email, str) and email.strip():
        sanitized = email.strip().upper().replace("@", "_AT_").replace(".", "_")
        v = os.environ.get(f"PHY_LAB_PASSWORD_{sanitized}")
        if isinstance(v, str) and v.strip():
            return v
    return None


def _read_password(cfg: Any) -> str | None:
    email = getattr(getattr(cfg, "account", None), "email", "")
    pw = _read_password_from_env(str(email or ""))
    if pw is not None:
        return pw
    # Optional fallback for testing: allow account.password in config.json.
    cfg_pw = getattr(getattr(cfg, "account", None), "password", None)
    if isinstance(cfg_pw, str) and cfg_pw.strip():
        return cfg_pw.strip()
    return None


def _format_exception_brief(e: BaseException) -> str:
    msg = str(e) or e.__class__.__name__
    msg = msg.replace("\n", " ").strip()
    if len(msg) > 300:
        msg = msg[:299] + "…"
    return f"{e.__class__.__name__}: {msg}".rstrip(": ").strip()

def _effective_system_prompt(*, cfg: Any, user_text: str) -> str:
    base = str(getattr(getattr(cfg, "agent", None), "system_prompt", "") or "").rstrip()
    # Always reinforce reply scope; optionally enforce one-shot reply semantics.
    is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in (user_text or ""))
    one_shot = bool(getattr(getattr(cfg, "agent", None), "reply_once", True))
    if is_cjk:
        extra = (
            "关键约束（务必遵守）\n"
            "- 只回复当前提问者（当前这条评论的作者），不要面向其他人说话。\n"
            "- 不要 @ 提及任何其他用户（如需引用他人昵称/用户名，用普通文字即可，但不要 @）。\n"
            "- 请优先根据“当前这一条用户消息”作答；不要假设存在更早的对话或更早的问题。\n"
            "- 如果关键信息缺失/人名目标不明确：在这一条回复里最多问 1 个澄清问题；不要瞎猜。\n"
            "- 系统提示词中出现的开发者/项目名（例如 MacroModel）不是用户问题的默认对象；只有用户明确提到时才作为查询/回答对象。\n"
        )
        if one_shot:
            extra += "- 本次为一次性回复；回复后会直接关闭对话，你将不会再继续跟进。\n"
    else:
        extra = (
            "Critical constraints (must follow)\n"
            "- Reply ONLY to the author of the current comment.\n"
            "- Do NOT @mention any other users (you may refer to a user by plain text name if needed, but no @mentions).\n"
            "- Prioritize answering the CURRENT user message; do not assume earlier unseen conversation.\n"
            "- If essential info is missing or the target person/work is ambiguous, ask at most ONE short clarifying question; do not guess.\n"
            "- Developer/project names in the system prompt (e.g., MacroModel) are NOT the default subject unless the user explicitly asks about them.\n"
        )
        if one_shot:
            extra += "- This is a one-shot reply; after replying the conversation is closed and you will not follow up.\n"

    return (base + "\n\n" + extra).strip() if base else extra.strip()


def _is_conversation_closed(
    state: AgentState, *, key: str, now_ms: int, ttl_sec: int
) -> bool:
    if not key:
        return False
    # ttl_sec <= 0 means "no cooldown": do not block future replies.
    # reply_once is enforced per incoming comment anyway (we only post once per comment key).
    if int(ttl_sec) <= 0:
        return False
    ts = state.closed_conversations.get(key)
    if not isinstance(ts, int) or ts <= 0:
        return False
    return (int(now_ms) - int(ts)) < int(ttl_sec) * 1000


def _mark_conversation_closed(state: AgentState, *, key: str, now_ms: int) -> None:
    if not key:
        return
    state.closed_conversations[key] = int(now_ms)
    # "Close" by dropping memory for this conversation.
    state.conversations.pop(key, None)
    # Keep the map bounded.
    if len(state.closed_conversations) > 5000:
        items = sorted(state.closed_conversations.items(), key=lambda kv: int(kv[1] or 0))
        state.closed_conversations = dict(items[-4000:])


def _guess_simulation_issue_codes(lines: list[str]) -> list[str]:
    """Return stable, user-facing issue codes based on collected debug lines."""
    blob = "\n".join(lines).casefold()
    codes: list[str] = []
    if any(x in blob for x in ("connection refused", "failed to establish", "timeout", "timed out", "connectionerror")):
        codes.append("OLLAMA_UNREACHABLE_OR_TIMEOUT")
    if "empty 'message.content'" in blob:
        codes.append("OLLAMA_EMPTY_CONTENT")
    if any(x in blob for x in ("ensure_phyengine_lib", "cmake", "libphyengine", "phyengine_lib_path", "not found")):
        codes.append("PHYENGINE_LIB_MISSING_OR_BUILD_FAILED")
    if any(x in blob for x in ("i couldn't parse the command script", "命令脚本解析", "pe-script", "pescripterror")):
        codes.append("AI_SCRIPT_PARSE_FAILED")
    if any(x in blob for x in ("jsondecodeerror", "invalid json", "parse json", "strict json")):
        codes.append("AI_JSON_SPEC_INVALID")
    if any(x in blob for x in ("statussave", "plsav missing statussave", "get_experiment", "get_summary")):
        codes.append("STATUSSAVE_FETCH_FAILED")
    return codes[:4]


def _is_probable_user_not_found_error(e: BaseException) -> bool:
    msg = (str(e) or "").casefold()
    return ("status=404" in msg) or ("notfound" in msg) or ("not found" in msg)

def _is_probable_content_not_found_error(e: BaseException) -> bool:
    msg = (str(e) or "").casefold()
    return ("content.not.found" in msg) or ("status=404" in msg) or ("not found" in msg)

def _parse_plar_lookup_query(query: str) -> dict[str, str]:
    """Parse a restricted PLAR lookup query.

    Supported:
      - user name: @name, user:<name>, 用户:<name>
      - user id: uid:<id>, user_id:<id>, 用户id:<id>
      - content id: experiment:<id>, discussion:<id>, 实验:<id>, 讨论:<id>, or a bare 24-hex ID (treated as content id)

    Returns a dict with keys:
      - kind: user_name|user_id|content_id
      - value: normalized lookup key
      - category: Experiment|Discussion|both (only for content_id)
    """
    q = (query or "").strip()
    if not q:
        return {"kind": "user_name", "value": ""}

    # @user
    if q.startswith(("@", "＠")):
        return {"kind": "user_name", "value": q[1:].strip()}

    low = q.casefold()

    def _split_after_prefix(prefixes: tuple[str, ...]) -> str | None:
        for p in prefixes:
            if low.startswith(p):
                return q[len(p) :].strip()
        return None

    # user name
    name = _split_after_prefix(("user:", "user ", "用户:", "用户 ", "nickname:", "name:"))
    if name is not None:
        return {"kind": "user_name", "value": name}

    # user id
    uid = _split_after_prefix(("uid:", "uid ", "user_id:", "user_id ", "userid:", "用户id:", "用户id ", "用户id：", "用户id "))
    if uid is not None:
        return {"kind": "user_id", "value": uid}

    # content id (explicit)
    exp_id = _split_after_prefix(("experiment:", "experiment ", "exp:", "exp ", "实验:", "实验 ", "实验："))
    if exp_id is not None:
        return {"kind": "content_id", "value": exp_id, "category": "Experiment"}
    disc_id = _split_after_prefix(("discussion:", "discussion ", "disc:", "disc ", "讨论:", "讨论 ", "讨论："))
    if disc_id is not None:
        return {"kind": "content_id", "value": disc_id, "category": "Discussion"}

    # bare id (prefer treating as content id)
    if _HEX24_RE.match(q) is not None:
        return {"kind": "content_id", "value": q, "category": "both"}

    # Fallback: treat as user name (NOT keyword search).
    return {"kind": "user_name", "value": q}


def _render_simulation_failure(
    *,
    sim_text: str,
    cfg: Any,
    failures: list[str],
    ai_parse_failure: str | None = None,
) -> str:
    is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in (sim_text or ""))
    codes = _guess_simulation_issue_codes(failures + ([ai_parse_failure] if ai_parse_failure else []))
    header = "仿真失败（诊断信息）" if is_cjk else "Simulation failed (diagnostics)"

    cfg_lines = [
        f"ollama.base_url={getattr(getattr(cfg, 'ollama', None), 'base_url', '')}",
        f"ollama.model={getattr(getattr(cfg, 'ollama', None), 'model', '')}",
        f"agent.simulation_enabled={bool(getattr(getattr(cfg, 'agent', None), 'simulation_enabled', True))}",
        f"agent.simulation_ai_enabled={bool(getattr(getattr(cfg, 'agent', None), 'simulation_ai_enabled', True))}",
        f"phy_engine.auto_build={bool(getattr(getattr(cfg, 'phy_engine', None), 'auto_build', False))}",
        f"phy_engine.phyengine_lib_path={getattr(getattr(cfg, 'phy_engine', None), 'phyengine_lib_path', '')}",
        f"phy_engine.cmake_source_dir={getattr(getattr(cfg, 'phy_engine', None), 'cmake_source_dir', '')}",
        f"phy_engine.cmake_build_dir={getattr(getattr(cfg, 'phy_engine', None), 'cmake_build_dir', '')}",
    ]

    out: list[str] = [header]
    if codes:
        out.append(("- 可能问题编号: " if is_cjk else "- Possible issue codes: ") + ", ".join(codes))
    out.append(("- 配置快照:" if is_cjk else "- Config snapshot:"))
    out.extend([f"  - {x}" for x in cfg_lines if str(x).strip()])
    out.append(("- 尝试路径/错误:" if is_cjk else "- Attempts/errors:"))
    for ln in failures[:10]:
        out.append(f"  - {ln}")

    if ai_parse_failure:
        out.append(("- AI 构建电路失败细节（截断）:" if is_cjk else "- AI circuit build details (truncated):"))
        out.append(truncate(ai_parse_failure, max_chars=900))

    out.append(
        (
            "你可以直接把上面内容发给我，我会按“问题编号”继续定位。"
            if is_cjk
            else "Paste the diagnostics above and I’ll pinpoint the root cause by the issue codes."
        )
    )
    return "\n".join(out)


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


def _agent_parse_tool_call(raw: str) -> tuple[str, dict[str, Any], str]:
    """Parse a tool call JSON from the LLM.

    Returns (tool_name, args, final_text).
    - For non-end tools, final_text is "".
    - For tool=end, args may be empty and final_text carries the intended reply.
    """
    obj = _try_parse_json_object(raw)
    if not obj:
        raise ValueError("no JSON tool call found")

    tool = obj.get("tool") or obj.get("name") or obj.get("action")
    if not isinstance(tool, str) or not tool.strip():
        raise ValueError("missing tool name")
    tool = tool.strip()

    args = obj.get("args")
    if not isinstance(args, dict):
        args = {}

    # Normalize common model variants:
    # - tool names with a "tool_" prefix (e.g. "tool_list_plar")
    # - wrapper objects: {"tool":"list_plar","args":{"tool":"list_plar","args":{...}}}
    # - aliases used by other router modes (e.g. "google" -> "web_search")
    def _normalize_tool_name(name: str) -> str:
        t = (name or "").strip()
        # Common namespaces/prefixes some models invent.
        for pfx in ("tool.", "tools.", "tool:", "tool/"):
            if t.startswith(pfx):
                t = t[len(pfx) :].strip()
                break
        if t.startswith("tool_"):
            t = t[len("tool_") :].strip()
        aliases = {
            "plar_list_plar": "list_plar",
            "plar_search_plar": "search_plar",
            "plar_web_search": "web_search",
            # Back-compat: some prompts/models call these "raw API" helpers.
            # We expose only the "context/open" tools, so map to those.
            "plar_get_experiment": "plar_open_content_page",
            "plar_get_summary": "plar_open_content_page",
            "get_experiment": "plar_open_content_page",
            "get_summary": "plar_open_content_page",
            "google": "web_search",
            "web": "web_search",
            "websearch": "web_search",
        }
        return aliases.get(t, t)

    tool = _normalize_tool_name(tool)
    if isinstance(args.get("tool"), str) and args.get("tool") and tool not in (
        "end",
        "list_plar",
        "search_plar",
        "web_search",
        "plar_query_experiments",
        "plar_get_user_by_name",
        "plar_get_user_by_id",
        "plar_get_experiment_context",
        "plar_get_status_save",
        "plar_get_comments",
        "simulate",
        "simulate_verilog",
        "simulate_status_save",
        "circuit",
    ):
        tool2 = _normalize_tool_name(str(args.get("tool") or ""))
        if tool2:
            tool = tool2
    # Unwrap nested args at most once.
    if isinstance(args.get("args"), dict) and (
        set(args.keys()).issubset({"tool", "args", "name", "action"})
        or ("query" in args.get("args", {}) and "query" not in args)
    ):
        args = dict(args.get("args") or {})

    # Heuristic repairs for common model schema mixups.
    # Example: tool=plar_get_user_by_name but args look like list_plar (kind/category/user_id/take).
    if tool == "plar_get_user_by_name" and not str(args.get("name") or "").strip():
        if any(k in args for k in ("kind", "category", "take", "skip", "from", "days", "tags", "user_id")):
            tool = "list_plar"

    final = ""
    if tool == "end":
        for k in ("final", "answer", "content", "message", "text"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                final = v.strip()
                break
        if not final and isinstance(args, dict):
            for k in ("final", "answer", "content", "message", "text"):
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    final = v.strip()
                    break
    return tool, args, final


def _agent_tool_prompt(*, max_seconds: int) -> str:
    tool_cutoff_seconds = max(0, int(max_seconds) - 60)
    return (
        "You are running in AGENT MODE (multi-step tool use).\n"
        f"Time budget: {int(max_seconds)}s. Tool-call cutoff: after {tool_cutoff_seconds}s, only end is allowed.\n"
        "\n"
        "Output format (STRICT JSON only, no prose):\n"
        "- Tool call: {\"tool\":\"<name>\",\"args\":{...}}\n"
        "- Finish:    {\"tool\":\"end\",\"final\":\"...\"}\n"
        "\n"
        "Tools:\n"
        "- web_search {query}\n"
        "- search_plar {query}  (LOOKUP ONLY: @name / uid:... / experiment:<id> / discussion:<id>; NOT keyword search)\n"
        "- list_plar {kind, category?, user_id?, take?, skip?, from?, days?, tags?}\n"
        "- plar_get_user_by_name {name}\n"
        "- plar_get_user_by_id {user_id}\n"
        "- plar_open_content_page {summary_id, category:\"Experiment\"|\"Discussion\", take?, skip?}\n"
        "- plar_get_experiment_context {summary_id, category}\n"
        "- plar_get_status_save {summary_id, category}\n"
        "- plar_get_comments {target_type, target_id, take?, skip?}\n"
        "- plar_get_user_board {user_id, take?, skip?}\n"
        "- store_get {key, offset?, limit?} | store_json {key, path}\n"
        "- simulate {text} | simulate_verilog {text} | simulate_status_save {summary_id, category, question?}\n"
        "- circuit {spec, publish?}\n"
        "\n"
        "Rules:\n"
        "- Treat the CURRENT user message as the only task (no earlier unseen context).\n"
        "- Names in the system prompt (e.g., MacroModel) are NOT default targets unless the user explicitly asks.\n"
        "- For lists/discovery use list_plar; use search_plar only for lookup by name/id.\n"
        "- To describe a specific work, you MUST open it first and use ONLY opened Context JSON.\n"
        "- Final answers about a work MUST include Category + SummaryID + Subject.\n"
        "- If tool output is stored, use store_get/store_json.\n"
        "- Keep args small (<~500 chars). Never paste large blobs into args. Never produce political content.\n"
        "\n"
        "Examples:\n"
        "- (first work) {\"tool\":\"list_plar\",\"args\":{\"kind\":\"latest\",\"category\":\"both\",\"user_id\":\"<uid>\",\"take\":1}}\n"
        "- (open) {\"tool\":\"plar_open_content_page\",\"args\":{\"summary_id\":\"<24hex>\",\"category\":\"Experiment\",\"take\":20,\"skip\":0}}\n"
    )


def _shrink_context_json_for_llm(context_json: dict[str, Any]) -> dict[str, Any]:
    """Shrink noisy/huge context fields to reduce accidental echo in tool calls.

    The full context is still available to tools via the `context_json` parameter passed
    to `_agent_execute_tool`; this function only affects what we embed in the LLM prompt.
    """

    def _truncate_str(s: str, *, max_chars: int) -> str:
        s2 = (s or "").strip("\n")
        if len(s2) <= max_chars:
            return s2
        return s2[: max(0, max_chars - 40)] + f"...<truncated len={len(s2)}>"

    def _walk(obj: Any, *, depth: int) -> Any:
        if depth <= 0:
            return "<omitted: depth_limit>"
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            return _truncate_str(obj, max_chars=4000)
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                ks = str(k)
                if ks in ("summary_data_excerpt", "experiment_data_excerpt"):
                    if isinstance(v, str) and v:
                        out[ks] = f"<omitted: {len(v)} chars>"
                    else:
                        out[ks] = "<omitted>"
                    continue
                out[ks] = _walk(v, depth=depth - 1)
            return out
        if isinstance(obj, list):
            max_items = 50
            items = obj[:max_items]
            out_list = [_walk(x, depth=depth - 1) for x in items]
            if len(obj) > max_items:
                out_list.append(f"<omitted: {len(obj) - max_items} more items>")
            return out_list
        try:
            return _truncate_str(str(obj), max_chars=4000)
        except Exception:
            return "<unprintable>"

    if not isinstance(context_json, dict):
        return {"_context": "<invalid>"}
    shrunk = _walk(context_json, depth=6)
    return shrunk if isinstance(shrunk, dict) else {"_context": shrunk}


def _llm_store_root(cache_dir: str) -> str:
    root = os.path.join(str(cache_dir or "").strip() or ".", "llm_store")
    os.makedirs(root, exist_ok=True)
    return root


def _llm_store_key_is_safe(key: str) -> bool:
    k = str(key or "").strip()
    if not k:
        return False
    # Only allow hex-ish keys (timestamp + random).
    return bool(re.fullmatch(r"[0-9a-f]{16,64}", k))


def _llm_store_put(*, cache_dir: str, tool: str, content: str) -> str:
    root = _llm_store_root(cache_dir)
    # 24-hex-ish key; keep it filesystem-friendly.
    key = f"{int(time.time() * 1000):x}{secrets.token_hex(8)}"
    path = os.path.join(root, f"{key}.txt")
    meta_path = os.path.join(root, f"{key}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(content or ""))
    meta = {"tool": str(tool or ""), "ts_ms": int(time.time() * 1000), "bytes": len(str(content or "").encode("utf-8"))}
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception:
        pass
    return key


def _llm_store_read(*, cache_dir: str, key: str) -> str:
    if not _llm_store_key_is_safe(key):
        raise ValueError("invalid store key")
    root = _llm_store_root(cache_dir)
    path = os.path.join(root, f"{key}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _json_path_get(obj: Any, path: str) -> Any:
    """Very small JSON path helper: dot segments + optional [index]."""
    p = (path or "").strip()
    if not p:
        return obj
    cur: Any = obj
    # Split by dots, but keep bracket indices.
    for seg in p.split("."):
        seg = seg.strip()
        if not seg:
            continue
        # Handle a.b[0].c
        m = re.fullmatch(r"([A-Za-z0-9_-]+)(\[[0-9]+\])?", seg)
        if not m:
            raise ValueError("invalid path segment")
        key = m.group(1)
        idx = m.group(2)
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            raise KeyError(key)
        if idx:
            if not isinstance(cur, list):
                raise TypeError("not a list")
            i = int(idx.strip("[]"))
            cur = cur[i]
    return cur


def _agent_tool_result_message(*, tool: str, result: str, cache_dir: str) -> dict[str, str]:
    # Keep tool output bounded to avoid blowing up prompt size.
    #
    # If the tool output is large, store it and return a small pointer + summary
    # so the LLM can query it via store_get/store_json without flooding the prompt.
    max_inline_chars = 3500
    raw = str(result or "")
    stored_key = None
    if len(raw) > max_inline_chars:
        try:
            stored_key = _llm_store_put(cache_dir=cache_dir, tool=tool, content=raw)
        except Exception:
            stored_key = None

    if stored_key:
        summary_obj: Any = None
        if tool == "list_plar":
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    out: dict[str, Any] = {"kind": obj.get("kind"), "category": obj.get("category")}
                    for section in ("experiment", "discussion"):
                        sec = obj.get(section)
                        if not isinstance(sec, dict):
                            continue
                        sec_out: dict[str, Any] = {}
                        for k in (
                            "user_id",
                            "take",
                            "skip",
                            "from",
                            "sort",
                            "days",
                            "tags",
                            "next_skip",
                            "next_from",
                        ):
                            if k in sec:
                                sec_out[k] = sec.get(k)
                        items = sec.get("items")
                        if isinstance(items, list):
                            sec_out["items_total"] = len(items)
                            sec_out["items"] = items[:3]
                            if items and isinstance(items[0], dict):
                                sec_out["first"] = {
                                    "id": items[0].get("id"),
                                    "subject": items[0].get("subject"),
                                }
                        out[section] = sec_out
                    summary_obj = out
            except Exception:
                summary_obj = None
        if summary_obj is None:
            summary_obj = {"preview": truncate(raw, max_chars=800)}
        pointer = {
            "stored": True,
            "key": stored_key,
            "tool": tool,
            "hint": (
                "Use store_get {key, offset, limit} or store_json {key, path}. "
                "For list_plar first item: experiment.items[0].subject / experiment.items[0].id"
            )
            if tool == "list_plar"
            else "Use store_get {key, offset, limit} or store_json {key, path}.",
            "summary": summary_obj,
        }
        return {"role": "system", "content": f"Tool result ({tool}):\n" + json.dumps(pointer, ensure_ascii=False, indent=2)}

    return {
        "role": "system",
        "content": f"Tool result ({tool}):\n" + truncate(raw, max_chars=4000),
    }


def _agent_execute_tool(
    *,
    tool: str,
    args: dict[str, Any],
    user: Any,
    ollama: OllamaClient,
    cfg: Any,
    cache_dir: str,
    config_base_dir: str,
    dry_run: bool,
    context_json: dict[str, Any] | None,
    logger: logging.Logger,
    requester_nickname: str | None = None,
    requester_user_id: str | None = None,
) -> str:
    try:
        tool = (tool or "").strip()
        args = args or {}

        if tool == "web_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return "ERROR: missing args.query"
            if _looks_political_sensitive(query):
                return _political_refusal_message(query)
            if not bool(getattr(cfg.agent, "web_search_enabled", False)):
                return "ERROR: web_search disabled (set agent.web_search_enabled=true)"
            return web_search(
                query=query,
                cache_dir=cache_dir,
                provider=str(getattr(cfg.agent, "web_search_provider", "google") or "google"),
                proxy=str(getattr(cfg.agent, "web_search_proxy", "") or ""),
                timeout_sec=int(getattr(cfg.agent, "web_search_timeout_sec", 20) or 20),
                ttl_sec=int(getattr(cfg.agent, "web_search_cache_ttl_sec", 3600) or 3600),
                max_results=int(getattr(cfg.agent, "web_search_max_results", 5) or 5),
                fallback_to_ddg=bool(getattr(cfg.agent, "web_search_fallback_to_ddg", True)),
                user_agent=str(getattr(cfg.agent, "web_search_user_agent", "") or ""),
                searxng_base_url=str(getattr(cfg.agent, "web_search_searxng_base_url", "") or ""),
            )

        if tool == "search_plar":
            query_v: Any = args.get("query")
            if isinstance(query_v, dict):
                for k in ("query", "q", "text", "value", "name", "id"):
                    vv = query_v.get(k)
                    if isinstance(vv, str) and vv.strip():
                        query_v = vv
                        break
            if not isinstance(query_v, str):
                return "ERROR: args.query must be a string"
            query = query_v.strip()
            if not query:
                return "ERROR: missing args.query"
            if _looks_political_sensitive(query):
                return _political_refusal_message(query)
            spec = _parse_plar_lookup_query(query)
            kind = spec.get("kind")
            value = (spec.get("value") or "").strip()
            if kind == "user_name":
                if not value:
                    return "ERROR: empty user name"
                try:
                    data = get_user_by_name(user, name=value)
                except Exception as e:
                    if _is_probable_user_not_found_error(e):
                        return "No user found."
                    return f"Lookup failed: {_format_exception_brief(e)}"
                u = data.get("User") if isinstance(data, dict) else None
                if not isinstance(u, dict):
                    return "No user found."
                out = {
                    "type": "user",
                    "id": best_effort_extract_text(u.get("ID"))
                    or best_effort_extract_text(u.get("UserID"))
                    or None,
                    "nickname": best_effort_extract_text(u.get("Nickname")) or None,
                    "verification": best_effort_extract_text(u.get("Verification")) or None,
                    "signature": best_effort_extract_text(u.get("Signature")) or None,
                }
                return json.dumps(out, ensure_ascii=False, indent=2)

            if kind == "user_id":
                if not value:
                    return "ERROR: empty user_id"
                try:
                    data = get_user_by_id(user, user_id=value)
                except Exception as e:
                    if _is_probable_user_not_found_error(e):
                        return "No user found."
                    return f"Lookup failed: {_format_exception_brief(e)}"
                u = data.get("User") if isinstance(data, dict) else None
                if not isinstance(u, dict):
                    return "No user found."
                out = {
                    "type": "user",
                    "id": best_effort_extract_text(u.get("ID"))
                    or best_effort_extract_text(u.get("UserID"))
                    or None,
                    "nickname": best_effort_extract_text(u.get("Nickname")) or None,
                    "verification": best_effort_extract_text(u.get("Verification")) or None,
                    "signature": best_effort_extract_text(u.get("Signature")) or None,
                }
                return json.dumps(out, ensure_ascii=False, indent=2)

            if kind == "content_id":
                if not value:
                    return "ERROR: empty content id"
                category = (spec.get("category") or "both").strip()
                tried: list[str] = []
                last_err: BaseException | None = None

                def _try(cat: str) -> dict[str, Any] | None:
                    nonlocal last_err
                    tried.append(cat)
                    try:
                        ctx = get_experiment_context(
                            user,
                            summary_id=value,
                            category_value=cat,
                            cache_dir=cache_dir,
                            ttl_sec=300,
                        )
                    except Exception as e:
                        last_err = e
                        return None
                    if not isinstance(ctx, dict):
                        return None
                    subj = best_effort_extract_text(ctx.get("subject") or ctx.get("Subject") or ctx.get("title"))
                    return {
                        "type": "content",
                        "id": value,
                        "category": cat,
                        "subject": subj or None,
                    }

                if category == "Experiment":
                    hit = _try("Experiment")
                    if hit is not None:
                        return json.dumps(hit, ensure_ascii=False, indent=2)
                elif category == "Discussion":
                    hit = _try("Discussion")
                    if hit is not None:
                        return json.dumps(hit, ensure_ascii=False, indent=2)
                else:
                    hit = _try("Experiment")
                    if hit is not None:
                        return json.dumps(hit, ensure_ascii=False, indent=2)
                    hit = _try("Discussion")
                    if hit is not None:
                        return json.dumps(hit, ensure_ascii=False, indent=2)

                if last_err is not None and _is_probable_content_not_found_error(last_err):
                    return json.dumps(
                        {"type": "content", "id": value, "found": False, "tried": tried},
                        ensure_ascii=False,
                        indent=2,
                    )
                return f"Lookup failed: {_format_exception_brief(last_err) if last_err is not None else 'unknown error'}"

            return "ERROR: unsupported search_plar query (lookup only)"

        if tool == "list_plar":
            kind = str(args.get("kind") or "").strip().lower()
            if not kind:
                return "ERROR: missing args.kind"

            def _parse_tags(v: Any) -> list[str] | None:
                if v is None:
                    return None
                if isinstance(v, list):
                    out = []
                    for x in v:
                        s = str(x or "").strip()
                        if s:
                            out.append(s)
                    return out
                s = str(v or "").strip()
                if not s:
                    return None
                if "," in s:
                    return [p.strip() for p in s.split(",") if p.strip()]
                return [s]

            def _compact(items: list[dict[str, Any]]) -> dict[str, Any]:
                compact: list[dict[str, Any]] = []
                last_id: str | None = None
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    sid = best_effort_extract_text(it.get("ID")) or best_effort_extract_text(it.get("Id"))
                    if sid:
                        last_id = sid
                    compact.append(
                        {
                            "id": sid,
                            "category": best_effort_extract_text(it.get("Category")) or None,
                            "subject": best_effort_extract_text(it.get("Subject"))
                            or best_effort_extract_text(it.get("Title"))
                            or None,
                            "user_id": best_effort_extract_text(it.get("UserID")) or None,
                            "tags": it.get("Tags") if isinstance(it.get("Tags"), list) else None,
                        }
                    )
                return {"items": compact, "next_from": last_id}

            # QueryExperiments-backed lists (plweb2-aligned).
            if kind in (
                "latest",
                "latest_experiments",
                "new",
                "recent",
                "hot",
                "popular",
                "hot_experiments",
                "popular_experiments",
                "featured",
                "精选",
                "random",
            ):
                category = str(args.get("category") or "").strip()
                if not category:
                    category = "Discussion" if kind in ("featured", "精选") else "Experiment"
                take = int(args.get("take") or 10)
                skip = int(args.get("skip") or 0)
                from_id = str(args.get("from") or args.get("from_skip") or "").strip() or None
                days = args.get("days")
                days_i = None
                if days is not None:
                    try:
                        days_i = int(days)
                    except Exception:
                        days_i = None
                if take < 1:
                    take = 10
                if take > 24:
                    take = 24
                if skip < 0:
                    skip = 0

                user_id = str(args.get("user_id") or "").strip() or None
                tags = _parse_tags(args.get("tags"))
                if tags is None:
                    tags = []
                sort: int | str | None
                if kind in ("hot", "popular", "hot_experiments", "popular_experiments"):
                    sort = "Popularity"
                elif kind in ("random",):
                    sort = "Random"
                else:
                    sort = "Default"

                if kind in ("featured", "精选") and not tags:
                    tags = ["精选"]

                def _query(cat: str) -> dict[str, Any]:
                    items = query_experiments(
                        user,
                        category=cat,
                        take=take,
                        skip=skip,
                        from_skip=from_id,
                        days=days_i,
                        sort=sort,
                        user_id=user_id,
                        tags=tags,
                    )
                    obj = _compact(items)
                    if user_id and isinstance(obj.get("items"), list):
                        for it in obj["items"]:
                            if isinstance(it, dict) and not it.get("user_id"):
                                it["user_id"] = user_id
                    obj.update({"category": cat, "take": take, "skip": skip, "from": from_id, "sort": sort, "days": days_i, "user_id": user_id, "tags": tags})
                    # plweb2 pagination typically uses skip += Take and from = last.ID
                    obj["next_skip"] = skip + min(len(obj.get("items") or []), take)
                    return obj

                if category.lower() in ("both", "all", "*"):
                    out = {"kind": kind, "category": "both", "experiment": _query("Experiment"), "discussion": _query("Discussion")}
                    return json.dumps(out, ensure_ascii=False, indent=2)

                out = {"kind": kind, **_query(category)}
                return json.dumps(out, ensure_ascii=False, indent=2)

            # Relations lists.
            if kind in ("following", "关注", "followings"):
                uid = str(args.get("user_id") or getattr(user, "user_id", "") or "").strip()
                if not uid:
                    return "ERROR: missing user_id and current user_id is unavailable"
                take = int(args.get("take") or 50)
                skip = int(args.get("skip") or 0)
                query = str(args.get("query") or "").strip()
                users = get_relations(user, user_id=uid, display_type=1, skip=skip, take=take, query=query)
                compact = []
                for u in users[:take]:
                    u2 = u.get("User") if isinstance(u, dict) and isinstance(u.get("User"), dict) else u
                    compact.append(
                        {
                            "id": best_effort_extract_text(u2.get("ID")) or best_effort_extract_text(u2.get("UserID")) or None,
                            "nickname": best_effort_extract_text(u2.get("Nickname")) or best_effort_extract_text(u2.get("Name")) or None,
                            "verification": best_effort_extract_text(u2.get("Verification")) or None,
                        }
                    )
                return json.dumps(compact, ensure_ascii=False, indent=2)

            if kind in ("followers", "粉丝", "follower"):
                uid = str(args.get("user_id") or getattr(user, "user_id", "") or "").strip()
                if not uid:
                    return "ERROR: missing user_id and current user_id is unavailable"
                take = int(args.get("take") or 50)
                skip = int(args.get("skip") or 0)
                query = str(args.get("query") or "").strip()
                users = get_relations(user, user_id=uid, display_type=0, skip=skip, take=take, query=query)
                compact = []
                for u in users[:take]:
                    u2 = u.get("User") if isinstance(u, dict) and isinstance(u.get("User"), dict) else u
                    compact.append(
                        {
                            "id": best_effort_extract_text(u2.get("ID")) or best_effort_extract_text(u2.get("UserID")) or None,
                            "nickname": best_effort_extract_text(u2.get("Nickname")) or best_effort_extract_text(u2.get("Name")) or None,
                            "verification": best_effort_extract_text(u2.get("Verification")) or None,
                        }
                    )
                return json.dumps(compact, ensure_ascii=False, indent=2)

            if kind in ("banned", "baned", "小黑屋", "blocked"):
                uid = str(args.get("user_id") or getattr(user, "user_id", "") or "").strip()
                if not uid:
                    return "ERROR: missing user_id and current user_id is unavailable"
                take = int(args.get("take") or 50)
                skip = int(args.get("skip") or 0)
                query = str(args.get("query") or "").strip()
                users = get_relations(user, user_id=uid, display_type=2, skip=skip, take=take, query=query)
                compact = []
                for u in users[:take]:
                    u2 = u.get("User") if isinstance(u, dict) and isinstance(u.get("User"), dict) else u
                    compact.append(
                        {
                            "id": best_effort_extract_text(u2.get("ID")) or best_effort_extract_text(u2.get("UserID")) or None,
                            "nickname": best_effort_extract_text(u2.get("Nickname")) or best_effort_extract_text(u2.get("Name")) or None,
                            "verification": best_effort_extract_text(u2.get("Verification")) or None,
                        }
                    )
                return json.dumps(compact, ensure_ascii=False, indent=2)

            if kind in ("volunteers", "volunteer", "志愿者", "义工"):
                uid = str(args.get("user_id") or getattr(user, "user_id", "") or "").strip()
                if not uid:
                    return "ERROR: missing user_id and current user_id is unavailable"
                take = int(args.get("take") or 80)
                skip = int(args.get("skip") or 0)
                query = str(args.get("query") or "").strip()
                users = get_relations(user, user_id=uid, display_type=3, skip=skip, take=take, query=query)
                compact = []
                for u in users[:take]:
                    u2 = u.get("User") if isinstance(u, dict) and isinstance(u.get("User"), dict) else u
                    compact.append(
                        {
                            "id": best_effort_extract_text(u2.get("ID")) or best_effort_extract_text(u2.get("UserID")) or None,
                            "nickname": best_effort_extract_text(u2.get("Nickname")) or best_effort_extract_text(u2.get("Name")) or None,
                            "verification": best_effort_extract_text(u2.get("Verification")) or None,
                        }
                    )
                return json.dumps(compact, ensure_ascii=False, indent=2)

            if kind in ("editors", "editor", "admins", "admin", "administrator", "administrators", "编辑", "管理员", "编辑和管理员"):
                uid = str(args.get("user_id") or getattr(user, "user_id", "") or "").strip()
                if not uid:
                    return "ERROR: missing user_id and current user_id is unavailable"
                take = int(args.get("take") or 80)
                skip = int(args.get("skip") or 0)
                query = str(args.get("query") or "").strip()
                users = get_relations(user, user_id=uid, display_type=4, skip=skip, take=take, query=query)
                compact = []
                for u in users[:take]:
                    u2 = u.get("User") if isinstance(u, dict) and isinstance(u.get("User"), dict) else u
                    compact.append(
                        {
                            "id": best_effort_extract_text(u2.get("ID")) or best_effort_extract_text(u2.get("UserID")) or None,
                            "nickname": best_effort_extract_text(u2.get("Nickname")) or best_effort_extract_text(u2.get("Name")) or None,
                            "verification": best_effort_extract_text(u2.get("Verification")) or None,
                        }
                    )
                return json.dumps(compact, ensure_ascii=False, indent=2)

            if kind in ("retired", "emeritus", "荣休", "退休", "荣誉"):
                uid = str(args.get("user_id") or getattr(user, "user_id", "") or "").strip()
                if not uid:
                    return "ERROR: missing user_id and current user_id is unavailable"
                take = int(args.get("take") or 80)
                skip = int(args.get("skip") or 0)
                query = str(args.get("query") or "").strip()
                users = get_relations(user, user_id=uid, display_type=5, skip=skip, take=take, query=query)
                compact = []
                for u in users[:take]:
                    u2 = u.get("User") if isinstance(u, dict) and isinstance(u.get("User"), dict) else u
                    compact.append(
                        {
                            "id": best_effort_extract_text(u2.get("ID")) or best_effort_extract_text(u2.get("UserID")) or None,
                            "nickname": best_effort_extract_text(u2.get("Nickname")) or best_effort_extract_text(u2.get("Name")) or None,
                            "verification": best_effort_extract_text(u2.get("Verification")) or None,
                        }
                    )
                return json.dumps(compact, ensure_ascii=False, indent=2)

            if kind in ("staff",):
                uid = str(args.get("user_id") or getattr(user, "user_id", "") or "").strip()
                if not uid:
                    return "ERROR: missing user_id and current user_id is unavailable"
                take = int(args.get("take") or 80)
                users = get_relations(user, user_id=uid, display_type=1, skip=0, take=take, query="")
                staff = []
                for u in users:
                    u2 = u.get("User") if isinstance(u, dict) and isinstance(u.get("User"), dict) else u
                    ver = best_effort_extract_text(u2.get("Verification"))
                    if ver in ("Volunteer", "Editor", "Emeritus", "Administrator"):
                        staff.append(
                            {
                                "id": best_effort_extract_text(u2.get("ID")) or best_effort_extract_text(u2.get("UserID")) or None,
                                "nickname": best_effort_extract_text(u2.get("Nickname")) or best_effort_extract_text(u2.get("Name")) or None,
                                "verification": ver,
                            }
                        )
                return json.dumps(staff, ensure_ascii=False, indent=2)

            return "ERROR: unknown list_plar kind (try: latest|hot|featured|random|following|followers|banned|volunteers|editors|retired|staff)"

        if tool == "plar_get_user_board":
            user_id = str(args.get("user_id") or "").strip()
            if not user_id:
                return "ERROR: missing args.user_id"
            take = int(args.get("take") or 20)
            skip = int(args.get("skip") or 0)
            if take < 1:
                take = 20
            if take > 50:
                take = 50
            if skip < 0:
                skip = 0
            comments = get_comments(user, target_id=user_id, target_type="User", take=take, skip=skip)
            ctx = _build_user_board_context(user=user, board_user_id=user_id, comments=comments)
            ctx["paging"] = {"take": take, "skip": skip}
            return json.dumps(ctx, ensure_ascii=False, indent=2)

        if tool == "plar_open_content_page":
            summary_id = str(args.get("summary_id") or "").strip()
            category = str(args.get("category") or "").strip() or "Experiment"
            if not summary_id:
                return "ERROR: missing args.summary_id"
            take = int(args.get("take") or 20)
            skip = int(args.get("skip") or 0)
            if take < 1:
                take = 20
            if take > 50:
                take = 50
            if skip < 0:
                skip = 0
            comments = []
            try:
                comments = get_comments(
                    user,
                    target_id=summary_id,
                    target_type="Experiment" if category == "Experiment" else "Discussion",
                    take=take,
                    skip=skip,
                )
            except Exception:
                comments = []
            try:
                ctx = get_experiment_context(
                    user,
                    summary_id=summary_id,
                    category_value="Experiment" if category == "Experiment" else "Discussion",
                    cache_dir=cache_dir,
                    ttl_sec=300,
                    max_json_chars=20_000,
                )
            except Exception as e:
                # Make errors machine-actionable for the agent loop.
                msg_low = (str(e) or "").casefold()
                if isinstance(e, PermissionError) and "login failed" in msg_low:
                    return json.dumps(
                        {"error": "auth_failed", "summary_id": summary_id, "category": category},
                        ensure_ascii=False,
                        indent=2,
                    )
                if _is_probable_content_not_found_error(e):
                    return json.dumps(
                        {"found": False, "summary_id": summary_id, "category": category},
                        ensure_ascii=False,
                        indent=2,
                    )
                return f"ERROR: {_format_exception_brief(e)}"
            obj = dict(ctx or {})
            obj["recent_comments"] = _recent_comments_context(comments, limit=8)
            obj["paging"] = {"take": take, "skip": skip}
            return json.dumps(obj, ensure_ascii=False, indent=2)

        if tool == "plar_query_experiments":
            category = str(args.get("category") or "").strip() or "Experiment"
            take = int(args.get("take") or 20)
            skip = int(args.get("skip") or 0)
            days = args.get("days")
            sort = str(args.get("sort") or "").strip() or None
            days_i = None
            if days is not None:
                try:
                    days_i = int(days)
                except Exception:
                    days_i = None
            if take < 1:
                take = 1
            if take > 50:
                take = 50
            if skip < 0:
                skip = 0
            items = query_experiments(
                user,
                category=category,
                take=take,
                skip=skip,
                from_skip=None,
                days=days_i,
                sort=sort,
            )
            compact: list[dict[str, Any]] = []
            for it in items[: min(len(items), 30)]:
                if not isinstance(it, dict):
                    continue
                compact.append(
                    {
                        "id": best_effort_extract_text(it.get("ID"))
                        or best_effort_extract_text(it.get("Id"))
                        or None,
                        "category": best_effort_extract_text(it.get("Category")) or None,
                        "subject": best_effort_extract_text(it.get("Subject"))
                        or best_effort_extract_text(it.get("Title"))
                        or None,
                    }
                )
            return json.dumps(compact, ensure_ascii=False, indent=2)

        if tool == "plar_get_user_by_name":
            name_v: Any = args.get("name")
            if isinstance(name_v, dict):
                # Some models may emit {"name": {"value": "..."} } in JSON mode.
                for k in ("name", "value", "text", "query"):
                    vv = name_v.get(k)
                    if isinstance(vv, str) and vv.strip():
                        name_v = vv
                        break
            if not isinstance(name_v, str):
                return "ERROR: args.name must be a string"
            name = name_v.strip()
            if not name:
                return "ERROR: missing args.name"
            data = get_user_by_name(user, name=name)
            u = data.get("User") if isinstance(data, dict) else None
            if not isinstance(u, dict):
                return "No user found."
            out = {
                "id": best_effort_extract_text(u.get("ID"))
                or best_effort_extract_text(u.get("UserID"))
                or None,
                "nickname": best_effort_extract_text(u.get("Nickname")) or None,
                "verification": best_effort_extract_text(u.get("Verification")) or None,
                "signature": truncate(
                    best_effort_extract_text(u.get("Signature")), max_chars=200
                )
                or None,
            }
            return json.dumps(out, ensure_ascii=False, indent=2)

        if tool == "plar_get_user_by_id":
            user_id_v: Any = args.get("user_id")
            if isinstance(user_id_v, dict):
                for k in ("user_id", "id", "value", "text"):
                    vv = user_id_v.get(k)
                    if isinstance(vv, str) and vv.strip():
                        user_id_v = vv
                        break
            if not isinstance(user_id_v, str):
                return "ERROR: args.user_id must be a string"
            user_id = user_id_v.strip()
            if not user_id:
                return "ERROR: missing args.user_id"
            data = get_user_by_id(user, user_id=user_id)
            u = data.get("User") if isinstance(data, dict) else None
            if not isinstance(u, dict):
                return "No user found."
            out = {
                "id": best_effort_extract_text(u.get("ID"))
                or best_effort_extract_text(u.get("UserID"))
                or None,
                "nickname": best_effort_extract_text(u.get("Nickname")) or None,
                "verification": best_effort_extract_text(u.get("Verification")) or None,
                "signature": truncate(
                    best_effort_extract_text(u.get("Signature")), max_chars=200
                )
                or None,
            }
            return json.dumps(out, ensure_ascii=False, indent=2)

        if tool == "plar_get_experiment_context":
            summary_id = str(args.get("summary_id") or "").strip()
            category = str(args.get("category") or "").strip() or "Experiment"
            if not summary_id:
                return "ERROR: missing args.summary_id"
            try:
                ctx = get_experiment_context(
                    user,
                    summary_id=summary_id,
                    category_value=category,
                    cache_dir=cache_dir,
                    ttl_sec=300,
                    max_json_chars=20_000,
                )
            except Exception as e:
                msg_low = (str(e) or "").casefold()
                if isinstance(e, PermissionError) and "login failed" in msg_low:
                    return json.dumps(
                        {"error": "auth_failed", "summary_id": summary_id, "category": category},
                        ensure_ascii=False,
                        indent=2,
                    )
                if _is_probable_content_not_found_error(e):
                    return json.dumps(
                        {"found": False, "summary_id": summary_id, "category": category},
                        ensure_ascii=False,
                        indent=2,
                    )
                return f"ERROR: {_format_exception_brief(e)}"
            return json.dumps(ctx, ensure_ascii=False, indent=2)

        if tool == "plar_get_status_save":
            summary_id = str(args.get("summary_id") or "").strip()
            category = str(args.get("category") or "").strip() or "Experiment"
            if not summary_id:
                return "ERROR: missing args.summary_id"
            status = get_status_save(
                user,
                summary_id=summary_id,
                category_value=category,
                cache_dir=cache_dir,
                ttl_sec=300,
            )
            # Do not dump the whole thing: it's huge.
            els = status.get("Elements")
            wires = status.get("Wires")
            sample: list[dict[str, Any]] = []
            if isinstance(els, list):
                for el in els[:30]:
                    if not isinstance(el, dict):
                        continue
                    sample.append(
                        {
                            "ModelID": best_effort_extract_text(el.get("ModelID")) or None,
                            "Label": best_effort_extract_text(el.get("Label")) or None,
                        }
                    )
            return json.dumps(
                {
                    "elements_count": len(els) if isinstance(els, list) else None,
                    "wires_count": len(wires) if isinstance(wires, list) else None,
                    "elements_sample": sample,
                },
                ensure_ascii=False,
                indent=2,
            )

        if tool == "plar_get_comments":
            target_type = str(args.get("target_type") or "").strip() or "Discussion"
            target_id = str(args.get("target_id") or "").strip()
            if not target_id:
                return "ERROR: missing args.target_id"
            take = int(args.get("take") or 20)
            skip = int(args.get("skip") or 0)
            if take < 1:
                take = 1
            if take > 50:
                take = 50
            if skip < 0:
                skip = 0
            comments = get_comments(
                user, target_id=target_id, target_type=target_type, take=take, skip=skip
            )
            return json.dumps(
                _recent_comments_context(comments, limit=12),
                ensure_ascii=False,
                indent=2,
            )

        if tool == "store_get":
            key = str(args.get("key") or "").strip()
            if not key:
                return "ERROR: missing args.key"
            offset = int(args.get("offset") or 0)
            limit = int(args.get("limit") or 1500)
            if offset < 0:
                offset = 0
            if limit <= 0:
                limit = 1500
            if limit > 4000:
                limit = 4000
            try:
                data = _llm_store_read(cache_dir=cache_dir, key=key)
            except Exception as e:
                return f"ERROR: store_get failed: {_format_exception_brief(e)}"
            chunk = data[offset : offset + limit]
            return json.dumps(
                {
                    "key": key,
                    "offset": offset,
                    "limit": limit,
                    "total_len": len(data),
                    "text": chunk,
                    "next_offset": offset + len(chunk),
                },
                ensure_ascii=False,
                indent=2,
            )

        if tool == "store_json":
            key = str(args.get("key") or "").strip()
            path = str(args.get("path") or "").strip()
            if not key:
                return "ERROR: missing args.key"
            if not path:
                return "ERROR: missing args.path"
            try:
                data = _llm_store_read(cache_dir=cache_dir, key=key)
                obj = json.loads(data)
                val = _json_path_get(obj, path)
            except Exception as e:
                return f"ERROR: store_json failed: {_format_exception_brief(e)}"
            out = json.dumps(val, ensure_ascii=False, indent=2)
            if len(out) > 4000:
                out = out[:3960] + f"...<truncated len={len(out)}>"
            return out

        if tool == "simulate":
            text = str(args.get("text") or "").strip()
            if not text:
                text = json.dumps(context_json, ensure_ascii=False) if context_json else ""
            if not text:
                return "ERROR: missing args.text"
            if bool(getattr(cfg.agent, "simulation_ai_enabled", True)):
                try:
                    reply = simulate_ai_script_circuit_with_phyengine(
                        ollama=ollama,
                        text=text,
                        context_json=context_json,
                        phy_engine_cfg=cfg.phy_engine,
                        config_base_dir=config_base_dir,
                        max_components=int(
                            getattr(cfg.agent, "simulation_ai_max_components", 30) or 30
                        ),
                        max_probes=int(getattr(cfg.agent, "simulation_ai_max_probes", 20) or 20),
                    )
                    if _looks_like_pe_script_parse_failure(reply):
                        reply = simulate_ai_circuit_with_phyengine(
                            ollama=ollama,
                            text=text,
                            context_json=context_json,
                            phy_engine_cfg=cfg.phy_engine,
                            config_base_dir=config_base_dir,
                            max_attempts=3,
                            max_components=int(
                                getattr(cfg.agent, "simulation_ai_max_components", 30) or 30
                            ),
                            max_probes=int(
                                getattr(cfg.agent, "simulation_ai_max_probes", 20) or 20
                            ),
                        )
                    return reply
                except Exception as e:
                    logger.info("Agent tool simulate failed; falling back to demo: %s", e)
            return simulate_series_vdc_resistors(
                text=text,
                phy_engine_cfg=cfg.phy_engine,
                config_base_dir=config_base_dir,
            )

        if tool == "simulate_verilog":
            text = str(args.get("text") or "").strip()
            if not text:
                return "ERROR: missing args.text"
            return simulate_ai_verilog_with_phyengine(
                ollama=ollama,
                text=text,
                context_json=context_json,
                phy_engine_cfg=cfg.phy_engine,
                config_base_dir=config_base_dir,
                cache_dir=cache_dir,
                max_attempts=2,
                max_elements=int(getattr(cfg.agent, "simulation_max_elements", 300) or 300),
            )

        if tool == "simulate_status_save":
            summary_id = str(args.get("summary_id") or "").strip()
            category = str(args.get("category") or "").strip() or "Experiment"
            question = str(args.get("question") or "").strip() or "simulate"
            if not summary_id:
                return "ERROR: missing args.summary_id"
            status = get_status_save(
                user,
                summary_id=summary_id,
                category_value=category,
                cache_dir=cache_dir,
                ttl_sec=300,
            )
            return simulate_status_save_with_phyengine(
                text=question,
                status_save=status,
                phy_engine_cfg=cfg.phy_engine,
                config_base_dir=config_base_dir,
                max_elements=int(getattr(cfg.agent, "simulation_max_elements", 300) or 300),
            )

        if tool == "circuit":
            spec = str(args.get("spec") or "").strip()
            if not spec:
                return "ERROR: missing args.spec"
            publish_wanted = bool(args.get("publish"))
            enable_publish_run = (
                bool(getattr(cfg.agent, "enable_publish", False))
                and bool(getattr(cfg.agent, "auto_publish", False))
                and publish_wanted
                and (not dry_run)
            )
            # Community rule: always publish to Discussion.
            publish_category_value = "Discussion"

            required_by = _format_required_by_prefix(
                nickname=requester_nickname,
                user_id=requester_user_id,
            )
            meta = _llm_generate_publish_title_intro(ollama=ollama, spec=spec) if enable_publish_run else None
            if meta is not None:
                title, intro_body = meta
            else:
                title = truncate(f"Auto Circuit: {spec}", max_chars=60)
                intro_body = truncate(f"Spec:\n{spec}", max_chars=520)
            introduction = truncate(required_by + "\n" + intro_body, max_chars=600)
            res = build_and_maybe_publish_circuit(
                ollama=ollama,
                user=user,
                spec=spec,
                cache_dir=cache_dir,
                phy_engine_cfg=cfg.phy_engine,
                config_base_dir=config_base_dir,
                keep_temp=bool(getattr(cfg.storage, "keep_temp", False)),
                enable_publish=bool(enable_publish_run),
                dry_run=dry_run,
                max_attempts=int(getattr(cfg.agent, "circuit_max_attempts", 3) or 3),
                publish_max_elements=int(getattr(cfg.agent, "publish_max_elements", 5000) or 5000),
                title=title,
                introduction=introduction,
                publish_category_value=publish_category_value,
                publish_tags=list(getattr(cfg.agent, "publish_tags", []) or []),
            )
            category = (
                str(getattr(res, "category", "") or "").strip()
                or publish_category_value.strip()
            )
            summary_id = getattr(res, "summary_id", None)
            open_hint = None
            if isinstance(summary_id, str) and summary_id.strip():
                prefix = "experiment" if category == "Experiment" else "discussion"
                open_hint = f"{prefix}:{summary_id.strip()}"
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "circuit.publish_result: published=%s category=%s summary_id=%s elements=%s block_reason=%s",
                    bool(getattr(res, "published", False)),
                    category or None,
                    summary_id,
                    getattr(res, "plsav_elements", None),
                    getattr(res, "publish_block_reason", None),
                )
            return json.dumps(
                {
                    "published": bool(getattr(res, "published", False)),
                    "summary_id": getattr(res, "summary_id", None),
                    "category": category or None,
                    "open_hint": open_hint,
                    "plsav_elements": getattr(res, "plsav_elements", None),
                    "publish_block_reason": getattr(res, "publish_block_reason", None),
                    "artifact_dir": getattr(res, "artifact_dir", None),
                    "artifact_sav_path": getattr(res, "artifact_sav_path", None),
                    "artifact_verilog_path": getattr(res, "artifact_verilog_path", None),
                },
                ensure_ascii=False,
                indent=2,
            )

        return f"ERROR: unknown tool '{tool}'"
    except Exception as e:
        return f"ERROR: {e.__class__.__name__}: {truncate(str(e) or '', max_chars=1800)}"


def agent_mode_run(
    *,
    ollama: OllamaClient,
    user: Any,
    cfg: Any,
    cache_dir: str,
    config_base_dir: str,
    dry_run: bool,
    logger: logging.Logger,
    task: str,
    context_json: dict[str, Any] | None,
    history: list[dict[str, str]],
    requester_nickname: str | None = None,
    requester_user_id: str | None = None,
    max_seconds: int = 600,
    max_steps: int = 100,
) -> str:
    task = (task or "").strip()
    if not task:
        return render_help(command_prefix=getattr(cfg.agent, "command_prefix", "!"), mode="agent")

    # In one-shot mode, prior conversation context is more likely to harm than help.
    # Keep the agent anchored to the current user message to reduce entity drift.
    if bool(getattr(getattr(cfg, "agent", None), "reply_once", True)):
        history = []

    wants_simulation = _looks_like_simulation_request(task)
    simulation_enabled = bool(getattr(getattr(cfg, "agent", None), "simulation_enabled", True))
    wants_content_intro = _looks_like_content_intro_request(task)
    wants_first_work = _looks_like_first_work_request(task)
    first_work_target_name = _extract_first_work_target_name(task) if wants_first_work else None
    wants_user_work_pick = _looks_like_user_work_pick_request(task)
    work_pick_target_name = _extract_work_pick_target_name(task) if wants_user_work_pick else None
    wants_circuit, _explicit_publish_intent = _fallback_route_for_circuit(task)

    start_ts = time.time()
    deadline = start_ts + float(max_seconds)
    tool_cutoff_ts = start_ts + float(max(0, int(max_seconds) - 60))
    system_prompt = _effective_system_prompt(cfg=cfg, user_text=task)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": _agent_tool_prompt(max_seconds=max_seconds)},
    ]
    if first_work_target_name:
        messages.append(
            {
                "role": "system",
                "content": (
                    "FIRST WORK TARGET USER (use EXACTLY; do not substitute based on prior context):\n"
                    f"- name: {first_work_target_name}\n"
                    "Use this exact string in plar_get_user_by_name.name (or in search_plar @handle)."
                ),
            }
        )
    if work_pick_target_name:
        messages.append(
            {
                "role": "system",
                "content": (
                    "USER WORK PICK REQUEST DETECTED (best/most interesting work).\n"
                    "Target user (use EXACTLY; do not substitute based on page context/history):\n"
                    f"- name: {work_pick_target_name}\n"
                    "Required flow:\n"
                    "1) plar_get_user_by_name {name:\"<name>\"}\n"
                    "2) list_plar {kind:\"latest\",category:\"Experiment\",user_id:\"<uid>\",take:12}\n"
                    "3) Pick ONE item and answer with Category + SummaryID + Subject.\n"
                    "Do NOT open or reference any unrelated page/content IDs."
                ),
            }
        )
    if wants_simulation and simulation_enabled:
        messages.append(
            {
                "role": "system",
                "content": (
                    "SIMULATION REQUEST DETECTED.\n"
                    "- You MUST use the simulation tools (simulate / simulate_verilog / simulate_status_save).\n"
                    "- Do NOT provide hand-waved 'by calculation' results unless tools are disabled or fail.\n"
                    "- If Context JSON provides summary_id/category, prefer simulate_status_save.\n"
                    "- Keep simulation tool args tiny; never paste circuit/status JSON into args."
                ),
            }
        )
    if wants_content_intro:
        messages.append(
            {
                "role": "system",
                "content": (
                    "CONTENT INTRO REQUEST DETECTED (介绍/简介/讲讲内容).\n"
                    "Goal: describe the requested work using ONLY opened Context JSON fields.\n"
                    "Required:\n"
                    "- If the user refers to a specific ID: plar_open_content_page(summary_id,...)\n"
                    "- If the user refers to a user's first work: plar_get_user_by_name -> list_plar(take=1) -> plar_open_content_page\n"
                    "Final answer must include:\n"
                    "- Category + SummaryID + Subject\n"
                    "- 1–3 sentences of introduction/summary from Context JSON (e.g., title/body_text/summary_text).\n"
                ),
            }
        )
    if wants_first_work:
        messages.append(
            {
                "role": "system",
                "content": (
                    "USER WORK LOOKUP DETECTED (first work / first item).\n"
                    "Goal: answer with the FIRST item in the user's LATEST works list.\n"
                    "Required steps:\n"
                    "1) Lookup user -> plar_get_user_by_name {name:\"...\"} (get uid)\n"
                    "2) List works -> list_plar {kind:\"latest\",category:\"both\",user_id:\"<uid>\",take:1}\n"
                    "3) End with {\"tool\":\"end\",\"final\":\"...\"} and include Category + SummaryID + Subject.\n"
                    "Notes:\n"
                    "- The target user is the one named in the CURRENT user request (not the system prompt / developer name).\n"
                    "- plar_get_user_board is留言板评论, NOT works. Do NOT use it for works listing.\n"
                    "- If list_plar result is stored (shows key), use store_json to extract fields like:\n"
                    "  experiment.items[0].subject / experiment.items[0].id (or discussion.* if experiment is empty).\n"
                ),
            }
        )
    if wants_circuit:
        messages.append(
            {
                "role": "system",
                "content": (
                    "CIRCUIT BUILD REQUEST DETECTED.\n"
                    "Goal: generate a Physics Lab .sav circuit via the circuit tool.\n"
                    "Rules:\n"
                    "- Call the circuit tool AT MOST ONCE for a given spec.\n"
                    "- After circuit returns, immediately end with {\"tool\":\"end\",\"final\":\"...\"}.\n"
                    "- Do NOT call circuit again with identical args; use the previous tool result.\n"
                    "- Keep publish=false unless the user explicitly asks to publish.\n"
                    "In your final answer, include artifact_sav_path, artifact_verilog_path, and plsav_elements."
                ),
            }
        )

    def _looks_like_page_context_reference(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        low = t.casefold()
        if any(x in low for x in ("summarize", "summary", "总结", "概括")):
            return True
        # Avoid matching overly-generic tokens like "当前" alone; use longer anchors.
        anchors = (
            "当前页面",
            "当前实验",
            "当前讨论",
            "本页面",
            "本实验",
            "本讨论",
            "这个页面",
            "这个实验",
            "这个讨论",
            "此页面",
            "此实验",
            "此讨论",
            "this page",
            "current page",
            "this experiment",
            "current experiment",
            "this discussion",
            "current discussion",
        )
        return any(a in low for a in anchors)

    include_context = (
        context_json is not None
        and (wants_simulation or wants_content_intro or _looks_like_page_context_reference(task))
    )
    if include_context and context_json is not None:
        context_for_llm = _shrink_context_json_for_llm(context_json)
        ctx_text = json.dumps(context_for_llm, ensure_ascii=False, indent=2)
        # Keep the embedded context bounded; the full context still exists in `context_json` for tools.
        ctx_text = truncate(ctx_text, max_chars=8000)
        messages.append(
            {
                "role": "system",
                "content": "Context JSON (current page, truncated):\n" + ctx_text,
            }
        )
    if history:
        # Keep it short; agent mode can re-query if needed.
        messages.extend(history[-8:])
    messages.append({"role": "user", "content": task})
    sticky_prefix_len = len(messages)

    debug_io = bool(getattr(getattr(cfg, "agent", None), "debug_log_llm_io", False)) and logger.isEnabledFor(
        logging.DEBUG
    )
    debug_max = int(getattr(getattr(cfg, "agent", None), "debug_llm_max_chars", 800) or 800)

    def _trim_messages_for_budget(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        # gpt-oss commonly runs with a 4k-ish context window; keep prompts tight.
        max_prompt_chars = int(getattr(getattr(cfg, "agent", None), "llm_prompt_max_chars", 14000) or 14000)

        def _chars(ms: list[dict[str, str]]) -> int:
            return sum(len(str(m.get("content") or "")) for m in ms if isinstance(m, dict))

        total = _chars(msgs)
        if total <= max_prompt_chars:
            return msgs

        prefix = list(msgs[:sticky_prefix_len])
        tail = list(msgs[sticky_prefix_len:])
        # Prefer keeping the most recent tool calls/results.
        keep_n = 24
        kept = tail[-keep_n:] if keep_n > 0 else []
        out = prefix + kept

        # If still too large, shrink the tail further.
        while _chars(out) > max_prompt_chars and keep_n > 8:
            keep_n = max(8, keep_n - 4)
            kept = tail[-keep_n:]
            out = prefix + kept

        # As a last resort, truncate any embedded Context JSON system message.
        if _chars(out) > max_prompt_chars:
            out2: list[dict[str, str]] = []
            for m in out:
                if not isinstance(m, dict):
                    continue
                c = str(m.get("content") or "")
                if (m.get("role") == "system") and ("Context JSON (current page" in c) and (len(c) > 2500):
                    c = truncate(c, max_chars=2500)
                out2.append({"role": str(m.get("role") or "system"), "content": c})
            out = out2

        return out

    def _timeout_reply() -> str:
        is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in (task or ""))
        mins = int(max(0, int(max_seconds)) // 60) or 10
        if is_cjk:
            return f"抱歉，Agent 已超时（{mins} 分钟），本次任务未能完成。你可以重新 @我 并简化需求再试一次。"
        return f"Sorry — agent timed out ({mins} minutes) and couldn't finish this task. Please @me again with a shorter request."

    def _llm_chat_json(*, messages: list[dict[str, str]], phase: str) -> str:
        try:
            return ollama.chat(messages=messages, response_format="json")
        except (OllamaError, Exception) as e:
            logger.warning("agent.llm_error: phase=%s err=%s", phase, e)
            is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in (task or ""))
            if is_cjk:
                final = "抱歉，本次模型没有返回可用输出（空响应/格式异常），请稍后重试或更换模型。"
            else:
                final = "Sorry — the model returned an unusable empty/invalid response. Please retry or switch models."
            return json.dumps({"tool": "end", "final": final}, ensure_ascii=False)

    last_raw = ""
    last_non_tool_output = ""
    did_use_simulation_tool = False
    did_use_circuit_tool = False
    last_circuit_info: dict[str, Any] | None = None
    did_open_content = False
    had_auth_failed = False
    allowed_open_ids: set[str] = set()
    did_list_plar_for_work_pick = False
    allowed_tools = {
        "web_search",
        "search_plar",
        "list_plar",
        "plar_query_experiments",
        "plar_get_user_by_name",
        "plar_get_user_by_id",
        "plar_get_user_board",
        "plar_get_experiment_context",
        "plar_open_content_page",
        "plar_get_status_save",
        "plar_get_comments",
        "store_get",
        "store_json",
        "simulate",
        "simulate_verilog",
        "simulate_status_save",
        "circuit",
        "end",
    }
    tool_sig_counts: dict[str, int] = {}
    tool_sig_last_result: dict[str, str] = {}

    def _maybe_add_planner_hint() -> None:
        planner_enabled = bool(getattr(getattr(cfg, "agent", None), "planner_enabled", False))
        if not planner_enabled:
            return
        # Avoid spending budget on planning when caller explicitly sets a tiny time budget.
        if int(max_seconds) < 90:
            return

        max_items = int(getattr(getattr(cfg, "agent", None), "planner_max_items", 6) or 6)
        if max_items <= 0:
            max_items = 6

        ctx_obj: dict[str, Any] | None = None
        if isinstance(context_json, dict) and context_json:
            try:
                ctx_obj = _shrink_context_json_for_llm(context_json)
            except Exception:
                ctx_obj = None

        planner_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "PLANNER MODE (internal).\n"
                    "- Do NOT call tools.\n"
                    "- Do NOT include chain-of-thought.\n"
                    "- Output STRICT JSON only.\n"
                    "Schema:\n"
                    "{\"plan\":[\"...\"]}\n"
                    "Rules:\n"
                    f"- plan must be a list of <= {max_items} short steps.\n"
                    "- steps must be actionable and reference tools by name when needed.\n"
                ),
            },
        ]
        if ctx_obj is not None:
            planner_messages.append(
                {
                    "role": "system",
                    "content": "Context JSON (current page):\n" + json.dumps(ctx_obj, ensure_ascii=False, indent=2),
                }
            )
        planner_messages.append(
            {
                "role": "system",
                "content": "Allowed tools: " + ", ".join(sorted(allowed_tools - {"end"})),
            }
        )
        planner_messages.append({"role": "user", "content": task})

        try:
            raw_plan = ollama.chat(messages=planner_messages, response_format="json")
        except Exception:
            return
        obj = _try_parse_json_object(raw_plan or "")
        if not obj:
            return
        steps = obj.get("plan")
        if not isinstance(steps, list):
            return
        cleaned: list[str] = []
        for s in steps:
            if not isinstance(s, str):
                continue
            s2 = (s or "").strip()
            if not s2:
                continue
            if len(s2) > 180:
                s2 = truncate(s2, max_chars=180)
            cleaned.append(s2)
            if len(cleaned) >= max_items:
                break
        if not cleaned:
            return
        plan_text = "\n".join([f"{i+1}. {x}" for i, x in enumerate(cleaned)])
        messages.append(
            {
                "role": "system",
                "content": "INTERNAL PLAN (do not reveal to user; follow it):\n" + plan_text,
            }
        )

    def _tool_sig(tool_name: str, tool_args: dict[str, Any]) -> str:
        try:
            blob = json.dumps(
                tool_args or {},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception:
            blob = str(tool_args or "")
        return f"{tool_name}:{blob}"

    def _looks_like_tool_call_text(text: str) -> bool:
        try:
            _agent_parse_tool_call(text)
            return True
        except Exception:
            return False

    _maybe_add_planner_hint()

    for step in range(1, int(max_steps) + 1):
        now = time.time()
        time_left = int(max(0.0, deadline - now))
        tools_enabled = now < tool_cutoff_ts
        if now > deadline:
            return _timeout_reply()

        # In the last 60s, tools are disabled: force final output and refuse tool calls.
        if not tools_enabled:
            def _tools_disabled_fallback(raw_text: str | None = None) -> str:
                is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in (task or ""))
                t = (raw_text or "").strip()
                # If the model produced plain text (not a tool call), accept it.
                obj = _try_parse_json_object(t) if t else None
                if t and not (isinstance(obj, dict) and isinstance(obj.get("tool"), str)):
                    out2 = safe_reply(t, max_chars=cfg.agent.max_reply_chars)
                    return out2 if out2.strip() else ("Done." if not is_cjk else "完成。")
                if is_cjk:
                    return (
                        "抱歉，当前时间预算过小/已进入最后 60 秒，工具调用已禁用，无法继续完成需要工具的步骤。"
                        "请提高 max_seconds 后重试。"
                    )
                return (
                    "Sorry — tools are disabled in the last 60 seconds (or the time budget is too small), "
                    "so I can't complete tool-required steps. Please retry with a larger max_seconds."
                )

            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"TOOLS DISABLED (last 60s). Time remaining: {time_left}s.\n"
                        "You must stop calling tools and immediately output:\n"
                        "{\"tool\":\"end\",\"final\":\"...\"}\n"
                        "If you attempt any other tool call, it will be rejected."
                    ),
                }
            )
            if debug_io:
                logger.debug(
                    "agent.llm_call: step=%d/%d tools_enabled=false time_left=%ds messages=%d",
                    step,
                    int(max_steps),
                    time_left,
                    len(messages),
                )
            messages = _trim_messages_for_budget(messages)
            raw_final = _llm_chat_json(messages=messages, phase="tools_disabled_end")
            if debug_io:
                logger.debug(
                    "agent.llm_out: step=%d len=%d preview=%r",
                    step,
                    len(raw_final or ""),
                    truncate(raw_final or "", max_chars=debug_max),
                )
            try:
                tool, _args, final = _agent_parse_tool_call(raw_final)
                if tool == "end":
                    if debug_io:
                        logger.debug("agent.tool_call: step=%d tool=end final_len=%d", step, len(final or ""))
                    out = safe_reply(final or "", max_chars=cfg.agent.max_reply_chars)
                    return out if out.strip() else "Done."
                # Reject any other tool call and force one last retry for end.
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "REJECTED: tool calls are disabled in the last 60 seconds. "
                            "Output ONLY: {\"tool\":\"end\",\"final\":\"...\"}"
                        ),
                    }
                )
                if debug_io:
                    logger.debug(
                        "agent.llm_call: step=%d/%d tools_enabled=false (retry_end) time_left=%ds messages=%d",
                        step,
                        int(max_steps),
                        time_left,
                    len(messages),
                )
                messages = _trim_messages_for_budget(messages)
                raw_final2 = _llm_chat_json(messages=messages, phase="tools_disabled_retry_end")
                if debug_io:
                    logger.debug(
                        "agent.llm_out: step=%d (retry_end) len=%d preview=%r",
                        step,
                        len(raw_final2 or ""),
                        truncate(raw_final2 or "", max_chars=debug_max),
                    )
                try:
                    tool2, _args2, final2 = _agent_parse_tool_call(raw_final2)
                    if tool2 == "end":
                        if debug_io:
                            logger.debug(
                                "agent.tool_call: step=%d tool=end (retry_end) final_len=%d",
                                step,
                                len(final2 or ""),
                            )
                        out = safe_reply(final2 or "", max_chars=cfg.agent.max_reply_chars)
                        return out if out.strip() else "Done."
                    return _tools_disabled_fallback(raw_text=final2 or raw_final2)
                except Exception:
                    return _tools_disabled_fallback(raw_text=raw_final2)
            except Exception:
                return _tools_disabled_fallback(raw_text=raw_final)

        # Normal tool-enabled phase.
        messages.append(
            {
                "role": "system",
                "content": f"Time remaining: {time_left}s. Tools enabled: true. Step {step}/{int(max_steps)}.",
            }
        )
        if now > deadline:
            return _timeout_reply()
        if debug_io:
            logger.debug(
                "agent.llm_call: step=%d/%d tools_enabled=true time_left=%ds messages=%d",
                step,
                int(max_steps),
                time_left,
                len(messages),
            )
        messages = _trim_messages_for_budget(messages)
        raw = _llm_chat_json(messages=messages, phase="tool_loop")
        last_raw = raw
        try:
            tool, args, final = _agent_parse_tool_call(raw)
        except Exception:
            # Reject: in agent mode, we require strict JSON tool calls until the final "end".
            if debug_io:
                logger.debug(
                    "agent.llm_out_unparsed: step=%d len=%d preview=%r",
                    step,
                    len(raw or ""),
                    truncate(raw or "", max_chars=debug_max),
                )
            last_non_tool_output = raw
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "REJECTED: You MUST output STRICT JSON only.\n"
                        "- Tool call: {\"tool\":\"<name>\",\"args\":{...}}\n"
                        "- Or finish: {\"tool\":\"end\",\"final\":\"...\"}\n"
                        "Do not output prose."
                    ),
                }
            )
            continue

        if tool not in allowed_tools:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"REJECTED: unknown tool '{tool}'. "
                        "Output ONLY a valid tool call JSON using one of the allowed tools: "
                        + ", ".join(sorted(allowed_tools))
                    ),
                }
            )
            continue

        if wants_first_work and first_work_target_name:
            def _norm_name(x: str) -> str:
                s = (x or "").strip()
                s = s.lstrip("@＠")
                s = re.sub(r"\s+", "", s)
                return s

            target_norm = _norm_name(first_work_target_name)
            if tool == "plar_get_user_by_name":
                got = _norm_name(str(args.get("name") or ""))
                if got and got != target_norm:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "REJECTED: wrong user name for first-work lookup.\n"
                                f"- required name: {first_work_target_name}\n"
                                f"- got: {str(args.get('name') or '').strip()}\n"
                                "Call plar_get_user_by_name again with the required name."
                            ),
                        }
                    )
                    continue
            if tool == "search_plar":
                q = str(args.get("query") or "").strip()
                try:
                    spec = _parse_plar_lookup_query(q)
                except Exception:
                    spec = {"kind": "", "value": ""}
                if spec.get("kind") == "user_name":
                    got = _norm_name(str(spec.get("value") or ""))
                    if got and got != target_norm:
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "REJECTED: wrong user in search_plar for first-work lookup.\n"
                                    f"- required: @{first_work_target_name}\n"
                                    f"- got: {q}\n"
                                    "Redo the lookup using the required user name."
                                ),
                            }
                        )
                        continue
            if tool == "list_plar":
                uid = str(args.get("user_id") or "").strip()
                if not uid:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "REJECTED: list_plar for 'first work' must include a specific user_id.\n"
                                f"First lookup the user '{first_work_target_name}' using plar_get_user_by_name, then call list_plar with user_id."
                            ),
                        }
                    )
                    continue

        if wants_user_work_pick and work_pick_target_name:
            def _norm_name2(x: str) -> str:
                s = (x or "").strip()
                s = s.lstrip("@＠")
                s = re.sub(r"\s+", "", s)
                return s

            target_norm2 = _norm_name2(work_pick_target_name)
            if tool == "plar_get_user_by_name":
                got = _norm_name2(str(args.get("name") or ""))
                if got and got != target_norm2:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "REJECTED: wrong user name for work-pick lookup.\n"
                                f"- required name: {work_pick_target_name}\n"
                                f"- got: {str(args.get('name') or '').strip()}\n"
                                "Call plar_get_user_by_name again with the required name."
                            ),
                        }
                    )
                    continue
            if tool == "list_plar":
                uid = str(args.get("user_id") or "").strip()
                if not uid:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "REJECTED: list_plar for work-pick must include a specific user_id.\n"
                                f"First lookup the user '{work_pick_target_name}' using plar_get_user_by_name, then call list_plar with user_id."
                            ),
                        }
                    )
                    continue
            if tool in ("plar_open_content_page", "plar_get_experiment_context") and allowed_open_ids:
                sid = str(args.get("summary_id") or args.get("id") or "").strip()
                if sid and sid not in allowed_open_ids:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "REJECTED: do not open unrelated content for this request.\n"
                                f"- requested user: {work_pick_target_name}\n"
                                f"- allowed ids (from list_plar): {', '.join(sorted(list(allowed_open_ids))[:12])}\n"
                                f"- got: {sid}\n"
                                "Open ONLY an id returned by list_plar for the target user, or answer directly from list_plar without opening."
                            ),
                        }
                    )
                    continue

        sig = ""
        if tool != "end":
            sig = _tool_sig(tool, args)
            prev_n = int(tool_sig_counts.get(sig, 0) or 0)
            prev_res = tool_sig_last_result.get(sig, "")
            if tool == "circuit" and prev_n >= 1 and not str(prev_res or "").startswith("ERROR"):
                dup_msg = (
                    "ERROR: duplicate circuit tool call suppressed (identical args). "
                    "Use the previous circuit tool result already returned, then call end."
                )
                if debug_io:
                    logger.debug("agent.tool_call: step=%d duplicate tool=%s suppressed", step, tool)
                messages.append({"role": "assistant", "content": raw})
                messages.append(_agent_tool_result_message(tool=tool, result=dup_msg, cache_dir=cache_dir))
                messages.append(
                    {
                        "role": "system",
                        "content": "Now call {\"tool\":\"end\",\"final\":\"...\"} using the previous circuit result.",
                    }
                )
                continue
            tool_sig_counts[sig] = prev_n + 1

        if tool == "end":
            if (
                wants_simulation
                and simulation_enabled
                and tools_enabled
                and (not did_use_simulation_tool)
            ):
                # Enforce tool usage while tools are still available.
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "REJECTED: You must run a simulation tool before ending.\n"
                            "Call one of:\n"
                            "- {\"tool\":\"simulate\",\"args\":{\"text\":\"...\"}}\n"
                            "- {\"tool\":\"simulate_verilog\",\"args\":{\"text\":\"...\"}}\n"
                            "- {\"tool\":\"simulate_status_save\",\"args\":{\"summary_id\":\"...\",\"category\":\"Experiment|Discussion\",\"question\":\"...\"}}\n"
                            "Then end with {\"tool\":\"end\",\"final\":\"...\"}."
                        ),
                    }
                )
                continue
            if wants_user_work_pick and tools_enabled and (not did_list_plar_for_work_pick) and (not had_auth_failed):
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "REJECTED: user asked you to pick an interesting experiment, but you did not list the user's works yet.\n"
                            "Required flow:\n"
                            "1) plar_get_user_by_name {name:\"...\"}\n"
                            "2) list_plar {kind:\"latest\",category:\"Experiment\",user_id:\"<uid>\",take:12}\n"
                            "Then pick ONE item and end with Category + SummaryID + Subject."
                        ),
                    }
                )
                continue
            if wants_content_intro and tools_enabled and (not did_open_content) and (not had_auth_failed):
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "REJECTED: You must OPEN the specific work/content before describing it.\n"
                            "- If user gave an ID: call plar_open_content_page.\n"
                            "- If user gave a nickname: plar_get_user_by_name -> list_plar(user_id=...) -> plar_open_content_page(summary_id,...).\n"
                            "If you cannot access (auth failed), say so and do NOT invent any details."
                        ),
                    }
                )
                continue
            if wants_content_intro and tools_enabled and did_open_content:
                final_text = str(final or "").strip()
                has_intro_hint = any(x in final_text for x in ("介绍", "简介", "内容", "summary", "Summary", "intro", "Intro"))
                if (len(final_text) < 120) and (not has_intro_hint):
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "REJECTED: user asked for an introduction/summary, but your end.final does not include it.\n"
                                "Include Category + SummaryID + Subject, plus 1–3 sentences summarizing the work from the opened Context JSON.\n"
                                "Now call {\"tool\":\"end\",\"final\":\"...\"} again."
                            ),
                        }
                    )
                    continue
            if wants_user_work_pick and tools_enabled:
                final_text = str(final or "")
                if _HEX24_RE.search(final_text or "") is None:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "REJECTED: your end.final is missing a SummaryID (24-hex id).\n"
                                "Pick one experiment from list_plar and include Category + SummaryID + Subject."
                            ),
                        }
                    )
                    continue
            if wants_circuit and tools_enabled and (not did_use_circuit_tool):
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "REJECTED: circuit build requested, but you did not run the circuit tool yet.\n"
                            "Call: {\"tool\":\"circuit\",\"args\":{\"spec\":\"...\",\"publish\":false}}\n"
                            "Then end with {\"tool\":\"end\",\"final\":\"...\"} including artifact_sav_path, artifact_verilog_path, plsav_elements."
                        ),
                    }
                )
                continue
            if wants_circuit and did_use_circuit_tool and isinstance(last_circuit_info, dict):
                sav = last_circuit_info.get("artifact_sav_path")
                vpath = last_circuit_info.get("artifact_verilog_path")
                elems = last_circuit_info.get("plsav_elements")
                missing: list[str] = []
                final_text = str(final or "")
                if isinstance(sav, str) and sav.strip() and sav.strip() not in final_text:
                    missing.append("artifact_sav_path")
                if isinstance(vpath, str) and vpath.strip() and vpath.strip() not in final_text:
                    missing.append("artifact_verilog_path")
                if isinstance(elems, int) and str(elems) not in final_text and "plsav_elements" not in final_text:
                    missing.append("plsav_elements")
                if missing:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "REJECTED: Your end.final is missing required circuit outputs: "
                                + ", ".join(missing)
                                + "\nUse these values from the circuit tool result:\n"
                                + f"- artifact_sav_path: {sav}\n"
                                + f"- artifact_verilog_path: {vpath}\n"
                                + f"- plsav_elements: {elems}\n"
                                "Now call {\"tool\":\"end\",\"final\":\"...\"} again."
                            ),
                        }
                    )
                    continue
            if debug_io:
                logger.debug("agent.tool_call: step=%d tool=end final_len=%d", step, len(final or ""))
            out = safe_reply(final or "", max_chars=cfg.agent.max_reply_chars)
            return out if out.strip() else "Done."

        if tool in ("simulate", "simulate_verilog", "simulate_status_save"):
            did_use_simulation_tool = True

        if debug_io:
            arg_keys = sorted([k for k in (args or {}).keys() if isinstance(k, str)])[:20]
            logger.debug(
                "agent.tool_call: step=%d tool=%s arg_keys=%s",
                step,
                tool,
                arg_keys,
            )

        result = _agent_execute_tool(
            tool=tool,
            args=args,
            user=user,
            ollama=ollama,
            cfg=cfg,
            cache_dir=cache_dir,
            config_base_dir=config_base_dir,
            dry_run=dry_run,
            context_json=context_json,
            logger=logger,
            requester_nickname=requester_nickname,
            requester_user_id=requester_user_id,
        )
        if sig:
            tool_sig_last_result[sig] = result
        if tool == "circuit":
            if not str(result or "").startswith("ERROR"):
                did_use_circuit_tool = True
            obj = _try_parse_json_object(result or "")
            if isinstance(obj, dict):
                last_circuit_info = obj
        if tool in ("plar_open_content_page", "plar_get_experiment_context"):
            obj = _try_parse_json_object(result or "")
            if isinstance(obj, dict) and obj.get("error") == "auth_failed":
                had_auth_failed = True
            elif isinstance(obj, dict) and obj.get("found") is False:
                pass
            elif isinstance(obj, dict):
                did_open_content = True
        if wants_user_work_pick and tool == "list_plar" and not str(result or "").startswith("ERROR"):
            obj = _try_parse_json_object(result or "")
            ids: list[str] = []
            if isinstance(obj, dict):
                items = obj.get("items")
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            sid = best_effort_extract_text(it.get("id")) or best_effort_extract_text(it.get("ID"))
                            if isinstance(sid, str) and sid.strip():
                                ids.append(sid.strip())
                for sec_key in ("experiment", "discussion"):
                    sec = obj.get(sec_key)
                    if isinstance(sec, dict) and isinstance(sec.get("items"), list):
                        for it in sec.get("items") or []:
                            if isinstance(it, dict):
                                sid = best_effort_extract_text(it.get("id")) or best_effort_extract_text(it.get("ID"))
                                if isinstance(sid, str) and sid.strip():
                                    ids.append(sid.strip())
            for sid in ids:
                allowed_open_ids.add(sid)
            if ids:
                did_list_plar_for_work_pick = True
        if debug_io:
            logger.debug(
                "agent.tool_result: step=%d tool=%s len=%d preview=%r",
                step,
                tool,
                len(result or ""),
                truncate(result or "", max_chars=debug_max),
            )
        messages.append({"role": "assistant", "content": raw})
        messages.append(_agent_tool_result_message(tool=tool, result=result, cache_dir=cache_dir))
        if tool == "circuit":
            sav = last_circuit_info.get("artifact_sav_path") if isinstance(last_circuit_info, dict) else None
            vpath = last_circuit_info.get("artifact_verilog_path") if isinstance(last_circuit_info, dict) else None
            elems = last_circuit_info.get("plsav_elements") if isinstance(last_circuit_info, dict) else None
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Circuit tool completed. Do NOT call any more tools.\n"
                        "Next message MUST be: {\"tool\":\"end\",\"final\":\"...\"}\n"
                        "Include these values:\n"
                        f"- artifact_sav_path: {sav}\n"
                        f"- artifact_verilog_path: {vpath}\n"
                        f"- plsav_elements: {elems}"
                    ),
                }
            )
        # Always remind remaining time after each tool result.
        now2 = time.time()
        time_left2 = int(max(0.0, deadline - now2))
        if now2 >= tool_cutoff_ts:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Time remaining: {time_left2}s. Tools will be DISABLED now; next message MUST be end."
                    ),
                }
            )
        else:
            cutoff_left = int(max(0.0, tool_cutoff_ts - now2))
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Time remaining: {time_left2}s. Tool-call cutoff in {cutoff_left}s (then only end allowed)."
                    ),
                }
            )

    # Budget exhausted: ask for end.
    if time.time() > deadline:
        return _timeout_reply()
    messages.append(
        {
            "role": "system",
            "content": "Agent mode budget reached. Call {\"tool\":\"end\",\"final\":\"...\"} with your best final answer now.",
        }
    )
    if debug_io:
        logger.debug(
            "agent.llm_call: budget_reached tools_enabled=%s messages=%d",
            (time.time() < tool_cutoff_ts),
            len(messages),
        )
    messages = _trim_messages_for_budget(messages)
    raw2 = _llm_chat_json(messages=messages, phase="budget_reached")
    try:
        tool, _args, final = _agent_parse_tool_call(raw2)
        if tool == "end":
            if debug_io:
                logger.debug("agent.tool_call: budget_reached tool=end final_len=%d", len(final or ""))
            out = safe_reply(final or "", max_chars=cfg.agent.max_reply_chars)
            return out if out.strip() else "Done."
        raw2 = ""
    except Exception:
        if debug_io:
            logger.debug(
                "agent.llm_out_unparsed: budget_reached len=%d preview=%r",
                len(raw2 or ""),
                truncate(raw2 or "", max_chars=debug_max),
            )
        raw2 = ""

    # One last retry: force an end tool call.
    messages.append(
        {
            "role": "system",
            "content": "REJECTED: budget reached. Output ONLY {\"tool\":\"end\",\"final\":\"...\"}.",
        }
    )
    messages = _trim_messages_for_budget(messages)
    raw3 = _llm_chat_json(messages=messages, phase="budget_reached_retry_end")
    try:
        tool3, _args3, final3 = _agent_parse_tool_call(raw3)
        if tool3 == "end":
            out = safe_reply(final3 or "", max_chars=cfg.agent.max_reply_chars)
            return out if out.strip() else "Done."
        raw3 = ""
    except Exception:
        raw3 = ""

    candidate = ""
    for x in (last_non_tool_output, last_raw, raw2, raw3):
        if isinstance(x, str) and x.strip() and (not _looks_like_tool_call_text(x)):
            candidate = x
            break
    if candidate:
        out = safe_reply(candidate, max_chars=cfg.agent.max_reply_chars)
        return out if out.strip() else "Done."

    is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in (task or ""))
    return (
        "抱歉，本次生成未能按协议结束（缺少 end）。请重试或缩短需求再试一次。"
        if is_cjk
        else "Sorry — the model failed to produce a valid end response. Please retry with a shorter request."
    )

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
    allow_agent: bool,
) -> dict[str, Any] | None:
    allow_agent = bool(allow_agent)
    allowed_action_lines = [
        "- chat",
        "- summarize",
        "- search_plar  (LOOKUP ONLY: user name/id, or experiment/discussion id; NOT keyword search)",
        "- google  (web search)",
        "- circuit  (generate Verilog and compile to .sav)",
        "- simulate  (run a local circuit simulation demo)",
    ]
    if allow_agent:
        allowed_action_lines.insert(2, "- agent  (multi-step tool agent)")
    allowed_actions = "\n".join(allowed_action_lines) + "\n"

    schema_actions = "chat|summarize"
    if allow_agent:
        schema_actions += "|agent"
    schema_actions += "|search_plar|google|circuit|simulate"

    agent_example = (
        "- User: 'Do a full investigation: search web + search PLAR and then answer' -> action=agent\n"
        if allow_agent
        else ""
    )
    router_prompt = (
        "You are a tool router for a Physics Lab AR community agent.\n"
        "Select the best action for the user request.\n\n"
        "Allowed actions:\n"
        + allowed_actions
        + "\n"
        "Internal search policy (important):\n"
        "- search_plar is LOOKUP-ONLY; it does NOT support keyword search across experiments.\n"
        "- Choose search_plar ONLY when the user provides (or clearly asks for) an ID or a user handle/name.\n"
        "- For discovery/recommendations (latest/hot/featured), choose agent so you can use list_plar.\n"
        "- If the user is not clearly asking for lookup/search, prefer chat (or ask a short clarifying question).\n\n"
        "Publishing policy:\n"
        "- Only set publish=true if the user explicitly asks to publish/share/post the experiment.\n"
        "- Otherwise publish=false.\n\n"
        "Examples:\n"
        "- User: 'Please implement a 4-bit adder and publish it as an experiment' -> action=circuit, publish=true\n"
        "- User: 'Summarize this' (on an experiment page) -> action=summarize\n"
        + agent_example
        + "- User: 'Find @someone' -> action=search_plar\n"
        "- User: 'Open experiment 0123456789abcdef01234567' -> action=search_plar\n"
        "- User: 'Look this up online' -> action=google\n\n"
        "Output STRICT JSON only, no prose. Schema:\n"
        + "{\"action\":\""
        + schema_actions
        + "\",\"arg\":\"...\",\"publish\":true|false}\n"
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
    if action not in ("chat", "summarize", "agent", "search_plar", "google", "circuit", "simulate"):
        return None
    if action == "agent" and not allow_agent:
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


def _looks_like_simulation_request(user_text: str) -> bool:
    t = (user_text or "").casefold()
    if not t:
        return False
    return any(
        x in t
        for x in (
            "simulate",
            "simulation",
            "transient",
            "dc analysis",
            "ac analysis",
            "time=",
            "t=",
            "仿真",
            "模拟",
            "瞬态",
            "时域",
            "直流分析",
            "交流分析",
        )
    )


def _looks_like_first_work_request(user_text: str) -> bool:
    t = (user_text or "").strip()
    if not t:
        return False
    low = t.casefold()
    return any(
        x in low
        for x in (
            "第一个作品",
            "第1个作品",
            "第一个实验",
            "第1个实验",
            "first work",
            "first experiment",
        )
    )

def _extract_first_work_target_name(user_text: str) -> str | None:
    """Best-effort extract target nickname for 'first work/experiment' requests.

    We keep this conservative: return None if we cannot confidently identify a single name.
    The goal is to prevent entity drift when prior context mentions other names.
    """
    t = (user_text or "").strip()
    if not t:
        return None

    low = t.casefold()
    # If the user already provided an explicit handle/id prefix, let the agent follow it.
    if any(p in low for p in ("uid:", "user_id:", "userid:", "experiment:", "discussion:", "user:", "nickname:")):
        return None
    if _HEX24_RE.search(t):
        return None

    # Prefer explicit @handle when present.
    m_at = _AT_HANDLE_RE.search(t)
    if m_at:
        cand = (m_at.group(1) or "").strip()
        if cand:
            return cand

    def _cleanup_name(name: str) -> str:
        s = (name or "").strip()
        # Strip common leading instruction phrases that can be accidentally captured.
        for p in (
            "请你告诉我",
            "麻烦你告诉我",
            "告诉我",
            "请问",
            "你认识",
            "你知道",
            "你了解",
        ):
            if s.startswith(p):
                s = s[len(p) :].strip()
                break
        # Strip common trailing verbs accidentally attached to the name.
        for suf in ("发布", "发表", "制作", "做"):
            if s.endswith(suf) and len(s) > len(suf):
                s = s[: -len(suf)].strip()
                break
        return s

    patterns = [
        # “告诉我紫兰斋的第一个作品…”
        r"(?:告诉我|请你告诉我|请问|麻烦你告诉我)(?P<name>[^\s，,。？！?]{1,32}?)的第(?:一|1)个(?:作品|实验)",
        # “紫兰斋发布的第一个实验…”
        r"(?P<name>[^\s，,。？！?]{1,32}?)(?:发布|发表|做|制作)的第(?:一|1)个(?:作品|实验)",
        # “紫兰斋的第一个作品/实验…”
        r"(?P<name>[^\s，,。？！?]{1,32}?)的第(?:一|1)个(?:作品|实验)",
        # “你认识紫兰斋吗？…第一个作品…”
        r"(?:你认识|你知道|你了解)(?P<name>[^\s，,。？！?吗]{1,32}?)(?:吗|？|\\?)?.{0,24}第(?:一|1)个(?:作品|实验)",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if not m:
            continue
        name = _cleanup_name((m.group("name") or "").strip())
        if not name:
            continue
        # Avoid capturing leading pronouns.
        if name in ("我", "你", "他", "她", "它", "我们", "他们", "她们", "自己"):
            continue
        return name

    return None

def _looks_like_user_work_pick_request(user_text: str) -> bool:
    t = (user_text or "").strip()
    if not t:
        return False
    low = t.casefold()
    wants_pick = any(x in low for x in ("最有趣", "最好", "最值得", "推荐", "best", "most interesting", "recommend"))
    mentions_work = any(x in low for x in ("作品", "实验", "experiment", "work"))
    mentions_published = any(x in low for x in ("发布", "发表", "发布的", "published"))
    has_target_hint = ("@" in t) or ("＠" in t) or ("的" in t) or ("user:" in low) or ("uid:" in low)
    return bool(wants_pick and mentions_work and (mentions_published or has_target_hint))

def _extract_work_pick_target_name(user_text: str) -> str | None:
    """Extract a target user name for 'pick best/most interesting work' queries."""
    t = (user_text or "").strip()
    if not t:
        return None
    # Prefer explicit @handle.
    m_at = _AT_HANDLE_RE.search(t)
    if m_at:
        cand = (m_at.group(1) or "").strip()
        return cand or None

    def _cleanup(name: str) -> str:
        s = (name or "").strip()
        for p in ("请你告诉我", "麻烦你告诉我", "告诉我", "请问"):
            if s.startswith(p):
                s = s[len(p) :].strip()
                break
        for suf in ("发布", "发表"):
            if s.endswith(suf) and len(s) > len(suf):
                s = s[: -len(suf)].strip()
                break
        return s

    # “MacroModel发布的最有趣的实验…”
    m = re.search(r"(?P<name>[^\s，,。？！?]{1,32}?)发布的", t)
    if m:
        name = _cleanup(m.group("name") or "")
        if name:
            return name
    # “MacroModel 的最有趣的实验…”
    m2 = re.search(r"(?P<name>[^\s，,。？！?]{1,32}?)的最", t)
    if m2:
        name = _cleanup(m2.group("name") or "")
        if name:
            return name
    return None


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


def _extract_user_obj_from_get_user_data(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    u = data.get("User")
    if isinstance(u, dict):
        return u
    # Some wrappers may already return the user object as Data.
    if any(k in data for k in ("ID", "UserID", "Nickname", "Name", "Signature")):
        return data
    return None


def _build_user_board_context(
    *,
    user: Any,
    board_user_id: str,
    comments: list[dict[str, Any]],
) -> dict[str, Any]:
    owner_id: str | None = board_user_id.strip() or None
    owner_nickname: str | None = None
    owner_signature: str | None = None
    try:
        data = get_user_by_id(user, user_id=board_user_id)
        u = _extract_user_obj_from_get_user_data(data)
        if isinstance(u, dict):
            owner_id = best_effort_extract_text(u.get("ID")) or best_effort_extract_text(u.get("UserID")) or owner_id
            owner_nickname = best_effort_extract_text(u.get("Nickname")) or best_effort_extract_text(u.get("Name")) or None
            sig = best_effort_extract_text(u.get("Signature"))
            owner_signature = truncate(sig, max_chars=200) if sig else None
    except Exception:
        pass

    title = ""
    if owner_nickname:
        title = f"{owner_nickname} 的留言板"

    return {
        "page_type": "UserBoard",
        "title": title or None,
        "board_owner": {
            "id": owner_id,
            "nickname": owner_nickname,
            "signature": owner_signature,
        },
        "recent_comments": _recent_comments_context(comments, limit=12),
    }


def _extract_user_board_query(arg: str) -> tuple[str | None, str | None]:
    """Return (nickname, user_id)."""
    t = (arg or "").strip()
    if not t or len(t) > 120:
        return None, None
    if not any(x in t for x in ("留言板", "主页", "墙", "board", "wall")):
        return None, None

    # Explicit ID.
    m_id = re.search(r"\bUser[:： ]([A-Za-z0-9_-]{8,64})\b", t)
    if m_id:
        return None, m_id.group(1).strip()

    # @nickname
    m_at = re.search(r"[@＠]([^\s，。！？:：;；]{2,32})", t)
    if m_at:
        return m_at.group(1).strip(), None

    # 用户:xxx / user:xxx
    m_user = re.search(r"(?:用户|user)\s*[:： ]\s*([^\s，。！？:：;；]{2,32})", t, flags=re.IGNORECASE)
    if m_user:
        return m_user.group(1).strip(), None

    return None, None


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
        ("agent", ("agent", "agentmode", "autopilot")),
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
    if text.startswith(("代理", "Agent", "agent")) and ("操作" in text or "模式" in text):
        # e.g. "agent 操作：xxx" / "代理模式 xxx"
        return "agent", re.sub(r"^[^:：]*[:：]?", "", text).strip()
    if text.startswith(("搜索", "查找")):
        return "search", text[2:].strip()
    if text.startswith("生成电路"):
        return "circuit", text[len("生成电路") :].strip()
    if "仿真" in text or "模拟" in text:
        return "simulate", text
    if text.startswith("谷歌"):
        return "google", text[len("谷歌") :].strip()

    return "chat", text


def _is_generic_summarize_arg(arg: str) -> bool:
    """True when the user likely means 'summarize the current page', not literal text."""
    t = (arg or "").strip()
    if not t:
        return True
    # Remove common punctuation/spaces so "这个 实验" or "实验。" matches.
    t2 = re.sub(r"[\s\.,!?，。！？:：;；\-_—()（）\[\]{}<>《》\"'“”‘’]+", "", t).strip()
    low = t2.casefold()
    return low in {
        "实验",
        "讨论",
        "留言板",
        "留言",
        "墙",
        "主页",
        "这个",
        "当前",
        "本文",
        "这篇",
        "这个实验",
        "这个讨论",
        "这个留言板",
        "当前留言板",
        "本实验",
        "本讨论",
        "当前实验",
        "当前讨论",
        "本留言板",
        "此实验",
        "此讨论",
        "此留言板",
        "experiment",
        "discussion",
        "messageboard",
        "board",
        "wall",
        "this",
        "current",
        "thisexperiment",
        "thisdiscussion",
        "thisboard",
        "thiswall",
    }


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
    system_prompt = _effective_system_prompt(cfg=cfg, user_text=content)
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

    agent_mode = str(getattr(getattr(cfg, "agent", None), "mode", "agent") or "agent").strip().lower()
    if agent_mode == "agent":
        # Agent-only mode: every triggered comment becomes an agent task.
        if cmd in ("", "help", "h", "?"):
            return render_help(command_prefix=cfg.agent.command_prefix, mode="agent")

        task = text.strip()
        if cfg.agent.commands_enabled and cfg.agent.command_prefix and task.startswith(cfg.agent.command_prefix):
            cmd2, arg2 = parse_command(task, prefix=cfg.agent.command_prefix)
            if cmd2 in ("", "help", "h", "?"):
                return render_help(command_prefix=cfg.agent.command_prefix, mode="agent")
            # Treat any legacy command as a task; prefer the argument text.
            task = (arg2 or cmd2 or "").strip()

        if not task:
            if experiment_context is not None:
                task = "请总结当前页面，并直接回答用户可能关心的关键点。"
            else:
                task = "请根据最近对话，直接完成用户的任务；如果任务不明确，请只问 1 个澄清问题。"

        reply = agent_mode_run(
            ollama=ollama,
            user=user,
            cfg=cfg,
            cache_dir=cache_dir,
            config_base_dir=config_base_dir,
            dry_run=dry_run,
            logger=logger,
            task=task,
            context_json=experiment_context,
            history=history,
            requester_nickname=nickname,
            requester_user_id=author_id,
        )
        return reply if (reply or "").strip() else "Done."

    if cmd in ("", "help", "h", "?"):
        return render_help(command_prefix=cfg.agent.command_prefix, mode="traditional")

    if cmd == "chat":
        if not arg:
            if experiment_context is not None:
                logger.debug("Tool chat invoked (empty input, summarizing context)")
                messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
                messages.append(
                    {
                        "role": "system",
                        "content": "Context JSON (current page):\n"
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
            return render_help(command_prefix=cfg.agent.command_prefix, mode="traditional")
        logger.debug("Tool chat invoked (len=%d)", len(arg))

        # LLM-driven tool routing for natural language requests.
        if (not cfg.agent.commands_enabled) and bool(getattr(cfg.agent, "auto_tool_routing", True)):
            try:
                route = _llm_route_tool(
                    ollama=ollama,
                    system_prompt=system_prompt,
                    user_text=arg,
                    context_json=experiment_context,
                    allow_agent=(agent_mode == "agent"),
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

                if action == "search_plar" and not _looks_like_plar_search_request(arg):
                    q0 = (routed_arg or "").strip()
                    allow_implicit_user_lookup = q0.startswith(("@", "＠")) or q0.casefold().startswith(
                        ("user:", "user ", "用户:", "用户 ")
                    )
                    if not allow_implicit_user_lookup:
                        logger.info(
                            "Auto tool routing: ignoring search_plar (no explicit search intent; user_text_len=%d)",
                            len(arg),
                        )
                        action = "chat"

                if action == "summarize":
                    if experiment_context is not None and _is_generic_summarize_arg(routed_arg):
                        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
                        messages.append(
                            {
                                "role": "system",
                                "content": "Context JSON (current page):\n"
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
                        system_prompt=system_prompt,
                        text=routed_arg,
                    )
                    return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

                if action == "agent":
                    run_task = routed_arg or arg
                    if not run_task:
                        return "Provide a task description."
                    logger.info("Agent mode invoked (auto_routed=true task_len=%d)", len(run_task))
                    return agent_mode_run(
                        ollama=ollama,
                        user=user,
                        cfg=cfg,
                        cache_dir=cache_dir,
                        config_base_dir=config_base_dir,
                        dry_run=dry_run,
                        logger=logger,
                        task=run_task,
                        context_json=experiment_context,
                        history=history,
                    )

                if action == "search_plar":
                    if not routed_arg:
                        return "Provide a query string."
                    if _looks_political_sensitive(routed_arg):
                        return _political_refusal_message(routed_arg)

                    spec = _parse_plar_lookup_query(routed_arg)
                    kind = spec.get("kind")
                    value = (spec.get("value") or "").strip()

                    def _render_user(u: dict[str, Any]) -> str:
                        uid = best_effort_extract_text(u.get("ID")) or best_effort_extract_text(
                            u.get("UserID")
                        )
                        nick = best_effort_extract_text(u.get("Nickname")) or "(no nickname)"
                        ver = best_effort_extract_text(u.get("Verification"))
                        lines = ["User:", f"- Nickname: {nick}", f"- ID: {uid or '(unknown)'}"]
                        if ver:
                            lines.append(f"- Verification: {ver}")
                        return "\n".join(lines)

                    if kind == "user_name":
                        if not value:
                            return "Search is lookup-only. Try '@name' or 'user: name'."
                        try:
                            data = get_user_by_name(user, name=value)
                        except Exception as e:
                            if _is_probable_user_not_found_error(e):
                                return "No user found."
                            return f"Lookup failed: {e}"
                        u = data.get("User") if isinstance(data, dict) else None
                        if not isinstance(u, dict):
                            return "No user found."
                        return safe_reply(_render_user(u), max_chars=cfg.agent.max_reply_chars)

                    if kind == "user_id":
                        if not value:
                            return "Search is lookup-only. Try 'uid: <id>'."
                        try:
                            data = get_user_by_id(user, user_id=value)
                        except Exception as e:
                            if _is_probable_user_not_found_error(e):
                                return "No user found."
                            return f"Lookup failed: {e}"
                        u = data.get("User") if isinstance(data, dict) else None
                        if not isinstance(u, dict):
                            return "No user found."
                        return safe_reply(_render_user(u), max_chars=cfg.agent.max_reply_chars)

                    if kind == "content_id":
                        if not value:
                            return "Search is lookup-only. Try 'experiment:<id>' or 'discussion:<id>'."
                        category = (spec.get("category") or "both").strip()
                        tried: list[str] = []
                        last_err: BaseException | None = None

                        def _try(cat: str) -> dict[str, Any] | None:
                            nonlocal last_err
                            tried.append(cat)
                            try:
                                ctx = get_experiment_context(
                                    user,
                                    summary_id=value,
                                    category_value=cat,
                                    cache_dir=cache_dir,
                                    ttl_sec=300,
                                )
                            except Exception as e:
                                last_err = e
                                return None
                            return ctx if isinstance(ctx, dict) else None

                        ctx = None
                        if category == "Experiment":
                            ctx = _try("Experiment")
                        elif category == "Discussion":
                            ctx = _try("Discussion")
                        else:
                            ctx = _try("Experiment") or _try("Discussion")

                        if ctx is None:
                            if last_err is not None and _is_probable_content_not_found_error(last_err):
                                return safe_reply(
                                    f"No content found. (tried: {', '.join(tried)})",
                                    max_chars=cfg.agent.max_reply_chars,
                                )
                            return f"Lookup failed: {last_err}"

                        subject = (
                            best_effort_extract_text(
                                ctx.get("subject") or ctx.get("Subject") or ctx.get("title")
                            )
                            or "(no subject)"
                        )
                        cat = best_effort_extract_text(ctx.get("category") or ctx.get("Category")) or (
                            tried[-1] if tried else None
                        )
                        lines = ["Content:"]
                        if cat:
                            lines.append(f"- Category: {cat}")
                        lines.append(f"- Subject: {subject}")
                        lines.append(f"- ID: {value}")
                        return safe_reply("\n".join(lines), max_chars=cfg.agent.max_reply_chars)

                    return "search_plar is lookup-only: @name/user:name/uid:<id>/experiment:<id>/discussion:<id>."

                if action == "google":
                    if not routed_arg:
                        return "Provide a query string."
                    if _looks_political_sensitive(routed_arg):
                        return _political_refusal_message(routed_arg)
                    if not bool(getattr(cfg.agent, "web_search_enabled", False)):
                        return "Web search is disabled. Set agent.web_search_enabled=true in config."
                    try:
                        search_txt = web_search(
                            query=routed_arg,
                            cache_dir=cache_dir,
                            provider=str(getattr(cfg.agent, "web_search_provider", "google") or "google"),
                            proxy=str(getattr(cfg.agent, "web_search_proxy", "") or ""),
                            timeout_sec=int(getattr(cfg.agent, "web_search_timeout_sec", 20) or 20),
                            ttl_sec=int(getattr(cfg.agent, "web_search_cache_ttl_sec", 3600) or 3600),
                            max_results=int(getattr(cfg.agent, "web_search_max_results", 5) or 5),
                            fallback_to_ddg=bool(getattr(cfg.agent, "web_search_fallback_to_ddg", True)),
                            user_agent=str(getattr(cfg.agent, "web_search_user_agent", "") or ""),
                            searxng_base_url=str(getattr(cfg.agent, "web_search_searxng_base_url", "") or ""),
                        )
                    except Exception as e:
                        return f"Web search failed: {e}"
                    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
                    messages.append(
                        {
                            "role": "system",
                            "content": "Web search results (use as external references; include relevant links):\n"
                            + search_txt,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": "Answer the user's query using the web search results above.\n"
                            "- If the results are blocked/empty, say so briefly and suggest a fix (proxy or SearXNG).\n"
                            "- Include up to 5 relevant links.\n\n"
                            f"User query:\n{routed_arg}",
                        }
                    )
                    reply = ollama.chat(messages=messages)
                    return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

                if action == "simulate":
                    sim_text = routed_arg or arg
                    failures: list[str] = []
                    ai_parse_failure: str | None = None
                    prefers_verilog = any(
                        x in (sim_text or "").casefold()
                        for x in (
                            "verilog",
                            "module",
                            "endmodule",
                            "logic",
                            "gate",
                            "flipflop",
                            "与门",
                            "或门",
                            "非门",
                            "异或",
                            "逻辑",
                        )
                    )
                    # Prefer: LLM builds PE-SCRIPT -> simulate -> LLM interprets results.
                    if bool(getattr(cfg.agent, "simulation_ai_enabled", True)):
                        if prefers_verilog:
                            try:
                                replyv = simulate_ai_verilog_with_phyengine(
                                    ollama=ollama,
                                    text=sim_text,
                                    context_json=experiment_context,
                                    phy_engine_cfg=cfg.phy_engine,
                                    config_base_dir=config_base_dir,
                                    cache_dir=cache_dir,
                                    max_attempts=2,
                                    max_elements=int(getattr(cfg.agent, "simulation_max_elements", 300) or 300),
                                )
                                return safe_reply(replyv, max_chars=cfg.agent.max_reply_chars)
                            except Exception as e:
                                failures.append(f"ai_verilog_sim_failed: {_format_exception_brief(e)}")
                                logger.info("AI Verilog simulation failed; falling back: %s", e)
                        try:
                            reply = simulate_ai_script_circuit_with_phyengine(
                                ollama=ollama,
                                text=sim_text,
                                context_json=experiment_context,
                                phy_engine_cfg=cfg.phy_engine,
                                config_base_dir=config_base_dir,
                                max_components=int(
                                    getattr(cfg.agent, "simulation_ai_max_components", 30) or 30
                                ),
                                max_probes=int(
                                    getattr(cfg.agent, "simulation_ai_max_probes", 20) or 20
                                ),
                            )
                            if not _looks_like_pe_script_parse_failure(reply):
                                return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

                            # Script parse failed; try legacy JSON-spec LLM path before other fallbacks.
                            ai_parse_failure = reply
                            try:
                                reply2 = simulate_ai_circuit_with_phyengine(
                                    ollama=ollama,
                                    text=sim_text,
                                    context_json=experiment_context,
                                    phy_engine_cfg=cfg.phy_engine,
                                    config_base_dir=config_base_dir,
                                    max_attempts=3,
                                    max_components=int(
                                        getattr(cfg.agent, "simulation_ai_max_components", 30) or 30
                                    ),
                                    max_probes=int(
                                        getattr(cfg.agent, "simulation_ai_max_probes", 20) or 20
                                    ),
                                )
                                return safe_reply(reply2, max_chars=cfg.agent.max_reply_chars)
                            except Exception as e:
                                failures.append(f"ai_json_spec_failed: {_format_exception_brief(e)}")
                                logger.info("AI JSON-spec simulation failed; falling back: %s", e)
                        except Exception as e:
                            failures.append(f"ai_pe_script_failed: {_format_exception_brief(e)}")
                            logger.info("AI PE-SCRIPT simulation failed; falling back: %s", e)
                    else:
                        failures.append("ai_sim_disabled: agent.simulation_ai_enabled=false")

                    if (
                        bool(getattr(cfg.agent, "simulation_enabled", True))
                        and experiment_context is not None
                        and isinstance(experiment_context.get("summary_id"), str)
                        and isinstance(experiment_context.get("category"), str)
                    ):
                        summary_id = str(experiment_context.get("summary_id"))
                        category_value = str(experiment_context.get("category"))
                        try:
                            status = get_status_save(
                                user,
                                summary_id=summary_id,
                                category_value=category_value,
                                cache_dir=cache_dir,
                                ttl_sec=300,
                            )
                            reply = simulate_status_save_with_phyengine(
                                text=sim_text,
                                status_save=status,
                                phy_engine_cfg=cfg.phy_engine,
                                config_base_dir=config_base_dir,
                                max_elements=int(getattr(cfg.agent, "simulation_max_elements", 300) or 300),
                            )
                            return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)
                        except Exception as e:
                            failures.append(f"status_save_sim_failed: {_format_exception_brief(e)}")
                            logger.info("StatusSave simulation failed; falling back to demo: %s", e)
                    else:
                        failures.append("status_save_not_attempted: no experiment_context (summary_id/category)")
                    try:
                        reply = simulate_series_vdc_resistors(
                            text=sim_text,
                            phy_engine_cfg=cfg.phy_engine,
                            config_base_dir=config_base_dir,
                        )
                        if not _looks_like_series_demo_prompt(reply) or _looks_like_series_demo_request(sim_text):
                            return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)
                        failures.append("series_demo_not_applicable: needs V and resistor values")
                    except Exception as e:
                        failures.append(f"series_demo_failed: {_format_exception_brief(e)}")
                        logger.info("Fallback demo simulation failed: %s", e)

                    return safe_reply(
                        _render_simulation_failure(
                            sim_text=sim_text,
                            cfg=cfg,
                            failures=failures
                            or [
                                "no_details: all simulation paths returned 'not applicable' without exceptions"
                            ],
                            ai_parse_failure=ai_parse_failure,
                        ),
                        max_chars=cfg.agent.max_reply_chars,
                    )

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
                    # Community rule: publishing is forced to Discussion (server-side category_value).
                    publish_category_value = "Discussion"
                    required_by = _format_required_by_prefix(nickname=nickname, user_id=author_id)
                    meta = _llm_generate_publish_title_intro(ollama=ollama, spec=routed_arg) if enable_publish else None
                    if meta is not None:
                        title, intro_body = meta
                    else:
                        title = truncate(f"Auto Circuit: {routed_arg}", max_chars=60)
                        intro_body = truncate(f"Spec:\n{routed_arg}", max_chars=520)
                    introduction = truncate(required_by + "\n" + intro_body, max_chars=600)
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
                        logger.warning(
                            "Circuit compilation failed (auto_routed=true): %s",
                            _public_error_text(e, max_chars=2000),
                        )
                        return safe_reply(
                            "Sorry, I couldn't compile the circuit after multiple attempts.\n"
                            f"Error: {_public_error_text(e)}",
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
                        cat = "Discussion"
                        open_hint = None
                        if isinstance(res.summary_id, str) and res.summary_id.strip():
                            open_hint = f"discussion:{res.summary_id.strip()}"
                        return safe_reply(
                            "Done. I generated Verilog, compiled to .sav with -O4 and '--layout hier', and published it.\n"
                            f"Category: {cat}\nSummaryID: {res.summary_id}\nOpen hint: {open_hint or ''}".rstrip(),
                            max_chars=cfg.agent.max_reply_chars,
                        )
                    return safe_reply(
                        "Done. I generated Verilog and compiled a .sav with -O4 and '--layout hier'.\n"
                        + (
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
                if _looks_like_simulation_request(arg) and bool(
                    getattr(cfg.agent, "simulation_enabled", True)
                ):
                    if bool(getattr(cfg.agent, "simulation_ai_enabled", True)):
                        try:
                            reply = simulate_ai_script_circuit_with_phyengine(
                                ollama=ollama,
                                text=arg,
                                context_json=experiment_context,
                                phy_engine_cfg=cfg.phy_engine,
                                config_base_dir=config_base_dir,
                                max_components=int(
                                    getattr(cfg.agent, "simulation_ai_max_components", 30) or 30
                                ),
                                max_probes=int(
                                    getattr(cfg.agent, "simulation_ai_max_probes", 20) or 20
                                ),
                            )
                            if not _looks_like_pe_script_parse_failure(reply):
                                return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

                            try:
                                reply2 = simulate_ai_circuit_with_phyengine(
                                    ollama=ollama,
                                    text=arg,
                                    context_json=experiment_context,
                                    phy_engine_cfg=cfg.phy_engine,
                                    config_base_dir=config_base_dir,
                                    max_attempts=3,
                                    max_components=int(
                                        getattr(cfg.agent, "simulation_ai_max_components", 30) or 30
                                    ),
                                    max_probes=int(
                                        getattr(cfg.agent, "simulation_ai_max_probes", 20) or 20
                                    ),
                                )
                                return safe_reply(reply2, max_chars=cfg.agent.max_reply_chars)
                            except Exception as e:
                                logger.info("AI JSON-spec simulation failed; falling back: %s", e)
                        except Exception as e:
                            logger.info("AI PE-SCRIPT simulation failed; falling back: %s", e)

                    if (
                        experiment_context is not None
                        and isinstance(experiment_context.get("summary_id"), str)
                        and isinstance(experiment_context.get("category"), str)
                    ):
                        try:
                            status = get_status_save(
                                user,
                                summary_id=str(experiment_context.get("summary_id")),
                                category_value=str(experiment_context.get("category")),
                                cache_dir=cache_dir,
                                ttl_sec=300,
                            )
                            reply = simulate_status_save_with_phyengine(
                                text=arg,
                                status_save=status,
                                phy_engine_cfg=cfg.phy_engine,
                                config_base_dir=config_base_dir,
                                max_elements=int(
                                    getattr(cfg.agent, "simulation_max_elements", 300) or 300
                                ),
                            )
                            return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)
                        except Exception as e:
                            logger.info("Fallback simulation failed: %s", e)
                    try:
                        reply = simulate_series_vdc_resistors(
                            text=arg,
                            phy_engine_cfg=cfg.phy_engine,
                            config_base_dir=config_base_dir,
                        )
                        if not _looks_like_series_demo_prompt(reply) or _looks_like_series_demo_request(arg):
                            return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)
                    except Exception as e:
                        logger.info("Fallback demo simulation failed: %s", e)

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
                    publish_category_value = "Discussion"
                    required_by = _format_required_by_prefix(nickname=nickname, user_id=author_id)
                    meta = _llm_generate_publish_title_intro(ollama=ollama, spec=arg) if enable_publish else None
                    if meta is not None:
                        title, intro_body = meta
                    else:
                        title = truncate(f"Auto Circuit: {arg}", max_chars=60)
                        intro_body = truncate(f"Spec:\n{arg}", max_chars=520)
                    introduction = truncate(required_by + "\n" + intro_body, max_chars=600)
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
                        logger.warning(
                            "Circuit compilation failed (auto_routed=false): %s",
                            _public_error_text(e, max_chars=2000),
                        )
                        return safe_reply(
                            "Sorry, I couldn't compile the circuit after multiple attempts.\n"
                            f"Error: {_public_error_text(e)}",
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
                        cat = "Discussion"
                        open_hint = None
                        if isinstance(res.summary_id, str) and res.summary_id.strip():
                            open_hint = f"discussion:{res.summary_id.strip()}"
                        return safe_reply(
                            "Done. I generated Verilog, compiled to .sav with -O4 and '--layout hier', and published it.\n"
                            f"Category: {cat}\nSummaryID: {res.summary_id}\nOpen hint: {open_hint or ''}".rstrip(),
                            max_chars=cfg.agent.max_reply_chars,
                        )
                    return safe_reply(
                        "Done. I generated Verilog and compiled a .sav with -O4 and '--layout hier'.\n"
                        + (
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
                "You are replying to a comment on the current page. "
                "Use the following context to answer accurately.\n"
                + json.dumps(experiment_context, ensure_ascii=False, indent=2)
                + "\n\n"
            )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
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
                    system_prompt=system_prompt,
                    user_text=arg,
                    context_json=experiment_context,
                )
            except Exception as e:
                logger.debug("Web router failed: %s", e)
                use_web, query = False, ""

            if use_web:
                if _looks_political_sensitive(query or arg):
                    return _political_refusal_message(query or arg)
                logger.info(
                    "Auto web search triggered (query=%r)", truncate(query, max_chars=200)
                )
                try:
                    search_txt = web_search(
                        query=query,
                        cache_dir=cache_dir,
                        provider=str(getattr(cfg.agent, "web_search_provider", "google") or "google"),
                        proxy=str(getattr(cfg.agent, "web_search_proxy", "") or ""),
                        timeout_sec=int(getattr(cfg.agent, "web_search_timeout_sec", 20) or 20),
                        ttl_sec=int(getattr(cfg.agent, "web_search_cache_ttl_sec", 3600) or 3600),
                        max_results=int(getattr(cfg.agent, "web_search_max_results", 5) or 5),
                        fallback_to_ddg=bool(getattr(cfg.agent, "web_search_fallback_to_ddg", True)),
                        user_agent=str(getattr(cfg.agent, "web_search_user_agent", "") or ""),
                        searxng_base_url=str(
                            getattr(cfg.agent, "web_search_searxng_base_url", "") or ""
                        ),
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

    if cmd in ("agent", "op", "autopilot"):
        run_task = (arg or "").strip()
        if not run_task:
            return "Provide a task description for agent mode."
        if agent_mode != "agent":
            return "Agent mode is disabled. Set agent.mode=agent in config to enable it."
        logger.info("Agent mode invoked (cmd=%s task_len=%d)", cmd, len(run_task))
        return agent_mode_run(
            ollama=ollama,
            user=user,
            cfg=cfg,
            cache_dir=cache_dir,
            config_base_dir=config_base_dir,
            dry_run=dry_run,
            logger=logger,
            task=run_task,
            context_json=experiment_context,
            history=history,
        )

    if cmd in ("summarize", "sum"):
        # Allow summarizing another user's message board by name/ID.
        nick, uid = _extract_user_board_query(arg)
        if nick or uid:
            try:
                board_user_id = ""
                board_user_obj: dict[str, Any] | None = None
                if uid:
                    board_user_id = uid
                    data = get_user_by_id(user, user_id=uid)
                    board_user_obj = _extract_user_obj_from_get_user_data(data)
                else:
                    data = get_user_by_name(user, name=str(nick or ""))
                    board_user_obj = _extract_user_obj_from_get_user_data(data)
                    if isinstance(board_user_obj, dict):
                        board_user_id = (
                            best_effort_extract_text(board_user_obj.get("ID"))
                            or best_effort_extract_text(board_user_obj.get("UserID"))
                        )

                if not board_user_id:
                    return "No user found."

                comments = get_comments(
                    user,
                    target_id=board_user_id,
                    target_type="User",
                    take=int(getattr(cfg.agent, "take", 20) or 20),
                    skip=0,
                )
                ctx = _build_user_board_context(
                    user=user,
                    board_user_id=board_user_id,
                    comments=comments,
                )
                if isinstance(board_user_obj, dict):
                    nickname = (
                        best_effort_extract_text(board_user_obj.get("Nickname"))
                        or best_effort_extract_text(board_user_obj.get("Name"))
                        or None
                    )
                    ctx["board_owner"] = {
                        "id": board_user_id,
                        "nickname": nickname or ctx.get("board_owner", {}).get("nickname"),
                        "signature": ctx.get("board_owner", {}).get("signature"),
                    }
                    if nickname:
                        ctx["title"] = f"{nickname} 的留言板"

                messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
                messages.append(
                    {
                        "role": "system",
                        "content": "Context JSON (current page):\n"
                        + json.dumps(ctx, ensure_ascii=False, indent=2),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Summarize this user's message board using the Context JSON. Keep it under 6 bullets.",
                    }
                )
                reply = ollama.chat(messages=messages)
                return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)
            except Exception as e:
                return f"Failed to summarize user board: {e}"

        use_page_context = experiment_context is not None and _is_generic_summarize_arg(arg)

        if (not arg) and (experiment_context is None):
            return "Provide text to summarize."

        if use_page_context:
            logger.debug("Tool summarize invoked (using page context)")
            messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
            messages.append(
                {
                    "role": "system",
                    "content": "Context JSON (current page):\n"
                    + json.dumps(experiment_context, ensure_ascii=False, indent=2),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Summarize the current page. Use Context JSON fields like title/body_text/content_text/recent_comments/plsav_summary. Keep it under 6 bullets.",
                }
            )
            reply = ollama.chat(messages=messages)
            return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

        logger.debug("Tool summarize invoked (len=%d)", len(arg))
        reply = llm_summarize(ollama=ollama, system_prompt=system_prompt, text=arg)
        return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

    if cmd in ("search", "find"):
        if not arg:
            return "Provide a query string."
        if _looks_political_sensitive(arg):
            return _political_refusal_message(arg)
        spec = _parse_plar_lookup_query(arg)
        kind = spec.get("kind")
        value = (spec.get("value") or "").strip()

        def _render_user(u: dict[str, Any]) -> str:
            uid = best_effort_extract_text(u.get("ID")) or best_effort_extract_text(u.get("UserID"))
            nick = best_effort_extract_text(u.get("Nickname")) or "(no nickname)"
            ver = best_effort_extract_text(u.get("Verification"))
            sig = best_effort_extract_text(u.get("Signature"))
            lines = ["User:"]
            lines.append(f"- Nickname: {nick}")
            if uid:
                lines.append(f"- ID: {uid}")
            if ver:
                lines.append(f"- Verification: {ver}")
            if sig:
                lines.append(f"- Signature: {truncate(sig, max_chars=120)}")
            return "\n".join(lines)

        if kind == "user_name":
            if not value:
                return "Search is lookup-only. Try '@name' or 'user: name'."
            logger.debug("Tool search(lookup user_name) invoked (name=%r)", value[:80])
            try:
                data = get_user_by_name(user, name=value)
            except Exception as e:
                if _is_probable_user_not_found_error(e):
                    return "No user found."
                logger.warning("User lookup failed: %s", _public_error_text(e, max_chars=2000))
                return f"Lookup failed: {e}"
            u = data.get("User") if isinstance(data, dict) else None
            if not isinstance(u, dict):
                return "No user found."
            return safe_reply(_render_user(u), max_chars=cfg.agent.max_reply_chars)

        if kind == "user_id":
            if not value:
                return "Search is lookup-only. Try 'uid: <id>'."
            logger.debug("Tool search(lookup user_id) invoked (id=%r)", value[:80])
            try:
                data = get_user_by_id(user, user_id=value)
            except Exception as e:
                if _is_probable_user_not_found_error(e):
                    return "No user found."
                logger.warning("User lookup failed: %s", _public_error_text(e, max_chars=2000))
                return f"Lookup failed: {e}"
            u = data.get("User") if isinstance(data, dict) else None
            if not isinstance(u, dict):
                return "No user found."
            return safe_reply(_render_user(u), max_chars=cfg.agent.max_reply_chars)

        if kind == "content_id":
            if not value:
                return "Search is lookup-only. Try 'experiment:<id>' or 'discussion:<id>'."
            category = (spec.get("category") or "both").strip()
            logger.debug(
                "Tool search(lookup content_id) invoked (id=%r category=%s)", value[:80], category
            )
            tried: list[str] = []
            last_err: BaseException | None = None

            def _try(cat: str) -> dict[str, Any] | None:
                nonlocal last_err
                tried.append(cat)
                try:
                    ctx = get_experiment_context(
                        user,
                        summary_id=value,
                        category_value=cat,
                        cache_dir=cache_dir,
                        ttl_sec=300,
                    )
                except Exception as e:
                    last_err = e
                    return None
                return ctx if isinstance(ctx, dict) else None

            ctx = None
            if category == "Experiment":
                ctx = _try("Experiment")
            elif category == "Discussion":
                ctx = _try("Discussion")
            else:
                ctx = _try("Experiment") or _try("Discussion")

            if ctx is None:
                if last_err is not None and _is_probable_content_not_found_error(last_err):
                    return safe_reply(
                        f"No content found. (tried: {', '.join(tried)})",
                        max_chars=cfg.agent.max_reply_chars,
                    )
                return f"Lookup failed: {last_err}"

            subject = (
                best_effort_extract_text(ctx.get("subject") or ctx.get("Subject") or ctx.get("title"))
                or "(no subject)"
            )
            cat = best_effort_extract_text(ctx.get("category") or ctx.get("Category")) or (
                tried[-1] if tried else None
            )
            lines = ["Content:"]
            if cat:
                lines.append(f"- Category: {cat}")
            lines.append(f"- Subject: {subject}")
            lines.append(f"- ID: {value}")
            return safe_reply("\n".join(lines), max_chars=cfg.agent.max_reply_chars)

        return "Search is lookup-only: @name/user:name/uid:<id>/experiment:<id>/discussion:<id>."

    if cmd in ("google", "web", "websearch"):
        if not arg:
            return "Provide a query string."
        if _looks_political_sensitive(arg):
            return _political_refusal_message(arg)
        if not bool(getattr(cfg.agent, "web_search_enabled", False)):
            return "Web search is disabled. Set agent.web_search_enabled=true in config."
        logger.debug("Tool google invoked (query=%r)", arg[:200])
        try:
            search_txt = web_search(
                query=arg,
                cache_dir=cache_dir,
                provider=str(getattr(cfg.agent, "web_search_provider", "google") or "google"),
                proxy=str(getattr(cfg.agent, "web_search_proxy", "") or ""),
                timeout_sec=int(getattr(cfg.agent, "web_search_timeout_sec", 20) or 20),
                ttl_sec=int(getattr(cfg.agent, "web_search_cache_ttl_sec", 3600) or 3600),
                max_results=int(getattr(cfg.agent, "web_search_max_results", 5) or 5),
                fallback_to_ddg=bool(getattr(cfg.agent, "web_search_fallback_to_ddg", True)),
                user_agent=str(getattr(cfg.agent, "web_search_user_agent", "") or ""),
                searxng_base_url=str(getattr(cfg.agent, "web_search_searxng_base_url", "") or ""),
            )
        except Exception as e:
            return f"Web search failed: {e}"
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.append(
            {
                "role": "system",
                "content": "Web search results (use as external references; include relevant links):\n"
                + search_txt,
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "Answer the user's query using the web search results above.\n"
                "- If the results are blocked/empty, say so briefly and suggest a fix (proxy or SearXNG).\n"
                "- Include up to 5 relevant links.\n\n"
                f"User query:\n{arg}",
            }
        )
        reply = ollama.chat(messages=messages)
        return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)

    if cmd in ("circuit", "verilog"):
        if not arg:
            return "Provide a circuit specification after the command."

        publish_category_value = "Discussion"
        publish_tags = getattr(cfg.agent, "publish_tags", None)
        publish_tags_list = None
        if isinstance(publish_tags, list) and all(isinstance(x, str) for x in publish_tags):
            publish_tags_list = [x.strip() for x in publish_tags if x.strip()]

        required_by = _format_required_by_prefix(nickname=nickname, user_id=author_id)
        meta = (
            _llm_generate_publish_title_intro(ollama=ollama, spec=arg)
            if bool(cfg.agent.enable_publish) and (not dry_run)
            else None
        )
        if meta is not None:
            title, intro_body = meta
        else:
            title = truncate(f"Auto Circuit: {arg}", max_chars=60)
            intro_body = truncate(f"Spec:\n{arg}", max_chars=520)
        introduction = truncate(required_by + "\n\n" + intro_body, max_chars=600)
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
                publish_category_value=publish_category_value,
                publish_tags=publish_tags_list,
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
            cat = "Discussion"
            open_hint = None
            if isinstance(res.summary_id, str) and res.summary_id.strip():
                open_hint = f"discussion:{res.summary_id.strip()}"
            return safe_reply(
                f"Your circuit has been generated and published.\n"
                f"Category: {cat}\nSummaryID: {res.summary_id}\nOpen hint: {open_hint or ''}".rstrip()
                + "\n"
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
        failures: list[str] = []
        ai_parse_failure: str | None = None
        if bool(getattr(cfg.agent, "simulation_ai_enabled", True)):
            try:
                reply = simulate_ai_script_circuit_with_phyengine(
                    ollama=ollama,
                    text=arg or text,
                    context_json=experiment_context,
                    phy_engine_cfg=cfg.phy_engine,
                    config_base_dir=config_base_dir,
                    max_components=int(getattr(cfg.agent, "simulation_ai_max_components", 30) or 30),
                    max_probes=int(getattr(cfg.agent, "simulation_ai_max_probes", 20) or 20),
                )
                if _looks_like_pe_script_parse_failure(reply):
                    ai_parse_failure = reply
                    reply = simulate_ai_circuit_with_phyengine(
                        ollama=ollama,
                        text=arg or text,
                        context_json=experiment_context,
                        phy_engine_cfg=cfg.phy_engine,
                        config_base_dir=config_base_dir,
                        max_attempts=3,
                        max_components=int(
                            getattr(cfg.agent, "simulation_ai_max_components", 30) or 30
                        ),
                        max_probes=int(getattr(cfg.agent, "simulation_ai_max_probes", 20) or 20),
                    )
                return safe_reply(reply, max_chars=cfg.agent.max_reply_chars)
            except Exception as e:
                failures.append(f"ai_sim_failed: {_format_exception_brief(e)}")
                logger.info("AI simulation failed; falling back to demo: %s", e)
        else:
            failures.append("ai_sim_disabled: agent.simulation_ai_enabled=false")

        try:
            reply = simulate_series_vdc_resistors(
                text=arg or text,
                phy_engine_cfg=cfg.phy_engine,
                config_base_dir=config_base_dir,
            )
        except Exception as e:
            failures.append(f"series_demo_failed: {_format_exception_brief(e)}")
            return safe_reply(
                _render_simulation_failure(
                    sim_text=(arg or text),
                    cfg=cfg,
                    failures=failures,
                    ai_parse_failure=ai_parse_failure,
                ),
                max_chars=cfg.agent.max_reply_chars,
            )
        if _looks_like_series_demo_prompt(reply) and not _looks_like_series_demo_request(arg or text):
            failures.append("series_demo_not_applicable: needs V and resistor values")
            return safe_reply(
                _render_simulation_failure(
                    sim_text=(arg or text),
                    cfg=cfg,
                    failures=failures,
                    ai_parse_failure=ai_parse_failure,
                ),
                max_chars=cfg.agent.max_reply_chars,
            )
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
            reply_to_self=bool(reply_to_self and getattr(cfg.agent, "trigger_on_reply_to_self", False)),
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

        conversation_key = None
        if isinstance(author_id, str) and author_id.strip():
            conversation_key = f"{target_key}|{author_id.strip()}"
        if bool(getattr(cfg.agent, "reply_once", True)) and conversation_key:
            ttl_sec = int(getattr(cfg.agent, "reply_once_ttl_sec", 0) or 0)
            if _is_conversation_closed(state, key=conversation_key, now_ms=now_ms, ttl_sec=ttl_sec):
                logger.info("[%s] Skip comment: reply cooldown active", target_key)
                target_state.processed_comment_keys.append(key)
                processed.add(key)
                target_state.last_seen_timestamp_ms = max(target_state.last_seen_timestamp_ms, ts)
                continue

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
                if bool(getattr(cfg.agent, "reply_once", True)) and conversation_key:
                    _mark_conversation_closed(state, key=conversation_key, now_ms=now_ms)
                target_state.processed_comment_keys.append(key)
                processed.add(key)
                target_state.last_seen_timestamp_ms = max(target_state.last_seen_timestamp_ms, ts)
                prune_processed_keys(target_state, keep_last=500)
                continue
        reply_once = bool(getattr(cfg.agent, "reply_once", True))
        history = [] if reply_once else (
            get_conversation_history(state, key=conversation_key, max_turns=12) if conversation_key else []
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
        elif target.type == "User":
            experiment_context = _build_user_board_context(
                user=user,
                board_user_id=target.id,
                comments=comments,
            )

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
            post_body = f"{prefix or ''}{_strip_public_mentions(reply)}".strip()
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
            if reply_once:
                if reply:
                    _mark_conversation_closed(state, key=conversation_key, now_ms=now_ms)
            else:
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


def _bootstrap_last_seen(
    *,
    target_state: Any,
    cfg: Any,
    logger: logging.Logger,
    target_key: str,
) -> None:
    if int(getattr(target_state, "last_seen_timestamp_ms", 0) or 0) != 0:
        return
    lookback_sec = float(getattr(cfg.agent, "bootstrap_lookback_sec", 0.0) or 0.0)
    if lookback_sec <= 0:
        return
    target_state.last_seen_timestamp_ms = max(
        0, int(time.time() * 1000) - int(lookback_sec * 1000)
    )
    logger.info(
        "[%s] Bootstrapping last_seen to %d (lookback_sec=%.1f)",
        target_key,
        target_state.last_seen_timestamp_ms,
        lookback_sec,
    )


def _update_last_seen_capped(
    *,
    target_state: Any,
    ts_ms: int,
    pending_min_ts_ms: int | None,
) -> None:
    cur = int(getattr(target_state, "last_seen_timestamp_ms", 0) or 0)
    ts_ms = int(ts_ms)
    candidate = max(cur, ts_ms)
    if isinstance(pending_min_ts_ms, int) and pending_min_ts_ms > 0 and candidate >= pending_min_ts_ms:
        cap = max(cur, pending_min_ts_ms - 1)
        candidate = min(candidate, cap)
    target_state.last_seen_timestamp_ms = candidate


def _fetch_comments_paged(
    *,
    user: Any,
    target: TargetConfig,
    take: int,
    last_seen_ms: int,
    max_pages: int,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    skip = 0
    for _ in range(max(1, int(max_pages))):
        try:
            page = get_comments(
                user,
                target_id=target.id,
                target_type=target.type,
                take=int(take),
                skip=int(skip),
            )
        except Exception as e:
            logger.warning("[%s:%s] Fetch comments failed (skip=%d take=%d): %s", target.type, target.id, skip, take, e)
            break
        if not page:
            break

        min_ts = None
        for c in page:
            if not isinstance(c, dict):
                continue
            k = _comment_key(c)
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
            ts = _comment_timestamp_ms(c)
            if isinstance(ts, int):
                min_ts = ts if min_ts is None else min(min_ts, ts)

        if last_seen_ms > 0 and isinstance(min_ts, int) and min_ts <= last_seen_ms:
            break
        skip += int(take)

    return out


def _scan_and_enqueue(
    *,
    user: Any,
    target: TargetConfig,
    state: AgentState,
    state_lock: threading.RLock,
    cfg: Any,
    pending: _PendingTracker,
    work_q: "Queue[_WorkItem]",
    max_enqueue: int,
    logger: logging.Logger,
) -> int:
    target_key = f"{target.type}:{target.id}"
    take = int(getattr(cfg.agent, "take", 20) or 20)
    max_pages = int(getattr(cfg.agent, "comment_scan_pages", 5) or 5)
    with state_lock:
        target_state = get_target_state(state, target_key)
        _bootstrap_last_seen(
            target_state=target_state, cfg=cfg, logger=logger, target_key=target_key
        )
        last_seen_ms = int(target_state.last_seen_timestamp_ms)

    comments = _fetch_comments_paged(
        user=user,
        target=target,
        take=take,
        last_seen_ms=last_seen_ms,
        max_pages=max_pages,
        logger=logger,
    )
    if not comments:
        return 0

    # Process in chronological order to keep behavior stable.
    comments = [c for c in comments if isinstance(c, dict)]
    comments.sort(key=lambda c: _comment_timestamp_ms(c) or 0)

    enq = 0

    with state_lock:
        target_state = get_target_state(state, target_key)
        processed = set(target_state.processed_comment_keys)

    for comment in comments:
        if enq >= max_enqueue:
            break

        ts = _comment_timestamp_ms(comment)
        if not isinstance(ts, int):
            continue
        key = _comment_key(comment)
        if not key:
            continue

        if pending.has(target_key, key):
            continue

        if key in processed:
            with state_lock:
                target_state = get_target_state(state, target_key)
                _update_last_seen_capped(
                    target_state=target_state,
                    ts_ms=ts,
                    pending_min_ts_ms=pending.min_ts(target_key),
                )
            continue

        author_id, nickname = _comment_author(comment)
        if author_id is not None and getattr(user, "user_id", None) == author_id:
            with state_lock:
                target_state = get_target_state(state, target_key)
                target_state.processed_comment_keys.append(key)
                processed.add(key)
                _update_last_seen_capped(
                    target_state=target_state,
                    ts_ms=ts,
                    pending_min_ts_ms=pending.min_ts(target_key),
                )
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
            reply_to_self=bool(reply_to_self and getattr(cfg.agent, "trigger_on_reply_to_self", False)),
        )
        if not triggered:
            with state_lock:
                target_state = get_target_state(state, target_key)
                target_state.processed_comment_keys.append(key)
                processed.add(key)
                _update_last_seen_capped(
                    target_state=target_state,
                    ts_ms=ts,
                    pending_min_ts_ms=pending.min_ts(target_key),
                )
            continue

        conversation_key = None
        if isinstance(author_id, str) and author_id.strip():
            conversation_key = f"{target_key}|{author_id.strip()}"
        if bool(getattr(cfg.agent, "reply_once", True)) and conversation_key:
            ttl_sec = int(getattr(cfg.agent, "reply_once_ttl_sec", 0) or 0)
            with state_lock:
                if _is_conversation_closed(state, key=conversation_key, now_ms=int(time.time() * 1000), ttl_sec=ttl_sec):
                    target_state = get_target_state(state, target_key)
                    target_state.processed_comment_keys.append(key)
                    processed.add(key)
                    _update_last_seen_capped(
                        target_state=target_state,
                        ts_ms=ts,
                        pending_min_ts_ms=pending.min_ts(target_key),
                    )
                    pending.remove(target_key, key)
                    logger.info("[%s] Skip enqueue: reply cooldown active", target_key)
                    continue

        # Reserve this comment for a worker.
        pending.add(target_key, key, ts)
        work_q.put(
            _WorkItem(target=target, comment=comment, comment_key=key, comment_ts_ms=ts)
        )
        enq += 1

    with state_lock:
        target_state = get_target_state(state, target_key)
        prune_processed_keys(target_state, keep_last=500)
    return enq


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
    state_lock: threading.RLock,
    cfg: Any,
    logger: logging.Logger,
) -> list[TargetConfig]:
    if not getattr(cfg.agent, "notifications_enabled", True):
        return []
    category_ids = list(getattr(cfg.agent, "notification_category_ids", [3]) or [3])
    take = int(getattr(cfg.agent, "notification_take", 20) or 20)

    discovered: dict[str, TargetConfig] = {}

    for cat in category_ids:
        msg_target_key = f"Messages:{int(cat)}"
        with state_lock:
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

        with state_lock:
            msg_state = get_target_state(state, msg_target_key)
            processed = set(msg_state.processed_comment_keys)
        for m in new_msgs:
            ts = _message_timestamp_ms(m)
            if ts is None:
                continue
            key = _message_key(m)
            if key in processed:
                with state_lock:
                    msg_state = get_target_state(state, msg_target_key)
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

            with state_lock:
                msg_state = get_target_state(state, msg_target_key)
                msg_state.processed_comment_keys.append(key)
                processed.add(key)
                msg_state.last_seen_timestamp_ms = max(msg_state.last_seen_timestamp_ms, ts)

        with state_lock:
            msg_state = get_target_state(state, msg_target_key)
            prune_processed_keys(msg_state, keep_last=500)

    if not discovered:
        return []

    logger.info("Discovered %d targets from notifications", len(discovered))
    return list(discovered.values())


def _process_work_item(
    *,
    user: Any,
    item: _WorkItem,
    state: AgentState,
    state_path: str,
    state_lock: threading.RLock,
    pending: _PendingTracker,
    cfg: Any,
    cache_dir: str,
    config_base_dir: str,
    dry_run: bool,
    logger: logging.Logger,
    ollama: Any,
) -> None:
    target = item.target
    comment = item.comment
    key = item.comment_key
    ts = int(item.comment_ts_ms)
    target_key = f"{target.type}:{target.id}"

    author_id, nickname = _comment_author(comment)
    if author_id is not None and getattr(user, "user_id", None) == author_id:
        pending.remove(target_key, key)
        return

    content = _comment_content(comment) or ""

    now_ms = int(time.time() * 1000)
    conversation_key = None
    if isinstance(author_id, str) and author_id.strip():
        conversation_key = f"{target_key}|{author_id.strip()}"
    if bool(getattr(cfg.agent, "reply_once", True)) and conversation_key:
        ttl_sec = int(getattr(cfg.agent, "reply_once_ttl_sec", 0) or 0)
        with state_lock:
            if _is_conversation_closed(state, key=conversation_key, now_ms=now_ms, ttl_sec=ttl_sec):
                target_state = get_target_state(state, target_key)
                target_state.processed_comment_keys.append(key)
                _update_last_seen_capped(
                    target_state=target_state,
                    ts_ms=ts,
                    pending_min_ts_ms=pending.min_ts(target_key),
                )
                prune_processed_keys(target_state, keep_last=500)
                pending.remove(target_key, key)
                save_state(state_path, state)
                logger.info("[%s] Skip reply: reply cooldown active", target_key)
                return
    if bool(getattr(cfg.agent, "overload_protection_enabled", True)):
        window_ms = int(getattr(cfg.agent, "overload_window_sec", 600) or 600) * 1000
        max_req = int(getattr(cfg.agent, "overload_max_requests", 40) or 40)
        with state_lock:
            recent_n = count_recent_requests(state, now_ms=now_ms, window_ms=window_ms)
        if max_req > 0 and recent_n >= max_req:
            busy_msg = str(getattr(cfg.agent, "overload_message_en", "") or "").strip()
            if not busy_msg:
                busy_msg = "Too many requests at the moment, please try again later."
            prefix = safe_mention_prefix(nickname) if nickname else None
            post_body = f"{prefix or ''}{busy_msg}".strip()
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

            with state_lock:
                record_request_timestamp(state, ts_ms=now_ms)
                if bool(getattr(cfg.agent, "reply_once", True)) and conversation_key:
                    _mark_conversation_closed(state, key=conversation_key, now_ms=now_ms)
                target_state = get_target_state(state, target_key)
                target_state.processed_comment_keys.append(key)
                _update_last_seen_capped(
                    target_state=target_state,
                    ts_ms=ts,
                    pending_min_ts_ms=pending.min_ts(target_key),
                )
                prune_processed_keys(target_state, keep_last=500)
                pending.remove(target_key, key)
                save_state(state_path, state)
            return

    with state_lock:
        reply_once = bool(getattr(cfg.agent, "reply_once", True))
        history = [] if reply_once else (
            get_conversation_history(state, key=conversation_key, max_turns=12) if conversation_key else []
        )

    experiment_context: dict[str, Any] | None = None
    if target.type in ("Experiment", "Discussion"):
        try:
            comments = get_comments(
                user,
                target_id=target.id,
                target_type=target.type,
                take=int(getattr(cfg.agent, "take", 20) or 20),
                skip=0,
            )
        except Exception:
            comments = []
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
        except Exception as e:
            logger.debug("[%s] Failed to load experiment context: %s", target_key, e)
    elif target.type == "User":
        try:
            comments = get_comments(
                user,
                target_id=target.id,
                target_type=target.type,
                take=int(getattr(cfg.agent, "take", 20) or 20),
                skip=0,
            )
        except Exception:
            comments = []
        experiment_context = _build_user_board_context(
            user=user,
            board_user_id=target.id,
            comments=comments,
        )

    reply: str | None = None
    try:
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
    except Exception as e:
        # Avoid getting stuck retrying the same comment forever: we will mark it processed below.
        details = getattr(e, "details", None)
        if isinstance(e, OllamaError) and isinstance(details, dict) and details:
            is_empty = "empty 'message.content'" in (str(e) or "").casefold()
            logger.warning(
                "[%s] LLM failed (comment=%s): %s details=%s",
                target_key,
                key,
                e,
                {
                    k: details.get(k)
                    for k in (
                        "attempt",
                        "max_attempts",
                        "had_thinking",
                        "thinking_len",
                        "content_len",
                        "done_reason",
                        "eval_count",
                        "prompt_eval_count",
                        "num_predict",
                    )
                },
                # Empty-content failures are expected/transient; avoid log spam with full tracebacks.
                exc_info=(False if is_empty else True),
            )
        else:
            logger.warning("[%s] Handle comment failed (comment=%s): %s", target_key, key, e, exc_info=True)

        # Public-facing fallback (one sentence) so the user gets an answer and the thread doesn't spam.
        msg_low = (str(e) or "").casefold()
        is_timeout = any(x in msg_low for x in ("timeout", "timed out", "readtimeout", "connecttimeout"))
        is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in (content or ""))
        if is_cjk:
            reply = "抱歉，本次请求超时（Agent 未能在限制时间内完成）。请稍后再试。" if is_timeout else "抱歉，我这次生成回复失败，请稍后再试。"
        else:
            reply = (
                "Sorry — the request timed out (agent couldn't finish within the time limit). Please try again later."
                if is_timeout
                else "Sorry — I failed to generate a reply this time. Please try again later."
            )

    with state_lock:
        record_request_timestamp(state, ts_ms=now_ms)

    posted_ok = False
    if reply:
        prefix = safe_mention_prefix(nickname) if nickname else None
        post_body = f"{prefix or ''}{_strip_public_mentions(reply)}".strip()
        if dry_run:
            logger.info("[%s] DRY RUN reply: %s", target_key, post_body)
            posted_ok = True
        else:
            try:
                post_comment(
                    user,
                    target_id=target.id,
                    target_type=target.type,
                    content=post_body,
                    reply_id=author_id,
                )
                posted_ok = True
                logger.info("[%s] Replied to %s", target_key, nickname or author_id or "unknown")
            except Exception as e:
                # Still mark the comment processed to avoid dead loops; the user can re-comment to retry.
                logger.warning("[%s] Posting reply failed (comment=%s): %s", target_key, key, e, exc_info=True)

    with state_lock:
        if conversation_key:
            if bool(getattr(cfg.agent, "reply_once", True)):
                if reply and posted_ok:
                    _mark_conversation_closed(state, key=conversation_key, now_ms=now_ms)
            else:
                user_text = strip_leading_mention(content, mention_tag=cfg.agent.mention_tag)
                normalized_user_turn = user_text
                if cfg.agent.commands_enabled:
                    cmd_name, cmd_arg = parse_command(user_text, prefix=cfg.agent.command_prefix)
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

        target_state = get_target_state(state, target_key)
        target_state.processed_comment_keys.append(key)
        _update_last_seen_capped(
            target_state=target_state,
            ts_ms=ts,
            pending_min_ts_ms=pending.min_ts(target_key),
        )
        prune_processed_keys(target_state, keep_last=500)
        pending.remove(target_key, key)
        save_state(state_path, state)


def _worker_loop(
    *,
    worker_id: int,
    user: Any,
    work_q: "Queue[_WorkItem]",
    capacity: threading.BoundedSemaphore,
    stop_event: threading.Event,
    state: AgentState,
    state_path: str,
    state_lock: threading.RLock,
    pending: _PendingTracker,
    cfg: Any,
    cache_dir: str,
    config_base_dir: str,
    dry_run: bool,
    logger: logging.Logger,
    ollama: Any,
) -> None:
    while not stop_event.is_set():
        try:
            item = work_q.get(timeout=0.5)
        except Empty:
            continue

        try:
            logger.info(
                "[worker-%d] Start %s comment=%s ts=%d",
                worker_id,
                f"{item.target.type}:{item.target.id}",
                item.comment_key,
                int(item.comment_ts_ms),
            )
            _process_work_item(
                user=user,
                item=item,
                state=state,
                state_path=state_path,
                state_lock=state_lock,
                pending=pending,
                cfg=cfg,
                cache_dir=cache_dir,
                config_base_dir=config_base_dir,
                dry_run=dry_run,
                logger=logger,
                ollama=ollama,
            )
            logger.info("[worker-%d] Done %s", worker_id, item.comment_key)
        except Exception as e:
            # Safety net: never allow a single broken comment to dead-loop forever.
            target_key = f"{item.target.type}:{item.target.id}"
            pending.remove(target_key, item.comment_key)
            logger.warning("[worker-%d] Failed %s: %s", worker_id, item.comment_key, e, exc_info=True)
            try:
                with state_lock:
                    target_state = get_target_state(state, target_key)
                    if item.comment_key not in target_state.processed_comment_keys:
                        target_state.processed_comment_keys.append(item.comment_key)
                    _update_last_seen_capped(
                        target_state=target_state,
                        ts_ms=int(item.comment_ts_ms),
                        pending_min_ts_ms=pending.min_ts(target_key),
                    )
                    prune_processed_keys(target_state, keep_last=500)
                    save_state(state_path, state)
            except Exception:
                # Do not crash the worker due to state issues.
                logger.debug("[worker-%d] Failed to mark comment processed after error", worker_id, exc_info=True)
        finally:
            work_q.task_done()
            capacity.release()


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

    password = _read_password(cfg)
    if password is None:
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
    endpoints = list(getattr(cfg.ollama, "base_urls", None) or []) or [cfg.ollama.base_url]
    max_parallel = int(getattr(cfg.ollama, "max_parallel_requests", 1) or 1)
    if max_parallel < 1:
        max_parallel = 1

    user_lock = threading.RLock()
    locked_user = _LockedUser(user, user_lock)
    state_lock = threading.RLock()
    pending = _PendingTracker()

    clients: list[OllamaClient] = []
    for i in range(max_parallel):
        url = str(endpoints[i % len(endpoints)]).strip() if endpoints else cfg.ollama.base_url
        if not url:
            url = cfg.ollama.base_url
        clients.append(
            OllamaClient(
                base_url=url,
                model=cfg.ollama.model,
                timeout_sec=cfg.ollama.request_timeout_sec,
                temperature=cfg.ollama.temperature,
                num_predict=int(getattr(cfg.ollama, "num_predict", 2048) or 2048),
                gptoss_optimization=bool(getattr(cfg.ollama, "gptoss_optimization", False)),
            )
        )

    logger.info("Targets: %s", format_targets(targets))
    if len(clients) > 1:
        logger.info(
            "Ollama workers: endpoints=%s model=%s max_parallel=%d",
            ", ".join(sorted(set(endpoints))),
            cfg.ollama.model,
            max_parallel,
        )
    else:
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
        if len(clients) <= 1:
            # Legacy single-worker polling loop (kept for simplicity).
            ollama = clients[0]
            while True:
                logger.debug("Polling cycle start")
                for target in targets:
                    _process_target(
                        user=locked_user,
                        target=target,
                        state=state,
                        ollama=ollama,
                        cfg=cfg,
                        cache_dir=cache_dir,
                        config_base_dir=base_dir,
                        dry_run=dry_run,
                        logger=logger,
                    )
                discovered = _process_notifications(
                    user=locked_user,
                    state=state,
                    state_lock=state_lock,
                    cfg=cfg,
                    logger=logger,
                )
                for t in discovered:
                    if f"{t.type}:{t.id}" not in {f"{x.type}:{x.id}" for x in targets}:
                        targets.append(t)
                logger.debug("Polling cycle end")

                try:
                    save_state(state_path, state)
                except OSError as e:
                    logger.warning("Failed to save state to %s: %s", state_path, e)

                if args.once:
                    return 0
                time.sleep(cfg.agent.poll_interval_sec)

        # Dispatch mode: poll only when at least one worker is available; enqueue work and process in parallel.
        capacity = threading.BoundedSemaphore(len(clients))
        work_q: "Queue[_WorkItem]" = Queue(maxsize=len(clients))
        stop_event = threading.Event()

        workers: list[threading.Thread] = []
        for idx, c in enumerate(clients, start=1):
            t = threading.Thread(
                target=_worker_loop,
                kwargs={
                    "worker_id": idx,
                    "user": locked_user,
                    "work_q": work_q,
                    "capacity": capacity,
                    "stop_event": stop_event,
                    "state": state,
                    "state_path": state_path,
                    "state_lock": state_lock,
                    "pending": pending,
                    "cfg": cfg,
                    "cache_dir": cache_dir,
                    "config_base_dir": base_dir,
                    "dry_run": dry_run,
                    "logger": logger,
                    "ollama": c,
                },
                daemon=True,
            )
            t.start()
            workers.append(t)

        fast_poll_sec = float(getattr(cfg.agent, "dispatch_fast_poll_interval_sec", 1.0) or 1.0)
        if fast_poll_sec <= 0:
            fast_poll_sec = 1.0

        targets_by_key: dict[str, TargetConfig] = {f"{t.type}:{t.id}": t for t in targets}

        while True:
            # Block until at least one worker slot is free, then reserve it.
            reserved = 0
            while reserved < len(clients) and capacity.acquire(blocking=False):
                reserved += 1
            if reserved == 0:
                capacity.acquire()
                reserved = 1

            enqueued = 0
            logger.debug("Dispatch poll start (reserved=%d queue=%d)", reserved, work_q.qsize())

            discovered = _process_notifications(
                user=locked_user,
                state=state,
                state_lock=state_lock,
                cfg=cfg,
                logger=logger,
            )
            for t in discovered:
                k = f"{t.type}:{t.id}"
                if k not in targets_by_key:
                    targets_by_key[k] = t
                    logger.info("Added target from notifications: %s", k)

            for t in list(targets_by_key.values()):
                if enqueued >= reserved:
                    break
                enqueued += _scan_and_enqueue(
                    user=locked_user,
                    target=t,
                    state=state,
                    state_lock=state_lock,
                    cfg=cfg,
                    pending=pending,
                    work_q=work_q,
                    max_enqueue=reserved - enqueued,
                    logger=logger,
                )

            # Release unused reserved slots.
            for _ in range(max(0, reserved - enqueued)):
                capacity.release()

            logger.debug("Dispatch poll end (enqueued=%d)", enqueued)

            with state_lock:
                try:
                    save_state(state_path, state)
                except OSError as e:
                    logger.warning("Failed to save state to %s: %s", state_path, e)

            if args.once:
                # Wait for queued work to finish, then exit.
                while not work_q.empty():
                    time.sleep(0.1)
                return 0

            # If we didn't enqueue anything, avoid hammering the server.
            if enqueued == 0:
                time.sleep(fast_poll_sec)
    except KeyboardInterrupt:
        logger.info("Stopping...")
        try:
            stop_event.set()  # type: ignore[name-defined]
        except Exception:
            pass
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
        state.closed_conversations = {}

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

    password = _read_password(cfg)
    if password is None:
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


_ABS_PATH_RE = re.compile(r"(?:(?:[A-Za-z]:\\\\)|/)[^\\s'\"<>]+")


def _public_error_text(err: Exception, *, max_chars: int = 600) -> str:
    msg = str(err).strip()
    if not msg:
        msg = err.__class__.__name__
    msg = _ABS_PATH_RE.sub("<path>", msg)
    return truncate(msg, max_chars=max_chars)


def _cmd_webtest(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"Failed to load config: {e}")
        return 2

    base_dir = config_dir(args.config)
    cache_dir = pick_cache_dir(args.config, config=cfg)
    os.makedirs(cache_dir, exist_ok=True)
    logger = _setup_logging(cache_dir=cache_dir, level=(args.log_level or "DEBUG"))
    logger.info("Web search test (config=%s, cache_dir=%s)", os.path.abspath(args.config), cache_dir)

    query = (args.query or "").strip() or "Physics Lab AR"
    proxy = (args.proxy if args.proxy is not None else getattr(cfg.agent, "web_search_proxy", "")) or ""
    user_agent = (args.user_agent if args.user_agent is not None else getattr(cfg.agent, "web_search_user_agent", "")) or ""
    searxng_base = (
        args.searxng_base_url
        if args.searxng_base_url is not None
        else getattr(cfg.agent, "web_search_searxng_base_url", "")
    ) or ""

    providers: list[str]
    if args.all:
        providers = ["bing", "duckduckgo", "baidu", "google", "searxng"]
    else:
        providers = [((args.provider or "") or getattr(cfg.agent, "web_search_provider", "google") or "google")]

    for p in providers:
        provider = str(p or "").strip().lower()
        if not provider:
            continue
        logger.info("Testing provider=%s proxy=%r searxng_base_url=%r", provider, proxy, searxng_base)
        try:
            res = web_search(
                query=query,
                cache_dir=cache_dir,
                provider=provider,
                proxy=str(proxy),
                timeout_sec=int(args.timeout_sec or getattr(cfg.agent, "web_search_timeout_sec", 20) or 20),
                ttl_sec=0,  # always fetch fresh for diagnostics
                max_results=int(args.max_results or getattr(cfg.agent, "web_search_max_results", 5) or 5),
                fallback_to_ddg=bool(getattr(cfg.agent, "web_search_fallback_to_ddg", True)),
                user_agent=str(user_agent),
                searxng_base_url=str(searxng_base),
            )
        except Exception as e:
            logger.error("Provider %s failed: %s", provider, e)
            continue

        print("\n" + ("=" * 72))
        print(f"Provider: {provider}")
        print(f"Query: {query}")
        print(res)
        print(("=" * 72) + "\n")

    return 0


def _cmd_simtest(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"Failed to load config: {e}")
        return 2

    cache_dir = pick_cache_dir(args.config, config=cfg)
    os.makedirs(cache_dir, exist_ok=True)
    logger = _setup_logging(cache_dir=cache_dir, level=(args.log_level or "DEBUG"))
    logger.info("Simulation smoke test (config=%s, cache_dir=%s)", os.path.abspath(args.config), cache_dir)

    text = str(args.text or "").strip() or "simulate V=5V R1=100ohm R2=200ohm"

    # Print proxy-related env vars (common root cause for localhost Ollama failures).
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        v = os.environ.get(k)
        if isinstance(v, str) and v.strip():
            print(f"{k}={v}")

    endpoints = list(getattr(cfg.ollama, "base_urls", None) or []) or [cfg.ollama.base_url]
    base_url = str(endpoints[0] or cfg.ollama.base_url).strip()
    client = OllamaClient(
        base_url=base_url,
        model=cfg.ollama.model,
        timeout_sec=int(getattr(cfg.ollama, "request_timeout_sec", 240) or 240),
        temperature=float(getattr(cfg.ollama, "temperature", 0.2) or 0.2),
        num_predict=int(getattr(cfg.ollama, "num_predict", 2048) or 2048),
        gptoss_optimization=bool(getattr(cfg.ollama, "gptoss_optimization", False)),
    )
    print(f"ollama.base_url={base_url}")
    print(f"ollama.model={cfg.ollama.model}")
    print(f"phy_engine.phyengine_lib_path={getattr(cfg.phy_engine, 'phyengine_lib_path', '')}")

    try:
        ping = client.chat(messages=[{"role": "user", "content": "Reply with exactly: OK"}])
        print(f"ollama.ping={ping!r}")
    except Exception as e:
        print(f"ollama.ping_failed={e}")
        return 3

    # LLM build PE-SCRIPT -> parse only (no simulation).
    try:
        from tools import llm_build_pe_sim_script  # local import to keep startup minimal
        from pe_cmd import parse_pe_script_to_spec_obj
        from pe_builder import parse_pe_sim_spec, build_circuit

        script = llm_build_pe_sim_script(
            ollama=client,
            user_text=text,
            context_json=None,
            max_components=int(args.max_components or 30),
            max_probes=int(args.max_probes or 20),
        )
        print("\nPE-SCRIPT (truncated):")
        print(truncate(script, max_chars=800))
        spec_obj = parse_pe_script_to_spec_obj(
            script,
            max_components=int(args.max_components or 30),
            max_probes=int(args.max_probes or 20),
        )
        spec = parse_pe_sim_spec(spec_obj, max_components=int(args.max_components or 30), max_probes=int(args.max_probes or 20))
        built = build_circuit(spec)
        print(f"pe_script.parse_ok=True components={len(spec.components)} nodes={len(built.node_to_pin)}")
    except Exception as e:
        print(f"pe_script.parse_ok=False error={e}")

    # LLM build strict JSON spec -> parse only (no simulation).
    try:
        from tools import llm_build_pe_sim_spec_json, _extract_json_object_text  # type: ignore
        from pe_builder import parse_spec_json, parse_pe_sim_spec, build_circuit

        raw = llm_build_pe_sim_spec_json(
            ollama=client,
            user_text=text,
            context_json=None,
            max_components=int(args.max_components or 30),
            max_probes=int(args.max_probes or 20),
        )
        extracted = _extract_json_object_text(raw)
        print("\nJSON-SPEC (truncated):")
        print(truncate(extracted, max_chars=800))
        obj = parse_spec_json(extracted)
        spec = parse_pe_sim_spec(obj, max_components=int(args.max_components or 30), max_probes=int(args.max_probes or 20))
        built = build_circuit(spec)
        print(f"json_spec.parse_ok=True components={len(spec.components)} nodes={len(built.node_to_pin)}")
    except Exception as e:
        print(f"json_spec.parse_ok=False error={e}")

    return 0


def _cmd_apitest(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"Failed to load config: {e}")
        return 2

    base_dir = config_dir(args.config)
    cache_dir = pick_cache_dir(args.config, config=cfg)
    os.makedirs(cache_dir, exist_ok=True)
    logger = _setup_logging(cache_dir=cache_dir, level=(args.log_level or "DEBUG"))
    logger.info("Physics Lab API smoke test (config=%s, cache_dir=%s)", os.path.abspath(args.config), cache_dir)

    password = _read_password(cfg)
    if password is None:
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

    try:
        from physicsLab import Category  # type: ignore
    except Exception as e:
        print(f"Failed to import physicsLab.Category: {e}")
        return 4

    def _print_items(items: list[dict[str, Any]], *, label: str) -> None:
        print("\n" + ("=" * 72))
        print(label)
        print(f"Count: {len(items)}")
        for it in items[:5]:
            sid = best_effort_extract_text(it.get("ID")) or best_effort_extract_text(it.get("Id")) or ""
            subj = best_effort_extract_text(it.get("Subject")) or best_effort_extract_text(it.get("Title")) or ""
            print(f"- {sid}  {subj}".rstrip())
        print(("=" * 72) + "\n")

    try:
        exp = query_experiments(user, category=Category.Experiment, take=int(args.take or 5))
        _print_items(exp, label="QueryExperiments: Experiment")
    except Exception as e:
        print(f"QueryExperiments Experiment failed: {e}")

    try:
        disc = query_experiments(user, category=Category.Discussion, take=int(args.take or 5))
        _print_items(disc, label="QueryExperiments: Discussion")
    except Exception as e:
        print(f"QueryExperiments Discussion failed: {e}")

    if args.user_name:
        name = str(args.user_name).strip().lstrip("@")
        if name:
            try:
                data = get_user_by_name(user, name=name)
                u = data.get("User") if isinstance(data, dict) else None
                uid = best_effort_extract_text(u.get("ID")) if isinstance(u, dict) else ""
                nick = best_effort_extract_text(u.get("Nickname")) if isinstance(u, dict) else ""
                print("\nUser lookup:")
                print(f"- Nickname: {nick}")
                print(f"- ID: {uid}")
            except Exception as e:
                print(f"GetUser failed: {e}")

    if args.query:
        q = str(args.query).strip()
        if q:
            try:
                hits = search_recent_experiments(user=user, query=q, max_scan=200, max_results=5)
                print("\nInternal search (best-effort recent scan):")
                print(format_experiment_hits(hits))
            except Exception as e:
                print(f"Internal search failed: {e}")

    return 0


def _cmd_publishsav(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"Failed to load config: {e}")
        return 2

    cache_dir = pick_cache_dir(args.config, config=cfg)
    os.makedirs(cache_dir, exist_ok=True)
    logger = _setup_logging(cache_dir=cache_dir, level=(args.log_level or "DEBUG"))
    logger.info("Publish .sav (config=%s, cache_dir=%s)", os.path.abspath(args.config), cache_dir)

    sav_path = os.path.abspath(str(args.sav_path or "").strip())
    if not sav_path:
        print("Missing --sav-path")
        return 2
    if not os.path.isfile(sav_path):
        print(f".sav not found: {sav_path}")
        return 2

    title = str(args.title or "").strip() or os.path.basename(sav_path)
    intro = str(args.introduction or "").strip()
    # Community rule: do not publish to Experiment via this tool.
    category_value = "Discussion"
    tags = []
    if args.tags:
        tags = [t.strip() for t in str(args.tags).split(",") if t.strip()]

    # Safety: do not publish unless explicitly confirmed.
    if not bool(args.yes):
        print("Dry-run (no publish). Would publish:")
        print(f"- sav: {sav_path}")
        print(f"- category: {category_value}")
        print(f"- title: {title}")
        if tags:
            print(f"- tags: {tags}")
        if intro:
            print(f"- introduction: {truncate(intro, max_chars=200)}")
        try:
            counts = load_plsav_counts(sav_path)
            print(f"- plsav elements: {counts.elements} wires: {counts.wires}")
        except Exception as e:
            print(f"- plsav inspect failed: {e}")
        print("Re-run with --yes to actually publish.")
        return 0

    password = _read_password(cfg)
    if password is None:
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

    try:
        info = upload_sav_as_experiment(
            user=user,
            sav_path=sav_path,
            title=title,
            introduction=intro,
            cache_dir=cache_dir,
            category_value=category_value,
            tags=tags or None,
        )
    except Exception as e:
        print(f"Publish failed: {e}")
        return 5

    print("Published:")
    print(f"- summary_id: {info.get('summary_id')}")
    print(f"- category: {info.get('category')}")
    return 0


def _parse_context_ref(text: str) -> tuple[str, str] | None:
    """Parse a context reference like `experiment:<24hex>` or `discussion:<24hex>`."""
    t = str(text or "").strip()
    if not t:
        return None
    if ":" in t:
        pfx, sid = t.split(":", 1)
        p = (pfx or "").strip().casefold()
        sid2 = (sid or "").strip()
        if not sid2:
            return None
        if p in ("experiment", "exp", "e"):
            return "Experiment", sid2
        if p in ("discussion", "discuss", "disc", "d"):
            return "Discussion", sid2
        # Accept already-normalized values.
        if p in ("experiment", "discussion"):
            return pfx.strip().title(), sid2
        return None
    # Default category when user provides only an ID.
    return "Experiment", t


def _cmd_oneshot(args: argparse.Namespace) -> int:
    """Run one local prompt (agent-mode or plain chat) for debugging."""
    try:
        cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"Failed to load config: {e}")
        return 2

    base_dir = config_dir(args.config)
    cache_dir = pick_cache_dir(args.config, config=cfg)
    os.makedirs(cache_dir, exist_ok=True)

    want_debug_io = bool(getattr(args, "debug_io", False))
    logger = _setup_logging(
        cache_dir=cache_dir,
        level=(args.log_level or ("DEBUG" if want_debug_io else getattr(cfg.agent, "log_level", "INFO"))),
    )
    logger.info("phy_lab oneshot (config=%s, cache_dir=%s)", os.path.abspath(args.config), cache_dir)

    mode = str(getattr(args, "mode", "agent") or "agent").strip().lower()
    if mode not in ("agent", "chat"):
        print("Invalid --mode. Use: agent, chat")
        return 2

    text = str(getattr(args, "text", "") or "").strip()
    if not text:
        try:
            text = (sys.stdin.read() or "").strip()
        except Exception:
            text = ""
    if not text:
        print("Missing --text (or provide stdin).")
        return 2

    context_json: dict[str, Any] | None = None
    context_ref = _parse_context_ref(str(getattr(args, "context", "") or "").strip()) if getattr(args, "context", None) else None

    need_login = (mode == "agent") or (context_ref is not None)
    user: Any = object()
    if need_login:
        password = _read_password(cfg)
        if password is None:
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

    # Optionally fetch context JSON (useful to reproduce comment-context behavior).
    if context_ref is not None:
        category_value, summary_id = context_ref
        try:
            context_json = get_experiment_context(
                user,
                summary_id=summary_id,
                category_value=category_value,
                cache_dir=cache_dir,
                ttl_sec=300,
                max_json_chars=20_000,
            )
        except Exception as e:
            print(f"Failed to load context {category_value}:{summary_id}: {_format_exception_brief(e)}")
            return 4

    # Force debug I/O if requested.
    if want_debug_io:
        try:
            setattr(cfg.agent, "debug_log_llm_io", True)
        except Exception:
            pass

    endpoints = list(getattr(cfg.ollama, "base_urls", None) or []) or [cfg.ollama.base_url]
    base_url = str(endpoints[0] or cfg.ollama.base_url).strip()
    if not base_url:
        print("Missing ollama.base_url in config.")
        return 2
    client = OllamaClient(
        base_url=base_url,
        model=cfg.ollama.model,
        timeout_sec=int(getattr(cfg.ollama, "request_timeout_sec", 240) or 240),
        temperature=float(getattr(cfg.ollama, "temperature", 0.2) or 0.2),
        num_predict=int(getattr(cfg.ollama, "num_predict", 2048) or 2048),
        gptoss_optimization=bool(getattr(cfg.ollama, "gptoss_optimization", False)),
    )
    logger.info("Ollama: %s (model=%s)", base_url, cfg.ollama.model)

    try:
        if mode == "chat":
            system_prompt = _effective_system_prompt(cfg=cfg, user_text=text)
            msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
            if context_json is not None:
                msgs.append(
                    {
                        "role": "system",
                        "content": "Context JSON (current page):\n"
                        + json.dumps(_shrink_context_json_for_llm(context_json), ensure_ascii=False, indent=2),
                    }
                )
            msgs.append({"role": "user", "content": text})
            out = client.chat(messages=msgs)
            print(out)
            return 0

        # Agent mode: run tools, but never publish (dry_run=True).
        reply = agent_mode_run(
            ollama=client,
            user=user,
            cfg=cfg,
            cache_dir=cache_dir,
            config_base_dir=base_dir,
            dry_run=True,
            logger=logger,
            task=text,
            context_json=context_json,
            history=[],
            requester_nickname=getattr(user, "nickname", None),
            requester_user_id=getattr(user, "user_id", None),
            max_seconds=int(getattr(args, "max_seconds", 900) or 900),
            max_steps=int(getattr(args, "max_steps", 30) or 30),
        )
        print(reply)
        return 0
    except Exception as e:
        logger.error("oneshot failed: %s", e, exc_info=True)
        print(f"ERROR: {_format_exception_brief(e)}")
        return 5


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

    p_web = sub.add_parser("webtest", help="Test web search providers (no login required)")
    p_web.add_argument("--config", required=True, help="Path to config JSON")
    p_web.add_argument("--query", default=None, help="Query to search (default: Physics Lab AR)")
    p_web.add_argument(
        "--provider",
        default=None,
        help="Override provider (duckduckgo-search|duckduckgo|google|bing|baidu|searxng). Defaults to config.",
    )
    p_web.add_argument(
        "--all",
        action="store_true",
        help="Test multiple providers in sequence (bing, duckduckgo, baidu, google, searxng).",
    )
    p_web.add_argument("--proxy", default=None, help="Override HTTP(S) proxy URL")
    p_web.add_argument("--user-agent", dest="user_agent", default=None, help="Override User-Agent")
    p_web.add_argument(
        "--searxng-base-url",
        default=None,
        help="Override SearXNG base URL (e.g. http://127.0.0.1:8080)",
    )
    p_web.add_argument("--timeout-sec", default=None, type=int, help="HTTP timeout seconds")
    p_web.add_argument("--max-results", default=None, type=int, help="Max results to parse (<=10)")
    p_web.add_argument("--log-level", default=None, help="Console log level (default: DEBUG)")
    p_web.set_defaults(func=_cmd_webtest)

    p_sim = sub.add_parser("simtest", help="Smoke test simulation LLM outputs (no login required)")
    p_sim.add_argument("--config", required=True, help="Path to config JSON")
    p_sim.add_argument("--text", default=None, help="Simulation request text")
    p_sim.add_argument("--max-components", default=30, type=int, help="Max components in LLM output")
    p_sim.add_argument("--max-probes", default=20, type=int, help="Max probes in LLM output")
    p_sim.add_argument("--log-level", default=None, help="Console log level (default: DEBUG)")
    p_sim.set_defaults(func=_cmd_simtest)

    p_api = sub.add_parser("apitest", help="Smoke test Physics Lab APIs (requires login)")
    p_api.add_argument("--config", required=True, help="Path to config JSON")
    p_api.add_argument("--take", default=5, type=int, help="Take count for QueryExperiments (<=24 recommended)")
    p_api.add_argument("--user-name", default=None, help="Optional username/nickname to lookup (Users/GetUser)")
    p_api.add_argument("--query", default=None, help="Optional internal search query to run (best-effort recent scan)")
    p_api.add_argument("--log-level", default=None, help="Console log level (default: DEBUG)")
    p_api.set_defaults(func=_cmd_apitest)

    p_pub = sub.add_parser("publishsav", help="Publish a local .sav as a community experiment (destructive)")
    p_pub.add_argument("--config", required=True, help="Path to config JSON")
    p_pub.add_argument("--sav-path", required=True, help="Path to .sav file")
    p_pub.add_argument("--category", default="Discussion", help="Experiment|Discussion (default: Discussion)")
    p_pub.add_argument("--title", default=None, help="Title (default: filename)")
    p_pub.add_argument("--introduction", default="", help="Introduction text")
    p_pub.add_argument("--tags", default=None, help="Comma-separated tags (e.g. SmallProject,Featured)")
    p_pub.add_argument("--yes", action="store_true", help="Actually publish (otherwise dry-run)")
    p_pub.add_argument("--log-level", default=None, help="Console log level (default: DEBUG)")
    p_pub.set_defaults(func=_cmd_publishsav)

    p_one = sub.add_parser("oneshot", help="Run a single local prompt (debug helper)")
    p_one.add_argument("--config", required=True, help="Path to config JSON")
    p_one.add_argument("--text", default=None, help="Prompt text (default: read from stdin)")
    p_one.add_argument("--mode", default="agent", help="agent|chat (default: agent)")
    p_one.add_argument(
        "--context",
        default=None,
        help="Optional context to load: experiment:<id> | discussion:<id> | <id> (defaults to Experiment)",
    )
    p_one.add_argument("--max-seconds", default=900, type=int, help="Agent-mode time budget seconds (default: 900)")
    p_one.add_argument("--max-steps", default=30, type=int, help="Agent-mode max tool steps (default: 30)")
    p_one.add_argument("--debug-io", action="store_true", help="Force debug logs for LLM I/O")
    p_one.add_argument("--log-level", default=None, help="Console log level (default: config or DEBUG if --debug-io)")
    p_one.set_defaults(func=_cmd_oneshot)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
