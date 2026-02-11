from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import plar  # noqa: E402
from aurex.agent import AurexAgent  # noqa: E402
from aurex.config import load_config  # noqa: E402
from aurex.contextdb import ContextDB  # noqa: E402
from aurex.logutil import setup_logger  # noqa: E402
from aurex.tools import create_registry  # noqa: E402


_HEX24_RE = re.compile(r"[0-9a-fA-F]{24}")


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    require_tools: bool = False
    require_local_context_tool: bool = False
    require_hex24_in_answer: bool = False


def _get_context_db_path(*, cfg, config_path: str) -> str:
    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=config_path)
    context_db_path_cfg = str(getattr(cfg.storage, "context_db_path", "") or "").strip()
    if context_db_path_cfg:
        return cfg.resolve_path(context_db_path_cfg, config_path=config_path)
    return os.path.join(cache_dir, "context_db.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local smoke test for aurex agent (planner+executor).")
    p.add_argument("--config", required=True, help="Path to config JSON")
    p.add_argument("--login", action="store_true", help="Enable PhysicsLab login for plar tools")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=args.config)
    logger = setup_logger(cache_dir=cache_dir, level=str(getattr(cfg.agent, "log_level", "INFO") or "INFO"))
    tools = create_registry()
    agent = AurexAgent(cfg=cfg, config_path=args.config, tools=tools, logger=logger)

    user = None
    if bool(args.login):
        email = (cfg.account.email or "").strip()
        pw = (getattr(cfg.account, "password", "") or "").strip()
        if not email:
            print("ERROR: config.account.email is empty")
            return 2
        if not pw:
            print("ERROR: config.account.password is empty (or set env AUREX_PASSWORD and use CLI instead)")
            return 2
        user = plar.email_login(email=email, password=pw, cache_dir=cache_dir, http_timeout_sec=60.0)

    # Pre-populate local context DB for “this board/thread” cases.
    context_db_path = _get_context_db_path(cfg=cfg, config_path=args.config)
    ContextDB(path=context_db_path).upsert_target_comments(
        target_key="User:U_TEST",
        target={"type": "User", "id": "U_TEST"},
        comments=[
            {"id": "c1", "ts_ms": int(time.time() * 1000) - 3000, "author_nickname": "Alice", "text": "你好！"},
            {"id": "c2", "ts_ms": int(time.time() * 1000) - 2000, "author_nickname": "Bob", "text": "今天讨论一下串联电路。"},
            {"id": "c3", "ts_ms": int(time.time() * 1000) - 1000, "author_nickname": "Alice", "text": "能不能总结一下要点？"},
        ],
        keep_last=int(getattr(cfg.agent, "context_db_keep_last_comments", 200) or 200),
    )

    cases: list[Case] = [
        Case(
            name="zh_greeting_like_notification",
            text=(
                "CONTEXT_JSON:\n"
                + '{"target":{"type":"User","id":"U_TEST"},"comment":{"id":"c100","author_id":"u2","author_nickname":"揉碎星月"}}'
                + "\n\n回复<user=U_TEST>@aurex</user>: 你好"
            ),
        ),
        Case(
            name="zh_summarize_this_board",
            text=(
                "CONTEXT_JSON:\n"
                + '{"target":{"type":"User","id":"U_TEST"},"comment":{"id":"c101","author_id":"u2","author_nickname":"揉碎星月"}}'
                + "\n\n回复<user=U_TEST>@aurex</user>: 总结一下这个留言板的内容"
            ),
            require_tools=True,
            require_local_context_tool=True,
        ),
        Case(
            name="en_mixed_question",
            text="what is the 中 means in chinese",
        ),
    ]
    if user is not None:
        cases.append(
            Case(
                name="plar_query_hot_1",
                text="请列出 1 个热门实验，并附上 Category+SummaryID+Subject。",
                require_tools=True,
                require_hex24_in_answer=True,
            )
        )

    failures: list[str] = []
    for c in cases:
        print("=" * 80)
        print(f"CASE: {c.name}")
        try:
            out = agent.handle(user_text=c.text, user=user)
        except Exception as e:
            failures.append(f"{c.name}: exception {type(e).__name__}: {e}")
            print(f"ERROR: {type(e).__name__}: {e}")
            continue

        plan = out.get("plan")
        steps = getattr(plan, "steps", []) or []
        tool_results = out.get("tool_results") or []
        answer = str(out.get("answer") or "").strip()

        print(f"plan.user_lang={getattr(plan, 'user_lang', None)!r} steps={len(steps)} tool_results={len(tool_results)}")
        if steps:
            print("plan.tools=", [getattr(s, "tool", None) for s in steps])
        print("answer=", answer[:400].replace("\n", "\\n"))

        if c.require_tools and not steps:
            failures.append(f"{c.name}: expected tool steps, got none")
        if c.require_local_context_tool and ("local_get_target_context" not in [getattr(s, "tool", "") for s in steps]):
            failures.append(f"{c.name}: expected local_get_target_context in plan.steps")
        if c.require_hex24_in_answer and _HEX24_RE.search(answer) is None:
            failures.append(f"{c.name}: expected a 24-hex ID in answer")

        if "@aurex" in answer or "＠aurex" in answer:
            failures.append(f"{c.name}: answer contains @aurex (should be stripped/replaced before posting)")

    print("=" * 80)
    if failures:
        print("SMOKETEST FAILURES:")
        for f in failures:
            print("-", f)
        return 1
    print("SMOKETEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

