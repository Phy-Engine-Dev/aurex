from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from unittest import mock

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from config import config_dir, load_config, pick_cache_dir  # noqa: E402
from ollama import OllamaClient  # noqa: E402
from plar import (  # noqa: E402
    email_login,
    get_experiment_context,
    get_status_save,
    get_user_by_name,
    query_experiments,
)
from tools import (  # noqa: E402
    build_and_maybe_publish_circuit,
    simulate_series_vdc_two_resistors,
)

import agent as agent_mod  # noqa: E402


_HEX24_RE = re.compile(r"^[0-9a-fA-F]{24}$")


@dataclass(frozen=True)
class _Result:
    name: str
    ok: bool
    detail: str
    seconds: float


def _extract_id(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    for k in ("ID", "Id", "id", "SummaryID", "summary_id"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_subject(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    for k in ("Subject", "Title", "subject", "title"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _tool_smoketest(
    *,
    cfg: Any,
    user: Any,
    client: OllamaClient,
    cache_dir: str,
    base_dir: str,
    live_publish: bool,
) -> list[_Result]:
    out: list[_Result] = []

    def _run(name: str, fn) -> None:  # type: ignore[no-untyped-def]
        t0 = time.time()
        try:
            detail = fn()
            out.append(_Result(name=name, ok=True, detail=str(detail or "OK"), seconds=time.time() - t0))
        except Exception as e:
            out.append(_Result(name=name, ok=False, detail=f"{e.__class__.__name__}: {e}", seconds=time.time() - t0))

    def _check_user(name: str) -> str:
        data = get_user_by_name(user, name=name)
        u = data.get("User") if isinstance(data, dict) else None
        if not isinstance(u, dict):
            raise RuntimeError("missing User object")
        uid = _extract_id(u)
        if not uid or _HEX24_RE.match(uid) is None:
            raise RuntimeError(f"invalid user id: {uid!r}")
        nick = str(u.get("Nickname") or "").strip() or "(no nickname)"
        ver = str(u.get("Verification") or "").strip() or ""
        return f"{nick} id={uid} verification={ver}"

    _run("tools.user_lookup(MacroModel)", lambda: _check_user("MacroModel"))
    _run("tools.user_lookup(揉碎星月)", lambda: _check_user("揉碎星月"))

    hot_ref: dict[str, Any] = {}

    def _list_hot_experiments() -> str:
        items = query_experiments(user, category="Experiment", take=5, skip=0, sort="Popularity")
        if not isinstance(items, list) or not items:
            raise RuntimeError("no hot experiments returned")
        candidates: list[dict[str, str]] = []
        for it in items[:5]:
            if not isinstance(it, dict):
                continue
            sid = _extract_id(it)
            if not sid or _HEX24_RE.match(sid) is None:
                continue
            subj = _extract_subject(it) or ""
            candidates.append({"summary_id": sid, "subject": subj})
        if not candidates:
            raise RuntimeError("no valid hot experiment ids in response")
        hot_ref["candidates"] = candidates
        hot_ref["summary_id"] = candidates[0]["summary_id"]
        hot_ref["subject"] = candidates[0]["subject"]
        return f"{candidates[0]['summary_id']} {candidates[0]['subject']}".strip()

    _run("tools.list_hot_experiments", _list_hot_experiments)

    def _open_hot_context() -> str:
        sid = str(hot_ref.get("summary_id") or "").strip()
        if not sid:
            raise RuntimeError("missing hot summary_id")
        ctx = get_experiment_context(
            user,
            summary_id=sid,
            category_value="Experiment",
            cache_dir=cache_dir,
            ttl_sec=300,
        )
        if not isinstance(ctx, dict):
            raise RuntimeError("invalid context object")
        if str(ctx.get("summary_id") or "").strip() != sid:
            raise RuntimeError("context.summary_id mismatch")
        subject = str(ctx.get("subject") or "").strip()
        return f"Experiment {sid} {subject}".strip()

    _run("tools.open_hot_context", _open_hot_context)

    def _status_save_counts() -> str:
        candidates = hot_ref.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("missing hot candidates")
        last_err: Exception | None = None
        for c in candidates[:5]:
            if not isinstance(c, dict):
                continue
            sid = str(c.get("summary_id") or "").strip()
            if not sid:
                continue
            try:
                status = get_status_save(
                    user,
                    summary_id=sid,
                    category_value="Experiment",
                    cache_dir=cache_dir,
                    ttl_sec=300,
                )
            except Exception as e:
                last_err = e
                continue
            els = status.get("Elements")
            wires = status.get("Wires")
            if not isinstance(els, list) or not isinstance(wires, list):
                last_err = RuntimeError("missing Elements/Wires")
                continue
            hot_ref["summary_id"] = sid
            hot_ref["subject"] = str(c.get("subject") or "").strip()
            return f"summary_id={sid} elements={len(els)} wires={len(wires)}"
        if last_err is not None:
            raise last_err
        raise RuntimeError("no candidate experiment had a usable StatusSave")

    _run("tools.get_status_save_counts", _status_save_counts)

    def _simulate_two_res() -> str:
        return simulate_series_vdc_two_resistors(
            text="simulate V=5V R1=100ohm R2=200ohm",
            phy_engine_cfg=cfg.phy_engine,
            config_base_dir=base_dir,
        )

    _run("tools.simulate(V=5 R1=100 R2=200)", _simulate_two_res)

    def _publish_small_circuit() -> str:
        tags = list(getattr(getattr(cfg, "agent", None), "publish_tags", []) or [])
        if not tags:
            tags = ["SmallProject"]
        verilog = "module top(input a, input b, output y);\n  assign y = a & b;\nendmodule\n"
        spec = (
            "SMOKETEST: publish a small logic circuit.\n\n"
            "```verilog\n"
            + verilog
            + "```\n"
        )
        title = f"[SMOKETEST] AND gate ({int(time.time())})"
        intro = "SMOKETEST: automated circuit publish from aurex."
        res = build_and_maybe_publish_circuit(
            ollama=client,
            user=user,
            spec=spec,
            cache_dir=cache_dir,
            phy_engine_cfg=cfg.phy_engine,
            config_base_dir=base_dir,
            keep_temp=True,
            enable_publish=bool(live_publish),
            dry_run=(not live_publish),
            max_attempts=2,
            publish_max_elements=int(getattr(getattr(cfg, "agent", None), "publish_max_elements", 5000) or 5000),
            title=title,
            introduction=intro,
            publish_category_value="Discussion",
            publish_tags=tags,
        )
        published = bool(getattr(res, "published", False))
        sid = getattr(res, "summary_id", None)
        sav = getattr(res, "artifact_sav_path", None)
        if live_publish:
            if not published:
                raise RuntimeError(f"publish failed: block_reason={getattr(res, 'publish_block_reason', None)!r}")
            if not isinstance(sid, str) or _HEX24_RE.match(sid.strip()) is None:
                raise RuntimeError(f"invalid published summary_id: {sid!r}")
            return f"published summary_id={sid}"
        # Dry-run path: ensure artifacts exist.
        if not isinstance(sav, str) or not sav.strip() or (not os.path.isfile(sav)):
            raise RuntimeError(f"missing artifact_sav_path: {sav!r}")
        return f"dry_run artifact_sav_path={sav}"

    _run("tools.publish_circuit" if live_publish else "tools.circuit_dry_run", _publish_small_circuit)
    return out


def _agent_smoketest(
    *,
    cfg: Any,
    user: Any,
    client: OllamaClient,
    cache_dir: str,
    base_dir: str,
    max_seconds: int,
    max_steps: int,
    live_publish: bool,
) -> list[_Result]:
    logger = logging.getLogger("community_qa_smoketest.agent")
    logger.setLevel(logging.ERROR)

    tool_calls: list[tuple[str, dict[str, Any], str]] = []
    orig = agent_mod._agent_execute_tool

    def _wrap(  # type: ignore[no-untyped-def]
        *,
        tool: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        res = orig(tool=tool, args=args, **kwargs)
        tool_calls.append((str(tool), dict(args or {}), str(res or "")))
        return res

    prompt = (
        "请按顺序完成以下“社区问答回归测试”，必须使用工具完成，不要凭空猜测：\n"
        "1) 用 plar_get_user_by_name 查询用户 MacroModel，得到用户ID与认证信息。\n"
        "2) 用 plar_get_user_by_name 查询用户 揉碎星月，得到用户ID与认证信息。\n"
        "3) 用 list_plar 获取最热门的实验：kind=hot, category=Experiment, take=3。\n"
        "4) 用 simulate 做一次直流仿真：simulate V=5V R1=100ohm R2=200ohm。\n"
        "5) 用 circuit 发布一个 AND 门逻辑电路（spec 内必须包含 verilog 代码块），并设置 publish=true。\n"
        "最后输出简短总结，并以 DONE 结尾。"
    )

    t0 = time.time()
    with mock.patch.object(agent_mod, "_agent_execute_tool", side_effect=_wrap):
        out_text = agent_mod.agent_mode_run(
            ollama=client,
            user=user,
            cfg=cfg,
            cache_dir=cache_dir,
            config_base_dir=base_dir,
            dry_run=(not live_publish),
            logger=logger,
            task=prompt,
            context_json=None,
            history=[],
            requester_nickname=getattr(user, "nickname", None),
            requester_user_id=getattr(user, "user_id", None),
            max_seconds=int(max_seconds),
            max_steps=int(max_steps),
        )
    seconds = time.time() - t0

    # Validate expected tool usage.
    ok_macro = False
    ok_star = False
    ok_hot = False
    ok_sim = False
    ok_circuit = False
    circuit_detail = ""

    for tool, args, res in tool_calls:
        if tool in ("plar_get_user_by_name", "search_plar"):
            name = str(args.get("name") or args.get("query") or "").strip()
            if name == "MacroModel":
                ok_macro = True
            if name == "揉碎星月":
                ok_star = True
        if tool == "list_plar":
            if str(args.get("kind") or "").strip().lower() in ("hot", "popular"):
                if str(args.get("category") or "").strip() in ("Experiment", "experiment"):
                    ok_hot = True
        if tool == "simulate":
            ok_sim = True
        if tool == "circuit":
            ok_circuit = True
            circuit_detail = res

    problems: list[str] = []
    if not ok_macro:
        problems.append("missing user lookup: MacroModel")
    if not ok_star:
        problems.append("missing user lookup: 揉碎星月")
    if not ok_hot:
        problems.append("missing hot experiments list")
    if not ok_sim:
        problems.append("missing simulate tool call")
    if not ok_circuit:
        problems.append("missing circuit tool call")

    if ok_circuit:
        try:
            obj = json.loads(circuit_detail)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            if live_publish:
                if not bool(obj.get("published")):
                    problems.append(f"circuit not published: {obj.get('publish_block_reason')!r}")
                sid = str(obj.get("summary_id") or "").strip()
                if not sid or _HEX24_RE.match(sid) is None:
                    problems.append("circuit missing summary_id")
            else:
                sav = str(obj.get("artifact_sav_path") or "").strip()
                if not sav or not os.path.isfile(sav):
                    problems.append("circuit missing artifact_sav_path")

    ok = not problems
    detail = "OK" if ok else "; ".join(problems)
    # Keep output snippet short to avoid leaking extra data.
    preview = (out_text or "").strip().replace("\n", " ")
    if len(preview) > 180:
        preview = preview[:180] + "..."
    detail = detail + f" | reply={preview!r}"

    return [_Result(name="agent.full_flow", ok=ok, detail=detail, seconds=seconds)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Community Q&A regression smoketest for phy_lab agent.")
    p.add_argument("--config", required=True, help="Path to config JSON")
    p.add_argument("--max-seconds", default=900, type=int, help="Agent-mode max seconds per case (default: 900)")
    p.add_argument("--max-steps", default=100, type=int, help="Agent-mode max tool steps per case (default: 100)")
    p.add_argument("--live-publish", action="store_true", help="Actually publish circuit (default: dry-run)")
    p.add_argument("--tools-only", action="store_true", help="Run tool-only integration checks only")
    p.add_argument("--agent-only", action="store_true", help="Run agent-mode flow only")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    base_dir = config_dir(args.config)
    cache_dir = pick_cache_dir(args.config, config=cfg)
    os.makedirs(cache_dir, exist_ok=True)

    if not getattr(cfg.account, "password", None):
        print("ERROR: account.password is required for this smoketest")
        return 2

    # Login once.
    user = email_login(
        email=cfg.account.email,
        password=cfg.account.password,
        cache_dir=cache_dir,
        http_timeout_sec=60.0,
    )

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

    # Basic ping (fast fail).
    try:
        ping = client.chat(messages=[{"role": "user", "content": "Reply with exactly: OK"}])
    except Exception as e:
        print(f"ERROR: ollama ping failed: {e}")
        return 3
    if (ping or "").strip() != "OK":
        print(f"ERROR: ollama ping unexpected reply: {ping!r}")
        return 3

    run_tools = not bool(args.agent_only)
    run_agent = not bool(args.tools_only)

    results: list[_Result] = []
    if run_tools:
        results.extend(
            _tool_smoketest(
                cfg=cfg,
                user=user,
                client=client,
                cache_dir=cache_dir,
                base_dir=base_dir,
                live_publish=bool(args.live_publish),
            )
        )
        if args.fail_fast and any(not r.ok for r in results):
            run_agent = False

    if run_agent:
        results.extend(
            _agent_smoketest(
                cfg=cfg,
                user=user,
                client=client,
                cache_dir=cache_dir,
                base_dir=base_dir,
                max_seconds=int(args.max_seconds),
                max_steps=int(args.max_steps),
                live_publish=bool(args.live_publish),
            )
        )

    failures = [r for r in results if not r.ok]
    print("COMMUNITY QA SMOKETEST")
    print(f"ollama={base_url} model={cfg.ollama.model}")
    print(f"live_publish={bool(args.live_publish)}")
    print("-")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        sec = f"{r.seconds:.1f}s"
        detail = str(r.detail or "")
        if len(detail) > 220:
            detail = detail[:220] + "..."
        print(f"{status} {r.name} ({sec}): {detail}")
        if args.fail_fast and (not r.ok):
            break
    print("-")
    if failures:
        print(f"fail={len(failures)}")
        return 1
    print("pass=all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
