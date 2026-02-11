from __future__ import annotations

import argparse
import getpass
import os
import sys
from typing import Any

import plar

from .agent import AurexAgent
from .config import AurexConfig, ConfigError, load_config, save_config
from .logutil import setup_logger
from .runloop import default_state_path, normalize_targets, parse_target, run_forever
from .tools import create_registry


def _read_stdin_text() -> str:
    try:
        return sys.stdin.read()
    except Exception:
        return ""


def _env_password() -> str:
    for k in ("AUREX_PASSWORD", "PHYSICSLAB_PASSWORD", "PHY_LAB_PASSWORD"):
        v = os.environ.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _login_if_needed(cfg: AurexConfig, config_path: str, enabled: bool) -> Any | None:
    if not enabled:
        return None
    email = (cfg.account.email or "").strip()
    if not email:
        raise SystemExit("config.account.email is empty; cannot login")
    pw = _env_password()
    if not pw:
        pw = str(getattr(cfg.account, "password", "") or "").strip()
    if not pw:
        pw = getpass.getpass("PhysicsLab password: ")
    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=config_path)
    return plar.email_login(email=email, password=pw, cache_dir=cache_dir)


def cmd_init(args: argparse.Namespace) -> int:
    path = args.config
    cfg = AurexConfig()
    if args.email:
        cfg = AurexConfig(account=cfg.account.__class__(email=str(args.email).strip()))
    save_config(cfg, path)
    print(path)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=args.config)
    logger = setup_logger(cache_dir=cache_dir, level=str(getattr(cfg.agent, "log_level", "INFO") or "INFO"))
    tools = create_registry()
    agent = AurexAgent(cfg=cfg, config_path=args.config, tools=tools, logger=logger)
    user = _login_if_needed(cfg, args.config, enabled=bool(args.login))

    text = (args.text or "").strip()
    if not text:
        text = _read_stdin_text().strip()
    if not text:
        raise SystemExit("No input text provided (use --text or pipe stdin).")

    out = agent.handle(user_text=text, user=user)
    print(out["answer"])
    return 0


def cmd_console(args: argparse.Namespace) -> int:
    """Interactive one-shot console chat for quick local testing."""
    cfg = load_config(args.config)
    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=args.config)
    logger = setup_logger(cache_dir=cache_dir, level=str(getattr(cfg.agent, "log_level", "INFO") or "INFO"))
    tools = create_registry()
    agent = AurexAgent(cfg=cfg, config_path=args.config, tools=tools, logger=logger)
    user = _login_if_needed(cfg, args.config, enabled=bool(args.login))

    text = (args.text or "").strip()
    if not text:
        if hasattr(sys.stdin, "isatty") and not sys.stdin.isatty():
            text = _read_stdin_text().strip()
        else:
            try:
                text = input("You> ").strip()
            except EOFError:
                text = ""
    if not text:
        raise SystemExit("No input text provided.")

    out = agent.handle(user_text=text, user=user)
    print(out["answer"])
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=args.config)
    logger = setup_logger(cache_dir=cache_dir, level=str(getattr(cfg.agent, "log_level", "INFO") or "INFO"))
    tools = create_registry()
    agent = AurexAgent(cfg=cfg, config_path=args.config, tools=tools, logger=logger)

    user = _login_if_needed(cfg, args.config, enabled=True)
    targets_cli = [parse_target(x) for x in (args.target or [])]
    targets_cfg = normalize_targets(getattr(cfg.agent, "targets", []) or [])
    targets = normalize_targets(list(targets_cli) + list(targets_cfg))

    state_path = args.state or default_state_path(args.config)
    run_forever(
        cfg=cfg,
        config_path=args.config,
        agent=agent,
        user=user,
        targets=targets,
        state_path=state_path,
        once=bool(args.once),
        dry_run=(bool(args.dry_run) if args.dry_run is not None else None),
        logger=logger,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aurex", description="aurex dual-model agent (planner+executor)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="write a fresh config file")
    p_init.add_argument("--config", required=True, help="config path to write")
    p_init.add_argument("--email", default="", help="PhysicsLab account email")
    p_init.set_defaults(func=cmd_init)

    p_chat = sub.add_parser("chat", help="run a single agent task from stdin or --text")
    p_chat.add_argument("--config", required=True, help="config path to read")
    p_chat.add_argument("--text", default="", help="input text (if empty, read stdin)")
    p_chat.add_argument("--login", action="store_true", help="login to PhysicsLab (enables plar tools)")
    p_chat.set_defaults(func=cmd_chat)

    p_console = sub.add_parser("console", help="interactive one-shot console chat (for testing)")
    p_console.add_argument("--config", required=True, help="config path to read")
    p_console.add_argument("--text", default="", help="input text (if empty, prompt in console; if piped, read stdin)")
    p_console.add_argument("--login", action="store_true", help="login to PhysicsLab (enables plar tools)")
    p_console.set_defaults(func=cmd_console)

    p_run = sub.add_parser("run", help="poll targets and auto-reply in a loop (like v1)")
    p_run.add_argument("--config", required=True, help="config path to read")
    p_run.add_argument("--state", default="", help="state path (default: <config_stem>.state.json)")
    p_run.add_argument("--target", action="append", default=[], help="override/add target (e.g. Experiment:<id>)")
    p_run.add_argument("--once", action="store_true", help="poll once then exit")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="do not post comments (print actions only); overrides config.agent.dry_run",
    )
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        return int(ns.func(ns))
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2


__all__ = ["main"]
