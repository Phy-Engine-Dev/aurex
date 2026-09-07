from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
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


def _task_database(cfg: AurexConfig, config_path: str):
    from .sessiondb import SessionDB
    return SessionDB(cfg.resolve_path(cfg.tracking.database_path, config_path=config_path))


def _queued_local_task(*, cfg, config_path, agent, user, logger, text, publish=False) -> int:
    """Submit one administrator task; attach to the existing worker if it owns the DB.

    A CLI request never bypasses FIFO through agent.handle. The existing Web worker
    retains its own login/configuration; --login here cannot change that worker.
    """
    from .web import PersistentTaskQueue
    db = _task_database(cfg, config_path)
    queue = PersistentTaskQueue(db, agent, user=user, logger=logger)
    owns_worker, rid = False, None
    try:
        try:
            queue.start()
            owns_worker = True
        except BlockingIOError:
            # Producer-only attachment: never recover/steal a live worker's task.
            print('Using the existing worker queue; its configuration and login remain in effect.', file=sys.stderr)
        rid = queue.enqueue(None, text, source='admin', explicit_publish_requested=bool(publish),
                            metadata={'entrypoint': 'cli'})
        task = db.get_task(rid)
        sid = task['session_id']
        print('Queued task ' + rid + ' (session ' + sid + ')', file=sys.stderr, flush=True)
        while task['status'] in {'queued', 'running', 'cancelling'}:
            time.sleep(0.1)
            task = db.get_task(rid)
        # Read the persisted final event, not an in-memory callback that is lost on restart.
        with db.connect() as conn:
            answer = conn.execute("SELECT data FROM events WHERE run_id=? AND session_id=? AND kind='answer' ORDER BY id DESC LIMIT 1", (rid, sid)).fetchone()
            reviewed = conn.execute('''SELECT m.data FROM final_answers f JOIN messages m ON m.id=f.message_id
                AND m.session_id=f.session_id WHERE f.run_id=? AND f.session_id=?''', (rid, sid)).fetchone()
        if answer:
            print(str(json.loads(answer['data']).get('text', '')))
        elif reviewed:
            print(str(json.loads(reviewed['data']).get('content', '')))
        else:
            print('Task ' + rid + ': ' + task['status'] + '. Details remain in the task journal.')
        return 0 if task['status'] == 'completed' else 130 if task['status'] == 'cancelled' else 1
    except KeyboardInterrupt:
        if rid:
            db.request_cancel(db.get_task(rid)['session_id'], rid)
            print('Stop requested for task ' + rid + '; waiting for a safe boundary. Existing external receipts remain authoritative.', file=sys.stderr)
        raise
    finally:
        # A producer-only CLI does not close the other process's worker or lock.
        queue.close(wait=owns_worker)


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

    if cfg.llm.enabled:
        return _queued_local_task(cfg=cfg, config_path=args.config, agent=agent, user=user, logger=logger,
                                  text=text, publish=getattr(args, 'publish', False))
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

    if cfg.llm.enabled:
        return _queued_local_task(cfg=cfg, config_path=args.config, agent=agent, user=user, logger=logger,
                                  text=text, publish=getattr(args, 'publish', False))
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
    options = dict(cfg=cfg, config_path=args.config, agent=agent, user=user, targets=targets,
                   state_path=state_path, once=bool(args.once),
                   dry_run=(bool(args.dry_run) if args.dry_run is not None else None), logger=logger)
    if cfg.llm.enabled:
        from .web import PersistentTaskQueue
        queue = PersistentTaskQueue(_task_database(cfg, args.config), agent, user=user, logger=logger)
        try:
            try:
                queue.start()
            except BlockingIOError as exc:
                raise ConfigError('A task worker already owns this database. Use the running Web service with --poll; a second polling worker will not bypass its FIFO queue.') from exc
            run_forever(**options, enqueue=queue.enqueue)
            if args.once:
                queue.wait_idle()
        finally:
            queue.close(wait=True)
    else:
        run_forever(**options)
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from .web import serve
    cfg = load_config(args.config)
    if not cfg.llm.enabled:
        raise ConfigError('Web agent requires llm.enabled=true with a vLLM endpoint')
    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=args.config)
    logger = setup_logger(cache_dir=cache_dir, level=cfg.agent.log_level)
    agent = AurexAgent(cfg=cfg, config_path=args.config, tools=create_registry(), logger=logger)
    user = _login_if_needed(cfg, args.config, enabled=bool(args.login or args.poll))
    serve(cfg=cfg, config_path=args.config, agent=agent, user=user,
          hostname=args.hostname, port=args.port, poll=args.poll, logger=logger)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aurex", description="Aurex physical laboratory agent and conversation tracker")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="write a fresh config file")
    p_init.add_argument("--config", required=True, help="config path to write")
    p_init.add_argument("--email", default="", help="PhysicsLab account email")
    p_init.set_defaults(func=cmd_init)

    p_chat = sub.add_parser("chat", help="run a single agent task from stdin or --text")
    p_chat.add_argument("--config", required=True, help="config path to read")
    p_chat.add_argument("--text", default="", help="input text (if empty, read stdin)")
    p_chat.add_argument("--login", action="store_true", help="login to PhysicsLab (enables plar tools)")
    p_chat.add_argument("--publish", action="store_true", help="explicitly request reviewed experiment publication for this administrator task")
    p_chat.set_defaults(func=cmd_chat)

    p_console = sub.add_parser("console", help="interactive one-shot console chat (for testing)")
    p_console.add_argument("--config", required=True, help="config path to read")
    p_console.add_argument("--text", default="", help="input text (if empty, prompt in console; if piped, read stdin)")
    p_console.add_argument("--login", action="store_true", help="login to PhysicsLab (enables plar tools)")
    p_console.add_argument("--publish", action="store_true", help="explicitly request reviewed experiment publication for this administrator task")
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

    p_web = sub.add_parser('web', help='start the vLLM vision agent and durable per-session Web tracker')
    p_web.add_argument('--config', required=True)
    p_web.add_argument('--hostname', default=None)
    p_web.add_argument('--port', type=int, default=None)
    p_web.add_argument('--login', action='store_true', help='enable authenticated PhysicsLab read tools')
    p_web.add_argument('--poll', action='store_true', help='also poll and reply to explicitly addressed community mentions')
    p_web.set_defaults(func=cmd_web)

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
