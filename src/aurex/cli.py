from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import plar

from .agent import AurexAgent
from .config import AurexConfig, ConfigError, load_config, save_config
from .logutil import setup_logger
from .terminal import AurexWebClient, DashboardState, TerminalApp, WebAPIError
from .tools import create_registry


def _env_password() -> str:
    for key in ("AUREX_PASSWORD", "PHYSICSLAB_PASSWORD", "PHY_LAB_PASSWORD"):
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _login(cfg: AurexConfig, config_path: str) -> Any:
    email = (cfg.account.email or "").strip()
    if not email:
        raise SystemExit("config.account.email is empty; cannot login")
    password = _env_password() or str(getattr(cfg.account, "password", "") or "").strip()
    if not password:
        password = getpass.getpass("PhysicsLab password: ")
    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=config_path)
    return plar.email_login(email=email, password=password, cache_dir=cache_dir)


def _access_token(cfg: AurexConfig, config_path: str) -> str:
    token = os.environ.get(cfg.tracking.token_env, "").strip()
    if token:
        return token
    candidate = Path(config_path).resolve().with_name("web-token")
    try:
        return candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _local_url(cfg: AurexConfig, port: int | None = None) -> str:
    return f"http://127.0.0.1:{port or cfg.tracking.port}"


def cmd_init(args: argparse.Namespace) -> int:
    cfg = AurexConfig()
    if args.email:
        cfg = AurexConfig(account=cfg.account.__class__(email=str(args.email).strip()))
    save_config(cfg, args.config)
    print(args.config)
    return 0


def _spawn_shared_web(cfg: AurexConfig, config_path: str, *, hostname: str | None,
                      port: int | None, token: str) -> subprocess.Popen:
    """Start the canonical detached server used by both terminal and browser UIs."""
    config = str(Path(config_path).resolve())
    database = Path(cfg.resolve_path(cfg.tracking.database_path, config_path=config)).resolve()
    runtime = database.parent
    runtime.mkdir(parents=True, exist_ok=True)
    pidfile = runtime / "web.pid"
    try:
        old_pid = int(pidfile.read_text(encoding="utf-8").strip())
        os.kill(old_pid, 0)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        pidfile.unlink(missing_ok=True)
    except PermissionError as exc:
        raise SystemExit("现有Aurex服务进程不可检查：" + str(exc)) from exc
    else:
        raise SystemExit(f"Aurex服务进程 {old_pid} 已存在但健康检查失败；拒绝启动第二个server")

    command = [sys.executable, "-m", "aurex", "web", "--config", config]
    if hostname:
        command.extend(["--hostname", hostname])
    if port:
        command.extend(["--port", str(port)])
    environment = os.environ.copy()
    if token:
        environment[cfg.tracking.token_env] = token
    log_path = runtime / "web.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(Path(config).parent.parent), env=environment, start_new_session=True)
    pidfile.write_text(str(process.pid) + "\n", encoding="utf-8")
    return process


def cmd_cli(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if not cfg.llm.enabled:
        raise ConfigError("CLI agent requires llm.enabled=true with a vLLM endpoint")
    token = _access_token(cfg, args.config)
    url = (args.url or _local_url(cfg, args.port)).rstrip("/")
    api = AurexWebClient(url, token)
    spawned: subprocess.Popen | None = None

    if not api.healthy():
        if args.url:
            raise SystemExit("指定的 Aurex Web 不可用；--url 模式不会另起服务")
        spawned = _spawn_shared_web(
            cfg, args.config, hostname=args.hostname, port=args.port, token=token)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not api.healthy():
            if spawned.poll() is not None:
                raise SystemExit(f"Aurex后端启动失败，退出码 {spawned.returncode}；请检查 .aurex/web.log")
            time.sleep(0.1)
        if not api.healthy():
            raise SystemExit("Aurex 后端在 20 秒内未就绪")

    state = DashboardState(api)
    try:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            state.refresh()
            print(state.plain_text())
            return 0
        import curses
        curses.wrapper(TerminalApp(state).run)
        return 0
    except KeyboardInterrupt:
        return 130
    except WebAPIError as exc:
        print("Aurex CLI error: " + str(exc), file=sys.stderr)
        return 1


def cmd_web(args: argparse.Namespace) -> int:
    from .web import serve
    cfg = load_config(args.config)
    if not cfg.llm.enabled:
        raise ConfigError("Web agent requires llm.enabled=true with a vLLM endpoint")
    existing = AurexWebClient(_local_url(cfg, args.port), _access_token(cfg, args.config))
    if existing.healthy():
        print("Aurex统一server已经运行；Web和CLI将使用同一服务。")
        return 0
    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=args.config)
    logger = setup_logger(cache_dir=cache_dir, level=cfg.agent.log_level)
    agent = AurexAgent(cfg=cfg, config_path=args.config, tools=create_registry(), logger=logger)
    user = _login(cfg, args.config)
    serve(cfg=cfg, config_path=args.config, agent=agent, user=user,
          hostname=args.hostname, port=args.port, logger=logger)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aurex", description="Aurex physical laboratory agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="write a fresh config file")
    init.add_argument("--config", required=True, help="config path to write")
    init.add_argument("--email", default="", help="PhysicsLab account email")
    init.set_defaults(func=cmd_init)

    cli = sub.add_parser("cli", help="open the sectioned terminal UI and continuous community agent")
    cli.add_argument("--config", default=os.environ.get("AUREX_CONFIG", ".config/aurex3.json"))
    cli.add_argument("--url", default="", help="attach to an existing Aurex Web URL")
    cli.add_argument("--hostname", default=None, help="embedded backend listen address")
    cli.add_argument("--port", type=int, default=None, help="embedded backend/attachment port")
    cli.set_defaults(func=cmd_cli)

    web = sub.add_parser("web", help="start Web UI, durable queue and continuous community polling")
    web.add_argument("--config", required=True)
    web.add_argument("--hostname", default=None)
    web.add_argument("--port", type=int, default=None)
    web.set_defaults(func=cmd_web)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["cli"]
    parser = build_parser()
    ns = parser.parse_args(arguments)
    try:
        return int(ns.func(ns))
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
