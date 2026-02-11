from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


class ConfigError(RuntimeError):
    pass


def _as_str(v: Any, *, where: str) -> str:
    if isinstance(v, str):
        return v
    raise ConfigError(f"{where} must be a string")


def _as_int(v: Any, *, where: str) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    raise ConfigError(f"{where} must be an integer")


def _as_float(v: Any, *, where: str) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    raise ConfigError(f"{where} must be a number")


def _as_bool(v: Any, *, where: str) -> bool:
    if isinstance(v, bool):
        return v
    raise ConfigError(f"{where} must be a boolean")


def _as_dict(v: Any, *, where: str) -> dict[str, Any]:
    if isinstance(v, dict):
        return v
    raise ConfigError(f"{where} must be an object")


def _as_list(v: Any, *, where: str) -> list[Any]:
    if isinstance(v, list):
        return v
    raise ConfigError(f"{where} must be an array")


def _parse_int_list(value: Any, *, default: list[int], where: str) -> list[int]:
    if value is None:
        return list(default)
    raw = _as_list(value, where=where)
    out: list[int] = []
    for x in raw:
        if isinstance(x, bool):
            raise ConfigError(f"{where} items must be integers")
        if isinstance(x, (int, float)):
            out.append(int(x))
            continue
        if isinstance(x, str) and x.strip():
            try:
                out.append(int(x.strip(), 10))
            except Exception as e:
                raise ConfigError(f"{where} items must be integers") from e
            continue
        raise ConfigError(f"{where} items must be integers")
    return out


def _norm_base_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        return "http://" + u
    return u


@dataclass(frozen=True)
class AccountConfig:
    email: str = ""
    password: str = ""


@dataclass(frozen=True)
class OllamaModelConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = ""
    timeout_sec: int = 240
    temperature: float = 0.2
    num_predict: int = 2048
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StorageConfig:
    cache_dir: str = ".aurex/cache"
    context_db_path: str = ""


@dataclass(frozen=True)
class WebSearchConfig:
    provider: str = "duckduckgo"
    max_results: int = 5
    region: str = "cn-zh"
    safesearch: str = "moderate"
    time_range: str = "y"


@dataclass(frozen=True)
class PhyEngineConfig:
    auto_build: bool = False
    cmake_source_dir: str = "third-parties/Phy-Engine/src"
    cmake_build_dir: str = ".aurex/cache/phy-engine-build"
    cmake_build_type: str = "Release"
    build_timeout_sec: int = 900
    run_timeout_sec: int = 300
    verilog2plsav_path: str = ""
    phyengine_lib_path: str = ""
    verilog2plsav_args: list[str] = field(default_factory=lambda: ["-O4", "--layout", "hier"])


@dataclass(frozen=True)
class AgentConfig:
    mention_tag: str = "@aurex"
    require_mention: bool = True
    user_targets_require_mention: bool = True
    trigger_on_reply_to_self: bool = False
    trigger_on_own_wall_without_mention: bool = False
    strip_mention_tag_in_replies: bool = True
    max_plan_steps: int = 10
    max_tool_loops: int = 20
    max_final_chars: int = 2000
    dry_run: bool = False
    poll_interval_sec: float = 15.0
    comment_take: int = 20
    comment_scan_pages: int = 3
    bootstrap_lookback_sec: int = 600
    reply_once: bool = False
    reply_once_ttl_sec: int = 0
    notifications_enabled: bool = True
    notification_category_ids: list[int] = field(default_factory=lambda: [3])
    notification_take: int = 20
    log_level: str = "INFO"
    debug_llm_io: bool = False
    debug_llm_max_chars: int = 1200
    force_reply_prefix: bool = True
    final_reply_only: bool = True
    prefetch_context_in_plan: bool = True
    prefetch_context_take: int = 10
    context_db_enabled: bool = True
    context_db_keep_last_comments: int = 200
    targets: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class AurexConfig:
    schema_version: int = 2
    account: AccountConfig = field(default_factory=AccountConfig)
    planner: OllamaModelConfig = field(
        default_factory=lambda: OllamaModelConfig(
            base_url="http://127.0.0.1:11434",
            model="qwen3:30b-a3b-thinking-2507-q4_K_M",
            temperature=0.2,
            num_predict=2048,
        )
    )
    executor: OllamaModelConfig = field(
        default_factory=lambda: OllamaModelConfig(
            base_url="http://127.0.0.1:11435",
            model="command-r7b",
            temperature=0.2,
            num_predict=1024,
        )
    )
    storage: StorageConfig = field(default_factory=StorageConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    phy_engine: PhyEngineConfig = field(default_factory=PhyEngineConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def resolve_path(self, path: str, *, config_path: str) -> str:
        p = (path or "").strip()
        if not p:
            return ""
        if os.path.isabs(p):
            return p
        base_dir = os.path.dirname(os.path.abspath(config_path))
        return os.path.abspath(os.path.join(base_dir, p))


def load_config(path: str) -> AurexConfig:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Config not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON config: {e}") from e

    root = _as_dict(raw, where="config")
    schema_version = int(root.get("schema_version") or 2)

    account_raw = _as_dict(root.get("account") or {}, where="account")
    account = AccountConfig(
        email=str(account_raw.get("email") or "").strip(),
        password=str(account_raw.get("password") or "").strip(),
    )

    def parse_model(key: str, defaults: OllamaModelConfig) -> OllamaModelConfig:
        m_raw = _as_dict(root.get(key) or {}, where=key)
        base_url = _norm_base_url(str(m_raw.get("base_url") or defaults.base_url))
        model = str(m_raw.get("model") or defaults.model).strip()
        timeout_sec = int(m_raw.get("timeout_sec") or defaults.timeout_sec)
        temperature = float(m_raw.get("temperature") if "temperature" in m_raw else defaults.temperature)
        num_predict = int(m_raw.get("num_predict") or defaults.num_predict)
        extra_options = m_raw.get("extra_options")
        if extra_options is None:
            extra_options_obj: dict[str, Any] = dict(defaults.extra_options)
        else:
            extra_options_obj = _as_dict(extra_options, where=f"{key}.extra_options")
        return OllamaModelConfig(
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
            temperature=temperature,
            num_predict=num_predict,
            extra_options=extra_options_obj,
        )

    planner = parse_model("planner", AurexConfig().planner)
    executor = parse_model("executor", AurexConfig().executor)

    storage_raw = _as_dict(root.get("storage") or {}, where="storage")
    storage = StorageConfig(
        cache_dir=str(storage_raw.get("cache_dir") or AurexConfig().storage.cache_dir),
        context_db_path=str(storage_raw.get("context_db_path") or ""),
    )

    web_raw = _as_dict(root.get("web_search") or {}, where="web_search")
    web = WebSearchConfig(
        provider=str(web_raw.get("provider") or "duckduckgo"),
        max_results=int(web_raw.get("max_results") or 5),
        region=str(web_raw.get("region") or "cn-zh"),
        safesearch=str(web_raw.get("safesearch") or "moderate"),
        time_range=str(web_raw.get("time_range") or "y"),
    )
    if web.provider != "duckduckgo":
        raise ConfigError("Only web_search.provider=duckduckgo is supported")

    pe_raw = _as_dict(root.get("phy_engine") or {}, where="phy_engine")
    verilog2plsav_args = pe_raw.get("verilog2plsav_args")
    if verilog2plsav_args is None:
        verilog2plsav_args_list = AurexConfig().phy_engine.verilog2plsav_args
    else:
        verilog2plsav_args_list = [str(x) for x in _as_list(verilog2plsav_args, where="phy_engine.verilog2plsav_args")]
    phy_engine = PhyEngineConfig(
        auto_build=bool(pe_raw.get("auto_build") or False),
        cmake_source_dir=str(pe_raw.get("cmake_source_dir") or AurexConfig().phy_engine.cmake_source_dir),
        cmake_build_dir=str(pe_raw.get("cmake_build_dir") or AurexConfig().phy_engine.cmake_build_dir),
        cmake_build_type=str(pe_raw.get("cmake_build_type") or AurexConfig().phy_engine.cmake_build_type),
        build_timeout_sec=int(pe_raw.get("build_timeout_sec") or AurexConfig().phy_engine.build_timeout_sec),
        run_timeout_sec=int(pe_raw.get("run_timeout_sec") or AurexConfig().phy_engine.run_timeout_sec),
        verilog2plsav_path=str(pe_raw.get("verilog2plsav_path") or ""),
        phyengine_lib_path=str(pe_raw.get("phyengine_lib_path") or ""),
        verilog2plsav_args=verilog2plsav_args_list,
    )

    agent_raw = _as_dict(root.get("agent") or {}, where="agent")
    agent_defaults = AurexConfig().agent
    agent = AgentConfig(
        mention_tag=str(agent_raw.get("mention_tag") or agent_defaults.mention_tag),
        require_mention=bool(
            agent_raw.get("require_mention") if "require_mention" in agent_raw else agent_defaults.require_mention
        ),
        user_targets_require_mention=bool(
            agent_raw.get("user_targets_require_mention")
            if "user_targets_require_mention" in agent_raw
            else agent_defaults.user_targets_require_mention
        ),
        trigger_on_reply_to_self=bool(
            agent_raw.get("trigger_on_reply_to_self")
            if "trigger_on_reply_to_self" in agent_raw
            else agent_defaults.trigger_on_reply_to_self
        ),
        trigger_on_own_wall_without_mention=bool(
            agent_raw.get("trigger_on_own_wall_without_mention")
            if "trigger_on_own_wall_without_mention" in agent_raw
            else agent_defaults.trigger_on_own_wall_without_mention
        ),
        strip_mention_tag_in_replies=bool(
            agent_raw.get("strip_mention_tag_in_replies")
            if "strip_mention_tag_in_replies" in agent_raw
            else agent_defaults.strip_mention_tag_in_replies
        ),
        max_plan_steps=int(agent_raw.get("max_plan_steps") or 10),
        max_tool_loops=int(agent_raw.get("max_tool_loops") or 20),
        max_final_chars=int(agent_raw.get("max_final_chars") or 2000),
        dry_run=bool(agent_raw.get("dry_run") or False),
        poll_interval_sec=float(agent_raw.get("poll_interval_sec") or 15.0),
        comment_take=int(agent_raw.get("comment_take") or 20),
        comment_scan_pages=int(agent_raw.get("comment_scan_pages") or 3),
        bootstrap_lookback_sec=int(
            agent_raw.get("bootstrap_lookback_sec")
            if "bootstrap_lookback_sec" in agent_raw
            else agent_defaults.bootstrap_lookback_sec
        ),
        reply_once=bool(
            agent_raw.get("reply_once")
            if "reply_once" in agent_raw
            else AurexConfig().agent.reply_once
        ),
        reply_once_ttl_sec=int(agent_raw.get("reply_once_ttl_sec") or 0),
        notifications_enabled=bool(
            agent_raw.get("notifications_enabled")
            if "notifications_enabled" in agent_raw
            else agent_defaults.notifications_enabled
        ),
        notification_category_ids=_parse_int_list(
            agent_raw.get("notification_category_ids"),
            default=[3],
            where="agent.notification_category_ids",
        ),
        notification_take=int(agent_raw.get("notification_take") or 20),
        log_level=str(agent_raw.get("log_level") or "INFO"),
        debug_llm_io=bool(agent_raw.get("debug_llm_io") or False),
        debug_llm_max_chars=int(agent_raw.get("debug_llm_max_chars") or 1200),
        force_reply_prefix=bool(
            agent_raw.get("force_reply_prefix")
            if "force_reply_prefix" in agent_raw
            else agent_defaults.force_reply_prefix
        ),
        final_reply_only=bool(
            agent_raw.get("final_reply_only")
            if "final_reply_only" in agent_raw
            else agent_defaults.final_reply_only
        ),
        prefetch_context_in_plan=bool(
            agent_raw.get("prefetch_context_in_plan")
            if "prefetch_context_in_plan" in agent_raw
            else agent_defaults.prefetch_context_in_plan
        ),
        prefetch_context_take=int(agent_raw.get("prefetch_context_take") or agent_defaults.prefetch_context_take),
        context_db_enabled=bool(
            agent_raw.get("context_db_enabled")
            if "context_db_enabled" in agent_raw
            else agent_defaults.context_db_enabled
        ),
        context_db_keep_last_comments=int(agent_raw.get("context_db_keep_last_comments") or 200),
        targets=[
            {"type": str(x.get("type") or ""), "id": str(x.get("id") or "")}
            for x in _as_list(agent_raw.get("targets") or [], where="agent.targets")
            if isinstance(x, dict)
        ],
    )

    return AurexConfig(
        schema_version=schema_version,
        account=account,
        planner=planner,
        executor=executor,
        storage=storage,
        web_search=web,
        phy_engine=phy_engine,
        agent=agent,
    )


def save_config(cfg: AurexConfig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = asdict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
