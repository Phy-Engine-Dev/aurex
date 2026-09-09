from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
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
    # Empty derives a private sibling ``backups`` directory from the active
    # tracking database.  This keeps temporary/test databases isolated.
    history_dir: str = ""
    history_retention_enabled: bool = True
    history_compress_at_gib: float = 10.0
    history_delete_at_gib: float = 20.0
    history_check_interval_sec: int = 300
    history_min_age_sec: int = 300


@dataclass(frozen=True)
class WebSearchConfig:
    provider: str = "duckduckgo"
    max_results: int = 5
    region: str = "cn-zh"
    safesearch: str = "moderate"
    time_range: str = "y"
    base_url: str = ""
    api_key_env: str = "BRAVE_SEARCH_API_KEY"
    timeout_sec: int = 15


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen38-27b"
    timeout_sec: int = 600
    context_length: int = 65536
    max_output_tokens: int | None = None
    enable_thinking: bool = True
    # Explicit opt-in: bounded generated-token diagnostics for no-thinking tool
    # requests only. No raw IDs/decoded text, task limits, or automatic cancellation.
    stream_token_progress: bool = False
    reasoning_effort: str | None = None
    temperature: float = 0.2
    max_images: int = 2
    image_max_side: int = 1024
    compact_at_ratio: float = 0.80
    api_key_env: str = "AUREX_LLM_API_KEY"


@dataclass(frozen=True)
class ContextPolicyConfig:
    """Input management, independent of generation/thinking and durable storage.

    Profiles are administrator-selected flat overrides, not model instructions.
    None keeps the corresponding legacy/model-derived default.
    """
    auto_compact: bool = True
    prune: bool = False
    reserved_output_tokens: int | None = None
    safety_tokens: int = 2048
    compact_at_ratio: float | None = None
    document_budget_ratio: float = 0.40
    tool_output_tokens: int | None = None
    retain_recent_turns: int = 1
    retain_recent_tokens: int | None = None
    summary_max_tokens: int = 4096
    summary_thinking: bool = False
    prune_keep_tool_results: int = 4
    prune_images: bool = True
    # Retrieval/selection policy, not task/token budgets. Full fetched sources
    # remain archived even when only a related subset enters active context.
    community_recent_hours: float = 24.0
    community_max_comments: int = 100
    community_recent_comments: int = 20
    profile: str | None = None
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def resolved(self) -> "ContextPolicyConfig":
        _validate_context_values(asdict(self), where="context")
        if self.profile is None:
            return self
        if self.profile not in self.profiles:
            raise ConfigError(f"Unknown context.profile: {self.profile}")
        return replace(self, **self.profiles[self.profile], profile=None, profiles={})


def _validate_context_values(raw: dict, *, where: str, overrides: bool = False) -> None:
    fields = ContextPolicyConfig.__dataclass_fields__
    for key, value in raw.items():
        location = f"{where}.{key}"
        if key not in fields or (overrides and key in {"profile", "profiles"}):
            raise ConfigError(f"Unknown context policy field: {location}")
        if key in {"auto_compact", "prune", "summary_thinking", "prune_images"}:
            _as_bool(value, where=location)
        elif key in {"reserved_output_tokens", "tool_output_tokens", "retain_recent_tokens"}:
            if value is not None and (type(value) is not int or value < 0):
                raise ConfigError(f"{location} must be null or a non-negative integer")
            if key == "tool_output_tokens" and value == 0:
                raise ConfigError(f"{location} must be positive when configured")
        elif key in {"safety_tokens", "retain_recent_turns", "summary_max_tokens", "prune_keep_tool_results"}:
            minimum = 256 if key == "summary_max_tokens" else 0
            if type(value) is not int or value < minimum:
                raise ConfigError(f"{location} must be an integer >= {minimum}")
        elif key in {"compact_at_ratio", "document_budget_ratio"}:
            if key == "compact_at_ratio" and value is None:
                continue
            if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 1:
                raise ConfigError(f"{location} must be a finite ratio in (0, 1]")
        elif key == 'community_recent_hours':
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ConfigError(f'{location} must be a finite number >= 0')
        elif key in {'community_max_comments', 'community_recent_comments'}:
            maximum = 500 if key == 'community_max_comments' else 100
            if type(value) is not int or not 1 <= value <= maximum:
                raise ConfigError(f'{location} must be an integer in 1..{maximum}')
        elif key == "profile":
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{location} must be null or a non-empty profile name")
        elif key == "profiles":
            profiles = _as_dict(value, where=location)
            for name, policy in profiles.items():
                if not isinstance(name, str) or not name.strip():
                    raise ConfigError(f"{location} keys must be non-empty strings")
                _validate_context_values(_as_dict(policy, where=f"{location}.{name}"),
                                         where=f"{location}.{name}", overrides=True)


def parse_context_policy(raw: Any) -> ContextPolicyConfig:
    data = _as_dict(raw, where="context")
    _validate_context_values(data, where="context")
    policy = ContextPolicyConfig(**data)
    policy.resolved()  # Validate selected profile without discarding administrator configuration.
    return policy


@dataclass(frozen=True)
class TrackingConfig:
    database_path: str = ".aurex/aurex.sqlite3"
    hostname: str = "0.0.0.0"
    port: int = 4097
    token_env: str = "AUREX_WEB_TOKEN"
    # The durable scheduler may run independent sessions concurrently.  Keep
    # this at one for a single-sequence local vLLM deployment.
    max_parallel_tasks: int = 1


@dataclass(frozen=True)
class PhyEngineConfig:
    auto_build: bool = False
    cmake_source_dir: str = "third-parties/Phy-Engine/src"
    cmake_build_dir: str = ".aurex/cache/phy-engine-build"
    cmake_build_type: str = "Release"
    build_timeout_sec: int = 900
    run_timeout_sec: int = 300
    digital_component_limit: int = 4096
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
    # Per-task execution ceiling. Administrators may lower it, but a single
    # request must never occupy the worker for more than 30 minutes.
    task_timeout_sec: int = 1800
    dry_run: bool = False
    poll_interval_sec: float = 15.0
    comment_take: int = 20
    comment_scan_pages: int = 3
    # Active mention context is bounded by target kind.  Full comment scans
    # remain an operator/audit concern and never enter the first model turn.
    community_max_related_post_comments: int = 16
    community_max_related_user_messages: int = 32
    bootstrap_lookback_sec: int = 600
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
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextPolicyConfig = field(default_factory=ContextPolicyConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def resolve_path(self, path: str, *, config_path: str) -> str:
        p = (path or "").strip()
        if not p:
            return ""
        if os.path.isabs(p):
            return p
        # Primary base: config file directory (legacy behavior).
        base_cfg = os.path.dirname(os.path.abspath(config_path))
        cand_cfg = os.path.abspath(os.path.join(base_cfg, p))

        # Secondary base: current working directory.
        #
        # Many users keep configs under a repo-local ".config/" folder but still write
        # paths relative to the repo root (cwd). If we only resolve relative to the
        # config directory, common paths like "third-parties/..." become ".config/third-parties/..."
        # and break builds.
        base_cwd = os.getcwd()
        cand_cwd = os.path.abspath(os.path.join(base_cwd, p))

        if cand_cfg == cand_cwd:
            return cand_cfg

        def _exists(x: str) -> bool:
            try:
                return os.path.exists(x)
            except Exception:
                return False

        cfg_exists = _exists(cand_cfg)
        cwd_exists = _exists(cand_cwd)
        if cfg_exists and not cwd_exists:
            return cand_cfg
        if cwd_exists and not cfg_exists:
            return cand_cwd
        if cfg_exists and cwd_exists:
            # If both exist, keep legacy preference (config-dir).
            return cand_cfg

        # Neither exists: choose the candidate whose parent hierarchy matches the
        # current filesystem better (i.e., fewer missing path segments).
        def _missing_steps(x: str) -> int:
            steps = 0
            cur = x
            # Hard cap to avoid pathological loops on strange paths.
            while steps < 64:
                if _exists(cur):
                    break
                parent = os.path.dirname(cur)
                if not parent or parent == cur:
                    break
                cur = parent
                steps += 1
            return steps

        ms_cfg = _missing_steps(cand_cfg)
        ms_cwd = _missing_steps(cand_cwd)
        if ms_cwd < ms_cfg:
            return cand_cwd
        if ms_cfg < ms_cwd:
            return cand_cfg
        # Tie-break: keep legacy behavior.
        return cand_cfg


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
    storage_defaults = AurexConfig().storage
    history_enabled = storage_raw.get(
        "history_retention_enabled", storage_defaults.history_retention_enabled)
    history_compress = storage_raw.get(
        "history_compress_at_gib", storage_defaults.history_compress_at_gib)
    history_delete = storage_raw.get(
        "history_delete_at_gib", storage_defaults.history_delete_at_gib)
    history_interval = storage_raw.get(
        "history_check_interval_sec", storage_defaults.history_check_interval_sec)
    history_min_age = storage_raw.get(
        "history_min_age_sec", storage_defaults.history_min_age_sec)
    if type(history_enabled) is not bool:
        raise ConfigError("storage.history_retention_enabled must be a boolean")
    for key, value in (("history_compress_at_gib", history_compress),
                       ("history_delete_at_gib", history_delete)):
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ConfigError(f"storage.{key} must be a finite number > 0")
    if history_delete < history_compress:
        raise ConfigError(
            "storage.history_delete_at_gib must be >= storage.history_compress_at_gib")
    if type(history_interval) is not int or not 1 <= history_interval <= 86400:
        raise ConfigError("storage.history_check_interval_sec must be an integer in 1..86400")
    if type(history_min_age) is not int or not 0 <= history_min_age <= 604800:
        raise ConfigError("storage.history_min_age_sec must be an integer in 0..604800")
    history_dir = storage_raw.get("history_dir", storage_defaults.history_dir)
    if not isinstance(history_dir, str):
        raise ConfigError("storage.history_dir must be a string")
    storage = StorageConfig(
        cache_dir=str(storage_raw.get("cache_dir") or storage_defaults.cache_dir),
        context_db_path=str(storage_raw.get("context_db_path") or ""),
        history_dir=history_dir.strip(),
        history_retention_enabled=history_enabled,
        history_compress_at_gib=float(history_compress),
        history_delete_at_gib=float(history_delete),
        history_check_interval_sec=history_interval,
        history_min_age_sec=history_min_age,
    )

    web_raw = _as_dict(root.get("web_search") or {}, where="web_search")
    web = WebSearchConfig(
        provider=str(web_raw.get("provider") or "duckduckgo"),
        max_results=int(web_raw.get("max_results") or 5),
        region=str(web_raw.get("region") or "cn-zh"),
        safesearch=str(web_raw.get("safesearch") or "moderate"),
        time_range=str(web_raw.get("time_range") or "y"),
        base_url=str(web_raw.get("base_url") or ""),
        api_key_env=str(web_raw.get("api_key_env") or "BRAVE_SEARCH_API_KEY"),
        timeout_sec=int(web_raw.get("timeout_sec") or 15),
    )
    if web.provider not in {"auto", "brave", "searxng", "bing", "duckduckgo", "crossref"}:
        raise ConfigError("Unsupported web_search.provider")

    llm_raw = _as_dict(root.get("llm") or {}, where="llm")
    llm = LLMConfig(**{k: v for k, v in llm_raw.items() if k in LLMConfig.__dataclass_fields__})
    _as_bool(llm.stream_token_progress, where="llm.stream_token_progress")
    if not 4096 <= llm.context_length <= 1048576:
        raise ConfigError("llm.context_length must be between 4096 and 1048576")
    if llm.max_output_tokens is not None and (type(llm.max_output_tokens) is not int or not 256 <= llm.max_output_tokens < llm.context_length):
        raise ConfigError("llm.max_output_tokens must be null or an integer below llm.context_length")
    if llm.reasoning_effort not in {None, "low", "medium", "xhigh"}:
        raise ConfigError("llm.reasoning_effort must be null, low, medium, or xhigh for the Qwen template")
    if not 0.5 <= llm.compact_at_ratio <= 0.95:
        raise ConfigError("llm.compact_at_ratio must be between 0.5 and 0.95")
    if not 1 <= llm.max_images <= 8 or not 224 <= llm.image_max_side <= 1536:
        raise ConfigError("Invalid llm image budget")
    context = parse_context_policy(root.get("context", {}))
    policy = context.resolved()
    reserve = max(llm.max_output_tokens or 0, policy.reserved_output_tokens or 0)
    if reserve + policy.safety_tokens >= llm.context_length:
        raise ConfigError("context output reserve and safety_tokens must leave input space within llm.context_length")
    tracking_raw = _as_dict(root.get("tracking") or {}, where="tracking")
    max_parallel_tasks = tracking_raw.get(
        "max_parallel_tasks", AurexConfig().tracking.max_parallel_tasks)
    if (type(max_parallel_tasks) is not int or
            not 1 <= max_parallel_tasks <= 64):
        raise ConfigError(
            "tracking.max_parallel_tasks must be an integer in 1..64")
    tracking = TrackingConfig(**{
        k: v for k, v in tracking_raw.items()
        if k in TrackingConfig.__dataclass_fields__ and k != "max_parallel_tasks"
    }, max_parallel_tasks=max_parallel_tasks)

    pe_raw = _as_dict(root.get("phy_engine") or {}, where="phy_engine")
    verilog2plsav_args = pe_raw.get("verilog2plsav_args")
    if verilog2plsav_args is None:
        verilog2plsav_args_list = AurexConfig().phy_engine.verilog2plsav_args
    else:
        verilog2plsav_args_list = [str(x) for x in _as_list(verilog2plsav_args, where="phy_engine.verilog2plsav_args")]
    from .phy_engine.limits import configured_digital_limit
    try:
        digital_component_limit = configured_digital_limit(pe_raw.get("digital_component_limit", 4096))
    except ValueError as error:
        raise ConfigError(str(error)) from error
    phy_engine = PhyEngineConfig(
        auto_build=bool(pe_raw.get("auto_build") or False),
        cmake_source_dir=str(pe_raw.get("cmake_source_dir") or AurexConfig().phy_engine.cmake_source_dir),
        cmake_build_dir=str(pe_raw.get("cmake_build_dir") or AurexConfig().phy_engine.cmake_build_dir),
        cmake_build_type=str(pe_raw.get("cmake_build_type") or AurexConfig().phy_engine.cmake_build_type),
        build_timeout_sec=int(pe_raw.get("build_timeout_sec") or AurexConfig().phy_engine.build_timeout_sec),
        run_timeout_sec=int(pe_raw.get("run_timeout_sec") or AurexConfig().phy_engine.run_timeout_sec),
        digital_component_limit=digital_component_limit,
        verilog2plsav_path=str(pe_raw.get("verilog2plsav_path") or ""),
        phyengine_lib_path=str(pe_raw.get("phyengine_lib_path") or ""),
        verilog2plsav_args=verilog2plsav_args_list,
    )

    agent_raw = _as_dict(root.get("agent") or {}, where="agent")
    agent_defaults = AurexConfig().agent
    task_timeout_sec = agent_raw.get("task_timeout_sec", agent_defaults.task_timeout_sec)
    if type(task_timeout_sec) is not int or not 1 <= task_timeout_sec <= 1800:
        raise ConfigError("agent.task_timeout_sec must be an integer in 1..1800")
    community_post_comments = agent_raw.get(
        "community_max_related_post_comments",
        agent_defaults.community_max_related_post_comments,
    )
    community_user_messages = agent_raw.get(
        "community_max_related_user_messages",
        agent_defaults.community_max_related_user_messages,
    )
    if (type(community_post_comments) is not int or
            not 1 <= community_post_comments <= 100):
        raise ConfigError(
            "agent.community_max_related_post_comments must be an integer in 1..100")
    if (type(community_user_messages) is not int or
            not 1 <= community_user_messages <= 100):
        raise ConfigError(
            "agent.community_max_related_user_messages must be an integer in 1..100")
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
        task_timeout_sec=task_timeout_sec,
        dry_run=bool(agent_raw.get("dry_run") or False),
        poll_interval_sec=float(agent_raw.get("poll_interval_sec") or 15.0),
        comment_take=int(agent_raw.get("comment_take") or 20),
        comment_scan_pages=int(agent_raw.get("comment_scan_pages") or 3),
        community_max_related_post_comments=community_post_comments,
        community_max_related_user_messages=community_user_messages,
        bootstrap_lookback_sec=int(
            agent_raw.get("bootstrap_lookback_sec")
            if "bootstrap_lookback_sec" in agent_raw
            else agent_defaults.bootstrap_lookback_sec
        ),
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
        llm=llm,
        context=context,
        tracking=tracking,
    )


def save_config(cfg: AurexConfig, path: str) -> None:
    target = os.path.abspath(path)
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    data = asdict(cfg)
    fd, temporary = tempfile.mkstemp(prefix=".aurex-config-", suffix=".tmp", dir=parent)
    try:
        # Configuration contains the community password and API credentials.
        # mkstemp starts at 0600 irrespective of the process umask; fchmod also
        # documents and enforces that invariant on unusual platforms/filesystems.
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
