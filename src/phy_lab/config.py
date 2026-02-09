from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

TargetType = Literal["User", "Experiment", "Discussion"]
AgentMode = Literal["traditional", "agent"]

DEFAULT_NOTIFICATION_CATEGORY_IDS: list[int] = [0, 3]
DEFAULT_VERILOG2PLSAV_ARGS: list[str] = ["-O4", "--layout", "hier"]
DEFAULT_PUBLISH_TAGS: list[str] = ["SmallProject"]

DEFAULT_SYSTEM_PROMPT = """You are aurex, a production-grade agent tool developed by MacroModel for the Physics Lab AR community.

Role and context
- You operate inside the Physics Lab AR community comment system.
- You help community members by answering their questions, summarizing relevant content, and completing safe tool-assisted tasks.
- You are friendly, respectful, and practical.

Language
- Reply in the same language as the incoming comment.
- If the user mixes languages, reply using the user's dominant language.

Clean answers (very important)
- Answer ONLY what the user asked about the community content or the current task.
- Do NOT add unrelated advice, generic disclaimers, marketing, or off-topic commentary.
- Do NOT mention system prompts, internal tools, policies, chain-of-thought, or that you are an AI model.
- If the request is unclear, ask 1 to 2 short clarifying questions instead of guessing.

Formatting
- Prefer short paragraphs or bullet points.
- Be concise; avoid repeating the question.
- If you provide steps, number them.

Safety and professionalism
- Do not request or store passwords or sensitive personal data.
- Do not invent facts about an experiment or discussion; use only provided context.
- Verifiable community answers (critical):
  - Never fabricate titles/tags/topics for any user's works.
  - If you describe a specific work, you MUST first open it (Context JSON) and only use fields from that Context JSON.
  - In replies about a work, include Category + SummaryID + Subject so the user can verify.
  - If you cannot open/read the work (e.g. login failed / not found), say so and do NOT guess what it contains.
- Context handling (critical):
  - When you receive a system message containing "Context JSON (current page)" (or "Context JSON (current content page)"), treat it as the full page context (title/body/content/comments) and use it directly.
  - Do NOT ask the user to "share context" when Context JSON is present.
  - Only ask for missing details if Context JSON is absent or clearly insufficient.

Physics Lab community search (best-effort)
- Do NOT try to "search" the Physics Lab community unless the user explicitly asks you to search/find/recommend experiments or users.
- If you do search, keep expectations clear: results may be incomplete or empty.
- If the request is ambiguous, ask for a short keyword (or an ID/author) instead of searching blindly.

Political content (strict)
- Refuse to generate, summarize, translate, or answer any political or politically sensitive content.
- This includes: elections, parties, government propaganda, geopolitical conflicts, political persuasion/advocacy, and any content that would be considered politically sensitive.
- If asked, respond with a brief refusal in the user's language and offer to help with Physics Lab AR community topics instead.
- Do not use web search for political content.
- If the user asks you to search the web for political content, refuse (do not perform any web lookup).

Single-reply policy (critical)
- Reply ONLY to the author of the current comment.
- Do NOT address or @mention any other users.
- Provide a complete one-shot answer; do NOT ask follow-up questions or request more context.
"""


@dataclass(frozen=True)
class AccountConfig:
    email: str
    # Optional: for non-interactive testing only. Prefer env vars in production.
    password: str | None = None


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    base_urls: list[str] = field(default_factory=list)
    model: str = "llama3.1"
    request_timeout_sec: int = 240
    temperature: float = 0.2
    num_predict: int = 2048
    max_parallel_requests: int = 1
    # Enable gpt-oss prompt tuning (Harmony-style header + high reasoning hint).
    # This is applied client-side (does not require server support).
    gptoss_optimization: bool = False


@dataclass(frozen=True)
class StorageConfig:
    cache_dir: str = "cache"
    keep_temp: bool = False


@dataclass(frozen=True)
class PhyEngineConfig:
    verilog2plsav_path: str = ""
    phyengine_lib_path: str = ""
    auto_build: bool = False
    prebuild_on_start: bool = False
    verilog2plsav_args: list[str] = field(default_factory=lambda: list(DEFAULT_VERILOG2PLSAV_ARGS))
    cmake_source_dir: str = "third-parties/Phy-Engine/src"
    cmake_build_dir: str = "cache/phy-engine-build"
    cmake_build_type: str = "Release"
    build_timeout_sec: int = 900
    run_timeout_sec: int = 300


@dataclass(frozen=True)
class TargetConfig:
    type: TargetType
    id: str


@dataclass(frozen=True)
class AgentConfig:
    mode: AgentMode = "agent"
    include_self_wall: bool = True
    poll_interval_sec: float = 15.0
    take: int = 20
    comment_scan_pages: int = 5
    bootstrap_lookback_sec: float = 600.0
    dispatch_fast_poll_interval_sec: float = 0.2
    max_reply_chars: int = 2000
    dry_run: bool = False
    require_mention: bool = True
    user_targets_require_mention: bool = True
    mention_tag: str = "@aurex"
    notifications_enabled: bool = True
    notification_category_ids: list[int] = field(default_factory=lambda: list(DEFAULT_NOTIFICATION_CATEGORY_IDS))
    notification_take: int = 20
    web_search_enabled: bool = False
    web_search_provider: str = "google"
    auto_web_search: bool = True
    web_search_user_agent: str = ""
    web_search_proxy: str = ""
    web_search_timeout_sec: int = 20
    web_search_cache_ttl_sec: int = 3600
    web_search_max_results: int = 5
    web_search_fallback_to_ddg: bool = True
    web_search_searxng_base_url: str = ""
    auto_tool_routing: bool = True
    auto_publish: bool = False
    publish_by_default: bool = False
    circuit_max_attempts: int = 3
    publish_max_elements: int = 5000
    publish_category: str = "Discussion"
    publish_tags: list[str] = field(default_factory=lambda: list(DEFAULT_PUBLISH_TAGS))
    simulation_enabled: bool = True
    simulation_max_elements: int = 300
    simulation_ai_enabled: bool = True
    simulation_ai_max_components: int = 30
    simulation_ai_max_probes: int = 20
    overload_protection_enabled: bool = True
    overload_window_sec: int = 600
    overload_max_requests: int = 40
    overload_message_en: str = "Too many requests at the moment, please try again later."
    commands_enabled: bool = False
    command_prefix: str = "!"
    enable_publish: bool = False
    log_level: str = "INFO"
    log_include_comment_content: bool = False
    # Debug-only: log sanitized LLM/tool I/O (never logs chain-of-thought fields).
    debug_log_llm_io: bool = False
    debug_llm_max_chars: int = 800
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Reply constraints to reduce misunderstandings/spam in public threads.
    reply_once: bool = True
    # Cooldown for same (target,user) when reply_once=true. 0 disables cooldown (recommended).
    reply_once_ttl_sec: int = 0
    # If true, a direct reply to the agent account triggers even without an explicit @mention.
    # This can cause unwanted replies in multi-user threads, so default is false.
    trigger_on_reply_to_self: bool = False
    targets: list[TargetConfig] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    schema_version: int
    account: AccountConfig
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    phy_engine: PhyEngineConfig = field(default_factory=PhyEngineConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


class ConfigError(ValueError):
    pass


_HOSTPORT_RE = re.compile(r"^[A-Za-z0-9._-]+:\d{2,6}$")


def _normalize_ollama_endpoint(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    # Allow OLLAMA_HOST style values (e.g. "127.0.0.1:11435").
    if _HOSTPORT_RE.match(value):
        return "http://" + value
    return value


def _require_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be an object")
    return value


def _require_str(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value.strip()


def _optional_str(value: Any, *, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string")
    value = value.strip()
    return value if value else None


def _optional_bool(value: Any, *, where: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a boolean")
    return value


def _optional_int(value: Any, *, where: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer")
    return value


def _optional_int_list(value: Any, *, where: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ConfigError(f"{where} must be a list of integers")
    return list(value)


def _optional_str_list(value: Any, *, where: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ConfigError(f"{where} must be a list of strings")
    return [x for x in value if x.strip()]


def _optional_float(value: Any, *, where: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    return float(value)


def _parse_targets(value: Any, *, where: str) -> list[TargetConfig]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{where} must be a list")

    targets: list[TargetConfig] = []
    for idx, item in enumerate(value):
        item_where = f"{where}[{idx}]"
        obj = _require_mapping(item, where=item_where)
        target_type = _require_str(obj.get("type"), where=f"{item_where}.type")
        if target_type not in ("User", "Experiment", "Discussion"):
            raise ConfigError(
                f"{item_where}.type must be one of: User, Experiment, Discussion"
            )
        target_id = _require_str(obj.get("id"), where=f"{item_where}.id")
        targets.append(TargetConfig(type=target_type, id=target_id))

    return targets


def parse_config(data: dict[str, Any], *, source: str) -> Config:
    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ConfigError(
            f"{source}: unsupported schema_version {schema_version!r} (expected 1)"
        )

    account_obj = _require_mapping(data.get("account"), where="account")
    email = _require_str(account_obj.get("email"), where="account.email")
    password = _optional_str(account_obj.get("password"), where="account.password")

    ollama_obj = _require_mapping(data.get("ollama", {}), where="ollama")
    base_url = _optional_str(ollama_obj.get("base_url"), where="ollama.base_url")
    base_urls = _optional_str_list(ollama_obj.get("base_urls"), where="ollama.base_urls")
    model = _optional_str(ollama_obj.get("model"), where="ollama.model")
    request_timeout_sec = _optional_int(
        ollama_obj.get("request_timeout_sec"), where="ollama.request_timeout_sec"
    )
    temperature = _optional_float(
        ollama_obj.get("temperature"), where="ollama.temperature"
    )
    num_predict = _optional_int(ollama_obj.get("num_predict"), where="ollama.num_predict")
    max_parallel_requests = _optional_int(
        ollama_obj.get("max_parallel_requests"), where="ollama.max_parallel_requests"
    )
    gptoss_optimization = _optional_bool(
        ollama_obj.get("gptoss-optimization", ollama_obj.get("gptoss_optimization")),
        where="ollama.gptoss-optimization",
    )

    endpoints = [u for u in (base_urls or []) if isinstance(u, str) and u.strip()]
    if not endpoints:
        endpoints = [(base_url or OllamaConfig.base_url)]
    endpoints = [_normalize_ollama_endpoint(u) for u in endpoints if u.strip()]
    endpoints = [u.strip() for u in endpoints if u.strip()]
    if not endpoints:
        raise ConfigError("ollama.base_url must be a non-empty string")

    max_parallel_final = max_parallel_requests if max_parallel_requests is not None else OllamaConfig.max_parallel_requests
    if max_parallel_final <= 0:
        raise ConfigError("ollama.max_parallel_requests must be >= 1")
    ollama = OllamaConfig(
        base_url=endpoints[0],
        base_urls=endpoints,
        model=model or OllamaConfig.model,
        request_timeout_sec=request_timeout_sec
        if request_timeout_sec is not None
        else OllamaConfig.request_timeout_sec,
        temperature=temperature if temperature is not None else OllamaConfig.temperature,
        num_predict=num_predict if num_predict is not None else OllamaConfig.num_predict,
        max_parallel_requests=max_parallel_final,
        gptoss_optimization=bool(gptoss_optimization)
        if gptoss_optimization is not None
        else OllamaConfig.gptoss_optimization,
    )

    storage_obj = _require_mapping(data.get("storage", {}), where="storage")
    cache_dir = _optional_str(storage_obj.get("cache_dir"), where="storage.cache_dir")
    keep_temp = _optional_bool(storage_obj.get("keep_temp"), where="storage.keep_temp")
    storage = StorageConfig(
        cache_dir=cache_dir or StorageConfig.cache_dir,
        keep_temp=keep_temp if keep_temp is not None else StorageConfig.keep_temp,
    )

    phy_obj = _require_mapping(data.get("phy_engine", {}), where="phy_engine")
    verilog2plsav_path = _optional_str(
        phy_obj.get("verilog2plsav_path"), where="phy_engine.verilog2plsav_path"
    )
    phyengine_lib_path = _optional_str(
        phy_obj.get("phyengine_lib_path"), where="phy_engine.phyengine_lib_path"
    )
    auto_build = _optional_bool(phy_obj.get("auto_build"), where="phy_engine.auto_build")
    prebuild_on_start = _optional_bool(
        phy_obj.get("prebuild_on_start"), where="phy_engine.prebuild_on_start"
    )
    verilog2plsav_args = _optional_str_list(
        phy_obj.get("verilog2plsav_args"), where="phy_engine.verilog2plsav_args"
    )
    cmake_source_dir = _optional_str(
        phy_obj.get("cmake_source_dir"), where="phy_engine.cmake_source_dir"
    )
    cmake_build_dir = _optional_str(
        phy_obj.get("cmake_build_dir"), where="phy_engine.cmake_build_dir"
    )
    cmake_build_type = _optional_str(
        phy_obj.get("cmake_build_type"), where="phy_engine.cmake_build_type"
    )
    build_timeout_sec = _optional_int(
        phy_obj.get("build_timeout_sec"), where="phy_engine.build_timeout_sec"
    )
    run_timeout_sec = _optional_int(
        phy_obj.get("run_timeout_sec"), where="phy_engine.run_timeout_sec"
    )
    phy_engine = PhyEngineConfig(
        verilog2plsav_path=verilog2plsav_path or PhyEngineConfig.verilog2plsav_path,
        phyengine_lib_path=phyengine_lib_path or PhyEngineConfig.phyengine_lib_path,
        auto_build=auto_build if auto_build is not None else PhyEngineConfig.auto_build,
        prebuild_on_start=prebuild_on_start
        if prebuild_on_start is not None
        else PhyEngineConfig.prebuild_on_start,
        verilog2plsav_args=verilog2plsav_args
        if verilog2plsav_args is not None
        else list(DEFAULT_VERILOG2PLSAV_ARGS),
        cmake_source_dir=cmake_source_dir or PhyEngineConfig.cmake_source_dir,
        cmake_build_dir=cmake_build_dir or PhyEngineConfig.cmake_build_dir,
        cmake_build_type=cmake_build_type or PhyEngineConfig.cmake_build_type,
        build_timeout_sec=build_timeout_sec
        if build_timeout_sec is not None
        else PhyEngineConfig.build_timeout_sec,
        run_timeout_sec=run_timeout_sec
        if run_timeout_sec is not None
        else PhyEngineConfig.run_timeout_sec,
    )

    agent_obj = _require_mapping(data.get("agent", {}), where="agent")
    mode = _optional_str(agent_obj.get("mode"), where="agent.mode")
    include_self_wall = _optional_bool(
        agent_obj.get("include_self_wall"), where="agent.include_self_wall"
    )
    poll_interval_sec = _optional_float(
        agent_obj.get("poll_interval_sec"), where="agent.poll_interval_sec"
    )
    take = _optional_int(agent_obj.get("take"), where="agent.take")
    comment_scan_pages = _optional_int(
        agent_obj.get("comment_scan_pages"), where="agent.comment_scan_pages"
    )
    bootstrap_lookback_sec = _optional_float(
        agent_obj.get("bootstrap_lookback_sec"), where="agent.bootstrap_lookback_sec"
    )
    dispatch_fast_poll_interval_sec = _optional_float(
        agent_obj.get("dispatch_fast_poll_interval_sec"),
        where="agent.dispatch_fast_poll_interval_sec",
    )
    if comment_scan_pages is not None and comment_scan_pages <= 0:
        raise ConfigError("agent.comment_scan_pages must be >= 1")
    if dispatch_fast_poll_interval_sec is not None and dispatch_fast_poll_interval_sec <= 0:
        raise ConfigError("agent.dispatch_fast_poll_interval_sec must be > 0")
    max_reply_chars = _optional_int(
        agent_obj.get("max_reply_chars"), where="agent.max_reply_chars"
    )
    dry_run = _optional_bool(agent_obj.get("dry_run"), where="agent.dry_run")
    require_mention = _optional_bool(
        agent_obj.get("require_mention"), where="agent.require_mention"
    )
    user_targets_require_mention = _optional_bool(
        agent_obj.get("user_targets_require_mention"),
        where="agent.user_targets_require_mention",
    )
    mention_tag = _optional_str(agent_obj.get("mention_tag"), where="agent.mention_tag")
    notifications_enabled = _optional_bool(
        agent_obj.get("notifications_enabled"), where="agent.notifications_enabled"
    )
    notification_category_ids = _optional_int_list(
        agent_obj.get("notification_category_ids"),
        where="agent.notification_category_ids",
    )
    notification_take = _optional_int(
        agent_obj.get("notification_take"), where="agent.notification_take"
    )
    web_search_enabled = _optional_bool(
        agent_obj.get("web_search_enabled"), where="agent.web_search_enabled"
    )
    web_search_provider = _optional_str(
        agent_obj.get("web_search_provider"), where="agent.web_search_provider"
    )
    auto_web_search = _optional_bool(
        agent_obj.get("auto_web_search"), where="agent.auto_web_search"
    )
    web_search_user_agent = _optional_str(
        agent_obj.get("web_search_user_agent"), where="agent.web_search_user_agent"
    )
    web_search_proxy = _optional_str(
        agent_obj.get("web_search_proxy"), where="agent.web_search_proxy"
    )
    web_search_timeout_sec = _optional_int(
        agent_obj.get("web_search_timeout_sec"), where="agent.web_search_timeout_sec"
    )
    web_search_cache_ttl_sec = _optional_int(
        agent_obj.get("web_search_cache_ttl_sec"), where="agent.web_search_cache_ttl_sec"
    )
    web_search_max_results = _optional_int(
        agent_obj.get("web_search_max_results"), where="agent.web_search_max_results"
    )
    web_search_fallback_to_ddg = _optional_bool(
        agent_obj.get("web_search_fallback_to_ddg"),
        where="agent.web_search_fallback_to_ddg",
    )
    web_search_searxng_base_url = _optional_str(
        agent_obj.get("web_search_searxng_base_url"),
        where="agent.web_search_searxng_base_url",
    )
    auto_tool_routing = _optional_bool(
        agent_obj.get("auto_tool_routing"), where="agent.auto_tool_routing"
    )
    auto_publish = _optional_bool(
        agent_obj.get("auto_publish"), where="agent.auto_publish"
    )
    publish_by_default = _optional_bool(
        agent_obj.get("publish_by_default"), where="agent.publish_by_default"
    )
    circuit_max_attempts = _optional_int(
        agent_obj.get("circuit_max_attempts"), where="agent.circuit_max_attempts"
    )
    publish_max_elements = _optional_int(
        agent_obj.get("publish_max_elements"), where="agent.publish_max_elements"
    )
    publish_category = _optional_str(
        agent_obj.get("publish_category"), where="agent.publish_category"
    )
    publish_tags = _optional_str_list(
        agent_obj.get("publish_tags"), where="agent.publish_tags"
    )
    overload_protection_enabled = _optional_bool(
        agent_obj.get("overload_protection_enabled"),
        where="agent.overload_protection_enabled",
    )
    overload_window_sec = _optional_int(
        agent_obj.get("overload_window_sec"), where="agent.overload_window_sec"
    )
    overload_max_requests = _optional_int(
        agent_obj.get("overload_max_requests"), where="agent.overload_max_requests"
    )
    overload_message_en = _optional_str(
        agent_obj.get("overload_message_en"), where="agent.overload_message_en"
    )
    command_prefix = _optional_str(
        agent_obj.get("command_prefix"), where="agent.command_prefix"
    )
    enable_publish = _optional_bool(
        agent_obj.get("enable_publish"), where="agent.enable_publish"
    )
    commands_enabled = _optional_bool(
        agent_obj.get("commands_enabled"), where="agent.commands_enabled"
    )
    log_level = _optional_str(agent_obj.get("log_level"), where="agent.log_level")
    log_include_comment_content = _optional_bool(
        agent_obj.get("log_include_comment_content"),
        where="agent.log_include_comment_content",
    )
    debug_log_llm_io = _optional_bool(
        agent_obj.get("debug_log_llm_io"), where="agent.debug_log_llm_io"
    )
    debug_llm_max_chars = _optional_int(
        agent_obj.get("debug_llm_max_chars"), where="agent.debug_llm_max_chars"
    )
    if debug_llm_max_chars is not None and debug_llm_max_chars < 0:
        raise ConfigError("agent.debug_llm_max_chars must be >= 0")
    system_prompt = _optional_str(
        agent_obj.get("system_prompt"), where="agent.system_prompt"
    )
    reply_once = _optional_bool(agent_obj.get("reply_once"), where="agent.reply_once")
    reply_once_ttl_sec = _optional_int(
        agent_obj.get("reply_once_ttl_sec"), where="agent.reply_once_ttl_sec"
    )
    trigger_on_reply_to_self = _optional_bool(
        agent_obj.get("trigger_on_reply_to_self"), where="agent.trigger_on_reply_to_self"
    )
    if reply_once_ttl_sec is not None and reply_once_ttl_sec < 0:
        raise ConfigError("agent.reply_once_ttl_sec must be >= 0")

    publish_category_final = publish_category or AgentConfig.publish_category
    if publish_category_final not in ("Experiment", "Discussion"):
        raise ConfigError("agent.publish_category must be 'Experiment' or 'Discussion'")

    web_search_provider_final = (web_search_provider or AgentConfig.web_search_provider).strip().lower()
    if web_search_provider_final not in (
        "google",
        "duckduckgo",
        "duckduckgo-search",
        "duckduckgo_search",
        "ddg",
        "ddg-search",
        "ddg_search",
        "ddgsearch",
        "baidu",
        "bing",
        "bing_html",
        "searxng",
        "searx",
    ):
        raise ConfigError(
            "agent.web_search_provider must be one of: duckduckgo-search, duckduckgo, google, baidu, bing, searxng"
        )
    if web_search_provider_final in (
        "ddg",
        "duckduckgo-search",
        "duckduckgo_search",
        "ddg-search",
        "ddg_search",
        "ddgsearch",
    ):
        web_search_provider_final = "duckduckgo"
    if web_search_provider_final == "bing_html":
        web_search_provider_final = "bing"
    if web_search_provider_final == "searx":
        web_search_provider_final = "searxng"

    simulation_enabled = _optional_bool(
        agent_obj.get("simulation_enabled"), where="agent.simulation_enabled"
    )
    simulation_max_elements = _optional_int(
        agent_obj.get("simulation_max_elements"), where="agent.simulation_max_elements"
    )
    if simulation_max_elements is not None and simulation_max_elements < 0:
        raise ConfigError("agent.simulation_max_elements must be >= 0")
    simulation_ai_enabled = _optional_bool(
        agent_obj.get("simulation_ai_enabled"), where="agent.simulation_ai_enabled"
    )
    simulation_ai_max_components = _optional_int(
        agent_obj.get("simulation_ai_max_components"),
        where="agent.simulation_ai_max_components",
    )
    simulation_ai_max_probes = _optional_int(
        agent_obj.get("simulation_ai_max_probes"),
        where="agent.simulation_ai_max_probes",
    )
    if simulation_ai_max_components is not None and simulation_ai_max_components <= 0:
        raise ConfigError("agent.simulation_ai_max_components must be > 0")
    if simulation_ai_max_probes is not None and simulation_ai_max_probes < 0:
        raise ConfigError("agent.simulation_ai_max_probes must be >= 0")

    mode_final: AgentMode = AgentConfig.mode
    if mode is not None:
        m = mode.strip().lower()
        if m not in ("traditional", "agent"):
            raise ConfigError("agent.mode must be one of: traditional, agent")
        mode_final = "agent" if m == "agent" else "traditional"

    agent = AgentConfig(
        mode=mode_final,
        include_self_wall=include_self_wall
        if include_self_wall is not None
        else AgentConfig.include_self_wall,
        poll_interval_sec=poll_interval_sec
        if poll_interval_sec is not None
        else AgentConfig.poll_interval_sec,
        take=take if take is not None else AgentConfig.take,
        comment_scan_pages=comment_scan_pages
        if comment_scan_pages is not None
        else AgentConfig.comment_scan_pages,
        bootstrap_lookback_sec=bootstrap_lookback_sec
        if bootstrap_lookback_sec is not None
        else AgentConfig.bootstrap_lookback_sec,
        dispatch_fast_poll_interval_sec=dispatch_fast_poll_interval_sec
        if dispatch_fast_poll_interval_sec is not None
        else AgentConfig.dispatch_fast_poll_interval_sec,
        max_reply_chars=max_reply_chars
        if max_reply_chars is not None
        else AgentConfig.max_reply_chars,
        dry_run=dry_run if dry_run is not None else AgentConfig.dry_run,
        require_mention=require_mention
        if require_mention is not None
        else AgentConfig.require_mention,
        user_targets_require_mention=user_targets_require_mention
        if user_targets_require_mention is not None
        else AgentConfig.user_targets_require_mention,
        mention_tag=mention_tag or AgentConfig.mention_tag,
        notifications_enabled=notifications_enabled
        if notifications_enabled is not None
        else AgentConfig.notifications_enabled,
        notification_category_ids=notification_category_ids
        if notification_category_ids is not None
        else list(DEFAULT_NOTIFICATION_CATEGORY_IDS),
        notification_take=notification_take
        if notification_take is not None
        else AgentConfig.notification_take,
        web_search_enabled=web_search_enabled
        if web_search_enabled is not None
        else AgentConfig.web_search_enabled,
        web_search_provider=web_search_provider_final,
        auto_web_search=auto_web_search
        if auto_web_search is not None
        else AgentConfig.auto_web_search,
        web_search_user_agent=web_search_user_agent
        if web_search_user_agent is not None
        else AgentConfig.web_search_user_agent,
        web_search_proxy=web_search_proxy or AgentConfig.web_search_proxy,
        web_search_timeout_sec=web_search_timeout_sec
        if web_search_timeout_sec is not None
        else AgentConfig.web_search_timeout_sec,
        web_search_cache_ttl_sec=web_search_cache_ttl_sec
        if web_search_cache_ttl_sec is not None
        else AgentConfig.web_search_cache_ttl_sec,
        web_search_max_results=web_search_max_results
        if web_search_max_results is not None
        else AgentConfig.web_search_max_results,
        web_search_fallback_to_ddg=web_search_fallback_to_ddg
        if web_search_fallback_to_ddg is not None
        else AgentConfig.web_search_fallback_to_ddg,
        web_search_searxng_base_url=web_search_searxng_base_url
        if web_search_searxng_base_url is not None
        else AgentConfig.web_search_searxng_base_url,
        auto_tool_routing=auto_tool_routing
        if auto_tool_routing is not None
        else AgentConfig.auto_tool_routing,
        auto_publish=auto_publish if auto_publish is not None else AgentConfig.auto_publish,
        publish_by_default=publish_by_default
        if publish_by_default is not None
        else AgentConfig.publish_by_default,
        circuit_max_attempts=circuit_max_attempts
        if circuit_max_attempts is not None
        else AgentConfig.circuit_max_attempts,
        publish_max_elements=publish_max_elements
        if publish_max_elements is not None
        else AgentConfig.publish_max_elements,
        publish_category=publish_category_final,
        publish_tags=publish_tags if publish_tags is not None else list(DEFAULT_PUBLISH_TAGS),
        simulation_enabled=simulation_enabled
        if simulation_enabled is not None
        else AgentConfig.simulation_enabled,
        simulation_max_elements=simulation_max_elements
        if simulation_max_elements is not None
        else AgentConfig.simulation_max_elements,
        simulation_ai_enabled=simulation_ai_enabled
        if simulation_ai_enabled is not None
        else AgentConfig.simulation_ai_enabled,
        simulation_ai_max_components=simulation_ai_max_components
        if simulation_ai_max_components is not None
        else AgentConfig.simulation_ai_max_components,
        simulation_ai_max_probes=simulation_ai_max_probes
        if simulation_ai_max_probes is not None
        else AgentConfig.simulation_ai_max_probes,
        overload_protection_enabled=overload_protection_enabled
        if overload_protection_enabled is not None
        else AgentConfig.overload_protection_enabled,
        overload_window_sec=overload_window_sec
        if overload_window_sec is not None
        else AgentConfig.overload_window_sec,
        overload_max_requests=overload_max_requests
        if overload_max_requests is not None
        else AgentConfig.overload_max_requests,
        overload_message_en=overload_message_en or AgentConfig.overload_message_en,
        commands_enabled=commands_enabled
        if commands_enabled is not None
        else AgentConfig.commands_enabled,
        command_prefix=command_prefix or AgentConfig.command_prefix,
        enable_publish=enable_publish
        if enable_publish is not None
        else AgentConfig.enable_publish,
        log_level=log_level or AgentConfig.log_level,
        log_include_comment_content=log_include_comment_content
        if log_include_comment_content is not None
        else AgentConfig.log_include_comment_content,
        debug_log_llm_io=debug_log_llm_io
        if debug_log_llm_io is not None
        else AgentConfig.debug_log_llm_io,
        debug_llm_max_chars=debug_llm_max_chars
        if debug_llm_max_chars is not None
        else AgentConfig.debug_llm_max_chars,
        system_prompt=system_prompt or AgentConfig.system_prompt,
        reply_once=reply_once if reply_once is not None else AgentConfig.reply_once,
        reply_once_ttl_sec=reply_once_ttl_sec
        if reply_once_ttl_sec is not None
        else AgentConfig.reply_once_ttl_sec,
        trigger_on_reply_to_self=trigger_on_reply_to_self
        if trigger_on_reply_to_self is not None
        else AgentConfig.trigger_on_reply_to_self,
        targets=_parse_targets(agent_obj.get("targets"), where="agent.targets"),
    )

    return Config(
        schema_version=1,
        account=AccountConfig(email=email, password=password),
        ollama=ollama,
        storage=storage,
        phy_engine=phy_engine,
        agent=agent,
    )


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: config file must be a JSON object")
    return parse_config(data, source=path)


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".tmp.{os.path.basename(path)}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def write_config(path: str, config: Config) -> None:
    data: dict[str, Any] = {
        "schema_version": config.schema_version,
        "account": {
            "email": config.account.email,
            **(
                {"password": config.account.password}
                if isinstance(config.account.password, str) and config.account.password.strip()
                else {}
            ),
        },
        "ollama": {
            "base_url": config.ollama.base_url,
            "base_urls": list(config.ollama.base_urls),
            "model": config.ollama.model,
            "request_timeout_sec": config.ollama.request_timeout_sec,
            "temperature": config.ollama.temperature,
            "num_predict": config.ollama.num_predict,
            "max_parallel_requests": config.ollama.max_parallel_requests,
            "gptoss-optimization": bool(getattr(config.ollama, "gptoss_optimization", False)),
        },
        "storage": {
            "cache_dir": config.storage.cache_dir,
            "keep_temp": config.storage.keep_temp,
        },
        "phy_engine": {
            "verilog2plsav_path": config.phy_engine.verilog2plsav_path,
            "phyengine_lib_path": config.phy_engine.phyengine_lib_path,
            "auto_build": config.phy_engine.auto_build,
            "prebuild_on_start": config.phy_engine.prebuild_on_start,
            "verilog2plsav_args": list(config.phy_engine.verilog2plsav_args),
            "cmake_source_dir": config.phy_engine.cmake_source_dir,
            "cmake_build_dir": config.phy_engine.cmake_build_dir,
            "cmake_build_type": config.phy_engine.cmake_build_type,
            "build_timeout_sec": config.phy_engine.build_timeout_sec,
            "run_timeout_sec": config.phy_engine.run_timeout_sec,
        },
        "agent": {
            "mode": config.agent.mode,
            "include_self_wall": config.agent.include_self_wall,
            "poll_interval_sec": config.agent.poll_interval_sec,
            "take": config.agent.take,
            "comment_scan_pages": config.agent.comment_scan_pages,
            "bootstrap_lookback_sec": config.agent.bootstrap_lookback_sec,
            "dispatch_fast_poll_interval_sec": config.agent.dispatch_fast_poll_interval_sec,
            "max_reply_chars": config.agent.max_reply_chars,
            "dry_run": config.agent.dry_run,
            "require_mention": config.agent.require_mention,
            "user_targets_require_mention": config.agent.user_targets_require_mention,
            "mention_tag": config.agent.mention_tag,
            "notifications_enabled": config.agent.notifications_enabled,
            "notification_category_ids": list(config.agent.notification_category_ids),
            "notification_take": config.agent.notification_take,
            "web_search_enabled": config.agent.web_search_enabled,
            "web_search_provider": config.agent.web_search_provider,
            "auto_web_search": config.agent.auto_web_search,
            "web_search_user_agent": config.agent.web_search_user_agent,
            "web_search_proxy": config.agent.web_search_proxy,
            "web_search_timeout_sec": config.agent.web_search_timeout_sec,
            "web_search_cache_ttl_sec": config.agent.web_search_cache_ttl_sec,
            "web_search_max_results": config.agent.web_search_max_results,
            "web_search_fallback_to_ddg": config.agent.web_search_fallback_to_ddg,
            "web_search_searxng_base_url": config.agent.web_search_searxng_base_url,
            "auto_tool_routing": config.agent.auto_tool_routing,
            "auto_publish": config.agent.auto_publish,
            "publish_by_default": config.agent.publish_by_default,
            "circuit_max_attempts": config.agent.circuit_max_attempts,
            "publish_max_elements": config.agent.publish_max_elements,
            "publish_category": config.agent.publish_category,
            "publish_tags": list(config.agent.publish_tags),
            "simulation_enabled": config.agent.simulation_enabled,
            "simulation_max_elements": config.agent.simulation_max_elements,
            "simulation_ai_enabled": config.agent.simulation_ai_enabled,
            "simulation_ai_max_components": config.agent.simulation_ai_max_components,
            "simulation_ai_max_probes": config.agent.simulation_ai_max_probes,
            "overload_protection_enabled": config.agent.overload_protection_enabled,
            "overload_window_sec": config.agent.overload_window_sec,
            "overload_max_requests": config.agent.overload_max_requests,
            "overload_message_en": config.agent.overload_message_en,
            "commands_enabled": config.agent.commands_enabled,
            "command_prefix": config.agent.command_prefix,
            "enable_publish": config.agent.enable_publish,
            "log_level": config.agent.log_level,
            "log_include_comment_content": config.agent.log_include_comment_content,
            "debug_log_llm_io": bool(getattr(config.agent, "debug_log_llm_io", False)),
            "debug_llm_max_chars": int(getattr(config.agent, "debug_llm_max_chars", 800) or 800),
            "reply_once": bool(getattr(config.agent, "reply_once", True)),
            "reply_once_ttl_sec": int(getattr(config.agent, "reply_once_ttl_sec", 0) or 0),
            "trigger_on_reply_to_self": bool(getattr(config.agent, "trigger_on_reply_to_self", False)),
            "system_prompt": config.agent.system_prompt,
            "targets": [{"type": t.type, "id": t.id} for t in config.agent.targets],
        },
    }
    _atomic_write_json(path, data)


def init_config_interactive(path: str, *, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        raise ConfigError(f"{path} already exists (use --overwrite to replace it)")

    email = ""
    while not email:
        email = input("Physics Lab AR account email: ").strip()

    model = input("Ollama model (default: llama3.1): ").strip() or "llama3.1"
    base_url = (
        input("Ollama base URL (default: http://127.0.0.1:11434): ").strip()
        or "http://127.0.0.1:11434"
    )
    mention_tag = input("Mention tag (default: @aurex): ").strip() or "@aurex"
    mode = input("Agent mode (agent|traditional, default: agent): ").strip().lower() or "agent"
    if mode not in ("agent", "traditional"):
        mode = "agent"

    config = Config(
        schema_version=1,
        account=AccountConfig(email=email, password=None),
        ollama=OllamaConfig(base_url=base_url, model=model),
        agent=AgentConfig(mention_tag=mention_tag, mode=mode),  # type: ignore[arg-type]
    )
    write_config(path, config)


def config_dir(config_path: str) -> str:
    return os.path.dirname(os.path.abspath(config_path))


def resolve_path(path: str, *, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def pick_state_path(config_path: str) -> str:
    return os.path.join(config_dir(config_path), "state.json")


def pick_cache_dir(config_path: str, *, config: Config) -> str:
    return resolve_path(config.storage.cache_dir, base_dir=config_dir(config_path))


def format_targets(targets: Sequence[TargetConfig]) -> str:
    if not targets:
        return "(none)"
    return ", ".join(f"{t.type}:{t.id}" for t in targets)
