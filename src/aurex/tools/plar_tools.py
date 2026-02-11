from __future__ import annotations

from typing import Any

import plar

from .registry import ToolError, ToolRuntime


def _require_user(runtime: ToolRuntime) -> Any:
    if runtime.user is None:
        raise ToolError("plar tool requires a logged-in PhysicsLab user in runtime.user")
    return runtime.user


def plar_query_experiments(runtime: ToolRuntime, args: dict[str, Any]) -> list[dict[str, Any]]:
    user = _require_user(runtime)
    category = args.get("category") or "Experiment"
    return plar.query_experiments(
        user,
        category=category,
        take=int(args.get("take") or 20),
        skip=int(args.get("skip") or 0),
        from_skip=args.get("from_skip"),
        days=args.get("days"),
        sort=args.get("sort"),
        user_id=args.get("user_id"),
        tags=args.get("tags"),
        exclude_tags=args.get("exclude_tags"),
        languages=args.get("languages"),
        exclude_languages=args.get("exclude_languages"),
    )


def plar_get_user(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    user = _require_user(runtime)
    name = args.get("name")
    user_id = args.get("user_id")
    if isinstance(name, str) and name.strip():
        return plar.get_user_by_name(user, name=name)
    if isinstance(user_id, str) and user_id.strip():
        return plar.get_user_by_id(user, user_id=user_id)
    raise ToolError("plar_get_user requires either name or user_id")


def plar_get_relations(runtime: ToolRuntime, args: dict[str, Any]) -> list[dict[str, Any]]:
    user = _require_user(runtime)
    user_id = str(args.get("user_id") or "").strip()
    if not user_id:
        raise ToolError("plar_get_relations: user_id is required")
    return plar.get_relations(
        user,
        user_id=user_id,
        display_type=args.get("display_type") or "Following",
        skip=int(args.get("skip") or 0),
        take=int(args.get("take") or 20),
        query=str(args.get("query") or ""),
    )


def plar_get_experiment_context(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    user = _require_user(runtime)
    summary_id = str(args.get("summary_id") or "").strip()
    category = str(args.get("category") or "Experiment").strip() or "Experiment"
    if not summary_id:
        raise ToolError("plar_get_experiment_context: summary_id is required")
    ttl_sec = int(args.get("ttl_sec") or 300)
    max_json_chars = int(args.get("max_json_chars") or 20_000)
    return plar.get_experiment_context(
        user,
        summary_id=summary_id,
        category_value=category,
        cache_dir=runtime.cache_dir,
        ttl_sec=ttl_sec,
        max_json_chars=max_json_chars,
    )


def plar_get_status_save(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    user = _require_user(runtime)
    summary_id = str(args.get("summary_id") or "").strip()
    category = str(args.get("category") or "Experiment").strip() or "Experiment"
    if not summary_id:
        raise ToolError("plar_get_status_save: summary_id is required")
    ttl_sec = int(args.get("ttl_sec") or 300)
    return plar.get_status_save(
        user,
        summary_id=summary_id,
        category_value=category,
        cache_dir=runtime.cache_dir,
        ttl_sec=ttl_sec,
    )


def plar_upload_sav(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    user = _require_user(runtime)
    sav_path = str(args.get("sav_path") or "").strip()
    if not sav_path:
        raise ToolError("plar_upload_sav: sav_path is required")
    title = str(args.get("title") or "").strip()
    introduction = str(args.get("introduction") or "").strip()
    category = str(args.get("category") or "Discussion").strip() or "Discussion"
    tags = args.get("tags")
    if tags is not None and not isinstance(tags, list):
        raise ToolError("plar_upload_sav: tags must be a list of strings")
    tags_list = [str(x) for x in (tags or []) if str(x).strip()]
    return plar.upload_sav_as_experiment(
        user=user,
        sav_path=sav_path,
        title=title,
        introduction=introduction,
        cache_dir=runtime.cache_dir,
        category_value=category,
        tags=tags_list or None,
    )


PLAR_QUERY_TOOL = {
    "name": "plar_query_experiments",
    "description": "List experiments/discussions from PhysicsLab community (QueryExperiments).",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": ["Experiment", "Discussion"]},
            "take": {"type": "integer", "minimum": 1, "maximum": 24, "default": 20},
            "skip": {"type": "integer", "minimum": 0, "default": 0},
            "from_skip": {"type": ["string", "null"]},
            "days": {"type": ["integer", "string", "null"]},
            "sort": {"type": ["integer", "string", "null"]},
            "user_id": {"type": ["string", "null"]},
            "tags": {"type": ["array", "null"], "items": {"type": "string"}},
            "exclude_tags": {"type": ["array", "null"], "items": {"type": "string"}},
            "languages": {"type": ["array", "null"], "items": {"type": "string"}},
            "exclude_languages": {"type": ["array", "null"], "items": {"type": "string"}},
        },
        "required": ["category"],
    },
}

PLAR_GET_USER_TOOL = {
    "name": "plar_get_user",
    "description": "Get a PhysicsLab user by @name or user_id.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": ["string", "null"], "description": "User nickname/handle (with or without @)."},
            "user_id": {"type": ["string", "null"], "description": "User ID."},
        },
    },
}

PLAR_RELATIONS_TOOL = {
    "name": "plar_get_relations",
    "description": "List a user's relations (following/followers/banned/volunteers/editors/retired).",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "display_type": {"type": ["string", "integer"], "default": "Following"},
            "skip": {"type": "integer", "minimum": 0, "default": 0},
            "take": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            "query": {"type": "string", "default": ""},
        },
        "required": ["user_id"],
    },
}

PLAR_CONTEXT_TOOL = {
    "name": "plar_get_experiment_context",
    "description": "Open an experiment/discussion by summary_id and return a compact context (title, text excerpts, plsav summary).",
    "parameters": {
        "type": "object",
        "properties": {
            "summary_id": {"type": "string"},
            "category": {"type": "string", "enum": ["Experiment", "Discussion"], "default": "Experiment"},
            "ttl_sec": {"type": "integer", "minimum": 0, "default": 300},
            "max_json_chars": {"type": "integer", "minimum": 1000, "default": 20000},
        },
        "required": ["summary_id"],
    },
}

PLAR_STATUS_SAVE_TOOL = {
    "name": "plar_get_status_save",
    "description": "Fetch and parse StatusSave JSON for an experiment/discussion.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary_id": {"type": "string"},
            "category": {"type": "string", "enum": ["Experiment", "Discussion"], "default": "Experiment"},
            "ttl_sec": {"type": "integer", "minimum": 0, "default": 300},
        },
        "required": ["summary_id"],
    },
}

PLAR_UPLOAD_SAV_TOOL = {
    "name": "plar_upload_sav",
    "description": "Upload a local .sav as a PhysicsLab Experiment/Discussion and confirm it.",
    "parameters": {
        "type": "object",
        "properties": {
            "sav_path": {"type": "string"},
            "title": {"type": "string"},
            "introduction": {"type": "string"},
            "category": {"type": "string", "enum": ["Experiment", "Discussion"], "default": "Discussion"},
            "tags": {"type": ["array", "null"], "items": {"type": "string"}},
        },
        "required": ["sav_path", "title", "introduction"],
    },
}

