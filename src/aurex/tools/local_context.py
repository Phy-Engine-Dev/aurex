from __future__ import annotations

import os
from typing import Any

from ..contextdb import ContextDB
from .registry import ToolError, ToolRuntime


def local_get_target_context(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    target_key = str(args.get("target_key") or "").strip()
    if target_key and ":" not in target_key and "/" in target_key:
        target_key = target_key.replace("/", ":", 1)
    if not target_key:
        ttype = str(args.get("target_type") or "").strip()
        tid = str(args.get("target_id") or "").strip()
        if not ttype or not tid:
            raise ToolError("local_get_target_context: provide target_key or (target_type + target_id)")
        target_key = f"{ttype}:{tid}"
    if ":" not in target_key:
        raise ToolError("local_get_target_context: target_key must be like User:<id> | Experiment:<id> | Discussion:<id>")
    take = int(args.get("take") or 20)
    if take <= 0:
        take = 20
    if take > 200:
        take = 200

    cfg = runtime.config
    context_db_path_cfg = str(getattr(getattr(cfg, "storage", None), "context_db_path", "") or "").strip()
    if context_db_path_cfg:
        path = cfg.resolve_path(context_db_path_cfg, config_path=runtime.config_path)
    else:
        path = os.path.join(runtime.cache_dir, "context_db.json")

    db = ContextDB(path=path)
    out = db.get_target_context(target_key=target_key, take=take)
    comments = out.get("comments")
    if isinstance(comments, list):
        out["comments_count"] = len(comments)
    return out


LOCAL_GET_TARGET_CONTEXT_TOOL = {
    "name": "local_get_target_context",
    "description": "Get cached target context (recent comments) from a local JSON DB built by the runloop. Use when user refers to “this board/thread/notification”.",
    "parameters": {
        "type": "object",
        "properties": {
            "target_key": {
                "type": ["string", "null"],
                "description": "Type:ID, e.g. User:... / Experiment:... / Discussion:... (or User/... will be normalized).",
            },
            "target_type": {"type": ["string", "null"], "enum": ["User", "Experiment", "Discussion"]},
            "target_id": {"type": ["string", "null"]},
            "take": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
        },
        "required": [],
    },
}
