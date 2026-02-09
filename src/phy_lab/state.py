from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


class StateError(ValueError):
    pass


@dataclass
class TargetState:
    last_seen_timestamp_ms: int = 0
    processed_comment_keys: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    schema_version: int = 1
    targets: dict[str, TargetState] = field(default_factory=dict)
    conversations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    closed_conversations: dict[str, int] = field(default_factory=dict)
    recent_request_timestamps_ms: list[int] = field(default_factory=list)


def _require_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StateError(f"{where} must be an object")
    return value


def _require_int(value: Any, *, where: str) -> int:
    if not isinstance(value, int):
        raise StateError(f"{where} must be an integer")
    return value


def _require_str_list(value: Any, *, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise StateError(f"{where} must be a list of strings")
    return value


def parse_state(data: dict[str, Any], *, source: str) -> AgentState:
    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise StateError(
            f"{source}: unsupported schema_version {schema_version!r} (expected 1)"
        )

    targets_obj = _require_mapping(data.get("targets", {}), where="targets")
    targets: dict[str, TargetState] = {}
    for key, value in targets_obj.items():
        if not isinstance(key, str):
            raise StateError("targets keys must be strings")
        obj = _require_mapping(value, where=f"targets[{key!r}]")
        last_seen = _require_int(
            obj.get("last_seen_timestamp_ms", 0),
            where=f"targets[{key!r}].last_seen_timestamp_ms",
        )
        processed = _require_str_list(
            obj.get("processed_comment_keys", []),
            where=f"targets[{key!r}].processed_comment_keys",
        )
        targets[key] = TargetState(
            last_seen_timestamp_ms=last_seen,
            processed_comment_keys=processed,
        )

    conversations_obj = data.get("conversations", {})
    conversations: dict[str, list[dict[str, Any]]] = {}
    if conversations_obj is not None:
        if not isinstance(conversations_obj, dict):
            raise StateError("conversations must be an object")
        for key, value in conversations_obj.items():
            if not isinstance(key, str):
                raise StateError("conversations keys must be strings")
            if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
                raise StateError(f"conversations[{key!r}] must be a list of objects")
            conversations[key] = list(value)

    closed_obj = data.get("closed_conversations", {})
    closed: dict[str, int] = {}
    if closed_obj is not None:
        if not isinstance(closed_obj, dict):
            raise StateError("closed_conversations must be an object")
        for key, value in closed_obj.items():
            if not isinstance(key, str):
                raise StateError("closed_conversations keys must be strings")
            if not isinstance(value, int):
                raise StateError(f"closed_conversations[{key!r}] must be an integer timestamp")
            closed[key] = int(value)

    recent_obj = data.get("recent_request_timestamps_ms", [])
    recent: list[int] = []
    if recent_obj is not None:
        if not isinstance(recent_obj, list) or not all(isinstance(x, int) for x in recent_obj):
            raise StateError("recent_request_timestamps_ms must be a list of integers")
        recent = list(recent_obj)

    return AgentState(
        schema_version=1,
        targets=targets,
        conversations=conversations,
        closed_conversations=closed,
        recent_request_timestamps_ms=recent,
    )


def load_state(path: str) -> AgentState:
    if not os.path.exists(path):
        return AgentState()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise StateError(f"{path}: state file must be a JSON object")
    return parse_state(data, source=path)


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".tmp.{os.path.basename(path)}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def save_state(path: str, state: AgentState) -> None:
    data = {
        "schema_version": state.schema_version,
        "targets": {
            key: {
                "last_seen_timestamp_ms": target.last_seen_timestamp_ms,
                "processed_comment_keys": target.processed_comment_keys,
            }
            for key, target in state.targets.items()
        },
        "conversations": state.conversations,
        "closed_conversations": state.closed_conversations,
        "recent_request_timestamps_ms": state.recent_request_timestamps_ms,
    }
    _atomic_write_json(path, data)


def get_target_state(state: AgentState, key: str) -> TargetState:
    if key not in state.targets:
        state.targets[key] = TargetState()
    return state.targets[key]


def prune_processed_keys(target_state: TargetState, *, keep_last: int = 500) -> None:
    if keep_last <= 0:
        target_state.processed_comment_keys = []
        return
    if len(target_state.processed_comment_keys) > keep_last:
        target_state.processed_comment_keys = target_state.processed_comment_keys[
            -keep_last:
        ]


def append_conversation_turn(
    state: AgentState,
    *,
    key: str,
    role: str,
    content: str,
    ts_ms: int | None = None,
    keep_last: int = 20,
) -> None:
    if not key:
        return
    role = (role or "").strip()
    if role not in ("user", "assistant", "system"):
        role = "user"
    content = (content or "").strip()
    if not content:
        return

    entry: dict[str, Any] = {"role": role, "content": content}
    if isinstance(ts_ms, int):
        entry["ts_ms"] = ts_ms

    if key not in state.conversations:
        state.conversations[key] = []
    state.conversations[key].append(entry)
    if keep_last > 0 and len(state.conversations[key]) > keep_last:
        state.conversations[key] = state.conversations[key][-keep_last:]


def get_conversation_history(
    state: AgentState, *, key: str, max_turns: int = 12
) -> list[dict[str, str]]:
    if not key or key not in state.conversations:
        return []
    items = state.conversations[key]
    if max_turns > 0 and len(items) > max_turns:
        items = items[-max_turns:]
    out: list[dict[str, str]] = []
    for item in items:
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str) and role in (
            "user",
            "assistant",
            "system",
        ):
            out.append({"role": role, "content": content})
    return out


def record_request_timestamp(
    state: AgentState,
    *,
    ts_ms: int,
    keep_last: int = 5000,
) -> None:
    if not isinstance(ts_ms, int) or ts_ms <= 0:
        return
    state.recent_request_timestamps_ms.append(ts_ms)
    if keep_last > 0 and len(state.recent_request_timestamps_ms) > keep_last:
        state.recent_request_timestamps_ms = state.recent_request_timestamps_ms[-keep_last:]


def count_recent_requests(
    state: AgentState,
    *,
    now_ms: int,
    window_ms: int,
) -> int:
    if window_ms <= 0:
        return 0
    cutoff = now_ms - window_ms
    xs = state.recent_request_timestamps_ms
    if not xs:
        return 0
    # Keep it simple; list is short and bounded.
    return sum(1 for x in xs if isinstance(x, int) and x >= cutoff)
