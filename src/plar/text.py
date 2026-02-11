from __future__ import annotations

import json
from typing import Any, Iterable


def best_effort_extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join([x for x in value if isinstance(x, str)])
    return ""


def iter_text_fields(item: dict[str, Any], keys: Iterable[str]) -> Iterable[str]:
    for key in keys:
        if key not in item:
            continue
        text = best_effort_extract_text(item.get(key)).strip()
        if text:
            yield text


_PRIORITY_TEXT_KEYS: tuple[str, ...] = (
    "Subject",
    "Title",
    "Name",
    "Description",
    "Content",
    "Text",
    "Body",
    "Markdown",
    "Html",
    "Introduction",
    "Intro",
    "Summary",
)

_SKIP_TEXT_KEYS: frozenset[str] = frozenset(
    {
        "StatusSave",
        "StatusSaveRaw",
        "Elements",
        "Wires",
        "Image",
        "Images",
        "Video",
        "Videos",
        "Audio",
        "Audios",
    }
)


def safe_json_excerpt(value: Any, *, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[: max(0, max_chars - 1)] + "…"
    return text


def extract_title(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    for key in ("Subject", "Title", "Name"):
        title = best_effort_extract_text(obj.get(key)).strip()
        if title:
            return title
    return ""


def collect_text(
    obj: Any,
    *,
    max_chars: int,
    max_nodes: int = 2500,
    max_depth: int = 7,
) -> str:
    """Walk a nested dict/list structure and collect human-readable strings."""
    if max_chars <= 0:
        return ""

    parts: list[str] = []
    seen_ids: set[int] = set()
    visited = 0
    stack: list[tuple[Any, int]] = [(obj, 0)]

    def _push(v: Any, depth: int) -> None:
        nonlocal visited
        if visited >= max_nodes:
            return
        stack.append((v, depth))
        visited += 1

    while stack and sum(len(p) for p in parts) < max_chars and visited < max_nodes:
        cur, depth = stack.pop()
        if cur is None or depth > max_depth:
            continue
        if isinstance(cur, str):
            s = cur.strip()
            if s:
                parts.append(s)
            continue
        if isinstance(cur, (int, float, bool)):
            continue

        cur_id = id(cur)
        if cur_id in seen_ids:
            continue
        seen_ids.add(cur_id)

        if isinstance(cur, list):
            for item in reversed(cur[:200]):
                _push(item, depth + 1)
            continue

        if isinstance(cur, dict):
            for key in reversed(_PRIORITY_TEXT_KEYS):
                if key in cur and key not in _SKIP_TEXT_KEYS:
                    _push(cur.get(key), depth + 1)
            for key in sorted(cur.keys(), key=lambda k: str(k), reverse=True):
                if key in _SKIP_TEXT_KEYS or key in _PRIORITY_TEXT_KEYS:
                    continue
                _push(cur.get(key), depth + 1)
            continue

    # stable de-dup
    out: list[str] = []
    seen_text: set[str] = set()
    total = 0
    for p in parts:
        p2 = p.strip()
        if not p2 or p2 in seen_text:
            continue
        seen_text.add(p2)
        out.append(p2)
        total += len(p2)
        if total >= max_chars:
            break

    joined = "\n".join(out).strip()
    if len(joined) <= max_chars:
        return joined
    return joined[: max(0, max_chars - 12)] + "...(truncated)"

