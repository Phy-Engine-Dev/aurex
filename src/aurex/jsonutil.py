from __future__ import annotations

import json
from typing import Any


class JsonExtractError(RuntimeError):
    pass


def extract_first_json(text: str) -> Any:
    """Extract and parse the first top-level JSON object/array found in text."""
    s = (text or "").strip()
    if not s:
        raise JsonExtractError("empty response")

    start = -1
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start < 0:
        raise JsonExtractError("no JSON found")

    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(s[start:])
        return obj
    except json.JSONDecodeError as e:
        raise JsonExtractError(f"invalid JSON: {e}") from e


def dumps_compact(obj: Any, *, max_chars: int = 4000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        s = str(obj)
    if max_chars > 0 and len(s) > max_chars:
        return s[: max_chars - 1] + "…"
    return s

