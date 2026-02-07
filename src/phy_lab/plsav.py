from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


class PlSavError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlSavCounts:
    elements: int | None
    wires: int | None


def _as_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        v = value.get("$values")
        if isinstance(v, list):
            return v
    return None


def _find_status_save(obj: Any, *, depth: int = 0, max_depth: int = 6) -> dict[str, Any] | None:
    if depth > max_depth:
        return None
    if isinstance(obj, dict):
        if "StatusSave" in obj and isinstance(obj.get("StatusSave"), dict):
            return obj.get("StatusSave")
        for v in obj.values():
            found = _find_status_save(v, depth=depth + 1, max_depth=max_depth)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_status_save(v, depth=depth + 1, max_depth=max_depth)
            if found is not None:
                return found
    return None


def plsav_counts_from_data(data: Any) -> PlSavCounts:
    status = None
    if isinstance(data, dict):
        exp = data.get("Experiment")
        if isinstance(exp, dict) and isinstance(exp.get("StatusSave"), dict):
            status = exp.get("StatusSave")
    if status is None:
        status = _find_status_save(data)

    if not isinstance(status, dict):
        return PlSavCounts(elements=None, wires=None)

    elements_list = None
    for k in ("Elements", "ElementSaves", "Models"):
        elements_list = _as_list(status.get(k))
        if elements_list is not None:
            break

    wires_list = None
    for k in ("Wires", "WireSaves", "Lines"):
        wires_list = _as_list(status.get(k))
        if wires_list is not None:
            break

    return PlSavCounts(
        elements=len(elements_list) if elements_list is not None else None,
        wires=len(wires_list) if wires_list is not None else None,
    )


def load_plsav_counts(
    path: str,
    *,
    max_bytes: int = 50 * 1024 * 1024,
) -> PlSavCounts:
    path = os.path.abspath(path)
    try:
        st = os.stat(path)
    except OSError as e:
        raise PlSavError(f"Cannot stat .sav: {e}") from e
    if max_bytes > 0 and st.st_size > max_bytes:
        raise PlSavError(f".sav is too large to inspect safely: {st.st_size} bytes")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise PlSavError(f"Cannot read .sav: {e}") from e
    except json.JSONDecodeError as e:
        raise PlSavError(f"Invalid .sav JSON: {e}") from e

    return plsav_counts_from_data(data)

