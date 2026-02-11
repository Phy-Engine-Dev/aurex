from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ContextDB:
    path: str

    def load(self) -> dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
        except FileNotFoundError:
            return {"version": 1, "targets": {}, "summaries": {}}
        except Exception:
            return {"version": 1, "targets": {}, "summaries": {}}
        return {"version": 1, "targets": {}, "summaries": {}}

    def save(self, data: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, self.path)

    def upsert_target_comments(
        self,
        *,
        target_key: str,
        target: dict[str, Any],
        comments: list[dict[str, Any]],
        keep_last: int = 200,
    ) -> None:
        if not target_key:
            return
        if keep_last <= 0:
            keep_last = 1

        data = self.load()
        targets = data.get("targets")
        if not isinstance(targets, dict):
            targets = {}
            data["targets"] = targets

        entry = targets.get(target_key)
        if not isinstance(entry, dict):
            entry = {}
            targets[target_key] = entry

        existing_target = entry.get("target")
        if isinstance(existing_target, dict):
            merged_target = dict(existing_target)
            merged_target.update(dict(target or {}))
            entry["target"] = merged_target
        else:
            entry["target"] = dict(target or {})
        entry["updated_at_ms"] = int(time.time() * 1000)

        existing = entry.get("comments")
        merged: dict[str, dict[str, Any]] = {}
        if isinstance(existing, list):
            for c in existing:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("id") or "").strip()
                if not cid:
                    continue
                merged[cid] = c

        for c in comments or []:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id") or "").strip()
            if not cid:
                continue
            merged[cid] = c

        def _ts_ms(obj: dict[str, Any]) -> int:
            v = obj.get("ts_ms")
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return int(v)
            return 0

        items = sorted(merged.values(), key=_ts_ms)
        if len(items) > keep_last:
            items = items[-keep_last:]
        entry["comments"] = items

        self.save(data)

    def upsert_target_meta(self, *, target_key: str, target: dict[str, Any]) -> None:
        if not target_key:
            return
        data = self.load()
        targets = data.get("targets")
        if not isinstance(targets, dict):
            targets = {}
            data["targets"] = targets

        entry = targets.get(target_key)
        if not isinstance(entry, dict):
            entry = {}
            targets[target_key] = entry

        existing_target = entry.get("target")
        if isinstance(existing_target, dict):
            merged_target = dict(existing_target)
            merged_target.update(dict(target or {}))
            entry["target"] = merged_target
        else:
            entry["target"] = dict(target or {})

        entry["target_meta_updated_at_ms"] = int(time.time() * 1000)
        self.save(data)

    def get_target_context(self, *, target_key: str, take: int = 20) -> dict[str, Any]:
        take_i = int(take or 20)
        if take_i <= 0:
            take_i = 20
        if take_i > 200:
            take_i = 200

        data = self.load()
        targets = data.get("targets")
        if not isinstance(targets, dict):
            return {"target_key": target_key, "found": False, "comments": []}
        entry = targets.get(target_key)
        if not isinstance(entry, dict):
            return {"target_key": target_key, "found": False, "comments": []}
        comments = entry.get("comments")
        out = [c for c in comments if isinstance(c, dict)] if isinstance(comments, list) else []
        if len(out) > take_i:
            out = out[-take_i:]
        return {
            "target_key": target_key,
            "found": True,
            "target": entry.get("target"),
            "updated_at_ms": entry.get("updated_at_ms"),
            "target_meta_updated_at_ms": entry.get("target_meta_updated_at_ms"),
            "comments": out,
        }
