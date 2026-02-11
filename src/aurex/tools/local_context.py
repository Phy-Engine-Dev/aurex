from __future__ import annotations

import os
from typing import Any

import plar

from ..contextdb import ContextDB
from .registry import ToolError, ToolRuntime


_TARGET_TYPE_CANON: dict[str, str] = {
    "user": "User",
    "experiment": "Experiment",
    "discussion": "Discussion",
}


def _canonical_target_key(key: str) -> str:
    s = str(key or "").strip()
    if not s:
        return ""
    if ":" not in s and "/" in s:
        s = s.replace("/", ":", 1)
    if ":" not in s:
        return s
    t, tid = s.split(":", 1)
    t_norm = _TARGET_TYPE_CANON.get(t.strip().casefold(), t.strip())
    return f"{t_norm}:{tid.strip()}"


def _canonical_target_type(ttype: str) -> str:
    return _TARGET_TYPE_CANON.get(str(ttype or "").strip().casefold(), str(ttype or "").strip())


def _comment_record(c: dict[str, Any]) -> dict[str, Any] | None:
    cid = str(c.get("ID") or c.get("Id") or "").strip()
    if not cid:
        return None
    ts_ms = c.get("Timestamp") or c.get("Time") or c.get("CreateTime") or c.get("CreatedAt") or c.get("Created") or 0
    try:
        ts_i = int(ts_ms) if isinstance(ts_ms, (int, float)) else 0
    except Exception:
        ts_i = 0
    author_id = plar.best_effort_extract_text(c.get("UserID") or c.get("AuthorID")).strip() or None
    author_nickname = plar.best_effort_extract_text(c.get("Nickname") or c.get("Author")).strip() or None
    text = plar.best_effort_extract_text(c.get("Content") or c.get("Text")).strip()
    if not text:
        return None
    # Best-effort: reply user id is not always present; keep None.
    return {
        "id": cid,
        "ts_ms": ts_i,
        "author_id": author_id,
        "author_nickname": author_nickname,
        "reply_user_id": None,
        "text": text,
    }


def local_get_target_context(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    target_key_in = str(args.get("target_key") or "").strip()
    target_key = _canonical_target_key(target_key_in)
    if not target_key:
        ttype = _canonical_target_type(args.get("target_type"))
        tid = str(args.get("target_id") or "").strip()
        if not ttype or not tid:
            raise ToolError("local_get_target_context: provide target_key or (target_type + target_id)")
        target_key = _canonical_target_key(f"{ttype}:{tid}")
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

    # If not cached yet, best-effort fetch live comments and populate the DB.
    if (not bool(out.get("found"))) and runtime.user is not None:
        ttype, tid = target_key.split(":", 1)
        ttype = _canonical_target_type(ttype)
        tid = tid.strip()
        if ttype and tid:
            try:
                # physicsLab server rejects take > 20 (400 Input.Field.Invalid).
                live = plar.get_comments(runtime.user, target_id=tid, target_type=ttype, take=20, skip=0)
            except Exception as e:
                raise ToolError(f"local_get_target_context: live fetch failed: {type(e).__name__}: {e}") from e
            records: list[dict[str, Any]] = []
            for c in live:
                if not isinstance(c, dict):
                    continue
                rec = _comment_record(c)
                if rec is not None:
                    records.append(rec)
            keep_last = int(getattr(getattr(getattr(runtime, "config", None), "agent", None), "context_db_keep_last_comments", 200) or 200)
            db.upsert_target_comments(
                target_key=target_key,
                target={"type": ttype, "id": tid},
                comments=records,
                keep_last=keep_last,
            )
            out = db.get_target_context(target_key=target_key, take=take)

    # Best-effort: enrich target metadata for Experiment/Discussion so the writer can reference title/author safely.
    if runtime.user is not None:
        try:
            ttype0, tid0 = target_key.split(":", 1)
        except ValueError:
            ttype0, tid0 = "", ""
        ttype0 = _canonical_target_type(ttype0)
        tid0 = tid0.strip()
        if ttype0 in ("Experiment", "Discussion") and tid0:
            try:
                ctx = plar.get_experiment_context(
                    runtime.user,
                    summary_id=tid0,
                    category_value=ttype0,
                    cache_dir=runtime.cache_dir,
                    ttl_sec=600,
                    max_json_chars=2000,
                )
            except Exception:
                ctx = None
            if isinstance(ctx, dict):
                meta: dict[str, Any] = {"type": ttype0, "id": tid0}
                title = ctx.get("title")
                author = ctx.get("author")
                plsav_summary = ctx.get("plsav_summary")
                summary_text = ctx.get("summary_text") or ctx.get("body_text")
                experiment_text = ctx.get("experiment_text") or ctx.get("content_text")
                if isinstance(title, str) and title.strip():
                    meta["title"] = title.strip()
                if isinstance(author, dict):
                    meta["author"] = author
                if isinstance(plsav_summary, dict):
                    meta["plsav_summary"] = plsav_summary
                if isinstance(summary_text, str) and summary_text.strip():
                    st = summary_text.strip()
                    meta["summary_text_excerpt"] = st if len(st) <= 1200 else (st[:1199] + "…")
                if isinstance(experiment_text, str) and experiment_text.strip():
                    et = experiment_text.strip()
                    meta["experiment_text_excerpt"] = et if len(et) <= 1200 else (et[:1199] + "…")
                if len(meta) > 2:
                    try:
                        db.upsert_target_meta(target_key=target_key, target=meta)
                    except Exception:
                        pass
                    tgt = out.get("target")
                    if isinstance(tgt, dict):
                        merged = dict(tgt)
                        merged.update(meta)
                        out["target"] = merged
                    else:
                        out["target"] = meta

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
