from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import plar

from .agent import AurexAgent
from .config import AurexConfig
from .contextdb import ContextDB
from .logutil import truncate
from .replyfmt import prefix_user_mention
from .shutdown import GracefulShutdown


class RunLoopError(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    type: str
    id: str

    @property
    def key(self) -> str:
        return f"{self.type}:{self.id}"


def parse_target(value: str) -> Target:
    s = (value or "").strip()
    if not s or ":" not in s:
        raise RunLoopError("target must be like Experiment:<id> | Discussion:<id> | User:<id>")
    t, tid = s.split(":", 1)
    t = t.strip()
    tid = tid.strip()
    if t not in ("Experiment", "Discussion", "User"):
        raise RunLoopError("target type must be Experiment|Discussion|User")
    if not tid:
        raise RunLoopError("target id is empty")
    return Target(type=t, id=tid)


def normalize_targets(items: Iterable[Any]) -> list[Target]:
    out: list[Target] = []
    seen: set[str] = set()
    for it in items:
        if isinstance(it, Target):
            t = it
        elif isinstance(it, str):
            t = parse_target(it)
        elif isinstance(it, dict):
            ttype = str(it.get("type") or "").strip()
            tid = str(it.get("id") or "").strip()
            if not ttype or not tid:
                continue
            t = Target(type=ttype, id=tid)
        else:
            continue
        if t.key in seen:
            continue
        seen.add(t.key)
        out.append(t)
    return out


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=False, default=str)
    except Exception:
        return str(obj)


def _extract_comment_id(c: dict[str, Any]) -> str:
    for k in ("ID", "Id", "CommentID", "CommentId"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _comment_key(c: dict[str, Any]) -> str:
    cid = _extract_comment_id(c)
    if cid:
        return cid
    try:
        blob = json.dumps(c, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(c).encode("utf-8", errors="replace")
    return "hash:" + hashlib.sha256(blob).hexdigest()


def _extract_comment_text(c: dict[str, Any]) -> str:
    for k in ("Content", "Text", "Body", "Message"):
        v = c.get(k)
        t = plar.best_effort_extract_text(v).strip()
        if t:
            return t
    return ""


def _extract_author(c: dict[str, Any]) -> tuple[str, str]:
    u = c.get("User")
    if isinstance(u, dict):
        uid = plar.best_effort_extract_text(u.get("ID") or u.get("UserID")).strip()
        nick = plar.best_effort_extract_text(u.get("Nickname") or u.get("Name")).strip()
        return uid, nick
    uid = plar.best_effort_extract_text(c.get("UserID") or c.get("AuthorID")).strip()
    nick = plar.best_effort_extract_text(c.get("Nickname") or c.get("Author")).strip()
    return uid, nick


def _extract_timestamp_sec(c: dict[str, Any]) -> float | None:
    # Best-effort: different clients may use different keys. We only need approximate freshness.
    for k in ("Timestamp", "Time", "CreateTime", "CreatedAt", "Created"):
        v = c.get(k)
        if isinstance(v, (int, float)):
            # assume ms if large
            ts = float(v)
            if ts > 10_000_000_000:
                ts /= 1000.0
            return ts
    return None


def _comment_timestamp_ms(c: dict[str, Any]) -> int | None:
    for k in ("Timestamp", "Time", "CreateTime", "CreatedAt", "Created"):
        v = c.get(k)
        if isinstance(v, (int, float)):
            # Keep values as-is (physicsLab uses ms).
            return int(v)
    return None


def _normalize_post_text(text: str) -> str:
    """Normalize reply text for Physics Lab AR posting.

    - Preserve newlines (platform is plain-text; no Markdown rendering)
    - Collapse excessive spaces/tabs within each line
    - Collapse multiple blank lines
    """
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    out: list[str] = []
    blank_run = 0
    for ln in lines:
        norm = " ".join(str(ln).split()).strip()
        if not norm:
            blank_run += 1
            if blank_run <= 1:
                out.append("")
            continue
        blank_run = 0
        out.append(norm)
    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out).strip()


_TRIGGER_CONTEXT_SPLIT_RE = re.compile(r"\n\s*\n", re.MULTILINE)


def _extract_trigger_text(text: str) -> str:
    """Normalize comment text for mention checks."""
    s = (text or "").strip()
    if not s:
        return ""
    # Strip simple XML-ish tags but keep inner text (e.g. "@aurex").
    s = re.sub(r"</?user[^>]*>", " ", s, flags=re.IGNORECASE)
    s = " ".join(s.split()).strip()
    return s


def _has_explicit_mention(*, text: str, mention_tag: str) -> bool:
    mt = (mention_tag or "").strip()
    if not mt:
        return False
    mt_cf = mt.casefold()
    s = _extract_trigger_text(text).casefold()

    # Prefer a robust @mention match (allow optional whitespace after '@' and avoid matching '@aurex2').
    if mt_cf.startswith("@") or mt_cf.startswith("＠"):
        name = mt_cf[1:].strip()
        if not name:
            return False
        pat = r"[@＠]\s*" + re.escape(name) + r"(?![0-9a-z_])"
        return re.search(pat, s) is not None

    # Fallback: substring match for non-@ tags.
    return mt_cf in s


def _extract_reply_user_id(c: dict[str, Any]) -> str:
    for rk in ("ReplyID", "ReplyId", "ReplyUserID", "ReplyUserId"):
        rv = c.get(rk)
        if isinstance(rv, str) and rv.strip():
            return rv.strip()
    return ""


def _context_comment_record(c: dict[str, Any]) -> dict[str, Any] | None:
    cid = _extract_comment_id(c)
    if not cid:
        return None
    ts_ms = _comment_timestamp_ms(c)
    if ts_ms is None:
        ts_ms = 0
    author_id, author_nick = _extract_author(c)
    text = _extract_comment_text(c)
    return {
        "id": cid,
        "ts_ms": int(ts_ms),
        "author_id": author_id or None,
        "author_nickname": author_nick or None,
        "reply_user_id": _extract_reply_user_id(c) or None,
        "text": text,
    }


@dataclass
class RunState:
    schema_version: int = 1
    started_at_sec: float = 0.0
    seen_comment_ids: dict[str, float] = field(default_factory=dict)
    replied_conversations: dict[str, float] = field(default_factory=dict)
    comments_last_seen_ms: dict[str, int] = field(default_factory=dict)
    comments_processed_keys: dict[str, list[str]] = field(default_factory=dict)
    messages_last_seen_ms: dict[str, int] = field(default_factory=dict)
    messages_processed_keys: dict[str, list[str]] = field(default_factory=dict)
    seen_message_keys: dict[str, float] = field(default_factory=dict)


def load_state(path: str) -> RunState:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return RunState(schema_version=1, started_at_sec=0.0)
    except Exception:
        return RunState(schema_version=1, started_at_sec=0.0)

    if not isinstance(raw, dict):
        return RunState(schema_version=1, started_at_sec=0.0)
    started = float(raw.get("started_at_sec") or 0.0)
    seen = raw.get("seen_comment_ids")
    replied = raw.get("replied_conversations")
    c_last = raw.get("comments_last_seen_ms")
    c_proc = raw.get("comments_processed_keys")
    msg_last = raw.get("messages_last_seen_ms")
    msg_proc = raw.get("messages_processed_keys")
    msg_seen = raw.get("seen_message_keys")
    seen2 = {str(k): float(v) for k, v in seen.items()} if isinstance(seen, dict) else {}
    rep2 = {str(k): float(v) for k, v in replied.items()} if isinstance(replied, dict) else {}
    c_last2 = {str(k): int(v) for k, v in c_last.items()} if isinstance(c_last, dict) else {}
    c_proc2: dict[str, list[str]] = {}
    if isinstance(c_proc, dict):
        for k, v in c_proc.items():
            if isinstance(k, str) and isinstance(v, list):
                c_proc2[k] = [str(x) for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    msg_last2 = {str(k): int(v) for k, v in msg_last.items()} if isinstance(msg_last, dict) else {}
    msg_proc2: dict[str, list[str]] = {}
    if isinstance(msg_proc, dict):
        for k, v in msg_proc.items():
            if isinstance(k, str) and isinstance(v, list):
                msg_proc2[k] = [str(x) for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    msg_seen2 = {str(k): float(v) for k, v in msg_seen.items()} if isinstance(msg_seen, dict) else {}
    return RunState(
        schema_version=1,
        started_at_sec=started,
        seen_comment_ids=seen2,
        replied_conversations=rep2,
        comments_last_seen_ms=c_last2,
        comments_processed_keys=c_proc2,
        messages_last_seen_ms=msg_last2,
        messages_processed_keys=msg_proc2,
        seen_message_keys=msg_seen2,
    )


def save_state(state: RunState, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = {
        "schema_version": int(state.schema_version),
        "started_at_sec": float(state.started_at_sec),
        "seen_comment_ids": dict(state.seen_comment_ids),
        "replied_conversations": dict(state.replied_conversations),
        "comments_last_seen_ms": dict(state.comments_last_seen_ms),
        "comments_processed_keys": dict(state.comments_processed_keys),
        "messages_last_seen_ms": dict(state.messages_last_seen_ms),
        "messages_processed_keys": dict(state.messages_processed_keys),
        "seen_message_keys": dict(state.seen_message_keys),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def default_state_path(config_path: str) -> str:
    base = os.path.abspath(config_path)
    d = os.path.dirname(base)
    stem = os.path.splitext(os.path.basename(base))[0]
    return os.path.join(d, stem + ".state.json")


def _prune_map(m: dict[str, float], *, max_items: int) -> None:
    if max_items <= 0:
        return
    if len(m) <= max_items:
        return
    items = sorted(m.items(), key=lambda kv: kv[1])
    drop = len(items) - max_items
    for k, _v in items[:drop]:
        m.pop(k, None)


def _prune_list_map(m: dict[str, list[str]], *, keep_last: int) -> None:
    if keep_last <= 0:
        return
    for k, v in list(m.items()):
        if not isinstance(v, list):
            m.pop(k, None)
            continue
        if len(v) > keep_last:
            m[k] = [str(x) for x in v[-keep_last:] if str(x).strip()]


def _message_timestamp_ms(message: dict[str, Any]) -> int | None:
    ts = message.get("Timestamp")
    if isinstance(ts, (int, float)):
        t = int(ts)
        # physicsLab uses ms; keep values as-is (v1 semantics).
        return t
    return None


def _message_key(message: dict[str, Any]) -> str:
    for k in ("ID", "Id", "MessageID", "MessageId"):
        v = message.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    try:
        blob = json.dumps(message, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(message).encode("utf-8", errors="replace")
    return "hash:" + hashlib.sha256(blob).hexdigest()


_HEX24_RE = re.compile(r"\b[0-9a-fA-F]{24}\b")
_TYPE_ID_RE_1 = re.compile(r"\b(Experiment|Discussion|User)\b.*?\b([0-9a-fA-F]{24})\b")
_TYPE_ID_RE_2 = re.compile(r"\b([0-9a-fA-F]{24})\b.*?\b(Experiment|Discussion|User)\b")


def _extract_target_from_notification(obj: Any) -> Target | None:
    allowed = {"Experiment", "Discussion", "User"}

    def scan_urlish(s: str) -> Target | None:
        m = _TYPE_ID_RE_1.search(s)
        if m:
            return Target(type=m.group(1), id=m.group(2))
        m = _TYPE_ID_RE_2.search(s)
        if m:
            return Target(type=m.group(2), id=m.group(1))
        return None

    stack: list[Any] = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            tt: str | None = None
            tid: str | None = None

            category_hint: str | None = None
            content_id_hint: str | None = None

            for k, v in cur.items():
                if isinstance(v, str):
                    t0 = scan_urlish(v.strip())
                    if t0 is not None:
                        return t0

                if isinstance(k, str):
                    lk = k.casefold()
                    if lk in ("targettype", "target_type") and isinstance(v, str) and v.strip() in allowed:
                        tt = v.strip()
                    if lk in ("targetid", "target_id") and isinstance(v, str) and v.strip():
                        tid = v.strip()
                    if lk in ("category", "categoryvalue") and isinstance(v, str) and v.strip() in ("Experiment", "Discussion"):
                        category_hint = v.strip()
                    if lk in ("summaryid", "summary_id", "contentid", "content_id") and isinstance(v, str) and v.strip():
                        content_id_hint = v.strip()

                    if lk.endswith("experimentid") and isinstance(v, str) and v.strip():
                        return Target(type="Experiment", id=v.strip())
                    if lk.endswith("discussionid") and isinstance(v, str) and v.strip():
                        return Target(type="Discussion", id=v.strip())
                    if lk.endswith("userid") and isinstance(v, str) and v.strip():
                        return Target(type="User", id=v.strip())

                if isinstance(v, (dict, list)):
                    stack.append(v)

            if tt and tid:
                return Target(type=tt, id=tid)
            if category_hint and content_id_hint:
                return Target(type=category_hint, id=content_id_hint)
            continue

        if isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append(v)
            continue

    # Broad fallback: scan all strings for "Type"+"24-hex".
    stack2: list[Any] = [obj]
    while stack2:
        cur = stack2.pop()
        if isinstance(cur, str):
            t0 = scan_urlish(cur.strip())
            if t0 is not None:
                return t0
            continue
        if isinstance(cur, dict):
            for v in cur.values():
                if isinstance(v, (dict, list, str)):
                    stack2.append(v)
            continue
        if isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list, str)):
                    stack2.append(v)
            continue

    return None


def _fallback_targets_from_notification(obj: Any) -> list[Target]:
    hex_ids: set[str] = set()
    user_ids: set[str] = set()
    hints: set[str] = set()

    stack: list[Any] = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            s = cur.strip()
            if not s:
                continue
            low = s.casefold()
            if "experiment" in low:
                hints.add("Experiment")
            if "discussion" in low:
                hints.add("Discussion")
            if "user" in low:
                hints.add("User")
            for mid in _HEX24_RE.findall(s):
                hex_ids.add(mid)
            continue

        if isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list, str)):
                    stack.append(v)
            continue

        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(k, str):
                    lk = k.casefold()
                    if lk == "users" and isinstance(v, list):
                        for uid in v:
                            if isinstance(uid, str) and uid.strip():
                                user_ids.add(uid.strip())
                    if lk.endswith("userid") or lk.endswith("_user_id") or lk in ("user_id", "userid"):
                        if isinstance(v, str) and v.strip():
                            user_ids.add(v.strip())
                if isinstance(v, (dict, list, str)):
                    stack.append(v)
            continue

    candidates = [x for x in hex_ids if x not in user_ids]
    if len(candidates) != 1:
        return []
    cid = candidates[0]

    if "Experiment" in hints and "Discussion" not in hints:
        return [Target(type="Experiment", id=cid)]
    if "Discussion" in hints and "Experiment" not in hints:
        return [Target(type="Discussion", id=cid)]
    return [Target(type="Experiment", id=cid), Target(type="Discussion", id=cid)]


def _discover_targets_from_notifications(
    *,
    user: Any,
    cfg: AurexConfig,
    state: RunState,
    logger: Any | None = None,
) -> list[Target]:
    if not bool(getattr(cfg.agent, "notifications_enabled", True)):
        return []

    if logger is None:
        import logging

        logger = logging.getLogger("aurex2")

    take = int(getattr(cfg.agent, "notification_take", 20) or 20)
    if take <= 0:
        take = 20
    if take > 100:
        take = 100

    cat_ids = list(getattr(cfg.agent, "notification_category_ids", [0, 3]) or [0, 3])
    out: list[Target] = []
    now_ms = int(time.time() * 1000)

    for cat in cat_ids:
        cat_key = str(int(cat))
        lookback_sec = int(getattr(cfg.agent, "bootstrap_lookback_sec", 0) or 0)
        back_ms = int(lookback_sec) * 1000
        if back_ms <= 0:
            # Small slack to avoid missing messages due to client/server clock skew.
            back_ms = 2000
        if cat_key not in state.messages_last_seen_ms or int(state.messages_last_seen_ms.get(cat_key) or 0) <= 0:
            state.messages_last_seen_ms[cat_key] = max(0, now_ms - back_ms)
        if cat_key not in state.messages_processed_keys:
            state.messages_processed_keys[cat_key] = []

        last_seen = int(state.messages_last_seen_ms.get(cat_key) or 0)
        processed_list = state.messages_processed_keys.get(cat_key) or []
        processed = set(processed_list)
        try:
            messages, templates = plar.get_messages(user, category_id=int(cat), skip=0, take=take, no_templates=False)
        except Exception:
            continue
        logger.debug(
            "[Messages:%s] fetched=%d templates=%d last_seen=%d processed=%d",
            cat_key,
            len(messages),
            len(templates),
            last_seen,
            len(processed_list),
        )

        tmpl_by_id: dict[str, dict[str, Any]] = {}
        for t in templates:
            if not isinstance(t, dict):
                continue
            tid = t.get("ID") or t.get("Id")
            if isinstance(tid, (str, int)):
                tmpl_by_id[str(tid)] = t

        # v1-style scan: timestamp watermark + per-category processed keys.
        new_msgs: list[dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            ts = _message_timestamp_ms(m)
            if ts is None:
                continue
            if int(ts) >= last_seen:
                new_msgs.append(m)
        new_msgs.sort(key=lambda m: _message_timestamp_ms(m) or 0)

        for m in new_msgs:
            ts = int(_message_timestamp_ms(m) or 0)
            key = _message_key(m)
            if key in processed:
                last_seen = max(last_seen, ts)
                continue

            tmpl_id = m.get("TemplateID") or m.get("TemplateId")
            tmpl = tmpl_by_id.get(str(tmpl_id)) if isinstance(tmpl_id, (str, int)) else None
            combined: dict[str, Any] = {"Message": m}
            if tmpl is not None:
                combined["Template"] = tmpl

            t0 = _extract_target_from_notification(combined)
            if t0 is not None:
                out.append(t0)
                logger.debug("[Messages:%s] new message key=%s ts=%d target=%s", cat_key, key, ts, t0.key)
                # Bootstrap comment watermark for newly discovered targets so we don't miss the triggering comment.
                if ts > 0:
                    slack_ms = max(30_000, int(float(getattr(cfg.agent, "poll_interval_sec", 15.0) or 15.0) * 2.0 * 1000.0) + 5000)
                    cur = int(state.comments_last_seen_ms.get(t0.key) or 0)
                    if cur <= 0 or cur > ts:
                        state.comments_last_seen_ms[t0.key] = max(0, ts - slack_ms)
            else:
                cand = _fallback_targets_from_notification(combined)
                out.extend(cand)
                logger.debug(
                    "[Messages:%s] new message key=%s ts=%d target=%s",
                    cat_key,
                    key,
                    ts,
                    [t.key for t in cand] if cand else None,
                )
                if ts > 0:
                    slack_ms = max(30_000, int(float(getattr(cfg.agent, "poll_interval_sec", 15.0) or 15.0) * 2.0 * 1000.0) + 5000)
                    for t1 in cand:
                        cur = int(state.comments_last_seen_ms.get(t1.key) or 0)
                        if cur <= 0 or cur > ts:
                            state.comments_last_seen_ms[t1.key] = max(0, ts - slack_ms)

            processed_list.append(key)
            processed.add(key)
            last_seen = max(last_seen, ts)

        state.messages_last_seen_ms[cat_key] = int(last_seen)
        state.messages_processed_keys[cat_key] = processed_list
        _prune_list_map(state.messages_processed_keys, keep_last=500)

    return normalize_targets(out)


def run_forever(
    *,
    cfg: AurexConfig,
    config_path: str,
    agent: AurexAgent,
    user: Any,
    targets: list[Target],
    state_path: str,
    once: bool,
    dry_run: bool | None = None,
    logger: Any | None = None,
) -> None:
    if logger is None:
        import logging

        logger = logging.getLogger("aurex2")
    notifications_enabled = bool(getattr(cfg.agent, "notifications_enabled", True))
    if not targets and not notifications_enabled:
        raise RunLoopError("No targets configured. Add agent.targets in config or pass --target.")

    take = int(cfg.agent.comment_take)
    if take <= 0:
        take = 20
    # physicsLab server rejects take > 20 (400 Input.Field.Invalid).
    if take > 20:
        take = 20

    pages = int(cfg.agent.comment_scan_pages)
    if pages <= 0:
        pages = 1
    if pages > 20:
        pages = 20

    poll_sec = float(cfg.agent.poll_interval_sec)
    if poll_sec <= 0:
        poll_sec = 5.0

    require_mention = bool(cfg.agent.require_mention)
    mention_tag = (cfg.agent.mention_tag or "").strip()
    user_targets_require_mention = bool(cfg.agent.user_targets_require_mention)
    lookback = int(cfg.agent.bootstrap_lookback_sec)
    if lookback < 0:
        lookback = 0

    state = load_state(state_path)
    now = time.time()
    first_run = state.started_at_sec <= 0.0
    if first_run:
        state.started_at_sec = now

    self_id = str(getattr(user, "user_id", "") or getattr(user, "id", "") or "").strip()
    logger.info("self_user_id=%s", self_id or "<unknown>")

    if dry_run is None:
        dry_run = bool(cfg.agent.dry_run)

    cache_dir = cfg.resolve_path(cfg.storage.cache_dir, config_path=config_path)
    os.makedirs(cache_dir, exist_ok=True)
    context_db_enabled = bool(getattr(cfg.agent, "context_db_enabled", True))
    context_db_keep_last = int(getattr(cfg.agent, "context_db_keep_last_comments", 200) or 200)
    context_db_path_cfg = str(getattr(cfg.storage, "context_db_path", "") or "").strip()
    if context_db_path_cfg:
        context_db_path = cfg.resolve_path(context_db_path_cfg, config_path=config_path)
    else:
        context_db_path = os.path.join(cache_dir, "context_db.json")
    context_db = ContextDB(path=context_db_path) if context_db_enabled else None

    targets_by_key: dict[str, Target] = {t.key: t for t in targets}
    notification_only_mode = notifications_enabled and not bool(targets_by_key)

    logger.info(
        "runloop start (once=%s dry_run=%s poll=%.2fs notifications=%s targets=%s state=%s)",
        once,
        dry_run,
        poll_sec,
        notifications_enabled,
        [t.key for t in targets_by_key.values()],
        os.path.abspath(state_path),
    )

    shutdown = GracefulShutdown(logger=logger)
    shutdown.install()
    try:
        def _sleep_with_shutdown(seconds: float) -> None:
            end_at = time.time() + float(max(0.0, seconds))
            while time.time() < end_at:
                if shutdown.stop_requested:
                    return
                time.sleep(min(0.5, end_at - time.time()))

        while True:
            if shutdown.stop_requested:
                logger.info("shutdown: stop requested; exiting main loop")
                break

            cycle_now = time.time()
            enqueued = 0

            # Discover targets from notifications (if enabled).
            discovered: list[Target] = []
            if not shutdown.stop_requested:
                discovered = _discover_targets_from_notifications(user=user, cfg=cfg, state=state, logger=logger)
            if discovered:
                logger.info("notifications discovered %d target(s): %s", len(discovered), [t.key for t in discovered])

            if notification_only_mode:
                cycle_targets_by_key: dict[str, Target] = {t.key: t for t in discovered}
            else:
                for t in discovered:
                    if t.key not in targets_by_key:
                        targets_by_key[t.key] = t
                        logger.info("added target from notifications: %s", t.key)
                cycle_targets_by_key = targets_by_key

            if not cycle_targets_by_key:
                try:
                    save_state(state, state_path)
                except Exception:
                    pass
                first_run = False
                if once:
                    return
                _sleep_with_shutdown(poll_sec)
                continue

            for tgt in list(cycle_targets_by_key.values()):
                if shutdown.stop_requested:
                    logger.info("shutdown: stop requested; skipping remaining targets")
                    break

                back_ms = int(float(lookback) * 1000.0)
                if back_ms <= 0:
                    back_ms = 2000
                if tgt.key not in state.comments_last_seen_ms or int(state.comments_last_seen_ms.get(tgt.key) or 0) <= 0:
                    state.comments_last_seen_ms[tgt.key] = max(0, int(cycle_now * 1000) - back_ms)
                if tgt.key not in state.comments_processed_keys:
                    state.comments_processed_keys[tgt.key] = []

                last_seen_ms = int(state.comments_last_seen_ms.get(tgt.key) or 0)
                processed_list = state.comments_processed_keys.get(tgt.key) or []
                processed = set(processed_list)

                try:
                    all_comments: list[dict[str, Any]] = []
                    # physicsLab.web.User.get_comments uses `skip` as unix_ms timestamp (not an offset).
                    # Pagination strategy: keep fetching older pages by passing the minimum timestamp
                    # from the previous page (minus 1ms to guarantee progress).
                    skip_ts_ms = 0
                    for page in range(pages):
                        chunk = plar.get_comments(user, target_id=tgt.id, target_type=tgt.type, take=take, skip=skip_ts_ms)
                        all_comments.extend(chunk)
                        if len(chunk) < take:
                            break
                        if last_seen_ms > 0:
                            min_ts_ms = None
                            for c in chunk:
                                if not isinstance(c, dict):
                                    continue
                                ts_ms = _comment_timestamp_ms(c)
                                if ts_ms is None:
                                    continue
                                min_ts_ms = ts_ms if min_ts_ms is None else min(min_ts_ms, ts_ms)
                            if not isinstance(min_ts_ms, int):
                                break
                            if isinstance(min_ts_ms, int) and min_ts_ms <= last_seen_ms:
                                break
                            if isinstance(min_ts_ms, int) and min_ts_ms > 0:
                                skip_ts_ms = max(0, int(min_ts_ms) - 1)
                        else:
                            # Even without last_seen_ms, paginate a limited number of pages.
                            min_ts_ms = None
                            for c in chunk:
                                if not isinstance(c, dict):
                                    continue
                                ts_ms = _comment_timestamp_ms(c)
                                if ts_ms is None:
                                    continue
                                min_ts_ms = ts_ms if min_ts_ms is None else min(min_ts_ms, ts_ms)
                            if not isinstance(min_ts_ms, int):
                                break
                            if isinstance(min_ts_ms, int) and min_ts_ms > 0:
                                skip_ts_ms = max(0, int(min_ts_ms) - 1)
                    logger.info(
                        "[%s] fetched comments=%d (take=%d pages=%d)",
                        tgt.key,
                        len(all_comments),
                        take,
                        pages,
                    )
                    if context_db is not None:
                        records: list[dict[str, Any]] = []
                        for c in all_comments:
                            if not isinstance(c, dict):
                                continue
                            rec = _context_comment_record(c)
                            if rec is not None:
                                records.append(rec)
                        if records:
                            context_db.upsert_target_comments(
                                target_key=tgt.key,
                                target={"type": tgt.type, "id": tgt.id},
                                comments=records,
                                keep_last=context_db_keep_last,
                            )
                except Exception as e:
                    logger.warning("[%s] get_comments failed: %s", tgt.key, e)
                    continue

                candidates: list[dict[str, Any]] = []
                old_skipped = 0
                for c in all_comments:
                    if not isinstance(c, dict):
                        continue
                    ts_ms = _comment_timestamp_ms(c)
                    if not isinstance(ts_ms, int):
                        continue
                    if ts_ms >= last_seen_ms:
                        candidates.append(c)
                    else:
                        old_skipped += 1

                candidates.sort(key=lambda c: _comment_timestamp_ms(c) or 0)
                logger.debug(
                    "[%s] comment scan window: fetched=%d old_skipped=%d candidates=%d last_seen_ms=%d processed=%d",
                    tgt.key,
                    len(all_comments),
                    old_skipped,
                    len(candidates),
                    last_seen_ms,
                    len(processed_list),
                )

                for c in candidates:
                    if shutdown.stop_requested:
                        logger.info("shutdown: stop requested; skipping remaining comments for %s", tgt.key)
                        break

                    cid = _extract_comment_id(c)
                    if cid and cid in state.seen_comment_ids:
                        continue

                    ts = _extract_timestamp_sec(c)
                    if first_run and lookback > 0 and ts is not None and ts < (cycle_now - float(lookback)):
                        if cid:
                            state.seen_comment_ids[cid] = cycle_now
                        logger.debug("[%s] skip old comment=%s ts=%s (bootstrap lookback)", tgt.key, cid or "?", ts)
                        continue

                    ts_ms = int(_comment_timestamp_ms(c) or 0)
                    key = _comment_key(c)
                    if key in processed:
                        last_seen_ms = max(last_seen_ms, ts_ms)
                        continue

                    author_id, author_nick = _extract_author(c)
                    if self_id and author_id and author_id == self_id:
                        if cid:
                            state.seen_comment_ids[cid] = cycle_now
                        logger.debug("[%s] skip self comment=%s", tgt.key, cid or "?")
                        processed_list.append(key)
                        processed.add(key)
                        last_seen_ms = max(last_seen_ms, ts_ms)
                        continue

                    text = _extract_comment_text(c)
                    if not text:
                        if cid:
                            state.seen_comment_ids[cid] = cycle_now
                        logger.debug("[%s] skip empty comment=%s", tgt.key, cid or "?")
                        processed_list.append(key)
                        processed.add(key)
                        last_seen_ms = max(last_seen_ms, ts_ms)
                        continue

                    is_user_target = tgt.type == "User"
                    reply_user_id = _extract_reply_user_id(c)
                    reply_to_self = bool(self_id and reply_user_id and reply_user_id == self_id)

                    mention_required = require_mention and (user_targets_require_mention or not is_user_target)
                    if bool(getattr(cfg.agent, "trigger_on_reply_to_self", False)) and reply_to_self:
                        mention_required = False
                    if (
                        bool(getattr(cfg.agent, "trigger_on_own_wall_without_mention", False))
                        and is_user_target
                        and self_id
                        and tgt.id == self_id
                    ):
                        mention_required = False

                    if mention_required and mention_tag and not _has_explicit_mention(text=text, mention_tag=mention_tag):
                        if cid:
                            state.seen_comment_ids[cid] = cycle_now
                        logger.debug("[%s] skip no-explicit-mention comment=%s", tgt.key, cid or "?")
                        processed_list.append(key)
                        processed.add(key)
                        last_seen_ms = max(last_seen_ms, ts_ms)
                        continue

                    logger.info(
                        "[%s] received comment=%s author=%s text=%r",
                        tgt.key,
                        cid or "?",
                        author_nick or author_id or "unknown",
                        truncate(text, max_chars=220),
                    )

                    ctx = {
                        "target": {"type": tgt.type, "id": tgt.id},
                        "comment": {"id": cid, "author_id": author_id or None, "author_nickname": author_nick or None},
                    }
                    user_text = "CONTEXT_JSON:\n" + _safe_json(ctx) + "\n\n" + text
                    try:
                        out = agent.handle(user_text=user_text, user=user)
                        reply = str(out.get("answer") or "").strip()
                    except Exception as e:
                        logger.error("[%s] agent failed for comment=%s: %s", tgt.key, cid, e)
                        if cid:
                            state.seen_comment_ids[cid] = cycle_now
                        processed_list.append(key)
                        processed.add(key)
                        last_seen_ms = max(last_seen_ms, ts_ms)
                        continue

                    if not reply:
                        if cid:
                            state.seen_comment_ids[cid] = cycle_now
                        logger.warning("[%s] empty reply for comment=%s", tgt.key, cid)
                        processed_list.append(key)
                        processed.add(key)
                        last_seen_ms = max(last_seen_ms, ts_ms)
                        continue

                    if bool(getattr(cfg.agent, "force_reply_prefix", True)):
                        reply = prefix_user_mention(reply, nickname=author_nick)

                    if bool(getattr(cfg.agent, "strip_mention_tag_in_replies", True)):
                        mt = (cfg.agent.mention_tag or "").strip()
                        if mt:
                            # Keep the subject while preventing self-trigger loops:
                            # replace "@aurex"/"＠aurex" with "aurex" instead of removing it.
                            replacement = mt[1:] if mt.startswith("@") and len(mt) > 1 else "aurex"
                            reply = reply.replace(mt, replacement).replace(mt.replace("@", "＠"), replacement)

                    # Keep newlines for readability (Physics Lab AR comments are plain text).
                    reply = _normalize_post_text(reply)

                    if dry_run:
                        logger.info(
                            "[%s] DRY-RUN reply_to=%s author=%s reply=%r",
                            tgt.key,
                            cid,
                            author_nick or author_id,
                            truncate(reply, max_chars=400),
                        )
                    else:
                        try:
                            plar.post_comment(
                                user,
                                target_id=tgt.id,
                                target_type=tgt.type,
                                content=reply,
                                reply_id=cid,
                            )
                            logger.info("[%s] replied comment=%s author=%s", tgt.key, cid, author_nick or author_id)
                        except Exception as e:
                            logger.error("[%s] post_comment failed (comment=%s): %s", tgt.key, cid, e)
                            if cid:
                                state.seen_comment_ids[cid] = cycle_now
                            processed_list.append(key)
                            processed.add(key)
                            last_seen_ms = max(last_seen_ms, ts_ms)
                            continue

                    if cid:
                        state.seen_comment_ids[cid] = cycle_now
                    enqueued += 1

                    processed_list.append(key)
                    processed.add(key)
                    last_seen_ms = max(last_seen_ms, ts_ms)

                state.comments_last_seen_ms[tgt.key] = int(last_seen_ms)
                state.comments_processed_keys[tgt.key] = processed_list
                _prune_list_map(state.comments_processed_keys, keep_last=2000)

            _prune_map(state.seen_comment_ids, max_items=50_000)
            if len(state.comments_last_seen_ms) > 50_000:
                items = sorted(state.comments_last_seen_ms.items(), key=lambda kv: kv[1])
                drop = len(items) - 50_000
                for k, _v in items[:drop]:
                    state.comments_last_seen_ms.pop(k, None)
            _prune_map(state.seen_message_keys, max_items=50_000)
            try:
                save_state(state, state_path)
            except Exception as e:
                logger.warning("[state] save failed: %s", e)

            first_run = False
            if once:
                return

            sleep_total = poll_sec if enqueued == 0 else min(2.0, poll_sec)
            _sleep_with_shutdown(sleep_total)
    finally:
        shutdown.restore()
        try:
            save_state(state, state_path)
        except Exception:
            pass
