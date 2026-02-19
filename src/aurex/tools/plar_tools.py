from __future__ import annotations

import re
from typing import Any

import plar

from .registry import ToolError, ToolRuntime


_HEX24_RE = re.compile(r"[0-9a-fA-F]{24}")
_TAG_SPLIT_RE = re.compile(r"[,\n;，；]+")

_TAG_NAME_TO_VALUE: dict[str, str] | None = None


def _extract_hex24(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    m = _HEX24_RE.search(s)
    return m.group(0) if m else ""


def _require_hex24(value: Any, *, where: str) -> str:
    s = str(value or "").strip()
    out = _extract_hex24(s)
    if out:
        return out
    if s and re.fullmatch(r"[0-9a-fA-F]+", s) and len(s) != 24:
        raise ToolError(f"{where} looks truncated (expected 24-hex), got {s!r}")
    raise ToolError(f"{where} must include a 24-hex id, got {s!r}")


def _load_tag_name_to_value() -> dict[str, str]:
    """Load physicsLab.Tag enum mapping: name(casefold) -> value.

    This covers common built-in tags like Featured(精选), Circuit(Type-0), NoRemixes(禁止改编), etc.
    If physicsLab is unavailable, return an empty mapping.
    """
    global _TAG_NAME_TO_VALUE
    if _TAG_NAME_TO_VALUE is not None:
        return _TAG_NAME_TO_VALUE
    # Minimal built-in mapping for the most common tag names, so tools/tests work even
    # when `physicsLab` isn't importable (e.g. in CI or offline environments).
    mapping: dict[str, str] = {
        "featured": "精选",
    }
    try:
        from physicsLab import Tag as PLTag  # type: ignore

        for t in PLTag:
            name = getattr(t, "name", None)
            val = getattr(t, "value", None)
            if isinstance(name, str) and name.strip() and isinstance(val, str) and val.strip():
                mapping[name.strip().casefold()] = val.strip()
    except Exception:
        pass
    _TAG_NAME_TO_VALUE = mapping
    return mapping


def _normalize_tag_list(value: Any) -> list[str] | None:
    """Normalize tag filters from LLM/user args.

    Accepts: null | string | list. Converts Tag.<Name> / <Name> to the Tag enum value when available.
    Unknown/custom tags are preserved as-is.
    """
    if value is None:
        return None

    raw_items: list[str] = []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        raw_items = [x.strip() for x in _TAG_SPLIT_RE.split(s) if x.strip()] or [s]
    elif isinstance(value, list):
        for x in value:
            if x is None:
                continue
            if isinstance(x, str) and x.strip():
                raw_items.append(x.strip())
            else:
                sx = str(x).strip()
                if sx:
                    raw_items.append(sx)
    else:
        s = str(value).strip()
        if not s:
            return None
        raw_items = [s]

    name_to_value = _load_tag_name_to_value()
    out: list[str] = []
    seen: set[str] = set()
    for it in raw_items:
        s = (it or "").strip()
        if not s:
            continue
        if s.startswith("Tag.") and len(s) > 4:
            s = s[4:].strip()
        mapped = name_to_value.get(s.casefold())
        tag = mapped or s
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out or None


def _require_user(runtime: ToolRuntime) -> Any:
    if runtime.user is None:
        raise ToolError("plar tool requires a logged-in PhysicsLab user in runtime.user")
    return runtime.user


def plar_query_experiments(runtime: ToolRuntime, args: dict[str, Any]) -> list[dict[str, Any]]:
    user = _require_user(runtime)
    category = args.get("category") or "Experiment"
    user_id_raw = args.get("user_id")
    user_id_s: str | None = None
    if isinstance(user_id_raw, str) and user_id_raw.strip():
        user_id_s = _require_hex24(user_id_raw, where="plar_query_experiments.user_id")
    tags = _normalize_tag_list(args.get("tags"))
    exclude_tags = _normalize_tag_list(args.get("exclude_tags"))
    items = plar.query_experiments(
        user,
        category=category,
        take=int(args.get("take") or 20),
        skip=int(args.get("skip") or 0),
        from_skip=args.get("from_skip"),
        days=args.get("days"),
        sort=args.get("sort"),
        user_id=user_id_s,
        tags=tags,
        exclude_tags=exclude_tags,
        languages=args.get("languages"),
        exclude_languages=args.get("exclude_languages"),
    )
    cat_hint = category if category in ("Experiment", "Discussion") else None
    out: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(_compact_qe_item(it, category_hint=cat_hint))
    return out


def plar_get_user(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    user = _require_user(runtime)
    name = args.get("name")
    user_id = args.get("user_id")
    if isinstance(name, str) and name.strip():
        pkg = plar.get_user_by_name(user, name=name)
        return _compact_user_pkg(pkg)
    if isinstance(user_id, str) and user_id.strip():
        uid = _require_hex24(user_id, where="plar_get_user.user_id")
        pkg = plar.get_user_by_id(user, user_id=uid)
        return _compact_user_pkg(pkg)
    raise ToolError("plar_get_user requires either name or user_id")


def plar_get_comments(runtime: ToolRuntime, args: dict[str, Any]) -> list[dict[str, Any]]:
    user = _require_user(runtime)
    target_type = str(args.get("target_type") or "").strip()
    if target_type.casefold() in ("user", "experiment", "discussion"):
        target_type = target_type[:1].upper() + target_type[1:].casefold()
    if target_type not in ("User", "Experiment", "Discussion"):
        raise ToolError("plar_get_comments: target_type must be User|Experiment|Discussion")
    target_id = _require_hex24(args.get("target_id"), where="plar_get_comments.target_id")
    take = int(args.get("take") or 20)
    if take <= 0:
        take = 20
    # physicsLab server rejects take > 20 (400 Input.Field.Invalid).
    if take > 20:
        take = 20
    skip = int(args.get("skip") or 0)
    if skip < 0:
        skip = 0
    raw = plar.get_comments(user, target_id=target_id, target_type=target_type, take=take, skip=skip)
    out: list[dict[str, Any]] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        rec = _compact_comment(c)
        if rec is not None:
            out.append(rec)
    return out


def plar_get_oldest_comment(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Scan the whole comment section (best-effort) and return the oldest comment.

    Notes:
    - PhysicsLab get_comments(skip=...) uses unix_ms timestamp pagination (not offset).
    - We page backwards by taking the minimum timestamp on each page and setting
      next_skip_ts_ms = min_ts_ms - 1.
    """
    user = _require_user(runtime)
    target_type = str(args.get("target_type") or "").strip()
    if target_type.casefold() in ("user", "experiment", "discussion"):
        target_type = target_type[:1].upper() + target_type[1:].casefold()
    if target_type not in ("User", "Experiment", "Discussion"):
        raise ToolError("plar_get_oldest_comment: target_type must be User|Experiment|Discussion")
    target_id = _require_hex24(args.get("target_id"), where="plar_get_oldest_comment.target_id")

    take = int(args.get("take") or 50)
    if take <= 0:
        take = 20
    # physicsLab server rejects take > 20 (400 Input.Field.Invalid).
    if take > 20:
        take = 20

    max_pages = int(args.get("max_pages") or 200)
    if max_pages < 1:
        max_pages = 1
    if max_pages > 800:
        max_pages = 800

    # Optional starting skip (unix_ms). If an offset-like small number is provided, ignore it.
    skip_in = args.get("skip") if "skip" in args else 0
    try:
        skip_ts_ms = int(skip_in) if skip_in is not None else 0
    except Exception:
        skip_ts_ms = 0
    if 0 < skip_ts_ms < 10_000_000_000:
        skip_ts_ms = 0
    if skip_ts_ms < 0:
        skip_ts_ms = 0

    pages = 0
    scanned = 0
    oldest: dict[str, Any] | None = None
    oldest_ts: int | None = None
    prev_page_min_ts: int | None = None
    seen_ids: set[str] = set()
    stopped_by_limit = False

    while pages < max_pages:
        raw = plar.get_comments(user, target_id=target_id, target_type=target_type, take=take, skip=int(skip_ts_ms))
        pages += 1
        if not raw:
            break

        page_min_ts: int | None = None
        any_new = False
        for c in raw:
            if not isinstance(c, dict):
                continue
            rec = _compact_comment(c)
            if rec is None:
                continue
            cid = str(rec.get("id") or "").strip()
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            any_new = True
            scanned += 1
            try:
                ts = int(rec.get("ts_ms") or 0)
            except Exception:
                ts = 0
            if page_min_ts is None or ts < page_min_ts:
                page_min_ts = ts
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
                oldest = rec

        if page_min_ts is None:
            break
        # If the page is short, we likely reached the end.
        if len(raw) < take:
            break
        # If we didn't make progress, avoid infinite loops.
        if prev_page_min_ts is not None and page_min_ts >= prev_page_min_ts:
            break
        if not any_new and prev_page_min_ts is not None:
            break

        prev_page_min_ts = page_min_ts
        skip_ts_ms = max(0, int(page_min_ts) - 1)

    if pages >= max_pages:
        stopped_by_limit = True

    return {
        "target": {"type": target_type, "id": target_id},
        "pages_scanned": pages,
        "comments_scanned": scanned,
        "incomplete": bool(stopped_by_limit),
        # Keep a "comments" list for compatibility with generic comment-based post-processing.
        "comments": [oldest] if isinstance(oldest, dict) else [],
        "oldest_comment": oldest,
    }


def _compact_qe_item(item: dict[str, Any], *, category_hint: str | None = None) -> dict[str, Any]:
    def _get_str(*keys: str) -> str:
        for k in keys:
            v = item.get(k)
            s = plar.best_effort_extract_text(v).strip()
            if s:
                return s
        return ""

    def _get_int(key: str) -> int | None:
        v = item.get(key)
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            try:
                return int(v.strip(), 10)
            except Exception:
                return None
        return None

    subject = _get_str("Subject", "Title", "Name")
    desc = plar.best_effort_extract_text(item.get("Description")).strip()
    if isinstance(item.get("Description"), list):
        desc = "\n".join([x for x in item.get("Description") if isinstance(x, str)]).strip()
    if len(desc) > 240:
        desc = desc[:239] + "…"

    user_obj = item.get("User")
    user_id = ""
    user_nick = ""
    if isinstance(user_obj, dict):
        user_id = plar.best_effort_extract_text(user_obj.get("ID") or user_obj.get("UserID")).strip()
        user_nick = plar.best_effort_extract_text(user_obj.get("Nickname") or user_obj.get("Name")).strip()
    if not user_id:
        user_id = _get_str("UserID")
    cat = _get_str("Category") or (category_hint or "")

    tags = item.get("Tags")
    tags_list = [str(x) for x in tags if isinstance(x, (str, int, float))] if isinstance(tags, list) else []

    return {
        "id": _get_str("ID", "Id"),
        "category": cat,
        "subject": subject,
        "description": desc,
        "user_id": user_id or None,
        "user_nickname": user_nick or None,
        "creation_date": _get_int("CreationDate"),
        "update_date": _get_int("UpdateDate"),
        "sorting_date": _get_int("SortingDate"),
        "popularity": _get_int("Popularity"),
        "stars": _get_int("Stars"),
        "supports": _get_int("Supports"),
        "visits": _get_int("Visits"),
        "tags": tags_list,
    }


def _compact_comment(c: dict[str, Any]) -> dict[str, Any] | None:
    cid = plar.best_effort_extract_text(c.get("ID") or c.get("Id")).strip()
    if not cid:
        return None
    ts_ms = c.get("Timestamp") or c.get("Time") or c.get("CreateTime") or c.get("CreatedAt") or c.get("Created") or 0
    try:
        ts_i = int(ts_ms) if isinstance(ts_ms, (int, float, str)) and str(ts_ms).strip() else 0
    except Exception:
        ts_i = 0
    author_id = plar.best_effort_extract_text(c.get("UserID") or c.get("AuthorID")).strip() or None
    author_nickname = plar.best_effort_extract_text(c.get("Nickname") or c.get("Author")).strip() or None
    text = plar.best_effort_extract_text(c.get("Content") or c.get("Text")).strip()
    if not text:
        return None
    if len(text) > 500:
        text = text[:499] + "…"
    return {
        "id": cid,
        "ts_ms": ts_i,
        "author_id": author_id,
        "author_nickname": author_nickname,
        "text": text,
    }


def _compact_user_pkg(pkg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pkg, dict):
        return {"id": None, "nickname": None}
    u = pkg.get("User") if isinstance(pkg.get("User"), dict) else {}
    s = pkg.get("Statistic") if isinstance(pkg.get("Statistic"), dict) else {}

    def _int(v: Any) -> int | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            try:
                return int(v.strip(), 10)
            except Exception:
                return None
        return None

    user_id = plar.best_effort_extract_text(u.get("ID") or u.get("Id") or u.get("UserID")).strip() or None
    nickname = plar.best_effort_extract_text(u.get("Nickname") or u.get("Name")).strip() or None
    signature = plar.best_effort_extract_text(u.get("Signature")).strip() or None

    return {
        "id": user_id,
        "nickname": nickname,
        "signature": signature,
        "level": _int(u.get("Level")),
        "experience": _int(u.get("Experience")),
        "stats": {
            "comment_count": _int(s.get("CommentCount")),
            "experiment_count": _int(s.get("ExperimentCount")),
            "following_count": _int(s.get("FollowingCount")),
            "follower_count": _int(s.get("FollowerCount")),
            "star_count": _int(s.get("StarCount")),
            "support_count": _int(s.get("SupportCount")),
        },
        "raw_type": str(pkg.get("$type") or "").strip() or None,
    }


def plar_oldest_by_user(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Find a user's oldest work by scanning QueryExperiments pages (best-effort)."""
    user = _require_user(runtime)
    user_id = _require_hex24(args.get("user_id"), where="plar_oldest_by_user.user_id")
    if not user_id:
        raise ToolError("plar_oldest_by_user: user_id is required")
    category = str(args.get("category") or "Experiment").strip() or "Experiment"
    if category not in ("Experiment", "Discussion", "both"):
        raise ToolError("plar_oldest_by_user: category must be Experiment|Discussion|both")

    take = int(args.get("take") or 24)
    if take <= 0:
        take = 24
    if take > 24:
        take = 24

    max_pages = int(args.get("max_pages") or 200)
    if max_pages < 1:
        max_pages = 1
    if max_pages > 800:
        max_pages = 800

    tags = args.get("tags")
    if tags is not None and not isinstance(tags, (list, str)):
        raise ToolError("plar_oldest_by_user: tags must be an array of strings or null")
    tags_list = _normalize_tag_list(tags)

    def _scan(cat: str) -> dict[str, Any]:
        skip = 0
        from_id: str | None = None
        pages = 0
        oldest: dict[str, Any] | None = None
        oldest_cd: int | None = None
        hit_limit = False

        while pages < max_pages:
            items = plar.query_experiments(
                user,
                category=cat,
                take=take,
                skip=skip,
                from_skip=from_id,
                days=None,
                sort="Default",
                user_id=user_id,
                tags=tags_list,
            )
            pages += 1
            if not items:
                break
            for it in items:
                if not isinstance(it, dict):
                    continue
                cd = it.get("CreationDate")
                try:
                    cd_i = int(cd) if isinstance(cd, (int, float, str)) and str(cd).strip() else None
                except Exception:
                    cd_i = None
                if cd_i is None:
                    continue
                if oldest_cd is None or cd_i < oldest_cd:
                    oldest_cd = cd_i
                    oldest = it

            last = items[-1] if isinstance(items[-1], dict) else None
            if last is not None:
                last_id = plar.best_effort_extract_text(last.get("ID") or last.get("Id")).strip()
                if last_id:
                    from_id = last_id
            skip += min(len(items), take)
            if len(items) < take:
                break
            if pages >= max_pages:
                hit_limit = True
                break

        return {
            "category": cat,
            "user_id": user_id,
            "tags": tags_list or [],
            "take": take,
            "pages_scanned": pages,
            "incomplete": bool(hit_limit),
            "item": _compact_qe_item(oldest or {}, category_hint=cat) if isinstance(oldest, dict) else None,
        }

    if category == "both":
        exp = _scan("Experiment")
        disc = _scan("Discussion")

        def _cd(d: dict[str, Any]) -> int | None:
            it = d.get("item")
            if not isinstance(it, dict):
                return None
            v = it.get("creation_date")
            try:
                return int(v) if v is not None else None
            except Exception:
                return None

        exp_cd = _cd(exp)
        disc_cd = _cd(disc)
        pick: dict[str, Any] | None = None
        if isinstance(exp.get("item"), dict) and isinstance(disc.get("item"), dict):
            if exp_cd is not None and disc_cd is not None:
                pick = exp["item"] if exp_cd <= disc_cd else disc["item"]
            else:
                pick = exp["item"]
        elif isinstance(exp.get("item"), dict):
            pick = exp["item"]
        elif isinstance(disc.get("item"), dict):
            pick = disc["item"]

        return {"category": "both", "experiment": exp, "discussion": disc, "pick": pick}

    return _scan(category)


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
    summary_id = _require_hex24(args.get("summary_id"), where="plar_get_experiment_context.summary_id")
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


def plar_check_following(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Check whether follower follows followee (best-effort, paginated)."""
    user = _require_user(runtime)

    follower_id_raw = args.get("follower_user_id")
    follower_name_raw = args.get("follower_name")
    followee_id_raw = args.get("followee_user_id")
    followee_name_raw = args.get("followee_name")

    def _resolve_user(*, uid_raw: Any, name_raw: Any, where: str) -> dict[str, Any]:
        uid = None
        name = None
        if isinstance(uid_raw, str) and uid_raw.strip():
            uid = _require_hex24(uid_raw, where=where + ".user_id")
        if isinstance(name_raw, str) and name_raw.strip():
            name = str(name_raw).strip().lstrip("@＠").strip() or None
        if uid:
            pkg = plar.get_user_by_id(user, user_id=uid)
            compact = _compact_user_pkg(pkg)
            return {"id": compact.get("id") or uid, "nickname": compact.get("nickname")}
        if name:
            pkg = plar.get_user_by_name(user, name=name)
            compact = _compact_user_pkg(pkg)
            cid = str(compact.get("id") or "").strip()
            if cid:
                return {"id": cid, "nickname": compact.get("nickname") or name}
            # Extremely defensive: if API returns no id, still return name.
            return {"id": None, "nickname": name}
        raise ToolError(f"{where}: provide follower_user_id/follower_name and followee_user_id/followee_name")

    follower = _resolve_user(uid_raw=follower_id_raw, name_raw=follower_name_raw, where="plar_check_following.follower")
    followee = _resolve_user(uid_raw=followee_id_raw, name_raw=followee_name_raw, where="plar_check_following.followee")
    follower_id = str(follower.get("id") or "").strip()
    followee_id = str(followee.get("id") or "").strip()
    follower_nick = str(follower.get("nickname") or "").strip()
    followee_nick = str(followee.get("nickname") or "").strip()

    if not follower_id or not followee_id:
        raise ToolError("plar_check_following: failed to resolve both users' IDs")
    if not _HEX24_RE.fullmatch(follower_id) or not _HEX24_RE.fullmatch(followee_id):
        raise ToolError("plar_check_following: resolved user ids are not 24-hex IDs")

    take = int(args.get("take") or 24)
    if take <= 0:
        take = 24
    # Backend rejects take > 24 (400 Input.Field.Invalid).
    if take > 24:
        take = 24
    max_pages = int(args.get("max_pages") or 50)
    if max_pages < 1:
        max_pages = 1
    if max_pages > 200:
        max_pages = 200

    checked: dict[str, Any] = {
        "display_type": "Following",
        "query_first": True,
        "take": take,
        "max_pages": max_pages,
    }

    def _extract_user_obj(it: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(it, dict):
            return {}
        u = it.get("User")
        return u if isinstance(u, dict) else it

    def _extract_user_id(it: dict[str, Any]) -> str:
        u = _extract_user_obj(it)
        uid = plar.best_effort_extract_text(u.get("ID") or u.get("Id") or u.get("UserID")).strip()
        return uid

    def _match_from_list(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        for it in items or []:
            if not isinstance(it, dict):
                continue
            uid = _extract_user_id(it)
            if uid == followee_id:
                u = _extract_user_obj(it)
                nick = plar.best_effort_extract_text(u.get("Nickname") or u.get("Name")).strip() or None
                return {"id": uid, "nickname": nick}
        return None

    # Phase 1: query filter (may be faster, but not always reliable).
    matched: dict[str, Any] | None = None
    if followee_nick:
        try:
            q_items = plar.get_relations(user, user_id=follower_id, display_type="Following", skip=0, take=take, query=followee_nick)
            matched = _match_from_list(q_items)
            checked["query"] = followee_nick
            checked["query_count"] = len(q_items) if isinstance(q_items, list) else 0
        except Exception as e:
            checked["query_error"] = f"{type(e).__name__}: {e}"
            matched = None
    else:
        checked["query"] = ""

    # Phase 2: scan pages until exhausted or matched.
    pages = 0
    scanned = 0
    if matched is None:
        skip = 0
        while pages < max_pages:
            pages += 1
            items = plar.get_relations(user, user_id=follower_id, display_type="Following", skip=skip, take=take, query="")
            if not isinstance(items, list):
                items = []
            scanned += len(items)
            m = _match_from_list(items)
            if m is not None:
                matched = m
                break
            if len(items) < take:
                break
            skip += take
    checked["pages_scanned"] = pages
    checked["items_scanned"] = scanned
    checked["incomplete"] = bool(matched is None and pages >= max_pages)

    return {
        "follower": {"id": follower_id, "nickname": follower_nick or None},
        "followee": {"id": followee_id, "nickname": followee_nick or None},
        "is_following": bool(matched is not None),
        "matched": matched,
        "checked": checked,
    }


def plar_get_status_save(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    user = _require_user(runtime)
    summary_id = _require_hex24(args.get("summary_id"), where="plar_get_status_save.summary_id")
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
    # NOTE: In aurex2, auto-publish is forced to the Discussion area.
    # Keep accepting `args.category` for backward compatibility, but ignore it.
    category = "Discussion"
    tags = args.get("tags")
    if tags is not None and not isinstance(tags, list):
        raise ToolError("plar_upload_sav: tags must be a list of strings")
    tags_list = [str(x) for x in (tags or []) if str(x).strip()]
    info = plar.upload_sav_as_experiment(
        user=user,
        sav_path=sav_path,
        title=title,
        introduction=introduction,
        cache_dir=runtime.cache_dir,
        category_value=category,
        tags=tags_list or None,
    )
    if not isinstance(info, dict):
        return {"published": True, "category": category}

    summary_id = str(info.get("summary_id") or "").strip()
    title_clean = " ".join(title.replace("\r", " ").replace("\n", " ").replace("\t", " ").split()).strip()
    title_clean = re.sub(r"[<>]", "", title_clean).strip()
    discussion_tag = f"<discussion={summary_id}>{title_clean}</discussion>" if (summary_id and title_clean) else ""

    out = dict(info)
    out["published"] = True
    out["category"] = "Discussion"
    if summary_id:
        out["discussion_id"] = summary_id
    if discussion_tag:
        out["discussion_tag"] = discussion_tag
        out["reply_suggestion_zh"] = f"您要的讨论 {discussion_tag} 已经发布！"
    elif summary_id:
        out["reply_suggestion_zh"] = f"已发布到讨论区（Discussion），ID：{summary_id}"
    return out


def plar_list_builtin_tags(_runtime: ToolRuntime, _args: dict[str, Any]) -> dict[str, Any]:
    """List physicsLab built-in Tag enum name/value pairs (best-effort)."""
    try:
        from physicsLab import Tag as PLTag  # type: ignore
    except Exception as e:
        raise ToolError(f"plar_list_builtin_tags: failed to import physicsLab.Tag: {type(e).__name__}: {e}") from e

    tags: list[dict[str, Any]] = []
    for t in PLTag:
        name = getattr(t, "name", None)
        val = getattr(t, "value", None)
        if isinstance(name, str) and name.strip() and isinstance(val, str) and val.strip():
            tags.append({"name": name.strip(), "value": val.strip()})
    return {"count": len(tags), "tags": tags}


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
            "tags": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Include tags (e.g. 精选 / Featured / Tag.Featured). Unknown/custom tags are allowed.",
            },
            "exclude_tags": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Exclude tags (same format as tags).",
            },
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

PLAR_GET_COMMENTS_TOOL = {
    "name": "plar_get_comments",
    "description": "List comments for a target (User wall / Experiment / Discussion).",
    "parameters": {
        "type": "object",
        "properties": {
            "target_type": {"type": "string", "enum": ["User", "Experiment", "Discussion"]},
            "target_id": {"type": "string"},
            "take": {"type": "integer", "minimum": 1, "maximum": 20, "default": 20},
            "skip": {"type": "integer", "minimum": 0, "default": 0},
        },
        "required": ["target_type", "target_id"],
    },
}

PLAR_GET_OLDEST_COMMENT_TOOL = {
    "name": "plar_get_oldest_comment",
    "description": "Scan and return the oldest comment for a target (best-effort, paginated). Use when the user asks for “最早/第一条/oldest/first comment”.",
    "parameters": {
        "type": "object",
        "properties": {
            "target_type": {"type": "string", "enum": ["User", "Experiment", "Discussion"]},
            "target_id": {"type": "string"},
            "take": {"type": "integer", "minimum": 1, "maximum": 20, "default": 20},
            "max_pages": {"type": "integer", "minimum": 1, "maximum": 800, "default": 200},
            "skip": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Optional unix_ms timestamp to start scanning from (0 means latest).",
            },
        },
        "required": ["target_type", "target_id"],
    },
}

PLAR_OLDEST_BY_USER_TOOL = {
    "name": "plar_oldest_by_user",
    "description": "Find a user's oldest (earliest published) Experiment/Discussion by scanning QueryExperiments pages (best-effort). Useful for queries like “<nickname>发布的第一个实验”.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "Target user ID."},
            "category": {"type": "string", "enum": ["Experiment", "Discussion", "both"], "default": "Experiment"},
            "take": {"type": "integer", "minimum": 1, "maximum": 24, "default": 24},
            "max_pages": {"type": "integer", "minimum": 1, "maximum": 800, "default": 200},
            "tags": {
                "type": ["array", "string", "null"],
                "items": {"type": "string"},
                "description": "Optional tag filter (e.g. 精选 / Featured / Tag.Featured). Unknown/custom tags are allowed.",
            },
        },
        "required": ["user_id"],
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
            "take": {"type": "integer", "minimum": 1, "maximum": 24, "default": 20},
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
    "description": "Upload a local .sav to PhysicsLab and confirm it (forced to Discussion in aurex2).",
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

PLAR_LIST_TAGS_TOOL = {
    "name": "plar_list_builtin_tags",
    "description": "List built-in PhysicsLab Tag enum names/values (e.g. Featured=精选). Useful when user asks what tags exist or how to filter by tags.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

PLAR_CHECK_FOLLOWING_TOOL = {
    "name": "plar_check_following",
    "description": "Check whether one user follows another user (best-effort, paginated). Useful for queries like “用户A有没有关注用户B”.",
    "parameters": {
        "type": "object",
        "properties": {
            "follower_user_id": {"type": ["string", "null"], "description": "Follower user ID (24-hex)."},
            "follower_name": {"type": ["string", "null"], "description": "Follower nickname (with or without @)."},
            "followee_user_id": {"type": ["string", "null"], "description": "Followee user ID (24-hex)."},
            "followee_name": {"type": ["string", "null"], "description": "Followee nickname (with or without @)."},
            "take": {"type": "integer", "minimum": 1, "maximum": 24, "default": 24},
            "max_pages": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
        },
        "required": [],
    },
}
