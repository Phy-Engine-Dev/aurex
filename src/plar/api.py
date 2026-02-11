from __future__ import annotations

import json
import os
import time
from typing import Any

from .errors import PLARError
from .http import _DEFAULT_TIMEOUT_SEC, post_json_no_env_proxy
from .physicslab import ensure_physicslab_importable, unwrap_user
from .text import collect_text, extract_title, safe_json_excerpt

_PLAR_BASE_CN = "https://physics-api-cn.turtlesim.com"


def email_login(
    *,
    email: str,
    password: str,
    cache_dir: str,
    http_timeout_sec: float | None = None,
) -> Any:
    ensure_physicslab_importable(cache_dir=cache_dir, http_timeout_sec=http_timeout_sec)
    from physicsLab import web  # type: ignore

    return web.email_login(email, password)


def get_comments(
    user: Any,
    *,
    target_id: str,
    target_type: str,
    take: int,
    skip: int = 0,
) -> list[dict[str, Any]]:
    result = user.get_comments(target_id=target_id, target_type=target_type, take=take, skip=int(skip))
    if not isinstance(result, dict):
        raise PLARError("get_comments returned a non-dict response")
    data = result.get("Data")
    if not isinstance(data, dict):
        raise PLARError("get_comments response missing Data")
    comments = data.get("Comments")
    if not isinstance(comments, list):
        raise PLARError("get_comments response missing Data.Comments")
    return [c for c in comments if isinstance(c, dict)]


def post_comment(
    user: Any,
    *,
    target_id: str,
    target_type: str,
    content: str,
    reply_id: str | None = None,
) -> Any:
    return user.post_comment(
        target_id=target_id,
        target_type=target_type,
        content=content,
        reply_id=reply_id,
    )


def query_experiments(
    user: Any,
    *,
    category: Any,
    take: int = 20,
    skip: int = 0,
    from_skip: str | None = None,
    days: int | str | None = None,
    sort: int | str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    languages: list[str] | None = None,
    exclude_languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    take_i = max(1, int(take))
    take_i = min(take_i, 24)
    skip_i = max(0, int(skip))
    from_skip_s = (from_skip or "").strip() or None
    user_id_s = (user_id or "").strip() or None

    if tags is not None and not isinstance(tags, list):
        raise PLARError("tags must be list[str] or None")
    if exclude_tags is not None and not isinstance(exclude_tags, list):
        raise PLARError("exclude_tags must be list[str] or None")
    if languages is not None and not isinstance(languages, list):
        raise PLARError("languages must be list[str] or None")
    if exclude_languages is not None and not isinstance(exclude_languages, list):
        raise PLARError("exclude_languages must be list[str] or None")

    def _normalize_days(v: int | str | None) -> int | str:
        if v is None:
            return 0
        if isinstance(v, int):
            return str(max(0, v))
        s = str(v).strip()
        if not s:
            return 0
        if s.isdigit():
            return s
        return s

    def _normalize_sort(v: int | str | None) -> int | str:
        if v is None:
            return 0
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if not s:
            return 0
        # Keep the caller's string (e.g. "Popularity") to match web clients.
        return s

    def _extract_values(obj: Any) -> list[dict[str, Any]]:
        if not isinstance(obj, dict):
            raise PLARError(f"QueryExperiments returned {type(obj).__name__}, expected dict")
        status = obj.get("Status")
        if status is not None:
            try:
                st = int(status)
            except Exception:
                st = None
            if st is not None and st != 200:
                msg = str(obj.get("Message") or "").strip()
                raise PLARError(f"QueryExperiments failed (status={st}): {msg}".rstrip())

        data = obj.get("Data")
        if data is None and "data" in obj:
            data = obj.get("data")
        if data is None:
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if not isinstance(data, dict):
            return []

        values = data.get("$values") or data.get("values") or data.get("Values")
        if not isinstance(values, list):
            return []
        return [x for x in values if isinstance(x, dict)]

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    can_http = isinstance(token, str) and token.strip() and isinstance(auth_code, str) and auth_code.strip()

    if can_http:
        cat_val = getattr(category, "value", category)
        status_code, data = post_json_no_env_proxy(
            url=f"{_PLAR_BASE_CN}/Contents/QueryExperiments",
            payload={
                "Query": {
                    "Category": cat_val,
                    "Languages": languages or [],
                    "ExcludeLanguages": exclude_languages,
                    "Tags": tags,
                    "ExcludeTags": exclude_tags,
                    "ModelTags": None,
                    "ModelID": None,
                    "ParentID": None,
                    "UserID": user_id_s,
                    "Special": None,
                    "From": from_skip_s,
                    "Skip": skip_i,
                    "Take": take_i,
                    "Days": _normalize_days(days),
                    "Sort": _normalize_sort(sort),
                    "ShowAnnouncement": False,
                }
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-API-Token": token,
                "x-API-AuthCode": auth_code,
                "User-Agent": "plar/2",
            },
            timeout_sec=_DEFAULT_TIMEOUT_SEC,
        )
        if status_code == 403:
            raise PermissionError("login failed")
        if status_code == 404:
            raise PLARError("QueryExperiments failed (status=404): not found")
        if status_code >= 400:
            raise PLARError(f"QueryExperiments failed (http={status_code}): {data}")
        return _extract_values(data)

    fn = getattr(user, "query_experiments", None)
    if not callable(fn):
        raise PLARError("query_experiments is not available (missing token/auth_code and wrapper method)")
    try:
        result = fn(
            category=category,
            tags=tags or [],
            exclude_tags=exclude_tags,
            languages=languages or [],
            exclude_languages=exclude_languages,
            user_id=user_id_s,
            take=take_i,
            skip=skip_i,
            from_skip=from_skip_s,
        )
    except Exception as e:
        raise PLARError(f"query_experiments failed via wrapper: {e}") from e
    return _extract_values(result)


def _extract_data_block(obj: Any, *, op: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise PLARError(f"{op} returned {type(obj).__name__}, expected dict")
    status = obj.get("Status")
    if status is not None:
        try:
            st = int(status)
        except Exception:
            st = None
        if st is not None and st != 200:
            msg = str(obj.get("Message") or "").strip()
            raise PLARError(f"{op} failed (status={st}): {msg}".rstrip())
    data = obj.get("Data")
    if not isinstance(data, dict):
        raise PLARError(f"{op} response missing Data")
    return data


def get_user_by_name(user: Any, *, name: str) -> dict[str, Any]:
    raw = (name or "").strip()
    if not raw:
        raise PLARError("name is empty")
    raw = raw.lstrip("@＠").strip()
    if not raw:
        raise PLARError("name is empty")

    def _candidates(n: str) -> list[str]:
        out = [n]
        low = n.casefold()
        if low.endswith("mium") and not low.endswith("nium") and len(n) > 4:
            out.append(n[:-4] + "nium")
        # unique, keep order
        uniq: list[str] = []
        seen: set[str] = set()
        for x in out:
            x2 = x.strip()
            if not x2 or x2 in seen:
                continue
            seen.add(x2)
            uniq.append(x2)
        return uniq

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    can_http = isinstance(token, str) and token.strip() and isinstance(auth_code, str) and auth_code.strip()

    wrapper = getattr(user, "get_user_by_name", None)
    last_error: BaseException | None = None

    for cand in _candidates(raw):
        if callable(wrapper):
            try:
                return _extract_data_block(wrapper(cand), op="GetUser")
            except TypeError as e:
                last_error = e
            except PLARError as e:
                last_error = e
                if "status=404" in str(e).casefold() or "not found" in str(e).casefold():
                    continue
                raise

        if not can_http:
            continue
        status_code, data = post_json_no_env_proxy(
            url=f"{_PLAR_BASE_CN}/Users/GetUser",
            payload={"Name": cand},
            headers={
                "Content-Type": "application/json",
                "x-API-Token": token,
                "x-API-AuthCode": auth_code,
            },
        )
        if status_code == 403:
            raise PermissionError("login failed")
        if status_code >= 400:
            last_error = PLARError(f"GetUser failed (http={status_code}): {data}")
            if status_code == 404:
                continue
            raise last_error
        try:
            return _extract_data_block(data, op="GetUser")
        except PLARError as e:
            last_error = e
            if "status=404" in str(e).casefold() or "not found" in str(e).casefold():
                continue
            raise

    if last_error is None:
        raise PLARError("GetUser failed (no wrapper and missing token/auth_code)")
    if isinstance(last_error, PLARError):
        raise last_error
    raise PLARError(str(last_error))


def get_user_by_id(user: Any, *, user_id: str) -> dict[str, Any]:
    user_id_s = (user_id or "").strip()
    if not user_id_s:
        raise PLARError("user_id is empty")

    wrapper = getattr(user, "get_user_by_id", None)
    if callable(wrapper):
        try:
            return _extract_data_block(wrapper(user_id_s), op="GetUser")
        except TypeError:
            pass

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    if not isinstance(token, str) or not token.strip() or not isinstance(auth_code, str) or not auth_code.strip():
        raise PLARError("get_user_by_id is not callable and token/auth_code are missing")

    status_code, data = post_json_no_env_proxy(
        url=f"{_PLAR_BASE_CN}/Users/GetUser",
        payload={"ID": user_id_s},
        headers={
            "Content-Type": "application/json",
            "x-API-Token": token,
            "x-API-AuthCode": auth_code,
        },
    )
    if status_code == 403:
        raise PermissionError("login failed")
    if status_code >= 400:
        raise PLARError(f"GetUser failed (http={status_code}): {data}")
    return _extract_data_block(data, op="GetUser")


def get_relations(
    user: Any,
    *,
    user_id: str,
    display_type: str | int = "Following",
    skip: int = 0,
    take: int = 20,
    query: str = "",
) -> list[dict[str, Any]]:
    uid = (user_id or "").strip()
    if not uid:
        raise PLARError("user_id is empty")

    def _display_type_code(v: str | int) -> int:
        if isinstance(v, int):
            return int(v)
        s = str(v).strip()
        if s.isdigit():
            return int(s)
        mapping = {
            "follower": 0,
            "followers": 0,
            "following": 1,
            "followings": 1,
            "banned": 2,
            "blocked": 2,
            "volunteer": 3,
            "editor": 4,
            "admin": 4,
            "admins": 4,
            "emeritus": 5,
            "retired": 5,
        }
        return mapping.get(s.casefold(), -1)

    dt = _display_type_code(display_type)
    if dt not in (0, 1, 2, 3, 4, 5):
        raise PLARError("display_type must be a known name or an integer 0..5")

    skip_i = max(0, int(skip))
    take_i = max(1, int(take))
    take_i = min(take_i, 100)
    q = (query or "").strip()

    def _extract_users(obj: Any) -> list[dict[str, Any]]:
        if not isinstance(obj, dict):
            raise PLARError(f"GetRelations returned {type(obj).__name__}, expected dict")
        status = obj.get("Status")
        if status is not None:
            try:
                st = int(status)
            except Exception:
                st = None
            if st is not None and st != 200:
                msg = str(obj.get("Message") or "").strip()
                raise PLARError(f"GetRelations failed (status={st}): {msg}".rstrip())

        data = obj.get("Data")
        if data is None and "data" in obj:
            data = obj.get("data")
        if isinstance(data, dict):
            values = data.get("$values")
            if isinstance(values, list):
                return [x for x in values if isinstance(x, dict)]
            for key in ("Users", "users", "Relations", "relations"):
                v = data.get(key)
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    wrapper = getattr(user, "get_relations", None)
    if callable(wrapper):
        last: Exception | None = None
        for dt_variant in (dt, str(dt)):
            try:
                return _extract_users(
                    wrapper(user_id=uid, display_type=dt_variant, skip=skip_i, take=take_i, query=q)
                )
            except Exception as e:
                last = e
        raise PLARError(f"get_relations failed via wrapper: {last}") from last

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    if not isinstance(token, str) or not token.strip() or not isinstance(auth_code, str) or not auth_code.strip():
        raise PLARError("get_relations is not callable and token/auth_code are missing")

    status_code, data = post_json_no_env_proxy(
        url=f"{_PLAR_BASE_CN}/Users/GetRelations",
        payload={"UserID": uid, "DisplayType": dt, "Skip": skip_i, "Take": take_i, "Query": q},
        headers={
            "Content-Type": "application/json",
            "x-API-Token": token,
            "x-API-AuthCode": auth_code,
        },
    )
    if status_code == 403:
        raise PermissionError("login failed")
    if status_code >= 400:
        raise PLARError(f"GetRelations failed (http={status_code}): {data}")
    return _extract_users(data)


def get_messages(
    user: Any,
    *,
    category_id: int,
    skip: int = 0,
    take: int = 20,
    no_templates: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = user.get_messages(category_id=category_id, skip=skip, take=take, no_templates=no_templates)
    if not isinstance(result, dict):
        raise PLARError("get_messages returned a non-dict response")
    data = result.get("Data")
    if not isinstance(data, dict):
        raise PLARError("get_messages response missing Data")
    messages = data.get("Messages")
    templates = data.get("Templates", [])
    if not isinstance(messages, list):
        raise PLARError("get_messages response missing Data.Messages")
    if not isinstance(templates, list):
        templates = []
    return (
        [m for m in messages if isinstance(m, dict)],
        [t for t in templates if isinstance(t, dict)],
    )


def upload_sav_as_experiment(
    *,
    user: Any,
    sav_path: str,
    title: str,
    introduction: str,
    cache_dir: str,
    category_value: str = "Experiment",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    ensure_physicslab_importable(cache_dir=cache_dir)
    from physicsLab import Category, Experiment, OpenMode  # type: ignore

    if category_value == "Experiment":
        category = Category.Experiment
    elif category_value == "Discussion":
        category = Category.Discussion
    else:
        raise PLARError("category_value must be 'Experiment' or 'Discussion'")

    exp = Experiment(OpenMode.load_by_filepath, sav_path)
    exp.edit_publish_info(title=title, introduction=introduction, wx=False)
    if tags:
        _apply_publish_tags(exp, tags)

    real_user = unwrap_user(user)
    try:
        submit_resp, submit_data = exp._Experiment__upload(real_user, category, None)  # type: ignore[attr-defined]
    except Exception as e:
        raise PLARError(f"SubmitExperiment failed: {e}") from e

    if not isinstance(submit_resp, dict):
        raise PLARError("SubmitExperiment returned unexpected type")
    st = submit_resp.get("Status")
    if st is not None:
        try:
            st_i = int(st)
        except Exception:
            st_i = None
        if st_i is not None and st_i != 200:
            msg = str(submit_resp.get("Message") or "").strip()
            raise PLARError(f"SubmitExperiment failed (status={st_i}): {msg}".rstrip())

    data = submit_resp.get("Data")
    if not isinstance(data, dict):
        raise PLARError("SubmitExperiment returned no Data")
    summary = data.get("Summary")
    if not isinstance(summary, dict):
        raise PLARError("SubmitExperiment returned no Data.Summary")
    summary_id = summary.get("ID")
    if not isinstance(summary_id, str) or not summary_id.strip():
        raise PLARError("SubmitExperiment returned empty summary id")

    image_counter = 0
    if isinstance(submit_data, dict):
        s = submit_data.get("Summary")
        if isinstance(s, dict) and isinstance(s.get("Image"), int):
            image_counter = int(s.get("Image"))

    try:
        real_user.confirm_experiment(summary_id.strip(), category, image_counter)
    except Exception as e:
        raise PLARError(f"ConfirmExperiment failed: {e}") from e

    return {"summary_id": summary_id.strip(), "category": category.value}


def _apply_publish_tags(exp: Any, tags: list[str]) -> None:
    cleaned = [t.strip() for t in (tags or []) if isinstance(t, str) and t.strip()]
    if not cleaned:
        return
    try:
        from physicsLab import Tag  # type: ignore
    except Exception:
        Tag = None  # type: ignore[assignment]

    enum_tags: list[Any] = []
    for t in cleaned:
        if Tag is None:
            continue
        for candidate in (t, t.removeprefix("Tag.")):
            key = candidate.strip()
            if not key:
                continue
            try:
                enum_tags.append(Tag[key])  # type: ignore[index]
                break
            except Exception:
                pass
            try:
                match = next((x for x in Tag if getattr(x, "value", None) == key), None)
            except Exception:
                match = None
            if match is not None:
                enum_tags.append(match)
                break

    if not enum_tags:
        return
    try:
        exp.edit_tags(*enum_tags)
    except Exception:
        return


def get_summary(user: Any, *, summary_id: str, category_value: str) -> dict[str, Any]:
    if category_value not in ("Experiment", "Discussion"):
        raise PLARError("category_value must be 'Experiment' or 'Discussion'")

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    if isinstance(token, str) and token.strip() and isinstance(auth_code, str) and auth_code.strip():
        status_code, data = post_json_no_env_proxy(
            url=f"{_PLAR_BASE_CN}/Contents/GetSummary",
            payload={"ContentID": summary_id, "Category": category_value},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-API-Token": token,
                "x-API-AuthCode": auth_code,
            },
        )
        if status_code == 403:
            raise PermissionError("login failed")
        if status_code == 404:
            raise PLARError("GetSummary failed (status=404): not found")
        if status_code >= 400:
            raise PLARError(f"GetSummary failed (http={status_code}): {data}")
        if not isinstance(data, dict):
            raise PLARError("GetSummary returned unexpected response type")
        return data

    try:
        from physicsLab import Category  # type: ignore
    except Exception as e:  # pragma: no cover
        raise PLARError(f"Failed to import physicsLab.Category: {e}") from e
    cat = Category.Experiment if category_value == "Experiment" else Category.Discussion
    result = user.get_summary(summary_id, cat)
    if not isinstance(result, dict):
        raise PLARError("get_summary returned a non-dict response")
    return result


def get_experiment(user: Any, *, summary_id: str, category_value: str) -> dict[str, Any]:
    if category_value not in ("Experiment", "Discussion"):
        raise PLARError("category_value must be 'Experiment' or 'Discussion'")

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    if isinstance(token, str) and token.strip() and isinstance(auth_code, str) and auth_code.strip():
        status_code, data = post_json_no_env_proxy(
            url=f"{_PLAR_BASE_CN}/Contents/GetExperiment",
            payload={"ContentID": summary_id, "Category": category_value},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-API-Token": token,
                "x-API-AuthCode": auth_code,
            },
        )
        if status_code == 403:
            raise PermissionError("login failed")
        if status_code == 404:
            raise PLARError("GetExperiment failed (status=404): not found")
        if status_code >= 400:
            raise PLARError(f"GetExperiment failed (http={status_code}): {data}")
        if not isinstance(data, dict):
            raise PLARError("GetExperiment returned unexpected response type")
        return data

    try:
        from physicsLab import Category  # type: ignore
    except Exception as e:  # pragma: no cover
        raise PLARError(f"Failed to import physicsLab.Category: {e}") from e
    cat = Category.Experiment if category_value == "Experiment" else Category.Discussion
    result = user.get_experiment(summary_id, cat)
    if not isinstance(result, dict):
        raise PLARError("get_experiment returned a non-dict response")
    return result


def _find_plsav_like(obj: Any, *, max_depth: int = 6) -> dict[str, Any] | None:
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if cur is None or depth > max_depth:
            continue
        if isinstance(cur, dict):
            if "Experiment" in cur and isinstance(cur.get("Experiment"), dict):
                exp = cur.get("Experiment")
                if isinstance(exp, dict) and ("StatusSave" in exp or "CameraSave" in exp):
                    return cur
            if "StatusSave" in cur or "CameraSave" in cur:
                return cur
            for v in cur.values():
                stack.append((v, depth + 1))
            continue
        if isinstance(cur, list):
            for v in cur:
                stack.append((v, depth + 1))
            continue
    return None


def _summarize_plsav(plsav: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    summary = plsav.get("Summary")
    if isinstance(summary, dict):
        out["subject"] = summary.get("Subject") if isinstance(summary.get("Subject"), str) else None
        out["tags"] = summary.get("Tags") if isinstance(summary.get("Tags"), list) else None
    exp = plsav.get("Experiment") if isinstance(plsav.get("Experiment"), dict) else plsav
    if isinstance(exp, dict):
        out["type"] = exp.get("Type")
        status_str = exp.get("StatusSave")
        if isinstance(status_str, str) and status_str.strip():
            try:
                status = json.loads(status_str)
            except Exception:
                status = None
            if isinstance(status, dict):
                els = status.get("Elements")
                wires = status.get("Wires")
                out["elements_count"] = len(els) if isinstance(els, list) else None
                out["wires_count"] = len(wires) if isinstance(wires, list) else None
    return out


def get_experiment_context(
    user: Any,
    *,
    summary_id: str,
    category_value: str,
    cache_dir: str,
    ttl_sec: int = 300,
    max_json_chars: int = 20_000,
) -> dict[str, Any]:
    context_version = 2
    os.makedirs(cache_dir, exist_ok=True)
    cache_root = os.path.join(cache_dir, "plar_cache")
    os.makedirs(cache_root, exist_ok=True)
    cache_path = os.path.join(cache_root, f"{category_value.lower()}_{summary_id}.json")

    now = time.time()
    try:
        st = os.stat(cache_path)
        if now - st.st_mtime <= ttl_sec:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (
                isinstance(cached, dict)
                and cached.get("context_version") == context_version
                and cached.get("summary_id") == summary_id
            ):
                return cached
    except Exception:
        pass

    summary_data: Any = None
    exp_data: Any = None
    plsav_summary: dict[str, Any] = {}

    # Preferred: use physicsLab's higher-level loader.
    try:
        from physicsLab import Category as PLCategory  # type: ignore
        from physicsLab import Experiment, OpenMode  # type: ignore

        cat = PLCategory.Experiment if category_value == "Experiment" else PLCategory.Discussion
        exp = Experiment(OpenMode.load_by_plar_app, summary_id, cat, user=unwrap_user(user))
        plsav = exp.PlSav if isinstance(getattr(exp, "PlSav", None), dict) else None
        if isinstance(plsav, dict):
            summary_data = plsav.get("Summary")
            exp_data = plsav.get("Experiment")
            plsav_summary = _summarize_plsav(plsav)
    except Exception:
        summary_res = get_summary(user, summary_id=summary_id, category_value=category_value)
        exp_res = get_experiment(user, summary_id=summary_id, category_value=category_value)
        summary_data = summary_res.get("Data") if isinstance(summary_res, dict) else None
        exp_data = exp_res.get("Data") if isinstance(exp_res, dict) else None
        plsav_like = _find_plsav_like(exp_data) or _find_plsav_like(exp_res)
        plsav_summary = _summarize_plsav(plsav_like) if isinstance(plsav_like, dict) else {}

    title = extract_title(summary_data) or extract_title(exp_data) or str(plsav_summary.get("subject") or "").strip()
    title = title or None

    summary_text = collect_text(summary_data, max_chars=6000)
    experiment_text = collect_text(exp_data, max_chars=8000)

    author_id: str | None = None
    author_nickname: str | None = None
    if isinstance(summary_data, dict):
        u = summary_data.get("User")
        if isinstance(u, dict):
            from .text import best_effort_extract_text

            author_id = (
                best_effort_extract_text(u.get("ID")).strip()
                or best_effort_extract_text(u.get("UserID")).strip()
                or None
            )
            author_nickname = (
                best_effort_extract_text(u.get("Nickname")).strip()
                or best_effort_extract_text(u.get("Name")).strip()
                or None
            )

    context = {
        "context_version": context_version,
        "summary_id": summary_id,
        "category": category_value,
        "title": title,
        "author": {"id": author_id, "nickname": author_nickname},
        "body_text": summary_text or None,
        "content_text": experiment_text or None,
        "summary_text": summary_text or None,
        "experiment_text": experiment_text or None,
        "summary_data_excerpt": safe_json_excerpt(summary_data, max_chars=max_json_chars),
        "experiment_data_excerpt": safe_json_excerpt(exp_data, max_chars=max_json_chars),
        "plsav_summary": plsav_summary,
    }

    try:
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, cache_path)
    except Exception:
        pass

    return context


def get_status_save(
    user: Any,
    *,
    summary_id: str,
    category_value: str,
    cache_dir: str,
    ttl_sec: int = 300,
) -> dict[str, Any]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_root = os.path.join(cache_dir, "plar_cache")
    os.makedirs(cache_root, exist_ok=True)
    cache_path = os.path.join(cache_root, f"status_{category_value.lower()}_{summary_id}.json")

    now = time.time()
    try:
        st = os.stat(cache_path)
        if now - st.st_mtime <= ttl_sec:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached.get("summary_id") == summary_id:
                payload = cached.get("status_save")
                if isinstance(payload, dict):
                    return payload
    except Exception:
        pass

    ensure_physicslab_importable(cache_dir=cache_dir)

    status_str: Any = None
    try:
        from physicsLab import Category as PLCategory  # type: ignore
        from physicsLab import Experiment, OpenMode  # type: ignore

        if category_value == "Experiment":
            cat = PLCategory.Experiment
        elif category_value == "Discussion":
            cat = PLCategory.Discussion
        else:
            raise PLARError("category_value must be 'Experiment' or 'Discussion'")

        exp = Experiment(OpenMode.load_by_plar_app, summary_id, cat, user=unwrap_user(user))
        plsav = exp.PlSav if isinstance(getattr(exp, "PlSav", None), dict) else None
        if isinstance(plsav, dict):
            exp_obj = plsav.get("Experiment") if isinstance(plsav.get("Experiment"), dict) else None
            status_str = exp_obj.get("StatusSave") if isinstance(exp_obj, dict) else plsav.get("StatusSave")
    except Exception:
        exp_res = get_experiment(user, summary_id=summary_id, category_value=category_value)
        data = exp_res.get("Data") if isinstance(exp_res, dict) else None
        plsav_like = _find_plsav_like(data) or _find_plsav_like(exp_res)
        if isinstance(plsav_like, dict):
            exp_obj = plsav_like.get("Experiment") if isinstance(plsav_like.get("Experiment"), dict) else None
            status_str = exp_obj.get("StatusSave") if isinstance(exp_obj, dict) else plsav_like.get("StatusSave")

    if not isinstance(status_str, str) or not status_str.strip():
        raise PLARError("PlSav missing StatusSave")
    try:
        status = json.loads(status_str)
    except Exception as e:
        raise PLARError(f"Failed to parse StatusSave JSON: {e}") from e
    if not isinstance(status, dict):
        raise PLARError("StatusSave JSON is not an object")

    try:
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"summary_id": summary_id, "status_save": status}, f, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, cache_path)
    except Exception:
        pass

    return status

