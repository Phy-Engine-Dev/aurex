from __future__ import annotations

import json
import hashlib
import copy
import os
import re
import tempfile
import time
from typing import Any

from .errors import PLARError
from .http import _DEFAULT_TIMEOUT_SEC, post_json_no_env_proxy
from . import official_publish_api
from .physicslab import ensure_physicslab_importable, unwrap_user
from .text import collect_text, extract_title, safe_json_excerpt

_PLAR_BASE_CN = "https://physics-api-cn.turtlesim.com"


class PublicationResponseError(PLARError):
    """Submission error with a deliberately non-secret diagnostic code."""

    def __init__(self, message: str, safe_diagnostic: str):
        super().__init__(message)
        self.safe_diagnostic = safe_diagnostic


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
    """SDK compatibility path: reply_id is a user ID, never a comment ID."""
    if reply_id and isinstance(content, str) and content.startswith('<user='):
        content = _sdk_reply_content(content, reply_id)
    return unwrap_user(user).post_comment(
        target_id=target_id,
        target_type=target_type,
        content=content,
        reply_id=reply_id,
    )


def _sdk_reply_content(content: str, requester_user_id: str) -> str:
    """Keep the reviewed ID binding while using the SDK's reply syntax on wire."""
    if not isinstance(requester_user_id, str) or not re.fullmatch(r'[0-9a-fA-F]{24}', requester_user_id):
        raise PLARError('Reply requires an exact requester user ID')
    if not isinstance(content, str):
        raise PLARError('Reply requires reviewed text')
    prefix = re.match(r'<user=' + re.escape(requester_user_id) + r'>(@[^<>\x00-\x1f]+)</user> ', content)
    if prefix is None or not content[prefix.end():].strip():
        raise PLARError('Reply must begin with its bound requester ID mention and contain an answer')
    nickname = prefix.group(1)[1:]
    if any(char.isspace() or char in ':：@＠' for char in nickname):
        raise PLARError('Requester nickname is ambiguous in the SDK reply syntax; refusing to guess')
    # Direct pre-rendered <user> mentions were visible but did not notify in
    # live tests. The official SDK's plain reply prefix did notify; the server
    # converted it back to an ID-based user tag. ReplyID remains explicit below.
    return '回复' + prefix.group(1) + ': ' + content[prefix.end():]


def _post_comment_sdk_once(user: Any, *, target_id: str, target_type: str,
                           requester_user_id: str, content: str) -> dict[str, Any]:
    """Isolate the blocking SDK transport; never retry an ambiguous send."""
    import subprocess
    import sys
    from pathlib import Path

    real = unwrap_user(user)
    _authenticated_headers(real)
    payload = {'token': real.token, 'auth_code': real.auth_code, 'target_id': target_id,
               'target_type': target_type, 'requester_user_id': requester_user_id, 'content': content}
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = source + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    try:
        # Authentication is private stdin, never process arguments or logs.
        # On timeout the child is killed. The request may already be accepted,
        # so the caller's durable ledger must mark it unknown and never retry.
        result = subprocess.run([sys.executable, '-m', 'plar.comment_worker'],
            input=json.dumps(payload, ensure_ascii=False), text=True,
            capture_output=True, timeout=_DEFAULT_TIMEOUT_SEC, env=env)
        if result.returncode != 0:
            raise ValueError('worker failed')
        value = json.loads(result.stdout)
    except Exception:
        raise PLARError('PostComment SDK did not confirm success; manual reconciliation may be required') from None
    if not isinstance(value, dict) or value.get('Status') != 200:
        raise PLARError('PostComment SDK did not confirm success; manual reconciliation may be required')
    return value


def post_task_comment_once(user: Any, *, target_id: str, target_type: str,
                           requester_user_id: str, content: str) -> dict[str, Any]:
    """One non-retrying external POST for the durable task-reply ledger.

    SDK _api.py post_comment's ReplyID is the replied-to USER ID, not a comment
    ID. Keep an original comment ID in local task metadata for deduplication.
    """
    if target_type not in {'User', 'Experiment', 'Discussion'} or not all(
        isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{24}', value)
        for value in (target_id, requester_user_id)):
        raise PLARError('Reply requires exact original target and requester user IDs')
    wire_content = _sdk_reply_content(content, requester_user_id)
    value = _post_comment_sdk_once(user, target_id=target_id, target_type=target_type,
                                   requester_user_id=requester_user_id, content=wire_content)
    data = value.get('Data')
    # The live endpoint returns the comment directly in Data. Retain support
    # for the nested form used by older adapters without losing the receipt ID.
    comment = data.get('Comment', data) if isinstance(data, dict) else {}
    comment_id = comment.get('ID') if isinstance(comment, dict) else None
    return {'posted': True, 'comment_id': comment_id if isinstance(comment_id, str) and re.fullmatch(r'[0-9a-fA-F]{24}', comment_id) else None,
            # Record the actual routing we submitted; acceptance does not prove
            # that a notification appeared on the recipient's device.
            'request_target_id': target_id, 'request_target_type': target_type,
            'request_reply_user_id': requester_user_id,
            'request_content_sha256': hashlib.sha256(wire_content.encode('utf-8')).hexdigest(),
            'reviewed_content_sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(),
            'request_format': 'sdk_plain_reply_prefix',
            'notification_status': 'unverified',
            'comment_hidden': comment.get('Hidden') if isinstance(comment, dict) and type(comment.get('Hidden')) is bool else None}


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
        # Common Chinese keywords (LLM / user input).
        if ("热门" in s) or ("最热" in s) or ("热度" in s):
            return "Popularity"
        if ("随机" in s) or ("乱序" in s):
            return "Random"
        if ("最新" in s) or ("最近" in s):
            return 0
        low = s.casefold()
        if low.isdigit():
            try:
                return int(low, 10)
            except Exception:
                return 0
        # The backend accepts these string values (case-insensitive):
        # - Default / Popularity / Random
        if low in ("default", "popularity", "random"):
            return s
        # Substring fallbacks for common phrases (e.g. "most popular", "history hot", "latest").
        if any(k in low for k in ("newest", "latest", "recent", "new ", "time")):
            return 0
        if any(k in low for k in ("popular", "popularity", "hot", "hottest", "trend")):
            return "Popularity"
        if any(k in low for k in ("random", "shuffle", "rand")):
            return "Random"
        # Common synonyms produced by LLMs; map them to supported values to avoid backend 500.
        if low in ("newest", "latest", "recent", "new", "time"):
            return 0
        if low in ("hot", "popular", "hottest", "trending"):
            return "Popularity"
        if low in ("rand", "shuffle"):
            return "Random"
        # Unknown sort strings can crash the backend; fall back to default ordering.
        return 0

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
    # Backend rejects take > 24 (400 Input.Field.Invalid).
    take_i = min(take_i, 24)
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

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    can_http = isinstance(token, str) and token.strip() and isinstance(auth_code, str) and auth_code.strip()

    wrapper = getattr(user, "get_relations", None)
    if callable(wrapper) and dt in (0, 1):
        # physicsLab.web.User.get_relations expects display_type as "Follower"|"Following",
        # while some wrappers accept numeric codes. Try both forms.
        last: Exception | None = None
        variants: list[str | int] = []
        if isinstance(display_type, str) and display_type.strip():
            variants.append(display_type.strip())
        variants.append("Follower" if dt == 0 else "Following")
        variants.extend([dt, str(dt)])
        seen: set[str] = set()
        for dt_variant in variants:
            k = str(dt_variant)
            if k in seen:
                continue
            seen.add(k)
            try:
                return _extract_users(wrapper(user_id=uid, display_type=dt_variant, skip=skip_i, take=take_i, query=q))
            except Exception as e:
                last = e
        # If wrapper failed, fall back to HTTP when possible.
        if not can_http:
            raise PLARError(f"get_relations failed via wrapper: {last}") from last

    if not can_http:
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


def _authenticated_headers(user: Any) -> dict[str, str]:
    real = unwrap_user(user)
    token, auth = getattr(real, "token", None), getattr(real, "auth_code", None)
    if not isinstance(token, str) or not token or not isinstance(auth, str) or not auth:
        raise PLARError("Publication requires an authenticated account")
    return {"x-API-Token": token, "x-API-AuthCode": auth}


def _publication_headers(user: Any) -> dict[str, str]:
    """Compatibility view of the official submission headers for diagnostics."""
    return {**_authenticated_headers(user), "x-API-Version": official_publish_api._version()}


def _official_call(operation: str, callback) -> dict[str, Any]:
    """Normalize the vendored official SDK result without exposing credentials."""
    try:
        value = callback()
    except Exception as error:
        code = getattr(error, "err_code", None)
        if isinstance(code, int):
            raise PublicationResponseError(f"{operation} was rejected by the official API",
                                           f"{operation}:api-{code}") from error
        raise PublicationResponseError(f"{operation} did not confirm success",
                                       f"{operation}:transport-unconfirmed") from error
    if not isinstance(value, dict) or value.get("Status") != 200:
        raise PublicationResponseError(f"{operation} did not confirm success",
                                       f"{operation}:api-status-not-success")
    return value


def submit_original_experiment(
    user: Any, *, source: dict[str, Any], title: str, introduction: str,
    cover_bytes: int,
) -> dict[str, Any]:
    """Create one new free Experiment (EXTERNAL WRITE).

    Internal building block for the server-side publication ledger, not an agent tool.
    Keeps the source StatusSave/CameraSave strings intact and sends it with the
    vendored official SDK request implementation; never invokes SDK eval.
    Electrical Type-0 requests one trusted cover slot. The fixed Type-3
    HDL-source carrier is text-only: no image Request and no image credential.
    A network exception is ambiguous: callers MUST NOT blindly retry SubmitExperiment.
    Image credentials in the return value are INTERNAL and must not reach the model.
    """
    experiment = source.get("Experiment") if isinstance(source, dict) else None
    kind = experiment.get("Type") if isinstance(experiment, dict) else None
    if type(kind) is not int or kind not in (0, 3) or type(source.get("Type")) is not int or source["Type"] != kind:
        raise PLARError("Publishing requires an explicit matching Type-0 or fixed Type-3 experiment")
    original_summary = source.get("Summary")
    if not isinstance(original_summary, dict) or original_summary.get("ID") or experiment.get("ID"):
        raise PLARError("Publishing only accepts a new original experiment, never an existing community ID")
    if type(cover_bytes) is not int or (kind == 0 and not 0 < cover_bytes <= 1024 * 1024) or (kind == 3 and cover_bytes != 0):
        raise PLARError("Type-0 requires one JPEG cover; fixed Type-3 is title/body only and forbids images")
    if kind == 3:
        try:
            status = json.loads(experiment["StatusSave"])
            camera = json.loads(experiment["CameraSave"])
        except (KeyError, TypeError, ValueError) as error:
            raise PLARError("Fixed Type-3 template is invalid") from error
        required_status = official_publish_api.hdl_source_carrier_status()
        required_camera = {"Mode": 2, "Distance": 2.75, "VisionCenter": "0,1.08,0",
                           "TargetRotation": "90,0,0"}
        if (status != required_status or camera != required_camera or experiment.get("Components") != 3
            or experiment.get("Version") != 2503 or original_summary.get("Type") != 3
            or original_summary.get("Tags") != ["Type-3", "高中", "教学实验"]
            or source.get("InternalName") != "Aurex HDL 源码载体"):
            raise PLARError("Only the exact fixed official Type-3 HDL-source carrier is accepted")
    real = unwrap_user(user)
    if not getattr(real, "user_id", None):
        raise PLARError("Publication requires a registered account ID")
    summary = copy.deepcopy(original_summary)
    summary.update({"ID": None, "ContentID": None, "Category": "Experiment", "Subject": title,
        "Description": introduction.split("\n"), "Language": "Chinese", "Price": 0,
        "Image": 0, "ImageRegion": 0, "Version": int(official_publish_api._version()), "Type": kind,
        "Tags": (["Type-3", "高中", "教学实验"] if kind == 3 else ["Type-0"]),
        "ParentID": None, "ParentName": None, "ParentCategory": None,
        "Coauthors": [], "Editor": None, "Anonymous": False,
        "CreationDate": int(time.time() * 1000), "UpdateDate": 0,
        **{k: 0 for k in ("Visits", "Stars", "Supports", "Remixes", "Comments", "Popularity")}})
    if kind == 3:
        # The official Celestial template does not define this Circuit-only
        # field; retaining it makes SubmitExperiment reject an otherwise exact
        # no-image Type-3 payload.
        summary.pop("Anonymous", None)
    summary["User"] = {key: getattr(real, attr, default) for key, attr, default in (
        ("ID", "user_id", None), ("Nickname", "nickname", ""), ("Signature", "signature", ""),
        ("Avatar", "avatar", 0), ("AvatarRegion", "avatar_region", 0),
        ("Decoration", "decoration", 0), ("Verification", "verification", None))}
    workspace = copy.deepcopy(source)
    workspace["Summary"] = None
    payload = {"Summary": summary, "Workspace": workspace}
    if kind == 0:
        payload["Request"] = {"FileSize": cover_bytes, "Extension": ".jpg"}
    value = _official_call("SubmitExperiment",
        lambda: official_publish_api.submit_experiment(real, payload))
    data = value.get("Data")
    summary_id = data.get("Summary", {}).get("ID") if isinstance(data, dict) else None
    if not isinstance(summary_id, str) or not re.fullmatch(r"[0-9a-fA-F]{24}", summary_id):
        raise PublicationResponseError("SubmitExperiment returned no valid summary ID; do not retry submission",
                                       "SubmitExperiment:missing-summary-id")
    token = data.get("Token")
    credential = {k: token.get(k) for k in ("Policy", "Authorization")} if isinstance(token, dict) else {}
    result = {"summary_id": summary_id, "category": "Experiment", "image_counter": 1 if kind == 0 else 0}
    if kind == 0:
        result["_cover_credential"] = credential
    return result


def upload_experiment_cover(*, user: Any, cover: bytes, credential: dict[str, Any]) -> None:
    """Upload the allocated cover slot through the official SDK API."""

    if not isinstance(cover, bytes) or not 0 < len(cover) <= 1024 * 1024:
        raise PLARError("Invalid cover image size")
    if not all(isinstance(credential.get(k), str) and credential[k] for k in ("Policy", "Authorization")):
        raise PLARError("The server did not grant a cover upload slot; publication remains unconfirmed")
    _official_call("UploadImage", lambda: official_publish_api.upload_image(
        user, credential["Policy"], credential["Authorization"], cover))


def confirm_original_experiment(user: Any, *, summary_id: str, image_counter: int = 1) -> None:
    """Confirm the SAME submitted experiment through the official SDK API."""

    if not re.fullmatch(r"[0-9a-fA-F]{24}", summary_id) or image_counter not in (0, 1):
        raise PLARError("Invalid publication confirmation receipt")
    _official_call("ConfirmExperiment",
        lambda: official_publish_api.confirm_experiment(user, summary_id, image_counter))


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
        # GetExperiment expects the content ID, not the community summary ID.
        # Match physicsLab.web._api.User.get_experiment(..., category)'s resolution.
        summary = _extract_data_block(get_summary(user, summary_id=summary_id, category_value=category_value), op="GetSummary")
        content_id = summary.get("ContentID")
        if not isinstance(content_id, str) or not content_id:
            raise PLARError("GetSummary returned no ContentID for the experiment")
        status_code, data = post_json_no_env_proxy(
            url=f"{_PLAR_BASE_CN}/Contents/GetExperiment",
            payload={"ContentID": content_id},
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


def get_experiment_file(
    user: Any, *, summary_id: str, category_value: str, cache_dir: str,
) -> dict[str, Any]:
    """Save an immutable original electrical PlSav for circuit_inspect/analyze.

    Read the raw API data without constructing SDK Experiment objects: the installed
    SDK evaluates camera coordinate strings and injects template fields on loading.
    Preserve all original experiment fields, including the exact StatusSave string;
    only the outer JSON encoding is generated locally. Never infer a missing Type.
    """
    if category_value not in ("Experiment", "Discussion"):
        raise PLARError("category_value must be Experiment or Discussion")
    if not re.fullmatch(r"[0-9a-fA-F]{24}", summary_id):
        raise PLARError("summary_id must be a complete 24-hex ID")
    summary = _extract_data_block(get_summary(user, summary_id=summary_id, category_value=category_value), op="GetSummary")
    response = get_experiment(user, summary_id=summary_id, category_value=category_value)
    data = _extract_data_block(response, op="GetExperiment")
    original = _find_plsav_like(data)
    if original is None:
        raise PLARError("GetExperiment returned no PlSav experiment/StatusSave")
    if isinstance(original.get("Experiment"), dict):
        plsav = dict(original)
        if not isinstance(plsav.get("Summary"), dict):
            plsav["Summary"] = summary
    else:
        plsav = {"Experiment": original, "Summary": summary}
    experiment = plsav["Experiment"]
    kind = experiment.get("Type")
    if not isinstance(kind, int) or isinstance(kind, bool) or kind != 0:
        raise PLARError(f"Only electrical experiments with explicit Type=0 can enter Phy-Engine; actual Type={kind!r}")
    status_raw = experiment.get("StatusSave")
    try:
        status = json.loads(status_raw) if isinstance(status_raw, str) else status_raw
    except ValueError as exc:
        raise PLARError("Original StatusSave is not valid JSON") from exc
    if not isinstance(status, dict) or not isinstance(status.get("Elements"), list) or not isinstance(status.get("Wires"), list):
        raise PLARError("Electrical StatusSave must contain original Elements and Wires lists")
    payload = (json.dumps(plsav, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > 32 * 1024 * 1024:
        raise PLARError("Original experiment exceeds the 32 MiB circuit input limit")
    checksum = hashlib.sha256(payload).hexdigest()
    folder = os.path.join(os.path.abspath(cache_dir), "plar_experiments")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{category_value.lower()}-{summary_id}-{checksum}.sav")
    with tempfile.NamedTemporaryFile(prefix=".download-", suffix=".sav", dir=folder, delete=False) as output:
        temporary = output.name
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    try:
        # Publish a complete file atomically without replacing an existing artifact.
        try:
            os.link(temporary, path)
        except FileExistsError:
            with open(path, "rb") as saved:
                if hashlib.sha256(saved.read()).hexdigest() != checksum:
                    raise PLARError("Existing content-addressed experiment was modified; refusing to overwrite")
    finally:
        os.unlink(temporary)
    return {
        "sav_path": path, "summary_id": summary_id, "category": category_value,
        "content_id": summary.get("ContentID"), "experiment_type": kind,
        "is_electrical": True, "bytes": len(payload), "sha256": checksum,
        "elements": len(status["Elements"]), "wires": len(status["Wires"]),
        "elements_with_original_position": sum(isinstance(x, dict) and "Position" in x for x in status["Elements"]),
        "source": "PhysicsLab raw GetSummary/GetExperiment API",
        "fidelity": "Original API fields and StatusSave preserved; outer JSON re-encoded without layout changes",
        "external_write_performed": False,
    }


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
