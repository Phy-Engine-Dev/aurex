from __future__ import annotations

import os
import sys
from typing import Any, Iterable
import json
import time


class PLARError(RuntimeError):
    pass


_DEFAULT_HTTP_TIMEOUT_SEC = 60.0
_requests_default_timeout_sec = _DEFAULT_HTTP_TIMEOUT_SEC
_requests_timeout_patched = False


def configure_requests_default_timeout(timeout_sec: float) -> None:
    global _requests_default_timeout_sec, _requests_timeout_patched
    _requests_default_timeout_sec = float(timeout_sec)

    if _requests_timeout_patched:
        return

    try:
        import requests
    except ImportError:
        return

    orig_request = requests.sessions.Session.request

    def request_with_default_timeout(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = _requests_default_timeout_sec
        return orig_request(self, method, url, **kwargs)

    requests.sessions.Session.request = request_with_default_timeout  # type: ignore[assignment]
    _requests_timeout_patched = True


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _unwrap_physicslab_user(user: Any) -> Any:
    """Best-effort unwrap for thin user proxies.

    The agent may wrap the PhysicsLab User object to serialize access across threads.
    The upstream physicsLab library does strict `isinstance(user, User)` checks, so we
    pass through the underlying object when available.
    """
    inner = getattr(user, "_user", None)
    return inner if inner is not None else user


def _vendored_physicslab_dir() -> str:
    return os.path.join(repo_root(), "third-parties", "physicsLab")


def ensure_physicslab_importable(*, cache_dir: str, http_timeout_sec: float | None = None) -> None:
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Ensure PhysicsLab local save path is controlled.
    os.environ["PHYSICSLAB_HOME_PATH"] = os.path.join(cache_dir, "physicsLabSav")
    configure_requests_default_timeout(
        _DEFAULT_HTTP_TIMEOUT_SEC if http_timeout_sec is None else float(http_timeout_sec)
    )

    try:
        import physicsLab  # noqa: F401

        return
    except ImportError:
        vendored = _vendored_physicslab_dir()
        if os.path.isdir(vendored) and vendored not in sys.path:
            sys.path.insert(0, vendored)

    try:
        import physicsLab  # noqa: F401
    except ImportError as e:
        raise PLARError(
            "Could not import 'physicsLab'. Install it with pip or ensure "
            "'third-parties/physicsLab' exists."
        ) from e


def email_login(
    *,
    email: str,
    password: str,
    cache_dir: str,
    http_timeout_sec: float | None = None,
) -> Any:
    ensure_physicslab_importable(cache_dir=cache_dir, http_timeout_sec=http_timeout_sec)
    from physicsLab import web

    return web.email_login(email, password)


def get_comments(
    user: Any,
    *,
    target_id: str,
    target_type: str,
    take: int,
    skip: int = 0,
) -> list[dict[str, Any]]:
    result = user.get_comments(
        target_id=target_id,
        target_type=target_type,
        take=take,
        skip=int(skip),
    )
    data = result.get("Data")
    if not isinstance(data, dict):
        raise PLARError("Unexpected get_comments response: missing Data object")
    comments = data.get("Comments")
    if not isinstance(comments, list):
        raise PLARError("Unexpected get_comments response: missing Data.Comments list")
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
    # plweb2 typing hints suggest `Take` is effectively capped (commonly 24).
    # Some servers reject larger values with `Input.Field.Invalid`.
    take = int(take)
    if take <= 0:
        take = 20
    if take > 24:
        take = 24
    skip = int(skip)
    if skip < 0:
        skip = 0
    if isinstance(from_skip, str) and not from_skip.strip():
        from_skip = None
    if isinstance(user_id, str) and not user_id.strip():
        user_id = None
    if tags is not None and not isinstance(tags, list):
        raise PLARError("tags must be a list[str] or null")
    if exclude_tags is not None and not isinstance(exclude_tags, list):
        raise PLARError("exclude_tags must be a list[str] or null")
    if languages is not None and not isinstance(languages, list):
        raise PLARError("languages must be a list[str] or null")
    if exclude_languages is not None and not isinstance(exclude_languages, list):
        raise PLARError("exclude_languages must be a list[str] or null")

    def _sort_variants(v: int | str | None) -> list[int | str]:
        if v is None:
            return [0]
        if isinstance(v, int):
            return [v]
        s = str(v).strip()
        if not s:
            return [0]
        low = s.casefold()
        mapping: dict[str, int] = {
            "default": 0,
            "popularity": 1,
            "popular": 1,
            "hot": 1,
            "random": 2,
        }
        if low in mapping:
            return [s, mapping[low]]
        return [s]

    def _days_variants(v: int | str | None) -> list[int | str]:
        if v is None:
            return [0]
        if isinstance(v, int):
            if v < 0:
                return [0]
            # App/web sometimes sends Days as a string.
            return [str(v), v]
        s = str(v).strip()
        if not s:
            return [0]
        if s.isdigit():
            return [s, int(s)]
        return [s]

    def _extract_values(result: Any) -> list[dict[str, Any]]:
        if not isinstance(result, dict):
            raise PLARError(f"Unexpected query_experiments response type: {type(result).__name__}")
        status = result.get("Status")
        message = result.get("Message")
        data = result.get("Data")
        if data is None and "data" in result:
            data = result.get("data")

        # plweb2 expects Result{Status,Message,Data}. Data can be null on error.
        if status is not None:
            try:
                st = int(status)
            except Exception:
                st = None
            if st is not None and st != 200:
                msg = str(message or "").strip()
                raise PLARError(f"QueryExperiments failed (status={st}): {msg}".rstrip())

        if data is None:
            return []
        if isinstance(data, list):
            return [v for v in data if isinstance(v, dict)]
        if not isinstance(data, dict):
            return []

        values = data.get("$values")
        if values is None:
            values = data.get("values")
        if values is None:
            values = data.get("Values")
        if values is None:
            return []
        if not isinstance(values, list):
            return []
        return [v for v in values if isinstance(v, dict)]

    def _direct_http_query() -> Any:
        token = getattr(user, "token", None)
        auth_code = getattr(user, "auth_code", None)
        if not isinstance(token, str) or not token.strip() or not isinstance(auth_code, str) or not auth_code.strip():
            raise PLARError("token/auth_code are missing for direct QueryExperiments")
        cat_val = getattr(category, "value", category)
        try:
            import requests  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise PLARError(
                "Missing dependency: requests (required for Physics Lab API calls). "
                "Install it with pip (e.g. 'pip install requests')."
            ) from e

        def _post(
            *,
            exclude_languages_value: Any,
            exclude_tags_value: Any,
            tags_value: Any,
            languages_value: Any,
            from_value: Any,
            skip_value: int,
            days_value: int | str,
            sort_value: int | str,
        ) -> Any:
            resp = requests.post(
                "https://physics-api-cn.turtlesim.com/Contents/QueryExperiments",
                json={
                    "Query": {
                        "Category": cat_val,
                        "Languages": languages_value,
                        # Some servers are picky about null vs [], so we allow retries.
                        "ExcludeLanguages": exclude_languages_value,
                        "Tags": tags_value,
                        "ExcludeTags": exclude_tags_value,
                        "ModelTags": None,
                        "ModelID": None,
                        "ParentID": None,
                        "UserID": user_id,
                        "Special": None,
                        "From": from_value,
                        "Skip": int(skip_value),
                        "Take": int(take),
                        "Days": days_value,
                        "Sort": sort_value,
                        "ShowAnnouncement": False,
                    }
                },
                headers={
                    "Content-Type": "application/json",
                    "x-API-Token": token,
                    "x-API-AuthCode": auth_code,
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) phy_lab/1.0",
                },
                timeout=_requests_default_timeout_sec,
            )
            resp.raise_for_status()
            return resp.json()

        # Try to follow plweb2 first (null exclude fields).
        exclude_lang_variants: list[Any]
        if exclude_languages is not None:
            exclude_lang_variants = [exclude_languages]
        else:
            exclude_lang_variants = [None, []]

        exclude_tag_variants: list[Any]
        if exclude_tags is not None:
            exclude_tag_variants = [exclude_tags]
        else:
            exclude_tag_variants = [None, []]

        lang_list = languages if languages is not None else []
        language_variants: list[Any] = [lang_list, []] if lang_list else [[]]

        # Many clients send Tags as [] when no tag filtering is intended.
        # Some servers accept null; we keep a compatibility fallback when tags is not specified.
        if tags is None:
            tags_variants: list[Any] = [None, [], []]
        else:
            # If caller requested tags filtering, do not silently change semantics.
            tags_variants = [tags]

        page_variants = [
            (from_skip, int(skip)),
            (None, int(skip)),
            (from_skip, 0),
            (None, 0),
        ]
        last: Any = None
        sort_variants = _sort_variants(sort)
        days_variants = _days_variants(days)
        for from_value, skip_value in page_variants:
            for tags_value in tags_variants:
                for languages_value in language_variants:
                    for ex_langs in exclude_lang_variants:
                        for ex_tags in exclude_tag_variants:
                            for days_value in days_variants:
                                for sort_value in sort_variants:
                                    last = _post(
                                        exclude_languages_value=ex_langs,
                                        exclude_tags_value=ex_tags,
                                        tags_value=tags_value,
                                        languages_value=languages_value,
                                        from_value=from_value,
                                        skip_value=int(skip_value),
                                        days_value=days_value,
                                        sort_value=sort_value,
                                    )
                                    if not isinstance(last, dict):
                                        continue
                                    st = last.get("Status")
                                    msg = str(last.get("Message") or "")
                                    if st == 400 and ("Input." in msg and "Invalid" in msg):
                                        continue
                                    return last
        return last

    # Prefer direct HTTP when possible to ensure request shape matches plweb2 (null vs []).
    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    if isinstance(token, str) and token.strip() and isinstance(auth_code, str) and auth_code.strip():
        return _extract_values(_direct_http_query())

    qe = getattr(user, "query_experiments", None)
    if callable(qe):
        # NOTE: The upstream API expects `Query.Tags` to be an array. Passing null can
        # cause `Input.Field.Invalid` (400). plweb2 always sends an array, even when
        # it's empty.
        try:
            result = qe(
                category=category,
                tags=tags or [],
                exclude_tags=exclude_tags,
                languages=languages or [],
                exclude_languages=exclude_languages,
                user_id=user_id,
                take=take,
                skip=skip,
                from_skip=from_skip,
            )
        except Exception:
            # Fall back to a direct HTTP call (below) if the wrapper is buggy or
            # rejects the request for any reason.
            result = None
    else:
        # Defensive fallback: some wrappers/mocks may expose a non-callable attribute with the
        # same name. In that case, call the underlying HTTP API directly using the user's
        # token/auth_code.
        try:
            result = _direct_http_query()
        except PLARError:
            raise
    if result is None:
        # Wrapper call failed; retry with a direct HTTP request.
        result = _direct_http_query()
    return _extract_values(result)


def get_user_by_name(
    user: Any,
    *,
    name: str,
) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise PLARError("name is empty")

    def _extract_data(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise PLARError(f"Unexpected get_user response type: {type(result).__name__}")
        status = result.get("Status")
        if status is not None:
            try:
                st = int(status)
            except Exception:
                st = None
            if st is not None and st != 200:
                msg = str(result.get("Message") or "").strip()
                raise PLARError(f"GetUser failed (status={st}): {msg}".rstrip())
        data = result.get("Data")
        if not isinstance(data, dict):
            raise PLARError("Unexpected get_user response: missing Data object")
        return data

    fn = getattr(user, "get_user_by_name", None)
    if callable(fn):
        return _extract_data(fn(name))

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    if not isinstance(token, str) or not token.strip() or not isinstance(auth_code, str) or not auth_code.strip():
        raise PLARError("get_user_by_name is not callable and token/auth_code are missing")
    try:
        import requests  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise PLARError(
            "Missing dependency: requests (required for Physics Lab API calls). "
            "Install it with pip (e.g. 'pip install requests')."
        ) from e
    resp = requests.post(
        "https://physics-api-cn.turtlesim.com/Users/GetUser",
        json={"Name": name},
        headers={
            "Content-Type": "application/json",
            "x-API-Token": token,
            "x-API-AuthCode": auth_code,
        },
        timeout=_requests_default_timeout_sec,
    )
    resp.raise_for_status()
    return _extract_data(resp.json())


def get_user_by_id(
    user: Any,
    *,
    user_id: str,
) -> dict[str, Any]:
    user_id = (user_id or "").strip()
    if not user_id:
        raise PLARError("user_id is empty")

    def _extract_data(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise PLARError(f"Unexpected get_user response type: {type(result).__name__}")
        status = result.get("Status")
        if status is not None:
            try:
                st = int(status)
            except Exception:
                st = None
            if st is not None and st != 200:
                msg = str(result.get("Message") or "").strip()
                raise PLARError(f"GetUser failed (status={st}): {msg}".rstrip())
        data = result.get("Data")
        if not isinstance(data, dict):
            raise PLARError("Unexpected get_user response: missing Data object")
        return data

    fn = getattr(user, "get_user_by_id", None)
    if callable(fn):
        return _extract_data(fn(user_id))

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    if not isinstance(token, str) or not token.strip() or not isinstance(auth_code, str) or not auth_code.strip():
        raise PLARError("get_user_by_id is not callable and token/auth_code are missing")
    try:
        import requests  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise PLARError(
            "Missing dependency: requests (required for Physics Lab API calls). "
            "Install it with pip (e.g. 'pip install requests')."
        ) from e
    resp = requests.post(
        "https://physics-api-cn.turtlesim.com/Users/GetUser",
        json={"ID": user_id},
        headers={
            "Content-Type": "application/json",
            "x-API-Token": token,
            "x-API-AuthCode": auth_code,
        },
        timeout=_requests_default_timeout_sec,
    )
    resp.raise_for_status()
    return _extract_data(resp.json())


def get_relations(
    user: Any,
    *,
    user_id: str,
    display_type: str | int = "Following",
    skip: int = 0,
    take: int = 20,
    query: str = "",
) -> list[dict[str, Any]]:
    """Fetch user's relations list (best-effort).

    display_type can be:
      - Names: "Follower", "Following", "Banned", "Volunteer", "Editor", "Emeritus"
      - Codes: 0..5 (plweb2-compatible):
          0=Follower, 1=Following, 2=Banned, 3=Volunteer, 4=Editor/Admins, 5=Emeritus/Retired
    """
    user_id = (user_id or "").strip()
    if not user_id:
        raise PLARError("user_id is empty")
    if isinstance(display_type, int):
        display_type_code = int(display_type)
    else:
        dt = (str(display_type or "Following") or "Following").strip()
        # Accept numeric strings too.
        try:
            display_type_code = int(dt)
        except Exception:
            display_type_code = -1
        if display_type_code < 0:
            mapping = {
                "follower": 0,
                "followers": 0,
                "following": 1,
                "followings": 1,
                "banned": 2,
                "baned": 2,
                "blocked": 2,
                "volunteer": 3,
                "volunteers": 3,
                "editor": 4,
                "editors": 4,
                "admin": 4,
                "admins": 4,
                "administrator": 4,
                "administrators": 4,
                "emeritus": 5,
                "retired": 5,
            }
            display_type_code = mapping.get(dt.casefold(), -1)
    if display_type_code not in (0, 1, 2, 3, 4, 5):
        raise PLARError("display_type must be a known name or an integer 0..5")
    skip = int(skip)
    if skip < 0:
        skip = 0
    take = int(take)
    if take <= 0:
        take = 20
    if take > 100:
        take = 100
    query = (query or "").strip()

    def _extract_users(result: Any) -> list[dict[str, Any]]:
        if not isinstance(result, dict):
            raise PLARError(f"Unexpected get_relations response type: {type(result).__name__}")
        status = result.get("Status")
        if status is not None:
            try:
                st = int(status)
            except Exception:
                st = None
            if st is not None and st != 200:
                msg = str(result.get("Message") or "").strip()
                raise PLARError(f"GetRelations failed (status={st}): {msg}".rstrip())
        data = result.get("Data")
        if data is None and "data" in result:
            data = result.get("data")

        if isinstance(data, dict):
            values = data.get("$values")
            if isinstance(values, list):
                return [x for x in values if isinstance(x, dict)]
            # Some wrappers might return users under other keys.
            for k in ("Users", "users", "Relations", "relations"):
                v = data.get(k)
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    fn = getattr(user, "get_relations", None)
    if callable(fn):
        last_e: Exception | None = None
        for dt_variant in (display_type_code, str(display_type_code)):
            try:
                return _extract_users(
                    fn(user_id=user_id, display_type=dt_variant, skip=skip, take=take, query=query)
                )
            except Exception as e:
                last_e = e
        raise PLARError(f"get_relations failed via user.get_relations: {last_e}") from last_e

    token = getattr(user, "token", None)
    auth_code = getattr(user, "auth_code", None)
    if not isinstance(token, str) or not token.strip() or not isinstance(auth_code, str) or not auth_code.strip():
        raise PLARError("get_relations is not callable and token/auth_code are missing")
    try:
        import requests  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise PLARError(
            "Missing dependency: requests (required for Physics Lab API calls). "
            "Install it with pip (e.g. 'pip install requests')."
        ) from e
    resp = requests.post(
        "https://physics-api-cn.turtlesim.com/Users/GetRelations",
        json={
            "UserID": user_id,
            "DisplayType": display_type_code,
            "Skip": skip,
            "Take": take,
            "Query": query,
        },
        headers={
            "Content-Type": "application/json",
            "x-API-Token": token,
            "x-API-AuthCode": auth_code,
        },
        timeout=_requests_default_timeout_sec,
    )
    resp.raise_for_status()
    return _extract_users(resp.json())


def get_messages(
    user: Any,
    *,
    category_id: int,
    skip: int = 0,
    take: int = 20,
    no_templates: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch notification messages.

    Returns:
        (messages, templates)
    """
    result = user.get_messages(
        category_id=category_id,
        skip=skip,
        take=take,
        no_templates=no_templates,
    )
    data = result.get("Data")
    if not isinstance(data, dict):
        raise PLARError("Unexpected get_messages response: missing Data object")

    messages = data.get("Messages")
    templates = data.get("Templates", [])
    if not isinstance(messages, list):
        raise PLARError("Unexpected get_messages response: missing Data.Messages list")
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
    from physicsLab import Category, Experiment, OpenMode

    real_user = _unwrap_physicslab_user(user)

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

    # Use the internal upload to retrieve the SummaryID for reporting.
    try:
        submit_response, submit_data = exp._Experiment__upload(real_user, category, None)  # type: ignore[attr-defined]
    except Exception as e:
        raise PLARError(f"SubmitExperiment failed: {e}") from e

    if not isinstance(submit_response, dict):
        raise PLARError("SubmitExperiment returned unexpected response type")
    status = submit_response.get("Status")
    if status is not None:
        try:
            st = int(status)
        except Exception:
            st = None
        if st is not None and st != 200:
            msg = str(submit_response.get("Message") or "").strip()
            raise PLARError(f"SubmitExperiment failed (status={st}): {msg}".rstrip())

    data = submit_response.get("Data")
    if not isinstance(data, dict):
        raise PLARError("SubmitExperiment returned no Data object")
    summary = data.get("Summary")
    if not isinstance(summary, dict):
        raise PLARError("SubmitExperiment returned no Data.Summary object")
    summary_id = summary.get("ID")
    if not isinstance(summary_id, str) or not summary_id.strip():
        raise PLARError("SubmitExperiment returned missing Data.Summary.ID")

    image_counter = None
    if isinstance(submit_data, dict):
        s2 = submit_data.get("Summary")
        if isinstance(s2, dict):
            image_counter = s2.get("Image")
    if not isinstance(image_counter, int):
        image_counter = 0

    try:
        real_user.confirm_experiment(summary_id, category, image_counter)
    except Exception as e:
        raise PLARError(f"ConfirmExperiment failed: {e}") from e
    return {
        "summary_id": summary_id.strip(),
        "category": category.value,
    }


def _apply_publish_tags(exp: Any, tags: list[str]) -> None:
    tags = [t.strip() for t in (tags or []) if isinstance(t, str) and t.strip()]
    if not tags:
        return
    try:
        from physicsLab import Tag
    except Exception:
        Tag = None  # type: ignore[assignment]

    enum_tags = []
    raw_tags: list[str] = []
    for t in tags:
        mapped = None
        if Tag is not None:
            # Accept either Tag enum name (e.g. "SmallProject") or Tag value (e.g. "小作品").
            for candidate in (t, t.replace("Tag.", "", 1)):
                key = candidate.strip()
                if not key:
                    continue
                try:
                    mapped = Tag[key]  # type: ignore[index]
                    break
                except Exception:
                    mapped = None
                try:
                    mapped = next((x for x in Tag if getattr(x, "value", None) == key), None)
                    if mapped is not None:
                        break
                except Exception:
                    mapped = None
        if mapped is not None:
            enum_tags.append(mapped)
        else:
            raw_tags.append(t)

    if enum_tags:
        try:
            exp.edit_tags(*enum_tags)
        except Exception:
            raw_tags.extend([getattr(t, "value", None) for t in enum_tags if getattr(t, "value", None)])

    # Do NOT inject unknown/raw tags into PlSav Summary: server-side validation may reject them.
    # Unknown tags are ignored.


def get_summary(user: Any, *, summary_id: str, category_value: str) -> dict[str, Any]:
    try:
        from physicsLab import Category
    except Exception as e:  # pragma: no cover
        raise PLARError(f"Failed to import physicsLab.Category: {e}") from e

    if category_value == "Experiment":
        category = Category.Experiment
    elif category_value == "Discussion":
        category = Category.Discussion
    else:
        raise PLARError("category_value must be 'Experiment' or 'Discussion'")

    result = user.get_summary(summary_id, category)
    if not isinstance(result, dict):
        raise PLARError("Unexpected get_summary response type")
    return result


def get_experiment(user: Any, *, summary_id: str, category_value: str) -> dict[str, Any]:
    try:
        from physicsLab import Category
    except Exception as e:  # pragma: no cover
        raise PLARError(f"Failed to import physicsLab.Category: {e}") from e

    if category_value == "Experiment":
        category = Category.Experiment
    elif category_value == "Discussion":
        category = Category.Discussion
    else:
        raise PLARError("category_value must be 'Experiment' or 'Discussion'")

    result = user.get_experiment(summary_id, category)
    if not isinstance(result, dict):
        raise PLARError("Unexpected get_experiment response type")
    return result


def _safe_json_dumps(value: Any, *, max_chars: int) -> str:
    try:
        s = json.dumps(value, ensure_ascii=False, sort_keys=False)
    except TypeError:
        s = str(value)
    if max_chars > 0 and len(s) > max_chars:
        return s[: max_chars - 1] + "…"
    return s


def _recursive_find_plsav_like(obj: Any, *, depth: int = 0, max_depth: int = 6) -> dict[str, Any] | None:
    if depth > max_depth:
        return None
    if isinstance(obj, dict):
        # Heuristic: a Physics Lab .sav root often has Summary+Experiment, or Experiment contains StatusSave/CameraSave.
        if "Experiment" in obj and isinstance(obj.get("Experiment"), dict):
            exp = obj.get("Experiment")
            if isinstance(exp, dict) and ("StatusSave" in exp or "CameraSave" in exp):
                return obj
        if "StatusSave" in obj or "CameraSave" in obj:
            return obj
        for v in obj.values():
            found = _recursive_find_plsav_like(v, depth=depth + 1, max_depth=max_depth)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _recursive_find_plsav_like(v, depth=depth + 1, max_depth=max_depth)
            if found is not None:
                return found
    return None


def _summarize_plsav(plsav: dict[str, Any]) -> dict[str, Any]:
    # Keep this summary small and stable: counts and top-level metadata only.
    exp = plsav.get("Experiment")
    summary = plsav.get("Summary")
    out: dict[str, Any] = {}

    if isinstance(summary, dict):
        subject = summary.get("Subject")
        desc = summary.get("Description")
        out["subject"] = subject if isinstance(subject, str) else None
        out["description"] = desc if isinstance(desc, list) else None
        tags = summary.get("Tags")
        out["tags"] = tags if isinstance(tags, list) else None

    if isinstance(exp, dict):
        out["type"] = exp.get("Type")
        status_save = exp.get("StatusSave")
        if isinstance(status_save, str):
            try:
                status = json.loads(status_save)
                elements = status.get("Elements")
                wires = status.get("Wires")
                out["elements_count"] = len(elements) if isinstance(elements, list) else None
                out["wires_count"] = len(wires) if isinstance(wires, list) else None
                if isinstance(elements, list):
                    model_counts: dict[str, int] = {}
                    for el in elements[:200]:
                        if not isinstance(el, dict):
                            continue
                        mid = el.get("ModelID")
                        if isinstance(mid, str) and mid:
                            model_counts[mid] = model_counts.get(mid, 0) + 1
                    top = sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
                    out["top_model_ids"] = [{"model_id": k, "count": v} for k, v in top]
            except Exception:
                out["elements_count"] = None
                out["wires_count"] = None
    else:
        # Sometimes the "plsav-like" dict is actually the Experiment subobject.
        status_save = plsav.get("StatusSave")
        if isinstance(status_save, str):
            try:
                status = json.loads(status_save)
                elements = status.get("Elements")
                wires = status.get("Wires")
                out["elements_count"] = len(elements) if isinstance(elements, list) else None
                out["wires_count"] = len(wires) if isinstance(wires, list) else None
                if isinstance(elements, list):
                    model_counts: dict[str, int] = {}
                    for el in elements[:200]:
                        if not isinstance(el, dict):
                            continue
                        mid = el.get("ModelID")
                        if isinstance(mid, str) and mid:
                            model_counts[mid] = model_counts.get(mid, 0) + 1
                    top = sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
                    out["top_model_ids"] = [{"model_id": k, "count": v} for k, v in top]
            except Exception:
                pass

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
    if os.path.exists(cache_path):
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

    # Preferred path: use physicsLab.Experiment(OpenMode.load_by_plar_app) to get a .sav-shaped object.
    try:
        from physicsLab import Category as PLCategory
        from physicsLab import Experiment, OpenMode

        category = PLCategory.Experiment if category_value == "Experiment" else PLCategory.Discussion
        exp = Experiment(
            OpenMode.load_by_plar_app,
            summary_id,
            category,
            user=_unwrap_physicslab_user(user),
        )
        plsav = exp.PlSav if isinstance(getattr(exp, "PlSav", None), dict) else None
        if isinstance(plsav, dict):
            summary_data = plsav.get("Summary")
            exp_data = plsav.get("Experiment")
            plsav_summary = _summarize_plsav(plsav)
    except Exception:
        # Fallback to raw API calls if the higher-level helper fails.
        summary_res = get_summary(user, summary_id=summary_id, category_value=category_value)
        exp_res = get_experiment(user, summary_id=summary_id, category_value=category_value)
        summary_data = summary_res.get("Data")
        exp_data = exp_res.get("Data")
        plsav_like = _recursive_find_plsav_like(exp_data) or _recursive_find_plsav_like(exp_res)
        plsav_summary = _summarize_plsav(plsav_like) if isinstance(plsav_like, dict) else {}

    title = _extract_title_from_obj(summary_data) or _extract_title_from_obj(exp_data)
    if not title and isinstance(plsav_summary, dict):
        title = str(plsav_summary.get("subject") or "").strip()

    # Best-effort text extraction for "正文/内容" style requests.
    # Keep these relatively small so we can always include them in LLM context.
    summary_text = _collect_text_recursive(summary_data, max_chars=6000)
    experiment_text = _collect_text_recursive(exp_data, max_chars=8000)

    author_id = ""
    author_nickname = ""
    if isinstance(summary_data, dict):
        u = summary_data.get("User")
        if isinstance(u, dict):
            author_id = best_effort_extract_text(u.get("ID")) or best_effort_extract_text(u.get("UserID"))
            author_nickname = (
                best_effort_extract_text(u.get("Nickname"))
                or best_effort_extract_text(u.get("Name"))
                or ""
            )

    context = {
        "context_version": context_version,
        "summary_id": summary_id,
        "category": category_value,
        # Friendly, structured fields for LLM prompting.
        "title": title or None,
        "author": {
            "id": author_id or None,
            "nickname": author_nickname or None,
        },
        # Aliases are intentional: different models/agents tend to prefer different words.
        "body_text": summary_text or None,
        "content_text": experiment_text or None,
        "summary_text": summary_text or None,
        "experiment_text": experiment_text or None,
        "summary_data_excerpt": _safe_json_dumps(summary_data, max_chars=max_json_chars),
        "experiment_data_excerpt": _safe_json_dumps(exp_data, max_chars=max_json_chars),
        "plsav_summary": plsav_summary,
    }

    try:
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, cache_path)
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
    """Fetch and parse the experiment's StatusSave JSON (Elements/Wires).

    This is used for local Phy-Engine simulation. Cached under cache_dir to reduce API calls.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_root = os.path.join(cache_dir, "plar_cache")
    os.makedirs(cache_root, exist_ok=True)
    cache_path = os.path.join(cache_root, f"status_{category_value.lower()}_{summary_id}.json")

    now = time.time()
    if os.path.exists(cache_path):
        try:
            st = os.stat(cache_path)
            if now - st.st_mtime <= ttl_sec:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, dict) and cached.get("summary_id") == summary_id:
                    data = cached.get("status_save")
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass

    ensure_physicslab_importable(cache_dir=cache_dir)
    try:
        from physicsLab import Category as PLCategory
        from physicsLab import Experiment, OpenMode
    except Exception as e:  # pragma: no cover
        raise PLARError(f"Failed to import physicsLab Experiment helpers: {e}") from e

    if category_value == "Experiment":
        category = PLCategory.Experiment
    elif category_value == "Discussion":
        category = PLCategory.Discussion
    else:
        raise PLARError("category_value must be 'Experiment' or 'Discussion'")

    status_str: Any = None
    try:
        exp = Experiment(
            OpenMode.load_by_plar_app,
            summary_id,
            category,
            user=_unwrap_physicslab_user(user),
        )
        plsav = exp.PlSav if isinstance(getattr(exp, "PlSav", None), dict) else None
        if isinstance(plsav, dict):
            exp_obj = plsav.get("Experiment")
            status_str = exp_obj.get("StatusSave") if isinstance(exp_obj, dict) else plsav.get("StatusSave")
    except Exception:
        # Fallback to raw API responses to avoid strict user type checks in the upstream library.
        exp_res = get_experiment(user, summary_id=summary_id, category_value=category_value)
        data = exp_res.get("Data") if isinstance(exp_res, dict) else None
        plsav_like = _recursive_find_plsav_like(data) or _recursive_find_plsav_like(exp_res)
        if isinstance(plsav_like, dict):
            exp_obj = plsav_like.get("Experiment")
            status_str = exp_obj.get("StatusSave") if isinstance(exp_obj, dict) else plsav_like.get("StatusSave")

    if not isinstance(status_str, str) or not status_str.strip():
        raise PLARError("PlSav missing StatusSave")

    try:
        status = json.loads(status_str)
    except Exception as e:
        raise PLARError(f"Failed to parse StatusSave JSON: {e}") from e
    if not isinstance(status, dict):
        raise PLARError("StatusSave JSON is not an object")

    to_cache = {"summary_id": summary_id, "status_save": status}
    try:
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(to_cache, f, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, cache_path)
    except Exception:
        pass

    return status


def best_effort_extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def iter_text_fields(item: dict[str, Any], keys: Iterable[str]) -> Iterable[str]:
    for key in keys:
        if key in item:
            text = best_effort_extract_text(item.get(key))
            if text:
                yield text


_PLAR_TEXT_PRI_KEYS: tuple[str, ...] = (
    # Common title-ish keys
    "Subject",
    "Title",
    "Name",
    # Common body-ish keys
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

_PLAR_TEXT_SKIP_KEYS: frozenset[str] = frozenset(
    {
        # Extremely large / not human-readable
        "StatusSave",
        "StatusSaveRaw",
        "Elements",
        "Wires",
        # Large media-ish / binary-ish
        "Image",
        "Images",
        "Video",
        "Videos",
        "Audio",
        "Audios",
    }
)


def _truncate_text(text: str, *, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 12)] + "...(truncated)"


def _extract_title_from_obj(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    for k in ("Subject", "Title", "Name"):
        v = obj.get(k)
        t = best_effort_extract_text(v).strip()
        if t:
            return t
    return ""


def _collect_text_recursive(
    obj: Any,
    *,
    max_chars: int,
    max_nodes: int = 2500,
    max_depth: int = 7,
) -> str:
    """Best-effort text extraction from physicsLab API objects.

    The upstream API returns a mix of nested dict/list structures. This function walks
    them and collects useful string leaves, skipping known huge/binary-ish fields.
    """
    if max_chars <= 0:
        return ""

    parts: list[str] = []
    seen: set[int] = set()
    nodes = 0
    stack: list[tuple[Any, int]] = [(obj, 0)]

    def push(v: Any, depth: int) -> None:
        nonlocal nodes
        if nodes >= max_nodes:
            return
        stack.append((v, depth))
        nodes += 1

    while stack and sum(len(p) for p in parts) < max_chars and nodes < max_nodes:
        cur, depth = stack.pop()
        if cur is None or depth > max_depth:
            continue

        if isinstance(cur, str):
            t = cur.strip()
            if t:
                parts.append(t)
            continue

        if isinstance(cur, (int, float, bool)):
            continue

        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)

        if isinstance(cur, list):
            # Preserve order.
            for item in reversed(cur[:200]):
                push(item, depth + 1)
            continue

        if isinstance(cur, dict):
            # Priority keys first (in order), then the rest (stable order).
            for key in reversed(_PLAR_TEXT_PRI_KEYS):
                if key in cur and key not in _PLAR_TEXT_SKIP_KEYS:
                    push(cur.get(key), depth + 1)

            for key in sorted(cur.keys(), key=lambda k: str(k), reverse=True):
                if key in _PLAR_TEXT_SKIP_KEYS or key in _PLAR_TEXT_PRI_KEYS:
                    continue
                push(cur.get(key), depth + 1)
            continue

    # De-dupe while keeping order, then truncate.
    out: list[str] = []
    seen_text: set[str] = set()
    for p in parts:
        p2 = p.strip()
        if not p2:
            continue
        # Avoid repeated boilerplate.
        if p2 in seen_text:
            continue
        seen_text.add(p2)
        out.append(p2)
        if sum(len(x) for x in out) >= max_chars:
            break

    return _truncate_text("\n".join(out), max_chars=max_chars)
