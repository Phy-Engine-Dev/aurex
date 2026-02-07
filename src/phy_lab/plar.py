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
) -> list[dict[str, Any]]:
    qe = getattr(user, "query_experiments", None)
    if callable(qe):
        result = qe(
            category=category,
            tags=None,
            exclude_tags=None,
            languages=[],
            exclude_languages=[],
            user_id=None,
            take=take,
            skip=skip,
            from_skip=from_skip,
        )
    else:
        # Defensive fallback: some wrappers/mocks may expose a non-callable attribute with the
        # same name. In that case, call the underlying HTTP API directly using the user's
        # token/auth_code.
        token = getattr(user, "token", None)
        auth_code = getattr(user, "auth_code", None)
        if not isinstance(token, str) or not token.strip() or not isinstance(auth_code, str) or not auth_code.strip():
            raise PLARError(
                "query_experiments is not callable on this user object, and token/auth_code are missing."
            )
        cat_val = getattr(category, "value", category)
        try:
            import requests  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise PLARError(
                "Missing dependency: requests (required for Physics Lab API calls). "
                "Install it with pip (e.g. 'pip install requests')."
            ) from e
        resp = requests.post(
            "https://physics-api-cn.turtlesim.com/Contents/QueryExperiments",
            json={
                "Query": {
                    "Category": cat_val,
                    "Languages": [],
                    "ExcludeLanguages": [],
                    "Tags": None,
                    "ExcludeTags": None,
                    "ModelTags": None,
                    "ModelID": None,
                    "ParentID": None,
                    "UserID": None,
                    "Special": None,
                    "From": from_skip,
                    "Skip": int(skip),
                    "Take": int(take),
                    "Days": 0,
                    "Sort": 0,
                    "ShowAnnouncement": False,
                }
            },
            headers={
                "Content-Type": "application/json",
                "x-API-Token": token,
                "x-API-AuthCode": auth_code,
            },
            timeout=_requests_default_timeout_sec,
        )
        resp.raise_for_status()
        result = resp.json()
    data = result.get("Data")
    if not isinstance(data, dict):
        raise PLARError("Unexpected query_experiments response: missing Data object")
    values = data.get("$values")
    if not isinstance(values, list):
        raise PLARError("Unexpected query_experiments response: missing Data.$values list")
    return [v for v in values if isinstance(v, dict)]


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
    submit_response, submit_data = exp._Experiment__upload(user, category, None)  # type: ignore[attr-defined]
    summary_id = submit_response["Data"]["Summary"]["ID"]
    image_counter = submit_data["Summary"]["Image"]

    user.confirm_experiment(summary_id, category, image_counter)
    return {
        "summary_id": summary_id,
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

    if raw_tags:
        try:
            plsav = getattr(exp, "PlSav", None)
            if isinstance(plsav, dict):
                summary = plsav.get("Summary")
                if isinstance(summary, dict):
                    existing = summary.get("Tags")
                    if not isinstance(existing, list):
                        existing = []
                    merged = [x for x in existing if isinstance(x, str) and x.strip()]
                    merged.extend(raw_tags)
                    summary["Tags"] = list(dict.fromkeys(merged))
        except Exception:
            return


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
                if isinstance(cached, dict) and cached.get("summary_id") == summary_id:
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
        exp = Experiment(OpenMode.load_by_plar_app, summary_id, category, user=user)
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

    context = {
        "summary_id": summary_id,
        "category": category_value,
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

    exp = Experiment(OpenMode.load_by_plar_app, summary_id, category, user=user)
    plsav = exp.PlSav if isinstance(getattr(exp, "PlSav", None), dict) else None
    if not isinstance(plsav, dict):
        raise PLARError("Failed to load PlSav for this content")

    exp_obj = plsav.get("Experiment")
    if isinstance(exp_obj, dict):
        status_str = exp_obj.get("StatusSave")
    else:
        status_str = plsav.get("StatusSave")
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
