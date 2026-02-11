from __future__ import annotations

from typing import Any

from .errors import PLARError

_DEFAULT_TIMEOUT_SEC = 60.0
_default_timeout_sec = _DEFAULT_TIMEOUT_SEC
_requests_timeout_patched = False


def configure_requests_default_timeout(timeout_sec: float) -> None:
    """Set a default `timeout=` for requests when callers forget to pass one.

    The upstream `physicsLab` client sometimes issues requests without a timeout, which
    can hang indefinitely. This function applies a best-effort monkey patch to
    `requests.sessions.Session.request` so a default timeout is injected when missing.
    """
    global _default_timeout_sec, _requests_timeout_patched

    _default_timeout_sec = float(timeout_sec)
    if _requests_timeout_patched:
        return

    try:
        import requests  # type: ignore
    except ImportError:
        return

    try:
        session_cls = requests.sessions.Session
        orig_request = session_cls.request
    except Exception:
        return

    def _request_with_timeout(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = _default_timeout_sec
        return orig_request(self, method, url, **kwargs)

    try:
        session_cls.request = _request_with_timeout  # type: ignore[assignment]
    except Exception:
        return
    _requests_timeout_patched = True


def post_json_no_env_proxy(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout_sec: float | None = None,
) -> tuple[int, Any]:
    """POST JSON, forcing requests to ignore proxy env vars.

    Many environments set HTTP(S)_PROXY for web access, but the PLAR backend is not
    meant to be called via those proxies. Using a Session with `trust_env = False`
    avoids accidental proxying.
    """
    try:
        import requests  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise PLARError(
            "Missing dependency: `requests` is required for direct PLAR HTTP calls. "
            "Install it (e.g. `pip install requests`)."
        ) from e

    headers_final: dict[str, str] = dict(headers or {})
    timeout_final = _default_timeout_sec if timeout_sec is None else float(timeout_sec)

    session_factory = getattr(requests, "Session", None)
    if callable(session_factory):
        session = session_factory()
        try:
            session.trust_env = False
        except Exception:
            pass
        resp = session.post(url, json=payload, headers=headers_final, timeout=timeout_final)
    else:  # pragma: no cover
        resp = requests.post(url, json=payload, headers=headers_final, timeout=timeout_final)

    status_code = int(getattr(resp, "status_code", 0) or 0)
    try:
        data = resp.json()
    except Exception:
        data = getattr(resp, "text", "")
    return status_code, data

