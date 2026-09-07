"""One isolated, non-retrying official-SDK comment submission.

Run only as a child process. Authentication is received on stdin, never argv.
The parent must persist its at-most-once ledger before spawning this worker and
treat any timeout, signal, or unsuccessful exit as an ambiguous submission.
"""
from __future__ import annotations

import json
import tempfile
import re
import socket
import sys
import urllib.error
import urllib.request


SOCKET_TIMEOUT_SECONDS = 50.0
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_CONTENT_CHARACTERS = 500_000
_FIELDS = frozenset({"token", "auth_code", "target_id", "target_type",
                     "requester_user_id", "content"})
_ID = re.compile(r"[0-9a-fA-F]{24}\Z")
_ERROR = "Comment submission failed; acceptance is unknown. Do not retry automatically.\n"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate input field")
        result[key] = value
    return result


def _validate(payload: object) -> dict:
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("Invalid input schema")
    if any(not isinstance(payload[key], str) for key in _FIELDS):
        raise ValueError("Invalid input types")
    for key in ("token", "auth_code"):
        value = payload[key]
        if not value or len(value) > 8192 or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
            raise ValueError("Invalid authentication field")
    if payload["target_type"] not in {"User", "Experiment", "Discussion"}:
        raise ValueError("Invalid target type")
    if any(not _ID.fullmatch(payload[key]) for key in ("target_id", "requester_user_id")):
        raise ValueError("Invalid routing identity")
    content = payload["content"]
    if len(content) > MAX_CONTENT_CHARACTERS or not content.startswith("回复@"):
        raise ValueError("Expected canonical reply content")
    nickname, delimiter, body = content[3:].partition(": ")
    if (not delimiter or not nickname or not body.strip()
        or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 or ch in "<>@＠:：" for ch in nickname)):
        raise ValueError("Ambiguous reply content")
    return payload


def _configure_transport() -> None:
    # These process-wide settings are safe ONLY in this dedicated child process.
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)
    urllib.request.install_opener(urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect()))


def _sdk_user(token: str, auth_code: str):
    # No login or nickname lookup: the parent has already bound an exact user ID.
    # This child starts with a fresh sys.path, so explicitly select the same
    # vendored SDK as the parent rather than accidentally requiring PyPI.
    try:
        from .physicslab import ensure_physicslab_importable
    except ImportError:  # Direct-file test loader; production uses -m plar.comment_worker.
        from plar.physicslab import ensure_physicslab_importable
    ensure_physicslab_importable(cache_dir=tempfile.gettempdir())
    from physicsLab import web

    user = object.__new__(web.User)
    user.token = token
    user.auth_code = auth_code
    return user


def _normalize_response(response: object) -> dict:
    if not isinstance(response, dict) or type(response.get("Status")) is not int or response["Status"] != 200:
        raise ValueError("Submission was not accepted")
    data = response.get("Data")
    if not isinstance(data, dict):
        raise ValueError("Unexpected submission response")
    legacy = data.get("Comment") if isinstance(data.get("Comment"), dict) else {}
    result = {}
    for source in (data, legacy):
        identifier = source.get("ID")
        if "ID" not in result and isinstance(identifier, str) and _ID.fullmatch(identifier):
            result["ID"] = identifier
        if "Hidden" not in result and type(source.get("Hidden")) is bool:
            result["Hidden"] = source["Hidden"]
    # Never return Token, AuthCode, content, server diagnostics, or URL/header data.
    # Acceptance and an ID still do not prove notification delivery.
    return {"Status": 200, "Data": result}


def submit_once(payload: object) -> dict:
    value = _validate(payload)
    _configure_transport()
    user = _sdk_user(value["token"], value["auth_code"])
    response = user.post_comment(
        target_id=value["target_id"], target_type=value["target_type"],
        content=value["content"], reply_id=value["requester_user_id"], special=None)
    return _normalize_response(response)


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("Input too large")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        result = submit_once(payload)
        sys.stdout.write(json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    except (Exception, KeyboardInterrupt):
        sys.stderr.write(_ERROR)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
