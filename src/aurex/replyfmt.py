from __future__ import annotations

import re


def _sanitize_nickname(nickname: str) -> str:
    s = (nickname or "").strip()
    if not s:
        return ""
    # Strip leading @ (half/full width).
    s = s.lstrip("@＠").strip()
    if not s:
        return ""
    # Avoid newlines/tabs breaking formatting; keep first token.
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    parts = [p for p in s.split(" ") if p]
    if not parts:
        return ""
    s2 = parts[0].strip()
    if not s2:
        return ""
    # Safety cap.
    return s2[:64]


def _sanitize_user_id(user_id: str) -> str:
    s = (user_id or "").strip()
    if not s:
        return ""
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    parts = [p for p in s.split(" ") if p]
    if not parts:
        return ""
    s2 = parts[0].strip()
    if not s2:
        return ""
    # Prefer the standard 24-hex ID.
    m = re.search(r"[0-9a-fA-F]{24}", s2)
    if m:
        return m.group(0)
    # Fallback: allow a small safe token to avoid tag injection.
    if re.fullmatch(r"[0-9a-zA-Z]{1,64}", s2):
        return s2[:64]
    return ""


def prefix_user_mention(reply: str, *, user_id: str, nickname: str) -> str:
    """Force reply to start with '<user=...>@<nickname></user> ' when possible."""
    nick = _sanitize_nickname(nickname)
    uid = _sanitize_user_id(user_id)
    body = (reply or "").lstrip()
    if not nick or not uid:
        return (reply or "").strip()

    mention = f"<user={uid}>@{nick}</user>"

    # Avoid duplicating the same prefix (mention tag form).
    if body.startswith(mention):
        rest = body[len(mention) :].lstrip()
        return f"{mention} {rest}".strip()

    # If the model used plain "@nick" prefix, replace it with the user-tag mention.
    pat = re.compile(rf"^[@＠]\s*{re.escape(nick)}(?![0-9a-z_])", re.IGNORECASE)
    m = pat.search(body)
    if m:
        rest = body[m.end() :].lstrip()
        rest = rest.lstrip(":：").lstrip()
        return f"{mention} {rest}".strip()

    return f"{mention} {body}".strip()
