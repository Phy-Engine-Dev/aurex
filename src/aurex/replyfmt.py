from __future__ import annotations


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


def prefix_user_mention(reply: str, *, nickname: str) -> str:
    """Force reply to start with '@<nickname> ' (space included) when nickname is available."""
    nick = _sanitize_nickname(nickname)
    body = (reply or "").lstrip()
    if not nick:
        return (reply or "").strip()

    # Avoid duplicating the same prefix.
    for pfx in (f"@{nick}", f"＠{nick}"):
        if body.startswith(pfx):
            # Ensure it has one trailing space for consistency.
            rest = body[len(pfx) :].lstrip()
            return f"@{nick} {rest}".strip()

    return f"@{nick} {body}".strip()

