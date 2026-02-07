from __future__ import annotations

import re


def truncate(text: str, *, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def safe_mention_prefix(nickname: str) -> str | None:
    nickname = (nickname or "").strip()
    if not nickname:
        return None
    if any(ch in nickname for ch in (":", " ")):
        return None
    return f"@{nickname} "


_FENCED_CODE_BLOCK_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]*)\n(?P<body>[\s\S]*?)\n```", re.MULTILINE
)


def extract_fenced_code(text: str, *, preferred_lang: str | None = None) -> str | None:
    if not text:
        return None
    matches = list(_FENCED_CODE_BLOCK_RE.finditer(text))
    if not matches:
        return None

    if preferred_lang:
        for m in matches:
            lang = (m.group("lang") or "").strip().lower()
            if lang == preferred_lang.lower():
                body = (m.group("body") or "").strip()
                return body or None

    body = (matches[0].group("body") or "").strip()
    return body or None


def strip_leading_mention(text: str, *, mention_tag: str) -> str:
    text = (text or "").lstrip()
    mention_tag = (mention_tag or "").strip()
    if not mention_tag:
        return text

    handle = mention_tag.lstrip("@＠").strip()
    if not handle:
        return text

    # Do not use \b here: CJK characters are treated as word chars, so "@aurex你好" would not match.
    # Instead, only prevent ASCII word-char continuation to avoid matching "@aurex2" when handle is "aurex".
    pattern = re.compile(
        rf"^(?:@|＠)\s*{re.escape(handle)}(?![A-Za-z0-9_])", re.IGNORECASE
    )
    m = pattern.match(text)
    if not m:
        return text
    return text[m.end() :].lstrip()


def contains_mention(text: str, *, mention_tag: str) -> bool:
    text = (text or "").strip()
    mention_tag = (mention_tag or "").strip()
    if not text or not mention_tag:
        return False
    handle = mention_tag.lstrip("@＠").strip()
    if not handle:
        return False
    pattern = re.compile(
        rf"(?:@|＠)\s*{re.escape(handle)}(?![A-Za-z0-9_])", re.IGNORECASE
    )
    return pattern.search(text) is not None


def parse_command(text: str, *, prefix: str) -> tuple[str | None, str]:
    text = (text or "").strip()
    prefix = prefix or "!"
    if not text.startswith(prefix):
        return None, text

    rest = text[len(prefix) :].lstrip()
    if not rest:
        return "", ""
    parts = rest.split(None, 1)
    cmd = parts[0].strip().lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return cmd, arg
