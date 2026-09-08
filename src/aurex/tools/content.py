"""Bounded community title/body readers.

The electrical payload of a PhysicsLab experiment is intentionally *not* read
by these helpers.  ``plar_read_title`` and ``plar_read_body`` only query the
public ``Summary`` record, which keeps prose lookup separate from the native
circuit pipeline (``plar_get_experiment_file`` + ``circuit_inspect`` /
``circuit_analyze``).

These tools are deliberately narrower than the old archive/context readers:
title is returned in one bounded response, while body supports an explicit
bounded window or a literal/regular-expression search.  Search results contain
small surrounding excerpts rather than the whole source document.
"""
from __future__ import annotations

import re
from typing import Any

import plar

from .registry import ToolError, ToolRuntime


_HEX24_RE = re.compile(r"^[0-9a-fA-F]{24}$")
_TITLE_MAX_CHARS = 4096
_BODY_MAX_CHARS = 4_000_000
_PATTERN_MAX_CHARS = 512
_DEFAULT_BODY_LENGTH = 4_000
_MAX_BODY_LENGTH = 12_000
_DEFAULT_CONTEXT_CHARS = 160
_MAX_CONTEXT_CHARS = 800
_MAX_EXCERPT_CHARS = 2_400


def _regex_atom_domain(atom: str) -> tuple[str, frozenset[str] | None]:
    """Return a conservative character domain for one restricted atom.

    The domain is used only to reject ambiguous quantified expressions before
    they reach Python's backtracking engine.  Unknown/negated domains are
    deliberately treated as ``any`` rather than accepted optimistically.
    """
    if atom == ".":
        return "any", None
    if atom in {r"\D", r"\W", r"\S"}:
        return "any", None
    if atom == r"\d":
        return "digit", None
    if atom == r"\w":
        return "word", None
    if atom == r"\s":
        return "space", None
    if atom.startswith("["):
        # Parse only simple positive classes for disjointness. The regular
        # expression compiler remains the source of truth for class syntax.
        negated = atom.startswith("[^")
        content = atom[2:-1] if negated else atom[1:-1]
        chars: set[str] = set()
        index = 0
        while index < len(content):
            if content[index] == "\\" and index + 1 < len(content):
                escaped = content[index:index + 2]
                if escaped in {r"\d", r"\w", r"\s", r"\D", r"\W", r"\S"}:
                    return "any", None
                chars.add(content[index + 1])
                index += 2
                continue
            if index + 2 < len(content) and content[index + 1] == "-":
                left, right = ord(content[index]), ord(content[index + 2])
                if left > right or right - left > 256:
                    return "any", None
                chars.update(chr(value) for value in range(left, right + 1))
                index += 3
                continue
            chars.add(content[index])
            index += 1
        return ("negset" if negated else "set"), frozenset(chars)
    if atom.startswith("\\") and len(atom) == 2:
        escaped_literals = {"n": "\n", "r": "\r", "t": "\t", "f": "\f", "v": "\v"}
        return "set", frozenset({escaped_literals.get(atom[1], atom[1])})
    return "set", frozenset({atom})


def _domains_overlap(left: tuple[str, frozenset[str] | None],
                     right: tuple[str, frozenset[str] | None]) -> bool:
    left_kind, left_chars = left
    right_kind, right_chars = right
    if "any" in {left_kind, right_kind}:
        return True
    if left_kind == "negset" and right_kind == "negset":
        return True
    if left_kind == "negset" and right_kind == "set":
        return any(char not in (left_chars or frozenset()) for char in (right_chars or frozenset()))
    if right_kind == "negset" and left_kind == "set":
        return any(char not in (right_chars or frozenset()) for char in (left_chars or frozenset()))
    if "negset" in {left_kind, right_kind}:
        return True
    if left_kind == "set" and right_kind == "set":
        return bool((left_chars or frozenset()) & (right_chars or frozenset()))

    def set_matches(chars: frozenset[str] | None, kind: str) -> bool:
        values = chars or frozenset()
        if kind == "digit":
            return any(char.isdigit() for char in values)
        if kind == "word":
            return any(char == "_" or char.isalnum() for char in values)
        if kind == "space":
            return any(char.isspace() for char in values)
        return True

    if left_kind == "set":
        return set_matches(left_chars, right_kind)
    if right_kind == "set":
        return set_matches(right_chars, left_kind)
    if {left_kind, right_kind} == {"digit", "word"}:
        return True
    return left_kind == right_kind


def _compile_safe_regex(pattern: str) -> re.Pattern[str]:
    """Compile Aurex's intentionally small, non-nested regex dialect.

    Supported constructs are literals, ``.``, positive/negative character
    classes, ``^``/``$``, the common ``\\d``/``\\w``/``\\s`` classes and the
    ``?``/``*``/``+`` quantifiers. Groups, alternation, backreferences,
    look-arounds and brace repetition are rejected. Ambiguous quantified atoms
    are also rejected so a public post cannot trigger catastrophic backtracking
    in the long-lived server process.
    """
    atoms: list[dict[str, Any]] = []
    anchored_start = pattern.startswith("^")
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "^":
            if index != 0:
                raise ToolError("Unsafe regular expression: ^ is allowed only at the start")
            index += 1
            continue
        if char == "$":
            if index != len(pattern) - 1:
                raise ToolError("Unsafe regular expression: $ is allowed only at the end")
            index += 1
            continue
        if char in "()|{}":
            raise ToolError(
                "Unsafe regular expression: groups, alternation and brace repetition are not supported; "
                "use literals, character classes, anchors and simple ?/*/+ quantifiers"
            )
        if char in "?*+":
            if not atoms or atoms[-1].get("quantifier") is not None:
                raise ToolError("Unsafe regular expression: misplaced or repeated quantifier")
            atoms[-1]["quantifier"] = char
            index += 1
            continue
        if char == "[":
            end = index + 1
            escaped = False
            while end < len(pattern):
                current = pattern[end]
                if current == "]" and not escaped and end > index + 1:
                    break
                escaped = current == "\\" and not escaped
                if current != "\\":
                    escaped = False
                end += 1
            if end >= len(pattern) or pattern[end] != "]":
                raise ToolError("Invalid regular expression: unterminated character class")
            atom = pattern[index:end + 1]
            index = end + 1
        elif char == "\\":
            if index + 1 >= len(pattern):
                raise ToolError("Invalid regular expression: trailing escape")
            escaped = pattern[index + 1]
            if escaped.isdigit() or escaped in {"A", "Z", "b", "B", "g", "k", "p", "P"}:
                raise ToolError("Unsafe regular expression: backreferences and special assertions are not supported")
            if escaped not in {"d", "D", "w", "W", "s", "S", "n", "r", "t", "f", "v",
                               ".", "^", "$", "[", "]", "\\", "?", "*", "+", "-"}:
                raise ToolError("Unsafe regular expression: unsupported escape in restricted regex dialect")
            atom = pattern[index:index + 2]
            index += 2
        else:
            atom = char
            index += 1
        atoms.append({"atom": atom, "domain": _regex_atom_domain(atom), "quantifier": None})

    if not atoms:
        raise ToolError("Unsafe regular expression: pattern must contain a matching atom")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"Invalid regular expression: {exc}") from exc
    if compiled.match("") is not None:
        raise ToolError("Unsafe regular expression: zero-width matches are not supported")

    variable = [position for position, atom in enumerate(atoms)
                if atom["quantifier"] in {"?", "*", "+"}]
    for position in variable:
        domain = atoms[position]["domain"]
        previous_required = next((atoms[other] for other in range(position - 1, -1, -1)
                                  if atoms[other]["quantifier"] is None), None)
        next_required = next((atoms[other] for other in range(position + 1, len(atoms))
                              if atoms[other]["quantifier"] is None), None)
        # An unanchored variable prefix followed by a suffix can make the engine
        # retry the same long run from every character (quadratic behavior).
        if not anchored_start and previous_required is None and next_required is not None:
            raise ToolError("Unsafe regular expression: add a fixed literal prefix or ^ before a variable-length atom")
        if next_required is not None and _domains_overlap(domain, next_required["domain"]):
            raise ToolError("Unsafe regular expression: quantified atom overlaps the following delimiter")
    for left, right in zip(variable, variable[1:]):
        separators = [atom for atom in atoms[left + 1:right] if atom["quantifier"] is None]
        if not separators and _domains_overlap(atoms[left]["domain"], atoms[right]["domain"]):
            raise ToolError("Unsafe regular expression: ambiguous quantified atoms are not supported")
        if separators and all(_domains_overlap(atoms[left]["domain"], atom["domain"])
                              or _domains_overlap(atoms[right]["domain"], atom["domain"])
                              for atom in separators):
            raise ToolError("Unsafe regular expression: ambiguous quantified atoms are not supported")
    return compiled


def _require_user(runtime: ToolRuntime) -> Any:
    if runtime.user is None:
        raise ToolError("community content tool requires a logged-in PhysicsLab user in runtime.user")
    return runtime.user


def _summary_id(value: Any, *, where: str) -> str:
    value = str(value or "").strip()
    if not _HEX24_RE.fullmatch(value):
        raise ToolError(f"{where} must be exactly 24 hexadecimal characters")
    return value


def _category(value: Any, *, where: str) -> str:
    value = str(value or "Experiment").strip().capitalize() or "Experiment"
    if value not in {"Experiment", "Discussion"}:
        raise ToolError(f"{where} must be Experiment or Discussion")
    return value


def _summary_record(response: Any, *, requested: str, tool_name: str) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ToolError(f"{tool_name}: API returned a non-object response")
    status = response.get("Status")
    if status is not None and str(status) not in {"200", "200.0"}:
        raise ToolError(f"{tool_name}: API returned status {status}")
    data = response.get("Data", response)
    if not isinstance(data, dict):
        raise ToolError(f"{tool_name}: API returned no Data object")
    summary = data.get("Summary") if isinstance(data.get("Summary"), dict) else data
    if not isinstance(summary, dict):
        raise ToolError(f"{tool_name}: API returned no Summary object")
    returned = summary.get("ID") or summary.get("SummaryID")
    if returned is not None and str(returned).strip().casefold() != requested.casefold():
        raise ToolError(f"{tool_name}: returned summary ID does not match the requested post")
    return summary


def _load_summary(user: Any, sid: str, args: dict[str, Any], *,
                  tool_name: str) -> tuple[dict[str, Any], str, bool]:
    """Load an exact post and resolve an omitted category without retry loops.

    Experiment and Discussion IDs share the same shape but live in different
    API namespaces.  A model may already have an exact ID while omitting the
    category.  The historical implementation silently defaulted to Experiment,
    returned 404 for a Discussion, and encouraged repeated identical reads.
    Try the other namespace once only when the caller did not explicitly bind
    a category; an explicitly wrong category still fails closed.
    """
    supplied = args.get("category")
    category = _category(supplied, where=f"{tool_name}.category")
    response = plar.get_summary(user, summary_id=sid, category_value=category)
    try:
        return (_summary_record(response, requested=sid, tool_name=tool_name),
                category, False)
    except ToolError:
        status = response.get("Status") if isinstance(response, dict) else None
        if supplied not in (None, "") or str(status) not in {"404", "404.0"}:
            raise
    fallback = "Discussion" if category == "Experiment" else "Experiment"
    fallback_response = plar.get_summary(
        user, summary_id=sid, category_value=fallback)
    return (_summary_record(fallback_response, requested=sid,
                            tool_name=tool_name), fallback, True)


def _text_field(summary: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, str | None]:
    """Return the first non-empty known text field and its source key."""
    for key in keys:
        value = summary.get(key)
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            text = "\n".join(value).strip()
        else:
            # The public prose contract is string/list-of-strings. Never walk
            # an arbitrary nested object: adjacent API payloads can contain a
            # serialized circuit under Content/StatusSave-like keys.
            text = ""
        if text:
            return text, key
    return "", None


def _bounded_title(text: str) -> str:
    if len(text) > _TITLE_MAX_CHARS:
        # Never return a partial title: the public API contract is much smaller
        # than this defensive ceiling, so an oversized value is malformed.
        raise ToolError(f"Summary title exceeds the verified {_TITLE_MAX_CHARS}-character bound")
    return text


def _bounded_body(text: str) -> str:
    if len(text) > _BODY_MAX_CHARS:
        raise ToolError(
            f"Summary body exceeds the bounded prose limit ({_BODY_MAX_CHARS} characters); "
            "do not use this tool for electrical content. Use the circuit engine for the experiment payload."
        )
    return text


def _validate_window(offset: Any, length: Any) -> tuple[int, int]:
    if type(offset) is not int or offset < 0:
        raise ToolError("offset must be a nonnegative integer")
    if type(length) is not int or not 1 <= length <= _MAX_BODY_LENGTH:
        raise ToolError(f"length must be an integer in 1..{_MAX_BODY_LENGTH}")
    return offset, length


def _validate_pattern(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ToolError("pattern is required for search/regex mode")
    if len(value) > _PATTERN_MAX_CHARS:
        raise ToolError(f"pattern must be at most {_PATTERN_MAX_CHARS} characters")
    return value


def _excerpt(text: str, start: int, end: int, context: int) -> dict[str, Any]:
    left = max(0, start - context)
    right = min(len(text), end + context)
    complete_right = right
    if right - left > _MAX_EXCERPT_CHARS:
        right = left + _MAX_EXCERPT_CHARS
    return {
        "start": start,
        "end": end,
        "text": text[left:right],
        "excerpt_start": left,
        "excerpt_end": right,
        "match_characters": end - start,
        "match_fully_shown": start >= left and end <= right,
        "excerpt_truncated": right < complete_right,
    }


def _map_casefold_spans(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Map spans in ``text.casefold()`` back to exact original-string spans."""
    output: list[tuple[int, int]] = []
    original_index = 0
    folded_index = 0
    for folded_start, folded_end in spans:
        if folded_start < folded_index:
            # A casefold expansion (for example ß -> ss) can contain more
            # than one folded match. They all map to the complete source char.
            original_start = max(0, original_index - 1)
        else:
            while original_index < len(text):
                width = len(text[original_index].casefold())
                if folded_index + width > folded_start:
                    break
                folded_index += width
                original_index += 1
            original_start = original_index
        while original_index < len(text) and folded_index < folded_end:
            folded_index += len(text[original_index].casefold())
            original_index += 1
        output.append((original_start, original_index))
    return output


def _casefold_literal_spans(text: str, pattern: str, *, start_offset: int,
                            max_matches: int) -> tuple[list[tuple[int, int]], bool]:
    folded = text.casefold()
    needle = pattern.casefold()
    if not needle:
        raise ToolError("case-insensitive pattern has no searchable characters after Unicode case folding")
    folded_cursor = sum(len(char.casefold()) for char in text[:min(start_offset, len(text))])
    raw_spans: list[tuple[int, int]] = []
    exhausted = False
    target = max_matches + 1
    while True:
        while len(raw_spans) < target:
            found = folded.find(needle, folded_cursor)
            if found < 0:
                exhausted = True
                break
            raw_spans.append((found, found + len(needle)))
            folded_cursor = found + len(needle)
        mapped = _map_casefold_spans(text, raw_spans)
        unique = list(dict.fromkeys(mapped))
        if len(unique) > max_matches:
            return unique[:max_matches], True
        if exhausted:
            return unique, False
        # One source character may expand to several folded characters. Fetch a
        # bounded next batch until we either find one additional source span or
        # prove the search complete.
        target += max_matches + 1


def _no_match(base: dict[str, Any], *, pattern: str, mode: str, start_offset: int,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an explicit terminal search result instead of inviting retries."""
    return {
        **base,
        "pattern": pattern,
        "start_offset": start_offset,
        **(extra or {}),
        "found": False,
        "match_count_returned": 0,
        "matches": [],
        "has_more_matches": False,
        "search_complete": True,
        "next_search_offset": None,
        "stop": True,
        "stop_reason": f"{mode}_pattern_not_found",
        "next_action": "Do not repeat this body search unchanged; answer from available evidence or use a materially different exact term.",
    }


def plar_read_title(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Read only the public title of one exact community post."""
    user = _require_user(runtime)
    sid = _summary_id(args.get("summary_id"), where="plar_read_title.summary_id")
    summary, category, category_inferred = _load_summary(
        user, sid, args, tool_name="plar_read_title")
    title, source_field = _text_field(summary, ("Subject", "Title", "Name"))
    title = _bounded_title(title)
    return {
        "summary_id": sid,
        "category": category,
        "category_inferred": category_inferred,
        "title": title or None,
        "title_characters": len(title),
        "title_truncated": False,
        "source_field": source_field,
        "source": "GetSummary.Summary",
        "untrusted_reference": True,
        "external_write_performed": False,
    }


def plar_read_body(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Read/search the public prose body without exposing electrical JSON."""
    user = _require_user(runtime)
    sid = _summary_id(args.get("summary_id"), where="plar_read_body.summary_id")
    mode = str(args.get("mode") or "read").strip().casefold()
    if mode not in {"read", "search", "regex"}:
        raise ToolError("plar_read_body.mode must be read, search or regex")
    summary, category, category_inferred = _load_summary(
        user, sid, args, tool_name="plar_read_body")
    # PhysicsLab prose lives in Summary.Description.  Never fall back to a
    # field named Content: depending on the endpoint that can be the serialized
    # experiment payload rather than human prose.
    body, source_field = _text_field(summary, ("Description",))
    body = _bounded_body(body)

    base: dict[str, Any] = {
        "summary_id": sid,
        "category": category,
        "category_inferred": category_inferred,
        "source_field": source_field,
        "source": "GetSummary.Summary",
        "mode": mode,
        "body_characters": len(body),
        "untrusted_reference": True,
        "external_write_performed": False,
    }

    if mode == "read":
        offset, length = _validate_window(args.get("offset", 0), args.get("length", _DEFAULT_BODY_LENGTH))
        text = body[offset:offset + length]
        end = offset + len(text)
        return {
            **base,
            "offset": offset,
            "length": length,
            "text": text,
            "has_more": end < len(body),
            "next_offset": end if end < len(body) else None,
            "stop": not text or end >= len(body),
            "stop_reason": "end_of_body" if not text or end >= len(body) else None,
        }

    pattern = _validate_pattern(args.get("pattern"))
    start_offset = args.get("start_offset", 0)
    if type(start_offset) is not int or start_offset < 0:
        raise ToolError("start_offset must be a nonnegative integer")
    context = args.get("context_chars", _DEFAULT_CONTEXT_CHARS)
    if type(context) is not int or not 0 <= context <= _MAX_CONTEXT_CHARS:
        raise ToolError(f"context_chars must be an integer in 0..{_MAX_CONTEXT_CHARS}")
    max_matches = args.get("max_matches", 6)
    if type(max_matches) is not int or not 1 <= max_matches <= 16:
        raise ToolError("max_matches must be an integer in 1..16")

    if mode == "search":
        # Literal matching is case-sensitive by default.  ``case_sensitive``
        # is explicit so a model cannot silently change the meaning of a query.
        case_sensitive = args.get("case_sensitive", True)
        if type(case_sensitive) is not bool:
            raise ToolError("case_sensitive must be boolean")
        matches: list[dict[str, Any]] = []
        has_more_matches = False
        if case_sensitive:
            cursor = min(start_offset, len(body))
            while len(matches) <= max_matches:
                found = body.find(pattern, cursor)
                found_end = found + len(pattern) if found >= 0 else -1
                if found < 0:
                    break
                if len(matches) == max_matches:
                    has_more_matches = True
                    break
                matches.append(_excerpt(body, found, found_end, context))
                cursor = found_end
        else:
            spans, has_more_matches = _casefold_literal_spans(
                body, pattern, start_offset=start_offset, max_matches=max_matches)
            matches = [_excerpt(body, start, end, context) for start, end in spans]
        next_search_offset = matches[-1]["end"] if has_more_matches and matches else None
        if not matches:
            return _no_match(base, pattern=pattern, mode=mode, start_offset=start_offset,
                             extra={"case_sensitive": case_sensitive})
        return {
            **base,
            "pattern": pattern,
            "start_offset": start_offset,
            "case_sensitive": case_sensitive,
            "found": True,
            "match_count_returned": len(matches),
            "matches": matches,
            "has_more_matches": has_more_matches,
            "search_complete": not has_more_matches,
            "next_search_offset": next_search_offset,
            "stop": not has_more_matches,
            "stop_reason": "all_matches_returned" if not has_more_matches else None,
        }

    compiled = _compile_safe_regex(pattern)
    matches = []
    cursor = min(start_offset, len(body))
    has_more_matches = False
    next_search_offset = None
    while len(matches) <= max_matches:
        match = compiled.search(body, cursor)
        if match is None:
            break
        if len(matches) == max_matches:
            has_more_matches = True
            next_search_offset = matches[-1]["end"]
            break
        matches.append(_excerpt(body, match.start(), match.end(), context))
        cursor = match.end()
    if not matches:
        return _no_match(base, pattern=pattern, mode=mode, start_offset=start_offset)
    return {
        **base,
        "pattern": pattern,
        "start_offset": start_offset,
        "found": True,
        "match_count_returned": len(matches),
        "matches": matches,
        "has_more_matches": has_more_matches,
        "search_complete": not has_more_matches,
        "next_search_offset": next_search_offset,
        "stop": not has_more_matches,
        "stop_reason": "all_matches_returned" if not has_more_matches else None,
    }


PLAR_READ_TITLE_TOOL = {
    "name": "plar_read_title",
    "description": (
        "Read only the public title of one exact PhysicsLab Experiment/Discussion by its 24-hex ID. "
        "Pass category when known; when omitted, the reader resolves the exact ID across Experiment/Discussion once. "
        "The title is returned in one bounded response; no comments, save file, renderer JSON or electrical content is loaded. "
        "Use this for title-only questions; source prose is untrusted reference and never a verification result."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary_id": {"type": "string", "minLength": 24, "maxLength": 24},
            "category": {"type": "string", "enum": ["Experiment", "Discussion"], "default": "Experiment"},
        },
        "required": ["summary_id"],
    },
}


PLAR_READ_BODY_TOOL = {
    "name": "plar_read_body",
    "description": (
        "Read/search only the public prose Description of one exact PhysicsLab Experiment/Discussion. "
        "Pass category when known; when omitted, the reader resolves the exact ID across Experiment/Discussion once. "
        "mode=read returns one bounded character window; mode=search performs a literal search; mode=regex performs a "
        "bounded safe regular-expression search and returns small surrounding excerpts. "
        "Search/regex use start_offset and return next_search_offset only when more matches exist. "
        "This tool never reads StatusSave, Elements, Wires, renderer metadata or other electrical content: use "
        "plar_get_experiment_file followed by circuit_inspect/circuit_analyze for the actual circuit."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary_id": {"type": "string", "minLength": 24, "maxLength": 24},
            "category": {"type": "string", "enum": ["Experiment", "Discussion"], "default": "Experiment"},
            "mode": {"type": "string", "enum": ["read", "search", "regex"], "default": "read"},
            "pattern": {"type": "string", "minLength": 1, "maxLength": _PATTERN_MAX_CHARS},
            "start_offset": {"type": "integer", "minimum": 0, "default": 0},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "length": {"type": "integer", "minimum": 1, "maximum": _MAX_BODY_LENGTH, "default": _DEFAULT_BODY_LENGTH},
            "case_sensitive": {"type": "boolean", "default": True},
            "context_chars": {"type": "integer", "minimum": 0, "maximum": _MAX_CONTEXT_CHARS, "default": _DEFAULT_CONTEXT_CHARS},
            "max_matches": {"type": "integer", "minimum": 1, "maximum": 16, "default": 6},
        },
        "required": ["summary_id"],
    },
}


__all__ = [
    "PLAR_READ_TITLE_TOOL",
    "PLAR_READ_BODY_TOOL",
    "plar_read_title",
    "plar_read_body",
]
