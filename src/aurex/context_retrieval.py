"""Exact, read-only retrieval from archived JSON; no evaluation or source edits."""
from __future__ import annotations

import hashlib
import json
import re


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Archived JSON contains duplicate keys; read the original text instead of an ambiguous JSON pointer')
        value[key] = item
    return value


def read_context(db, sid: str, args: dict) -> dict:
    allowed = {'document_id', 'find', 'json_pointer', 'select', 'json_search', 'offset', 'length'}
    unknown = set(args) - allowed
    if unknown:
        hint = ' Use length (not limit) for the returned character budget.' if 'limit' in unknown else ''
        raise ValueError('read_context accepts only document_id, find, json_pointer, select, json_search, offset and length.' + hint)
    document_id = args.get('document_id')
    if not isinstance(document_id, str) or not document_id:
        raise ValueError('document_id must be a nonempty string')
    pointer = args.get('json_pointer')
    needle = args.get('find')
    select = args.get('select')
    json_search = args.get('json_search')
    if 'select' in args and (not isinstance(select, dict) or pointer is None or needle is not None):
        raise ValueError('select requires a JSON array pointer and cannot be combined with find')
    if 'json_search' in args and (not isinstance(json_search, dict) or pointer is not None or
                                  needle is not None or select is not None):
        raise ValueError('json_search must be an object and cannot be combined with find, json_pointer or select')
    if 'find' in args and (not isinstance(needle, str) or not 1 <= len(needle) <= 256):
        raise ValueError('find must be a literal string containing 1..256 characters; regular expressions are not executed')
    if pointer is not None and (not isinstance(pointer, str) or (pointer and not pointer.startswith('/')) or len(pointer) > 2048):
        raise ValueError('json_pointer must be an RFC6901 pointer starting with /, or empty for the root')
    offset, length = args.get('offset', 0), args.get('length', 12000)
    if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 20000:
        raise ValueError('offset must be nonnegative and length must be an integer in 1..20000')
    with db.connect() as store:
        row = store.execute('SELECT title,content FROM documents WHERE id=? AND session_id=?',
                            (document_id, sid)).fetchone()
    if row is None:
        raise ValueError('Document does not belong to this session')
    raw = row['content']
    # A read_context outcome is a transport record containing an already-read
    # source page. Paging that JSON record creates escaped copies of escaped
    # copies and, worse, changes the apparent source identity. Fail closed and
    # give the exact immutable source cursor instead.
    if row['title'].startswith('Tool read_context'):
        source_id = source_pointer = None
        source_offset = 0
        try:
            recorded = json.loads(raw, object_pairs_hook=_unique_object)
            recorded = recorded.get('data') if isinstance(recorded, dict) else None
            if isinstance(recorded, dict):
                source_id = recorded.get('id')
                source_pointer = recorded.get('json_pointer')
                source_offset = recorded.get('offset', 0) + len(recorded.get('text', ''))
        except (ValueError, TypeError):
            pass
        recovery = {'document_id': source_id, 'offset': source_offset, 'length': length}
        if isinstance(source_pointer, str):
            recovery['json_pointer'] = source_pointer
        suffix = (' Continue only from the original source with: read_context(' +
                  json.dumps(recovery, ensure_ascii=False, separators=(',', ':'))[1:-1] + ').') if source_id else ''
        raise ValueError(
            f'document_id={document_id} is a derived Tool read_context outcome, not an original source; '
            'never page or select it. The tool-result document ID is diagnostics-only.' + suffix
        )
    # A rendered circuit's archived full netlist is an evidence artifact, not
    # an agent browsing surface. circuit_inspect provides bounded exact
    # component, node and type selection with importer-backed pin semantics.
    # Reject all reads here, including JSON select: otherwise a model can turn
    # one relevant lookup into an unbounded netlist traversal after compaction.
    selector = needle.strip('"') if isinstance(needle, str) else None
    if row['title'].endswith(': full netlist_path') and json_search is None:
        target = selector if selector is not None else 'the requested component/type'
        raise ValueError(
            f'Do not read or select the archived full circuit netlist for {target}. '
            f'Call circuit_inspect on the original circuit/state path with query={json.dumps(target)}; '
            'use its exact query offset/limit and importer-backed pin labels, or use bounded json_search for an '
            'exact archived field lookup. No source value was inferred.'
        )
    if pointer is None and needle is None and json_search is None:
        return {'id': document_id, 'title': row['title'], 'offset': offset, 'total_chars': len(raw),
                'text': raw[offset:offset + length], 'has_more': offset + length < len(raw)}
    if pointer is None and json_search is None:
        return _find_page(document_id, row['title'], raw, raw, None, needle, offset, length)
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError('This source is not JSON; omit json_pointer and read an ordinary text page') from exc
    if json_search is not None:
        text = json.dumps(_search_json_fields(value, json_search), ensure_ascii=False,
                          separators=(',', ':'), allow_nan=False)
        return {'id': document_id, 'title': row['title'], 'json_search': json_search,
                'document_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                'offset': offset, 'total_chars': len(text), 'text': text[offset:offset + length],
                'has_more': offset + length < len(text), 'recorded_not_resimulated': True,
                'offset_scope': 'Serialized bounded JSON field-search results, not the source document'}
    for token in pointer.split('/')[1:]:
        if re.search(r'~(?![01])', token):
            raise ValueError('Invalid JSON pointer escape')
        token = token.replace('~1', '/').replace('~0', '~')
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and re.fullmatch(r'0|[1-9][0-9]*', token) and len(token) <= 16 and int(token) < len(value):
            value = value[int(token)]
        else:
            raise ValueError('JSON pointer does not exist in the archived source; no value was inferred')
    if select is not None:
        text = json.dumps(_select_rows(value, select), ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        return {'id': document_id, 'title': row['title'], 'json_pointer': pointer,
                'select': select, 'document_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                'offset': offset, 'total_chars': len(text), 'text': text[offset:offset + length],
                'has_more': offset + length < len(text), 'recorded_not_resimulated': True,
                'offset_scope': 'Serialized selected rows; select.offset pages matching rows, offset pages this text'}
    # Canonical subtree text retains all JSON values. Offsets apply to this
    # selected serialization, NOT byte/character positions in the full source.
    text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if needle is not None:
        return _find_page(document_id, row['title'], raw, text, pointer, needle, offset, length)
    return {'id': document_id, 'title': row['title'], 'json_pointer': pointer,
            'document_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'offset': offset, 'total_chars': len(text), 'text': text[offset:offset + length],
            'has_more': offset + length < len(text), 'recorded_not_resimulated': True,
            'offset_scope': 'JSON serialization of the selected subtree, not the full source document'}


def _select_rows(value, spec):
    """Select exact records without an expression language or external access."""
    if not isinstance(value, list):
        raise ValueError('select requires a JSON array, such as /components')
    if set(spec) - {'where', 'fields', 'offset', 'limit'}:
        raise ValueError('select accepts only where, fields, offset and limit')
    where, fields = spec.get('where', {}), spec.get('fields')
    row_offset, limit = spec.get('offset', 0), spec.get('limit', 16)
    scalar = (str, bool, int, float, type(None))
    if not isinstance(where, dict) or len(where) > 4 or any(
        not isinstance(k, str) or not 1 <= len(k) <= 128 or not isinstance(v, scalar) or
        isinstance(v, (dict, list)) or (isinstance(v, str) and len(v) > 256)
        for k, v in where.items()):
        raise ValueError('where must contain at most four exact top-level scalar field matches')
    if fields is not None and (not isinstance(fields, list) or not 1 <= len(fields) <= 16 or
        any(not isinstance(k, str) or not 1 <= len(k) <= 128 for k in fields) or
        len(set(fields)) != len(fields)):
        raise ValueError('fields must contain 1..16 distinct top-level field names')
    if type(row_offset) is not int or row_offset < 0 or type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError('select offset must be nonnegative and limit must be 1..64')
    matches = [(i, row) for i, row in enumerate(value) if isinstance(row, dict) and all(
        k in row and type(row[k]) is type(v) and row[k] == v for k, v in where.items())]
    rows = []
    for i, row in matches[row_offset:row_offset + limit]:
        rows.append({'source_index': i,
                     'value': row if fields is None else {k: row[k] for k in fields if k in row},
                     **({'missing_fields': [k for k in fields if k not in row]} if fields is not None else {})})
    return {'total_matches': len(matches), 'offset': row_offset, 'rows': rows,
            'has_more_rows': row_offset + limit < len(matches),
            'next_row_offset': row_offset + len(rows) if row_offset + len(rows) < len(matches) else None,
            'note': 'Exact archived records only. Missing fields are not inferred; source indices identify original array entries. This is not a simulation or validation.'}


def _json_pointer_token(value: object) -> str:
    return str(value).replace('~', '~0').replace('/', '~1')


def _nested_field(row: dict, path: list[str]):
    value = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _exact_json_scalar(left, right) -> bool:
    # JSON has one number type; do not make 10 and 10.0 mysteriously differ.
    if (isinstance(left, (int, float)) and not isinstance(left, bool) and
            isinstance(right, (int, float)) and not isinstance(right, bool)):
        return left == right
    return type(left) is type(right) and left == right


def _search_json_fields(value, spec):
    """Recursively select mappings by a bounded, non-expression field query."""
    if set(spec) - {'field', 'match', 'value', 'fields', 'offset', 'limit'}:
        raise ValueError('json_search accepts only field, match, value, fields, offset and limit')
    field = spec.get('field')
    mode = spec.get('match', 'exact')
    fields = spec.get('fields')
    row_offset, limit = spec.get('offset', 0), spec.get('limit', 8)
    scalar = (str, bool, int, float, type(None))
    if not isinstance(field, str) or not 1 <= len(field) <= 512:
        raise ValueError('json_search.field must be a dotted field path containing 1..512 characters')
    path = field.split('.')
    if (len(path) > 16 or any(not key or len(key) > 128 for key in path)):
        raise ValueError('json_search.field must contain 1..16 nonempty dotted path segments of at most 128 characters')
    if mode not in {'exact', 'contains', 'exists'}:
        raise ValueError('json_search.match must be exact, contains or exists')
    has_value = 'value' in spec
    expected = spec.get('value')
    if mode == 'exists' and has_value:
        raise ValueError('json_search exists mode does not accept value')
    if mode != 'exists' and (not has_value or not isinstance(expected, scalar) or
                             isinstance(expected, str) and len(expected) > 512):
        raise ValueError('json_search exact/contains mode requires a scalar value of bounded size')
    if mode == 'contains' and not isinstance(expected, str):
        raise ValueError('json_search contains mode requires a string value')
    if (not isinstance(fields, list) or not 1 <= len(fields) <= 16 or
            any(not isinstance(key, str) or not 1 <= len(key) <= 128 for key in fields) or
            len(set(fields)) != len(fields)):
        raise ValueError('json_search.fields must contain 1..16 distinct top-level field names')
    if type(row_offset) is not int or row_offset < 0 or type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError('json_search offset must be nonnegative and limit must be 1..32')

    matches = []
    stack = [('', value, 0)]
    visited = 0
    while stack:
        pointer, current, depth = stack.pop()
        visited += 1
        if visited > 1_000_000 or depth > 128:
            raise ValueError('Archived JSON is too deeply nested or large for bounded field search')
        if isinstance(current, dict):
            present, candidate = _nested_field(current, path)
            matched = present and (mode == 'exists' or
                mode == 'exact' and _exact_json_scalar(candidate, expected) or
                mode == 'contains' and isinstance(candidate, str) and expected.casefold() in candidate.casefold())
            if matched:
                projected = {key: current[key] for key in fields if key in current}
                matches.append({'json_pointer': pointer or '', 'value': projected,
                                **({'missing_fields': [key for key in fields if key not in current]}
                                   if any(key not in current for key in fields) else {})})
            for key, child in reversed(list(current.items())):
                stack.append((pointer + '/' + _json_pointer_token(key), child, depth + 1))
        elif isinstance(current, list):
            for index in range(len(current) - 1, -1, -1):
                stack.append((pointer + '/' + str(index), current[index], depth + 1))
    rows = matches[row_offset:row_offset + limit]
    return {'field': field, 'match': mode, **({'value': expected} if has_value else {}),
            'total_matches': len(matches), 'offset': row_offset, 'rows': rows,
            'has_more_rows': row_offset + limit < len(matches),
            'next_row_offset': row_offset + len(rows) if row_offset + len(rows) < len(matches) else None,
            'note': 'Archived JSON field matches only; JSON pointers identify exact source objects. No source value was inferred and this is not a new measurement.'}


def _find_page(document_id, title, raw, text, pointer, needle, offset, length):
    """Literal, bounded retrieval only. Never evaluate patterns or source text."""
    match = text.find(needle, offset)
    start = max(0, match - min(160, length // 4)) if match >= 0 else offset
    end = start + length
    return {'id': document_id, 'title': title, 'document_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            **({'json_pointer': pointer} if pointer is not None else {}),
            'find': needle, 'found': match >= 0, 'match_offset': match if match >= 0 else None,
            'offset': start, 'total_chars': len(text), 'text': text[start:end] if match >= 0 else '',
            'next_search_offset': match + 1 if match >= 0 else None,
            'has_more': match >= 0 and end < len(text), 'recorded_not_resimulated': True,
            'offset_scope': 'JSON serialization of selected subtree' if pointer is not None else 'original document characters',
            'note': 'Literal source match, not validation or a new measurement. Use next_search_offset to find the next occurrence; do not traverse the whole netlist.'}
