"""Session-owned HDL files with atomic exact edits and immutable SQL revisions.

Model arguments never name host paths. Verification receives a frozen snapshot;
editing the workspace cannot change an in-flight compilation or certify a new
revision using an old report.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

from .registry import ToolError, ToolRegistry, ToolRuntime, ToolSpec


def _hashes(files):
    return {name: hashlib.sha256(item['content'].encode()).hexdigest() for name, item in sorted(files.items())}


def _bundle(files):
    return hashlib.sha256(json.dumps(_hashes(files), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _name(name):
    if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,80}\.(?:v|sv)', name)
            or name.startswith('aurex_profile')):
        raise ToolError('Use a flat .v/.sv filename, without paths or reserved verifier names')
    return name


def _validate(files):
    if not isinstance(files, dict) or not 1 <= len(files) <= 16:
        raise ToolError('Workspace requires 1..16 files')
    total = 0
    for name, item in files.items():
        _name(name)
        if not isinstance(item, dict) or set(item) != {'content', 'role'} or item['role'] not in ('source', 'testbench'):
            raise ToolError('File role must be source or testbench')
        content = item['content']
        if not isinstance(content, str) or not content.strip() or '\0' in content:
            raise ToolError('HDL files need nonempty text without NUL')
        size = len(content.encode())
        total += size
        if size > 256_000 or total > 512_000:
            raise ToolError('HDL file/bundle exceeds the existing 256000/512000 byte limits')
    if not any(item['role'] == 'source' for item in files.values()):
        raise ToolError('Keep at least one design source file')


@contextmanager
def _store(runtime: ToolRuntime):
    if not isinstance(runtime.session_id, str) or not runtime.session_id or not runtime.task_id:
        raise ToolError('HDL workspace requires a server-bound session and task')
    if runtime.check_cancel:
        runtime.check_cancel()
    root = Path(runtime.cache_dir).resolve() / 'hdl-workspaces'
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    db = sqlite3.connect(root / 'workspaces.sqlite3', timeout=10)
    db.row_factory = sqlite3.Row
    try:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS workspaces(
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, label TEXT NOT NULL,
                head INTEGER NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS revisions(
                workspace_id TEXT NOT NULL, revision INTEGER NOT NULL, task_id TEXT NOT NULL,
                files_json TEXT NOT NULL, source_sha256 TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY(workspace_id,revision));
            CREATE TABLE IF NOT EXISTS checks(
                workspace_id TEXT NOT NULL, revision INTEGER NOT NULL, report_path TEXT NOT NULL,
                verified INTEGER NOT NULL, profile TEXT NOT NULL, top TEXT NOT NULL, created REAL NOT NULL);
        ''')
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def _load(db, runtime, workspace_id, revision=None):
    if not isinstance(workspace_id, str) or not re.fullmatch(r'[0-9a-f]{32}', workspace_id):
        raise ToolError('Invalid workspace ID')
    row = db.execute('SELECT * FROM workspaces WHERE id=? AND session_id=?',
                     (workspace_id, runtime.session_id)).fetchone()
    if row is None:
        raise ToolError('Workspace not found in this session')
    revision = row['head'] if revision is None else revision
    if type(revision) is not int or revision < 1:
        raise ToolError('revision must be a positive integer')
    saved = db.execute('SELECT * FROM revisions WHERE workspace_id=? AND revision=?',
                       (workspace_id, revision)).fetchone()
    if saved is None:
        raise ToolError('Workspace revision does not exist')
    files = json.loads(saved['files_json'])
    _validate(files)
    if _bundle(files) != saved['source_sha256']:
        raise ToolError('Stored workspace integrity check failed')
    return dict(row), revision, files


def _info(db, row, revision, files):
    check = db.execute('SELECT report_path,verified,profile,top FROM checks WHERE workspace_id=? AND revision=? ORDER BY created DESC LIMIT 1',
                       (row['id'], revision)).fetchone()
    return {'workspace_id': row['id'], 'label': row['label'], 'workspace_revision': revision,
        'head_revision': row['head'], 'source_sha256': _bundle(files),
        'files': [{'name': name, 'role': item['role'], 'sha256': _hashes(files)[name],
                   'characters': len(item['content'])} for name, item in sorted(files.items())],
        'last_verification': ({**dict(check), 'verified': bool(check['verified'])} if check else None),
        'note': 'Source revision, not a functional PASS. Verify this revision; old reports do not certify later edits.'}


def workspace_snapshot(runtime, workspace_id, revision, *, require_head=True):
    if type(revision) is not int or revision < 1:
        raise ToolError('Pin an explicit positive workspace revision')
    with _store(runtime) as db:
        row, version, files = _load(db, runtime, workspace_id, revision)
        if require_head and row['head'] != version:
            raise ToolError(f'Stale workspace revision {version}; current revision is {row["head"]}. Read before verifying/exporting.')
        return _info(db, row, version, files), files


def workspace_check(runtime, report):
    """Record an actual immutable report, without promoting a stale revision."""
    with _store(runtime) as db:
        db.execute('BEGIN IMMEDIATE')
        row, revision, files = _load(db, runtime, report['workspace_id'], report['workspace_revision'])
        if _bundle(files) != report['source_sha256']:
            raise ToolError('Verification does not match its workspace source')
        db.execute('INSERT INTO checks VALUES(?,?,?,?,?,?,?)', (row['id'], revision, report['report_path'],
            int(report['verified']), report['profile'], report['top'], time.time()))
        return row['head'] == revision


def hdl_workspace_create(runtime, args):
    raw = args.get('files')
    if not isinstance(raw, list):
        raise ToolError('files must be an array')
    files = {}
    for item in raw:
        if not isinstance(item, dict) or set(item) - {'name', 'content', 'role'} or not {'name', 'content'} <= set(item):
            raise ToolError('Each file needs name, content and optionally role')
        name = _name(item['name'])
        if name in files:
            raise ToolError('Duplicate filename')
        files[name] = {'content': item['content'], 'role': item.get('role', 'source')}
    _validate(files)
    label = args.get('label', 'HDL workspace')
    if not isinstance(label, str) or not 1 <= len(label) <= 120:
        raise ToolError('label must contain 1..120 characters')
    with _store(runtime) as db:
        wid, now = uuid.uuid4().hex, time.time()
        db.execute('BEGIN IMMEDIATE')
        db.execute('INSERT INTO workspaces VALUES(?,?,?,?,?,?)', (wid, runtime.session_id, label, 1, now, now))
        db.execute('INSERT INTO revisions VALUES(?,?,?,?,?,?)',
                   (wid, 1, runtime.task_id, json.dumps(files, ensure_ascii=False), _bundle(files), now))
        row, revision, files = _load(db, runtime, wid)
        return _info(db, row, revision, files)


def hdl_workspace_read(runtime, args):
    with _store(runtime) as db:
        if 'workspace_id' not in args:
            if set(args):
                raise ToolError('Supply workspace_id before file selectors')
            rows = db.execute('SELECT id,label,head FROM workspaces WHERE session_id=? ORDER BY updated DESC LIMIT 50',
                              (runtime.session_id,)).fetchall()
            return {'workspaces': [dict(row) for row in rows], 'limit': 50}
        row, revision, files = _load(db, runtime, args['workspace_id'], args.get('revision'))
        out = _info(db, row, revision, files)
        if 'name' not in args:
            if any(key in args for key in ('find', 'offset', 'length')):
                raise ToolError('Supply a filename for text selectors')
            out['history'] = [dict(item) for item in db.execute('SELECT revision,source_sha256 FROM revisions WHERE workspace_id=? ORDER BY revision DESC LIMIT 10', (row['id'],))]
            return out
        name = _name(args['name'])
        if name not in files:
            raise ToolError('File not present in this revision')
        offset, length = args.get('offset', 0), args.get('length', 8000)
        if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 20000:
            raise ToolError('offset >=0 and length 1..20000 are required')
        text = files[name]['content']
        if 'find' in args:
            needle = args['find']
            if not isinstance(needle, str) or not needle or len(needle) > 2000:
                raise ToolError('find must be an exact substring of 1..2000 characters')
            hit = text.find(needle, offset)
            out.update(found=hit >= 0, match_offset=hit if hit >= 0 else None)
            if hit < 0:
                out.update(name=name, text='', offset=offset, total_chars=len(text), has_more=False)
                return out
            # Return verbatim source around the hit, never a synthetic summary.
            offset = max(0, hit - min(300, length // 4))
        out.update(name=name, offset=offset, total_chars=len(text), text=text[offset:offset + length],
                   has_more=offset + length < len(text), next_offset=min(len(text), offset + length),
                   source_kind='workspace_hdl_source', recorded_not_resimulated=True)
        return out


def _commit(db, runtime, row, old_revision, files):
    _validate(files)
    now, revision = time.time(), old_revision + 1
    db.execute('INSERT INTO revisions VALUES(?,?,?,?,?,?)', (row['id'], revision, runtime.task_id,
        json.dumps(files, ensure_ascii=False), _bundle(files), now))
    db.execute('UPDATE workspaces SET head=?,updated=? WHERE id=?', (revision, now, row['id']))
    return _info(db, {**row, 'head': revision}, revision, files)


def hdl_workspace_write(runtime, args):
    """Create a module/testbench, or explicitly replace one hash-checked file."""
    if type(args.get('expected_revision')) is not int or args['expected_revision'] < 1:
        raise ToolError('Pin an explicit positive expected_revision before writing')
    with _store(runtime) as db:
        db.execute('BEGIN IMMEDIATE')
        row, revision, files = _load(db, runtime, args['workspace_id'], args['expected_revision'])
        if revision != row['head']:
            raise ToolError(f'Stale edit; current revision is {row["head"]}')
        name = _name(args['name'])
        old = files.get(name)
        if 'expected_sha256' not in args or args['expected_sha256'] != (_hashes(files)[name] if old else None):
            raise ToolError('Expected file SHA differs; read again. Use null only for a new file.')
        files[name] = {'content': args['content'], 'role': args.get('role', old['role'] if old else 'source')}
        if old == files[name]:
            raise ToolError('No changes to write. This invalid no-op was not committed; read the current file '
                            'and submit changed content, not the same write arguments again.')
        return _commit(db, runtime, row, revision, files)


def hdl_workspace_edit(runtime, args):
    if type(args.get('expected_revision')) is not int or args['expected_revision'] < 1:
        raise ToolError('Pin an explicit positive expected_revision before editing')
    edits = args.get('edits')
    if not isinstance(edits, list) or not 1 <= len(edits) <= 32:
        raise ToolError('edits must contain 1..32 exact replacements')
    with _store(runtime) as db:
        db.execute('BEGIN IMMEDIATE')
        row, revision, files = _load(db, runtime, args['workspace_id'], args['expected_revision'])
        if revision != row['head']:
            raise ToolError(f'Stale edit; current revision is {row["head"]}. Read before editing.')
        changes = []
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) - {'name', 'old_text', 'new_text', 'replace_all'}:
                raise ToolError('Invalid edit fields')
            name = _name(edit.get('name'))
            if name not in files:
                raise ToolError('Use hdl_workspace_write to create a new file')
            old, new = edit.get('old_text'), edit.get('new_text')
            if not isinstance(old, str) or not old or not isinstance(new, str) or old == new:
                raise ToolError('old_text must be nonempty and differ from new_text. This invalid no-op was '
                                'not committed; read the current file and do not resubmit identical arguments.')
            replace_all = edit.get('replace_all', False)
            if type(replace_all) is not bool:
                raise ToolError('replace_all must be boolean')
            text = files[name]['content']
            count = text.count(old)
            if count == 0 or (count != 1 and not replace_all):
                raise ToolError(f'Exact edit found {count} matches in {name}; nothing was committed. Read the '
                                'current revision and use its exact text (or add context / explicitly set '
                                'replace_all); do not resubmit the same rejected arguments.')
            files[name] = {**files[name], 'content': text.replace(old, new, -1 if replace_all else 1)}
            changes.append({'name': name, 'replacements': count if replace_all else 1,
                'old_preview': old[:400], 'new_preview': new[:400],
                'preview_truncated': len(old) > 400 or len(new) > 400})
        out = _commit(db, runtime, row, revision, files)
        out.update(changes=changes, previous_revision=revision, atomic=True)
        return out


def register_hdl_workspace_tools(registry: ToolRegistry):
    string = {'type': 'string'}
    revision = {'type': 'integer', 'minimum': 1}
    role = {'enum': ['source', 'testbench']}
    def register(name, description, properties, required, handler):
        registry.register(ToolSpec(name, description, {'type': 'object', 'additionalProperties': False,
            'properties': properties, 'required': required}, handler))
    register('hdl_workspace_create', 'Create persistent session-owned HDL files once. Use exact edits for corrections instead of resending entire designs. Returns revision and hashes; nothing is simulated or published.',
        {'label': string, 'files': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': {
            'type': 'object', 'additionalProperties': False, 'properties': {'name': string, 'content': string, 'role': role},
            'required': ['name', 'content']}}}, ['files'], hdl_workspace_create)
    register('hdl_workspace_read', 'Read exact HDL text or list workspace files/history. Omit workspace_id to list this session workspaces; omit name for file hashes; revision optionally reads a historical immutable version. Optional find locates an exact substring within the named file, starting at offset, and returns surrounding verbatim text. Offset/length are characters, not lines.',
        {'workspace_id': string, 'revision': revision, 'name': string, 'offset': {'type': 'integer', 'minimum': 0},
         'find': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
         'length': {'type': 'integer', 'minimum': 1, 'maximum': 20000}}, [], hdl_workspace_read)
    register('hdl_workspace_write', 'Add a source/testbench file, or explicitly replace one entire file. Prefer edit for small fixes. Expected revision and existing SHA prevent overwriting newer code; expected_sha256=null means a new filename only.',
        {'workspace_id': string, 'expected_revision': revision, 'name': string, 'content': string, 'role': role,
         'expected_sha256': {'type': ['string', 'null']}},
        ['workspace_id', 'expected_revision', 'name', 'content', 'expected_sha256'], hdl_workspace_write)
    register('hdl_workspace_edit', 'Apply exact old_text/new_text replacements atomically to persistent HDL. One match required unless replace_all=true. On stale revision or ambiguous/missing match nothing is changed. Read before retrying; no fuzzy matching or automatic compilation.',
        {'workspace_id': string, 'expected_revision': revision, 'edits': {'type': 'array', 'minItems': 1, 'maxItems': 32,
         'items': {'type': 'object', 'additionalProperties': False, 'properties': {'name': string, 'old_text': string,
             'new_text': string, 'replace_all': {'type': 'boolean'}}, 'required': ['name', 'old_text', 'new_text']}}},
        ['workspace_id', 'expected_revision', 'edits'], hdl_workspace_edit)
