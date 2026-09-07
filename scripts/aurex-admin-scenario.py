#!/usr/bin/env python3
"""Operator-only admin dry-run scenarios in the existing Web FIFO.

prepare is read-only. enqueue requires --execute AND a new --receipt filename;
the existing Web worker WILL run an enqueued request. Never retry a receipt:
use reconcile to read its exact task ID after any failure or uncertain outcome.
This harness starts no worker and calls no model/community API. Same-session
follow-ups require an existing admin session; every submission has a new task ID.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
MAX_CHARACTERS = 500000
KIND = 'admin_dry_run_scenario_v1'
DEFAULT_TITLE = 'Admin dry-run scenario'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@contextmanager
def readonly(path):
    path = Path(path).resolve(strict=True)
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    try:
        yield db
    finally:
        db.close()


def prepare(database, request, *, session_id=None, title=None):
    if not isinstance(request, str) or not request.strip() or len(request) > MAX_CHARACTERS:
        raise ValueError('Request must be a nonempty string of at most 500000 characters')
    if title is not None and (not isinstance(title, str) or not title.strip() or len(title) > 120):
        raise ValueError('Title must be nonempty text of at most 120 characters')
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 200):
        raise ValueError('Session ID must be specific existing nonempty text')
    with readonly(database) as db:
        required = {'id', 'session_id', 'prompt', 'original_user_request', 'source', 'target',
                    'explicit_publish_requested', 'requester_user_id', 'requester_nickname',
                    'reply_id', 'metadata', 'input_data', 'cancel_requested'}
        if not required <= {r['name'] for r in db.execute('PRAGMA table_info(runs)')}:
            raise ValueError('Task schema is not current; application migration is required')
        if session_id is not None:
            row = db.execute('SELECT source FROM sessions WHERE id=?', (session_id,)).fetchone()
            if row is None:
                raise ValueError('Follow-up session does not exist; omit --session-id for a new session')
            if row['source'] != 'admin':
                raise ValueError('Follow-ups require an admin session, not a community/Web conversation')
    return {'prepared_only': True, 'enqueued': False,
            'database': str(Path(database).resolve(strict=True)),
            'session_id': session_id, 'existing_session': session_id is not None,
            'request_sha256': digest(request), 'request_characters': len(request),
            'title_sha256': digest(title or DEFAULT_TITLE),
            'source': 'admin', 'dry_run': True, 'explicit_publish_requested': False,
            'requester_user_id': None, 'requester_nickname': None, 'reply_id': None, 'target': None}


def write_receipt(path, record, *, exclusive=False):
    """Prewrite one private receipt; updates are atomic and never delete history."""
    path = Path(path)
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    if exclusive:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # O_EXCL also refuses existing symlinks, including dangling symlinks.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        sync_parent(path)
        return
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        sync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sync_parent(path):
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def enqueue(database, request, receipt, *, execute=False, session_id=None, title=None):
    if execute is not True:
        raise ValueError('enqueue requires explicit --execute; prepare never queues work')
    plan = prepare(database, request, session_id=session_id, title=title)
    destination = Path(os.path.abspath(os.fspath(receipt)))
    if destination.exists() or destination.is_symlink():
        raise ValueError('Receipt already exists: use reconcile; never repeat this submission')
    sid, rid = session_id or uuid.uuid4().hex, uuid.uuid4().hex
    metadata = {'admin_scenario': True, 'dry_run': True, 'scenario_id': rid,
                'request_sha256': plan['request_sha256']}
    record = {'kind': KIND, 'database': plan['database'], 'task_id': rid, 'session_id': sid,
              'existing_session': plan['existing_session'], 'request_sha256': plan['request_sha256'],
              'request_characters': plan['request_characters'], 'title_sha256': plan['title_sha256'],
              'metadata': metadata, 'submission': 'pending_do_not_retry', 'created': time.time(),
              'source': 'admin', 'dry_run': True, 'explicit_publish_requested': False,
              'worker_started_by_script': False}
    # All identifying information is durable BEFORE any DB write can occur.
    write_receipt(destination, record, exclusive=True)
    try:
        sys.path.insert(0, str(REPO / 'src'))
        from aurex.sessiondb import SessionDB

        class ExistingSessionDB(SessionDB):
            def __init__(self, path):
                # No migrations/legacy UPDATEs; only enqueue_task writes new work.
                self.path = str(Path(path).resolve(strict=True))

        result = ExistingSessionDB(database).enqueue_task(
            sid, request, prompt=request, task_id=rid, source='admin', images=[],
            title=title or DEFAULT_TITLE, target=None, explicit_publish_requested=False,
            requester_user_id=None, requester_nickname=None, reply_id=None, metadata=metadata)
        if result != rid:
            raise RuntimeError('Unexpected enqueue receipt; reconcile the prewritten task ID')
    except BaseException:
        record['submission'] = 'uncertain_do_not_repeat_reconcile_task_id'
        write_receipt(destination, record)
        raise
    record['submission'] = 'queued_in_existing_fifo'
    write_receipt(destination, record)
    return {**record, 'receipt': str(destination), 'assessment': 'not_scored'}


def ledger_audit(path, task_id):
    result = {'available': bool(path and Path(path).is_file()), 'publication_rows': 0,
              'schema_complete': False, 'legacy_publication_rows': 0,
              'final_reply_states': [], 'violations': []}
    if result['available']:
        with readonly(path) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            result['schema_complete'] = {'task_scopes', 'task_publications', 'approvals', 'task_final_answers'} <= tables
            if 'task_scopes' in tables:
                row = db.execute('SELECT binding FROM task_scopes WHERE task_id=?', (task_id,)).fetchone()
                if row:
                    scope = json.loads(row['binding'])
                    if (scope.get('source') != 'admin' or scope.get('dry_run') is not True
                        or scope.get('explicit_publish_requested') is not False
                        or any(scope.get(k) is not None for k in ('requester_user_id', 'requester_nickname', 'target'))):
                        result['violations'].append('Task action scope is not unbound admin/dry-run/nonpublishing')
            for table, key, out in [('task_publications', 'task_id', 'publication_rows'),
                                    ('approvals', 'run_id', 'legacy_publication_rows')]:
                if table in tables:
                    result[out] = db.execute(f'SELECT count(*) FROM {table} WHERE {key}=?', (task_id,)).fetchone()[0]
            if result['publication_rows'] or result['legacy_publication_rows']:
                result['violations'].append('Unexpected publication ledger entry')
            if 'task_final_answers' in tables:
                for row in db.execute('SELECT state,reply_receipt IS NOT NULL AS has_receipt FROM task_final_answers WHERE task_id=?', (task_id,)):
                    result['final_reply_states'].append(row['state'])
                    if row['state'] in {'replying', 'replied', 'unknown'} or row['has_receipt']:
                        result['violations'].append('Community reply attempted or uncertain/recorded receipt')
    result['no_recorded_external_writes'] = not result['violations'] if result['schema_complete'] else None
    result['assertion_scope'] = 'ledger_only_not_independent_community_verification'
    return result


def reconcile(receipt, *, ledger=None):
    path = Path(receipt).resolve(strict=True)
    if path.stat().st_size > 65536:
        raise ValueError('Receipt is oversized')
    record = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(record, dict) or record.get('kind') != KIND
        or record.get('source') != 'admin' or record.get('dry_run') is not True
        or record.get('explicit_publish_requested') is not False):
        raise ValueError('Not a recognized admin dry-run receipt')
    rid, sid = record['task_id'], record['session_id']
    with readonly(record['database']) as db:
        row = db.execute('SELECT * FROM runs WHERE id=?', (rid,)).fetchone()
        if row is None:
            return {'task_id': rid, 'session_id': sid, 'found': False,
                    'submission': record['submission'], 'reconciliation': 'not_found_do_not_resubmit',
                    'enqueued_by_reconcile': False}
        task = dict(row)
        violations = []
        if (task['session_id'] != sid or digest(task['prompt']) != record['request_sha256']
            or digest(task['original_user_request']) != record['request_sha256']
            or len(task['original_user_request']) != record['request_characters']
            or digest(task['title']) != record['title_sha256']):
            violations.append('Stored task does not match prewritten request/session/title binding')
        if (task['source'] != 'admin' or task['explicit_publish_requested']
            or json.loads(task['metadata']) != record['metadata']
            or json.loads(task['metadata']).get('dry_run') is not True
            or any(task[k] is not None for k in ('requester_user_id', 'requester_nickname', 'reply_id'))
            or json.loads(task['target']) or json.loads(task['input_data']).get('images')):
            violations.append('Stored task changed its admin/dry-run/no-destination policy')
        counts = {r['kind']: r['n'] for r in db.execute("SELECT kind,count(*) n FROM events WHERE run_id=? AND kind NOT IN ('reasoning_delta','text_delta') GROUP BY kind", (rid,))}
    return {'task_id': rid, 'session_id': sid, 'found': True, 'status': task['status'],
            'receipt_submission': record['submission'], 'reconciliation': 'task_found' if not violations else 'binding_mismatch',
            'binding_violations': violations, 'events_excluding_deltas': counts,
            'external_write_audit': ledger_audit(ledger, rid), 'enqueued_by_reconcile': False,
            'assessment': 'not_scored'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'enqueue', 'reconcile'], nargs='?', default='prepare')
    request = parser.add_mutually_exclusive_group()
    request.add_argument('--request')
    request.add_argument('--request-file', type=Path, help='UTF-8 user request only; never a JSON task-policy object')
    parser.add_argument('--database', type=Path, default=REPO / '.aurex/aurex.sqlite3')
    parser.add_argument('--session-id', help='Existing admin session for follow-up; omitted means new session')
    parser.add_argument('--title')
    parser.add_argument('--receipt', type=Path, help='New unique private receipt for enqueue; existing receipt for reconcile')
    parser.add_argument('--ledger', type=Path, default=REPO / '.aurex/cache/.publication/ledger.sqlite3')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'reconcile':
        if not args.receipt or any(x is not None for x in (args.request, args.request_file, args.session_id, args.title)) or args.execute:
            parser.error('reconcile requires only an existing --receipt and optional --ledger; it never submits')
        result = reconcile(args.receipt, ledger=args.ledger)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 4 if result.get('binding_violations') or result.get('external_write_audit', {}).get('violations') else 0
    if args.request is None and args.request_file is None:
        parser.error('--request or --request-file is required')
    text = args.request
    if args.request_file is not None:
        if args.request_file.stat().st_size > MAX_CHARACTERS * 4:
            parser.error('Request file is too large')
        text = args.request_file.read_text(encoding='utf-8')
    if args.action == 'prepare':
        result = prepare(args.database, text, session_id=args.session_id, title=args.title)
    else:
        if not args.execute or not args.receipt:
            parser.error('enqueue requires --execute and a new --receipt filename')
        result = enqueue(args.database, text, args.receipt, execute=True, session_id=args.session_id, title=args.title)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
