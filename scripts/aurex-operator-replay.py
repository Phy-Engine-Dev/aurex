#!/usr/bin/env python3
"""Operator-only, dry-run replays in the existing Web FIFO.

prepare/status use SQLite mode=ro and never initialize/migrate a database.
enqueue requires --execute and copies the exact stored request into a NEW admin
session/task with dry_run=true and publication intent=false. An explicit
--request-append adds the same user supplement to both request fields without
rewriting their original content. No worker is started.
This script never calls a model or a community API. The running Web worker WILL
execute an explicitly enqueued task, so enqueue is not a harmless preparation step.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import uuid


REPO = Path(__file__).resolve().parents[1]
CASES = {
    'cpu': '7c5bbf28821a6b7a5fc8b0b55fdfd161',
    'wall': '3a5b0e975d62dbb784dc785379b47802',
    'matrix': '38a18f0e3187604820724adf9c279bd0',
    'cover-a': '88fd1dd4071abeb7880f30fa2c3d5a7a',
    'cover-b': '6a65572427bbaa3f0222c1e63bf8a79a',
}
TERMINAL = {'completed', 'error', 'cancelled', 'interrupted', 'needs_attention', 'blocked'}
MAX_REQUEST_CHARACTERS = 500000
REQUEST_APPEND_SEPARATOR = '\n\n用户补充要求：\n'


def digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


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


def read_task(path, task_id):
    if not isinstance(task_id, str) or not task_id or len(task_id) > 200:
        raise ValueError('A specific existing task ID is required')
    with readonly(path) as db:
        row = db.execute('SELECT * FROM runs WHERE id=?', (task_id,)).fetchone()
    if row is None:
        raise ValueError('Source/task ID does not exist in this database')
    task = dict(row)
    for name in ('target', 'metadata', 'input_data'):
        task[name] = json.loads(task.get(name) or '{}')
    return task


def source_summary(task):
    return {'task_id': task['id'], 'session_id': task['session_id'], 'status': task['status'],
            'source': task['source'], 'prompt_characters': len(task['prompt']),
            'prompt_sha256': digest(task['prompt']),
            'original_user_request_characters': len(task['original_user_request']),
            'original_user_request_sha256': digest(task['original_user_request']),
            'image_count': len(task['input_data'].get('images', [])),
            'has_context_envelope': task['prompt'].startswith('CONTEXT_JSON:'),
            'replay_of': task['metadata'].get('replay_of')}


def replay_request(task, request_append=None):
    """Preserve source bytes; supplements are explicit user text, never a rewrite."""
    prompt, request = task['prompt'], task['original_user_request']
    if request_append is not None:
        if not isinstance(request_append, str) or not request_append.strip():
            raise ValueError('--request-append must be a nonempty string')
        suffix = REQUEST_APPEND_SEPARATOR + request_append
        if max(len(prompt) + len(suffix), len(request) + len(suffix)) > MAX_REQUEST_CHARACTERS:
            raise ValueError('Appended prompt and original_user_request must each be at most 500000 characters')
        prompt += suffix
        request += suffix
    provenance = {
        'request_append_present': request_append is not None,
        'request_append_sha256': digest(request_append) if request_append is not None else None,
        'request_append_characters': len(request_append) if request_append is not None else 0,
        'source_prompt_sha256': digest(task['prompt']),
        'source_original_request_sha256': digest(task['original_user_request']),
        'replay_prompt_sha256': digest(prompt),
        'replay_original_request_sha256': digest(request),
    }
    return prompt, request, provenance


def ledger_audit(path, task):
    result = {'available': Path(path).is_file(), 'publication_rows': 0, 'legacy_publication_rows': 0,
              'final_reply_states': [], 'scope_present': False, 'violations': []}
    is_replay = task['metadata'].get('operator_replay') is True
    if is_replay:
        if task['source'] != 'admin' or task['metadata'].get('dry_run') is not True or task['explicit_publish_requested']:
            result['violations'].append('Replay task metadata permits an unexpected external action')
    if result['available']:
        with readonly(path) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'task_scopes' in tables:
                row = db.execute('SELECT binding FROM task_scopes WHERE task_id=?', (task['id'],)).fetchone()
                result['scope_present'] = row is not None
                if row:
                    scope = json.loads(row['binding'])
                    if is_replay and (scope.get('source') != 'admin' or scope.get('dry_run') is not True
                                      or scope.get('explicit_publish_requested')):
                        result['violations'].append('Replay action scope is not admin/dry-run/nonpublishing')
            if 'task_publications' in tables:
                result['publication_rows'] = db.execute('SELECT COUNT(*) FROM task_publications WHERE task_id=?', (task['id'],)).fetchone()[0]
            if 'approvals' in tables:
                result['legacy_publication_rows'] = db.execute('SELECT COUNT(*) FROM approvals WHERE run_id=?', (task['id'],)).fetchone()[0]
            if is_replay and result['publication_rows'] + result['legacy_publication_rows']:
                result['violations'].append('Unexpected publication record exists for a nonpublishing replay')
            if 'task_final_answers' in tables:
                for row in db.execute('SELECT state,reply_receipt IS NOT NULL AS has_receipt FROM task_final_answers WHERE task_id=?', (task['id'],)):
                    result['final_reply_states'].append(row['state'])
                    if is_replay and (row['state'] in {'replying', 'replied', 'unknown'} or row['has_receipt']):
                        result['violations'].append('Community reply was attempted or has an uncertain/recorded receipt')
    # This is a ledger assertion, never a claim to independently inspect the community.
    result['no_recorded_external_writes'] = (not result['violations']) if is_replay and result['available'] else None
    result['assertion_scope'] = 'replay_ledger_only' if is_replay else 'not_a_replay_task'
    return result


def metrics(path, task_id):
    counts, repeated, finishes = Counter(), Counter(), Counter()
    tools, models, contexts, errors = [], [], [], []
    with readonly(path) as db:
        rows = db.execute("SELECT kind,data,created FROM events WHERE run_id=? AND kind NOT IN ('reasoning_delta','text_delta') ORDER BY id", (task_id,))
        for row in rows:
            kind, data = row['kind'], json.loads(row['data'])
            counts[kind] += 1
            if not isinstance(data, dict):
                continue
            if kind == 'tool_start':
                raw = data.get('arguments', {})
                signature = digest(json.dumps(raw, ensure_ascii=False, sort_keys=True))
                repeated[(str(data.get('name')), signature)] += 1
            elif kind == 'tool_end':
                tools.append({k: data[k] for k in ('name', 'call_id', 'ok', 'duration') if k in data})
            elif kind in {'model_start', 'model_end'}:
                model = {k: data[k] for k in ('step', 'thinking', 'reasoning_characters', 'content_characters', 'finish_reason') if k in data}
                if isinstance(data.get('usage'), dict):
                    model['usage'] = {k: data['usage'][k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens') if k in data['usage']}
                model['kind'] = kind
                models.append(model)
                if data.get('finish_reason'):
                    finishes[str(data['finish_reason'])] += 1
            elif kind in {'context_budget', 'compaction_start', 'compaction_end', 'compaction_fallback'}:
                contexts.append({'kind': kind, **{k: v for k, v in data.items() if isinstance(v, (int, float, bool)) and k != 'id'}})
            elif kind == 'error':
                # Do not copy user text, account information, or arbitrary exception payloads.
                errors.append({'type': data.get('type'), 'error_sha256': digest(str(data.get('error', '')))})
    return {'event_counts_excluding_deltas': dict(counts), 'tool_results': tools, 'model_requests': models,
            'finish_reasons': dict(finishes), 'context_measurements': contexts, 'errors': errors,
            'repeated_call_signatures': [{'name': name, 'arguments_sha256': signature, 'count': count}
                                         for (name, signature), count in repeated.items() if count > 1]}


def status(database, ledger, task_id):
    task = read_task(database, task_id)
    return {'task': source_summary(task), 'metadata_policy': {
                'operator_replay': task['metadata'].get('operator_replay') is True,
                'dry_run': task['metadata'].get('dry_run') is True,
                'explicit_publish_requested': bool(task['explicit_publish_requested'])},
            'metrics': metrics(database, task_id), 'external_write_audit': ledger_audit(ledger, task),
            'assessment': 'not_scored'}


def write_record(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def enqueue(database, source_task_id, records, *, execute=False, request_append=None):
    if not execute:
        raise ValueError('enqueue requires explicit --execute; prepare/status never enqueue')
    original = read_task(database, source_task_id)
    if not original['prompt'].strip() or not original['original_user_request'].strip():
        raise ValueError('Original prompt/request is empty; do not invent a replacement')
    prompt, request, provenance = replay_request(original, request_append)
    images = original['input_data'].get('images', [])
    if not isinstance(images, list) or any(not isinstance(x, str) or not Path(x).is_file() for x in images):
        raise ValueError('Original image input is unavailable; replay must not silently drop it')
    # Existing-database adapter deliberately avoids SessionDB.__init__, which runs
    # migrations and legacy UPDATEs. Only enqueue_task may write the two NEW rows.
    sys.path.insert(0, str(REPO / 'src'))
    from aurex.sessiondb import SessionDB
    class ExistingSessionDB(SessionDB):
        def __init__(self, path):
            self.path = str(Path(path).resolve(strict=True))
    writer = ExistingSessionDB(database)
    with readonly(database) as db:
        required = {'requester_nickname', 'reply_id', 'metadata', 'original_user_request', 'input_data', 'cancel_requested'}
        if not required <= {r['name'] for r in db.execute('PRAGMA table_info(runs)')}:
            raise ValueError('Existing task DB schema is not current; migrate through the application first')
    sid, rid = uuid.uuid4().hex, uuid.uuid4().hex
    original_summary = source_summary(original)
    metadata = {'operator_replay': True, 'replay_of': source_task_id, 'dry_run': True,
                'replay_source_session': original['session_id'],
                **provenance}
    record = {'source': original_summary, 'new_session_id': sid, 'new_task_id': rid,
              'metadata': metadata, 'submission': 'pending', 'assessment': 'not_scored'}
    records = Path(records)
    records.mkdir(parents=True, exist_ok=True)
    destination = records / (rid + '.json')
    write_record(destination, record)
    try:
        result = writer.enqueue_task(sid, request, prompt=prompt,
                    task_id=rid, images=images, source='admin', target=original['target'] or None,
                    title='Read-only replay of ' + source_task_id, explicit_publish_requested=False,
                    requester_user_id=None, requester_nickname=None, reply_id=None, metadata=metadata)
        if result != rid:
            raise RuntimeError('Unexpected enqueue task receipt')
    except BaseException:
        record['submission'] = 'uncertain_do_not_repeat_check_recorded_task_id'
        write_record(destination, record)
        raise
    record['submission'] = 'queued_in_existing_fifo'
    write_record(destination, record)
    return {'new_task_id': rid, 'new_session_id': sid, 'replay_of': source_task_id,
            'source': 'admin', 'dry_run': True, 'explicit_publish_requested': False,
            **provenance,
            'record': str(destination), 'worker_started_by_script': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'enqueue', 'status'], nargs='?', default='prepare')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--case', choices=CASES)
    selection.add_argument('--task-id')
    parser.add_argument('--database', type=Path, default=REPO / '.aurex/aurex.sqlite3')
    parser.add_argument('--ledger', type=Path, default=REPO / '.aurex/cache/.publication/ledger.sqlite3')
    parser.add_argument('--records', type=Path, default=REPO / '.aurex/cache/operator-replays')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--request-append', default=None,
                        help='Explicit user supplement for prepare/enqueue; append verbatim to both request fields')
    parser.add_argument('--observe', action='store_true', help='Read-only polling after explicit enqueue or in status mode')
    parser.add_argument('--timeout', type=float, default=60, help='Observer duration only, never a task budget')
    args = parser.parse_args(argv)
    if not 0 < args.timeout < float('inf'):
        parser.error('--timeout must be finite and positive')
    if args.request_append is not None and not args.request_append.strip():
        parser.error('--request-append must be a nonempty string')
    if args.request_append is not None and args.action == 'status':
        parser.error('--request-append is only valid for prepare/enqueue')
    task_id = args.task_id or CASES.get(args.case)
    if args.action == 'prepare':
        selected = [(args.case or 'explicit-task', task_id)] if task_id else list(CASES.items())
        summaries = []
        for case, old_id in selected:
            try:
                original = read_task(args.database, old_id)
                _, _, provenance = replay_request(original, args.request_append)
                summaries.append({'case': case, **source_summary(original), 'request_provenance': provenance})
            except ValueError as exc:
                summaries.append({'case': case, 'task_id': old_id, 'error': str(exc)})
        print(json.dumps({'prepared_only': True, 'enqueued': False, 'sources': summaries}, ensure_ascii=False, indent=2))
        return 0
    if not task_id:
        parser.error('--case or --task-id is required')
    if args.action == 'enqueue':
        if not args.execute:
            parser.error('enqueue requires --execute; the existing Web worker will run it')
        receipt = enqueue(args.database, task_id, args.records, execute=True, request_append=args.request_append)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        task_id = receipt['new_task_id']
        if not args.observe:
            return 0
    end = time.monotonic() + args.timeout
    while True:
        snapshot = status(args.database, args.ledger, task_id)
        print(json.dumps(snapshot, ensure_ascii=False, indent=2), flush=True)
        if snapshot['external_write_audit']['violations']:
            return 4
        if not args.observe or snapshot['task']['status'] in TERMINAL:
            return 0
        if time.monotonic() >= end:
            print(json.dumps({'observation': 'expired_task_unchanged', 'task_id': task_id,
                              'resume': 'status --task-id ' + task_id + ' --observe'}))
            return 2
        time.sleep(min(2, end - time.monotonic()))


if __name__ == '__main__':
    raise SystemExit(main())
