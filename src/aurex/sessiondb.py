"""Durable conversation and execution journal, shared by CLI, bot and Web."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class SessionDB:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL,
                    status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                    summary TEXT NOT NULL DEFAULT '', compacted_until INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, role TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_session ON events(session_id, id);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, prompt TEXT NOT NULL,
                    status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, path TEXT NOT NULL,
                    mime_type TEXT NOT NULL, label TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, title TEXT NOT NULL,
                    content TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_outcomes (
                    run_id TEXT NOT NULL, call_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    name TEXT NOT NULL, ok INTEGER NOT NULL, document_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL, created REAL NOT NULL,
                    PRIMARY KEY(run_id, call_id)
                );
                CREATE TABLE IF NOT EXISTS final_answers (
                    run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    review_id TEXT NOT NULL, message_id INTEGER NOT NULL, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_checkpoints (
                    run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    summary TEXT NOT NULL, compacted_until INTEGER NOT NULL,
                    updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_plan_items (
                    run_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    item_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    title TEXT NOT NULL, status TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '', evidence TEXT NOT NULL DEFAULT '[]',
                    created REAL NOT NULL, updated REAL NOT NULL,
                    PRIMARY KEY(run_id, item_id)
                );
                CREATE INDEX IF NOT EXISTS task_plan_order
                    ON task_plan_items(run_id, ordinal);
                CREATE TABLE IF NOT EXISTS subagent_runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    parent_run_id TEXT NOT NULL, depth INTEGER NOT NULL,
                    objective TEXT NOT NULL, context TEXT NOT NULL,
                    status TEXT NOT NULL, report TEXT NOT NULL DEFAULT '{}',
                    deadline_at REAL, created REAL NOT NULL, updated REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS subagent_runs_parent
                    ON subagent_runs(parent_run_id, created, id);
                CREATE TABLE IF NOT EXISTS subagent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, subagent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, parent_run_id TEXT NOT NULL,
                    role TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS subagent_messages_child
                    ON subagent_messages(subagent_id, id);
                CREATE TABLE IF NOT EXISTS subagent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, subagent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, parent_run_id TEXT NOT NULL,
                    kind TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS subagent_events_child
                    ON subagent_events(subagent_id, id);
                CREATE TABLE IF NOT EXISTS subagent_tool_outcomes (
                    subagent_id TEXT NOT NULL, call_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, parent_run_id TEXT NOT NULL,
                    name TEXT NOT NULL, ok INTEGER NOT NULL,
                    document_id TEXT NOT NULL, message_id INTEGER NOT NULL,
                    created REAL NOT NULL, PRIMARY KEY(subagent_id, call_id)
                );
                CREATE INDEX IF NOT EXISTS subagent_tools_parent
                    ON subagent_tool_outcomes(parent_run_id, created);
            """)
            columns = {r['name'] for r in db.execute('PRAGMA table_info(runs)')}
            if 'input_data' not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN input_data TEXT NOT NULL DEFAULT '{}'")
            if 'cancel_requested' not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")
            for name, declaration in {
                'requester_user_id': 'TEXT', 'original_user_request': "TEXT NOT NULL DEFAULT ''",
                'source': "TEXT NOT NULL DEFAULT 'web'", 'target': "TEXT NOT NULL DEFAULT '{}'",
                'explicit_publish_requested': 'INTEGER NOT NULL DEFAULT 0',
                'title': "TEXT NOT NULL DEFAULT ''",
                'requester_nickname': 'TEXT', 'reply_id': 'TEXT',
                'metadata': "TEXT NOT NULL DEFAULT '{}'",
            }.items():
                if name not in columns:
                    db.execute('ALTER TABLE runs ADD COLUMN ' + name + ' ' + declaration)
            db.execute("UPDATE runs SET original_user_request=prompt WHERE original_user_request=''")
            db.execute('CREATE INDEX IF NOT EXISTS runs_fifo ON runs(status,created,id)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def session(self, session_id: str | None = None, *, title: str = "New conversation", source: str = "web") -> str:
        sid = session_id or uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO sessions(id,title,source,status,created,updated) VALUES(?,?,?,'idle',?,?)",
                       (sid, title[:120], source, time.time(), time.time()))
        return sid

    def get(self, sid: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("""SELECT s.*, (SELECT r.id FROM runs r WHERE r.session_id=s.id
                AND r.status IN ('running','cancelling') ORDER BY r.created,r.id LIMIT 1) AS active_run_id,
                (SELECT COUNT(*) FROM runs r WHERE r.session_id=s.id AND r.status='queued' AND r.cancel_requested=0) AS queued_tasks
                FROM sessions s WHERE s.id=?""", (sid,)).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("""SELECT s.id,s.title,s.source,s.status,s.created,s.updated,
                (SELECT r.id FROM runs r WHERE r.session_id=s.id AND r.status IN ('running','cancelling')
                 ORDER BY r.created,r.id LIMIT 1) AS active_run_id,
                (SELECT COUNT(*) FROM runs r WHERE r.session_id=s.id AND r.status='queued' AND r.cancel_requested=0) AS queued_tasks
                FROM sessions s ORDER BY s.updated DESC LIMIT 200""")]

    def status(self, sid: str, status: str):
        with self.connect() as db:
            self._session_status(db, sid, status)

    @staticmethod
    def _session_status(db, sid: str, fallback: str | None = None, *, touch: bool = True):
        """A session is an aggregate; an old task must not hide its active sibling."""
        active = db.execute("SELECT status,cancel_requested FROM runs WHERE session_id=? AND status IN ('running','cancelling') ORDER BY created,id LIMIT 1", (sid,)).fetchone()
        if active:
            status = 'cancelling' if active['cancel_requested'] else 'running'
        elif db.execute("SELECT 1 FROM runs WHERE session_id=? AND status='queued' AND cancel_requested=0", (sid,)).fetchone():
            status = 'queued'
        else:
            recent = db.execute('SELECT status FROM runs WHERE session_id=? ORDER BY created DESC,id DESC LIMIT 1', (sid,)).fetchone()
            status = recent['status'] if recent else (fallback or 'idle')
        if touch:
            db.execute('UPDATE sessions SET status=?,updated=? WHERE id=?', (status, time.time(), sid))
        else:
            # Recomputing a derived aggregate during startup is not user activity.
            db.execute('UPDATE sessions SET status=? WHERE id=?', (status, sid))

    @staticmethod
    def _task(row) -> dict | None:
        if row is None:
            return None
        task = dict(row)
        task['target'] = json.loads(task['target'])
        task['metadata'] = json.loads(task['metadata'])
        task['images'] = json.loads(task['input_data']).get('images', [])
        task['explicit_publish_requested'] = bool(task['explicit_publish_requested'])
        task['cancel_requested'] = bool(task['cancel_requested'])
        if task['cancel_requested'] and task['status'] == 'running':
            task['status'] = 'cancelling'
        return task

    def begin(self, sid: str, prompt: str, run_id: str | None = None, *, images=None) -> str:
        # Compatibility entry for the agent: existing durable task metadata is never overwritten.
        if run_id:
            old = self.get_task(run_id)
            if old:
                if old['session_id'] != sid or old['prompt'] != prompt:
                    raise ValueError('Run ID already belongs to a different session or request')
                return run_id
        return self.enqueue_task(sid, prompt, task_id=run_id, images=images, _legacy=True)

    def enqueue_task(self, sid: str, original_user_request: str, *, prompt: str | None = None,
                     task_id: str | None = None, images=None, title: str | None = None,
                     requester_user_id: str | None = None, source: str = 'web', target=None,
                     explicit_publish_requested: bool = False, requester_nickname: str | None = None,
                     reply_id: str | None = None, metadata: dict | None = None, _legacy: bool = False) -> str:
        if not isinstance(original_user_request, str) or not original_user_request.strip() or len(original_user_request) > 500000:
            raise ValueError('Original request must contain 1–500000 characters')
        if requester_user_id is not None and (not isinstance(requester_user_id, str) or not requester_user_id.strip() or len(requester_user_id) > 200):
            raise ValueError('requester_user_id must be a nonempty string or null')
        if source not in {'web', 'community', 'admin'} or type(explicit_publish_requested) is not bool:
            raise ValueError('Invalid source or publication request flag')
        if source == 'community' and not requester_user_id:
            raise ValueError('Community tasks require a trusted original author ID')
        for name, value in [('requester_nickname', requester_nickname), ('reply_id', reply_id)]:
            if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 500):
                raise ValueError(name + ' must be nonempty text or null')
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError('Server metadata must be an object')
        if target is not None and (not isinstance(target, dict) or set(target) != {'type', 'id'} or target.get('type') not in {'Experiment', 'Discussion', 'User'} or not isinstance(target.get('id'), str) or not target['id'].strip() or len(target['id']) > 200):
            raise ValueError('Invalid community target')
        if title is not None and not isinstance(title, str):
            raise ValueError('Task title must be text')
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError('Task prompt must be nonempty text')
        prompt = original_user_request if prompt is None else prompt
        rid = task_id or uuid.uuid4().hex
        if not isinstance(rid, str) or not rid or len(rid) > 200:
            raise ValueError('Invalid task ID')
        fields = (requester_user_id, original_user_request, source, encode(target or {}),
                  int(explicit_publish_requested), (title or original_user_request)[:120],
                  requester_nickname, reply_id, encode(metadata or {}))
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone() is None:
                now = time.time()
                db.execute("INSERT INTO sessions(id,title,source,status,created,updated) VALUES(?,?,?,'idle',?,?)",
                           (sid, (title or original_user_request)[:120], source, now, now))
            existing = db.execute('SELECT * FROM runs WHERE id=?', (rid,)).fetchone()
            if existing:
                if existing['session_id'] != sid or existing['prompt'] != prompt:
                    raise ValueError('Run ID already belongs to a different session or request')
                if not _legacy:
                    saved = tuple(existing[k] for k in ('requester_user_id','original_user_request','source','target','explicit_publish_requested','title','requester_nickname','reply_id','metadata'))
                    comparable = lambda values: (*values[:3], json.loads(values[3]), *values[4:8], json.loads(values[8]))
                    if comparable(saved) != comparable(fields):
                        raise ValueError('Task ID already has different immutable metadata')
                return rid
            now = time.time()
            db.execute("""INSERT INTO runs(id,session_id,prompt,status,created,updated,input_data,
                requester_user_id,original_user_request,source,target,explicit_publish_requested,title)
                VALUES(?,?,?,'queued',?,?,?,?,?,?,?,?,?)""",
                (rid, sid, prompt, now, now, encode({'images': list(images or [])}), *fields[:6]))
            db.execute('UPDATE runs SET requester_nickname=?,reply_id=?,metadata=? WHERE id=?', (*fields[6:], rid))
            self._session_status(db, sid)
        return rid

    def get_task(self, rid: str) -> dict | None:
        with self.connect() as db:
            return self._task(db.execute('SELECT * FROM runs WHERE id=?', (rid,)).fetchone())

    def tasks(self, sid: str | None = None, status: str | None = None, limit: int = 200) -> list[dict]:
        clauses, values = [], []
        if sid is not None:
            clauses.append('session_id=?'); values.append(sid)
        if status is not None:
            if status not in {'queued','running','cancelling','cancelled','completed','needs_attention','interrupted','error'}:
                raise ValueError('Invalid task status')
            clauses.append('status=?'); values.append('running' if status == 'cancelling' else status)
            if status == 'cancelling':
                clauses.append('cancel_requested=1')
        limit = max(1, min(int(limit), 1000))
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        with self.connect() as db:
            return [self._task(r) for r in db.execute('SELECT * FROM runs' + where + """ ORDER BY
                CASE WHEN status IN ('running','cancelling') THEN 0 WHEN status='queued' THEN 1 ELSE 2 END,
                CASE WHEN status IN ('queued','running','cancelling') THEN created ELSE -created END,id LIMIT ?""", (*values, limit))]

    def has_pending_tasks(self) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1 FROM runs WHERE status IN ('queued','running','cancelling') LIMIT 1").fetchone() is not None

    def claim_next_task(self) -> dict | None:
        """Atomically claim global FIFO; at most one task can own the worker."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM runs WHERE status IN ('running','cancelling') LIMIT 1").fetchone():
                return None
            row = db.execute("SELECT * FROM runs WHERE status='queued' AND cancel_requested=0 ORDER BY created,id LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE runs SET status='running',updated=? WHERE id=?", (time.time(), row['id']))
            self._session_status(db, row['session_id'])
            return self._task(db.execute('SELECT * FROM runs WHERE id=?', (row['id'],)).fetchone())

    def run_status(self, rid: str, status: str):
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?,updated=? WHERE id=?", (status, time.time(), rid))
            row = db.execute('SELECT session_id FROM runs WHERE id=?', (rid,)).fetchone()
            if row:
                self._session_status(db, row['session_id'])

    def request_cancel(self, sid: str, rid: str) -> dict:
        """Request a safe-boundary stop; never infer task completion or undo effects."""
        if not isinstance(rid, str) or not rid:
            raise ValueError('run_id must be a nonempty string')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            run = db.execute('SELECT * FROM runs WHERE id=? AND session_id=?', (rid, sid)).fetchone()
            if run is None:
                raise ValueError('Run does not belong to this session')
            active = run['status'] in {'queued', 'running', 'cancelling'}
            if active and not run['cancel_requested']:
                now = time.time()
                # A queued task owns no work, so cancellation is immediately terminal.
                new_status = 'cancelled' if run['status'] == 'queued' else run['status']
                db.execute('UPDATE runs SET cancel_requested=1,status=?,updated=? WHERE id=?', (new_status, now, rid))
                self._session_status(db, sid)
                db.execute('INSERT INTO events(session_id,run_id,kind,data,created) VALUES(?,?,?,?,?)',
                           (sid, rid, 'cancelled' if new_status == 'cancelled' else 'cancel_requested',
                            encode({'message': '排队任务已取消，未调用模型或工具。' if new_status == 'cancelled' else '已请求停止，等待模型或工具的安全边界；不会撤销已经提交的外部操作。'}), now))
            return {'session_id': sid, 'run_id': rid,
                    'cancel_requested': bool(run['cancel_requested'] or active),
                    'status': ('cancelled' if run['status'] == 'queued' else 'cancelling') if active else run['status'],
                    'message': ('排队任务已取消，未调用模型或工具。' if run['status'] == 'queued' else '正在等待安全边界停止；已提交的外部操作需按回执核对。') if active else '该运行已结束，未发送新的停止请求。'}

    def cancel_requested(self, rid: str) -> bool:
        with self.connect() as db:
            row = db.execute('SELECT cancel_requested FROM runs WHERE id=?', (rid,)).fetchone()
        return bool(row and row['cancel_requested'])

    def finish_run(self, sid: str, rid: str, status: str) -> str:
        """Atomically finish the run and its session; cancellation wins a concurrent final."""
        if status not in {'completed', 'needs_attention', 'cancelled', 'interrupted', 'error'}:
            raise ValueError('Invalid terminal run status')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            run = db.execute('SELECT * FROM runs WHERE id=? AND session_id=?', (rid, sid)).fetchone()
            if run is None:
                raise ValueError('Run does not belong to this session')
            if run['cancel_requested'] and status == 'completed':
                status = 'cancelled'
            now = time.time()
            db.execute('UPDATE runs SET status=?,updated=? WHERE id=?', (status, now, rid))
            self._session_status(db, sid, status)
        return status

    def recover(self) -> list[dict]:
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            cancelled = db.execute("SELECT * FROM runs WHERE cancel_requested=1 AND status IN ('queued','running','cancelling')").fetchall()
            for row in cancelled:
                now = time.time()
                db.execute("UPDATE runs SET status='cancelled',updated=? WHERE id=?", (now, row['id']))
                db.execute("UPDATE sessions SET status='cancelled',updated=? WHERE id=?", (now, row['session_id']))
                db.execute('INSERT INTO events(session_id,run_id,kind,data,created) VALUES(?,?,?,?,?)',
                           (row['session_id'], row['id'], 'cancelled', encode({'message': '重启前的停止请求已保留；不会恢复执行。已经提交的外部操作仍以持久化回执为准。'}), now))
            active = db.execute("SELECT * FROM runs WHERE status='running' AND cancel_requested=0").fetchall()
            for row in active:
                db.execute("UPDATE runs SET status='interrupted',updated=? WHERE id=?", (time.time(), row['id']))
                db.execute("UPDATE sessions SET status='interrupted',updated=? WHERE id=?", (time.time(), row['session_id']))
                db.execute("INSERT INTO events(session_id,run_id,kind,data,created) VALUES(?,?, 'interrupted',?,?)",
                           (row['session_id'], row['id'], encode({'message': 'Server restarted during this run. Completed tool results remain in the journal; resume explicitly.'}), time.time()))
            queued = [{**dict(r), 'images': json.loads(r['input_data']).get('images', [])}
                      for r in db.execute("SELECT * FROM runs WHERE status='queued' AND cancel_requested=0 ORDER BY created,id")]
            for row in db.execute('SELECT id FROM sessions').fetchall():
                self._session_status(db, row['id'], touch=False)
        for row in [*active, *cancelled]:
            self.repair_tools(row['session_id'], row['id'])
        return queued

    def tool_outcome(self, sid: str, rid: str, call_id: str, name: str,
                     full_json: str, ok: bool) -> tuple[str, int]:
        """Commit immutable raw output and its replay pointer in one transaction.

        An existing (run, call) is returned unchanged. This journal prevents
        duplicate result records, not re-execution: callers must consult
        get_tool_outcome BEFORE executing a previously observed call ID.
        """
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise ValueError('Tool call ID and name must be nonempty strings')
        if not isinstance(full_json, str) or type(ok) is not bool:
            raise ValueError('Tool output must be JSON text with a boolean outcome')
        json.loads(full_json)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM runs WHERE id=? AND session_id=?', (rid, sid)).fetchone() is None:
                raise ValueError('Run does not belong to this session')
            old = db.execute('SELECT * FROM tool_outcomes WHERE run_id=? AND call_id=?', (rid, call_id)).fetchone()
            if old is not None:
                return old['document_id'], old['message_id']
            did, now = uuid.uuid4().hex, time.time()
            db.execute('INSERT INTO documents(id,session_id,title,content,created) VALUES(?,?,?,?,?)',
                       (did, sid, 'Tool ' + name, full_json, now))
            message = {'role': 'tool', 'tool_call_id': call_id, 'content': encode({
                'ok': ok, 'document_id': did,
                'message': ('Raw tool outcome is archived for operator audit only. '
                            'A bounded model-facing projection follows before the next model turn; '
                            'do not infer values from this placeholder.')})}
            row = db.execute('INSERT INTO messages(session_id,run_id,role,data,created) VALUES(?,?,?,?,?)',
                             (sid, rid, 'tool', encode(message), now))
            mid = int(row.lastrowid)
            db.execute('INSERT INTO tool_outcomes VALUES(?,?,?,?,?,?,?,?)', (rid, call_id, sid, name, int(ok), did, mid, now))
            return did, mid

    @staticmethod
    def _subagent_run(row) -> dict | None:
        if row is None:
            return None
        value = dict(row)
        value['context'] = json.loads(value['context'])
        value['report'] = json.loads(value['report'])
        return value

    def create_subagent(self, sid: str, rid: str, child_id: str, objective: str,
                        context: dict, *, deadline_at: float | None = None) -> dict:
        """Create one depth-1 trace bound to an existing parent task.

        Subagents are deliberately not sessions or runs: they cannot enter the
        scheduler and their messages can never be selected as parent model history.
        """
        if (not isinstance(child_id, str) or not re.fullmatch(r'[0-9a-f]{32}', child_id)
                or not isinstance(objective, str) or not objective.strip()):
            raise ValueError('A subagent requires a UUID-hex ID and nonempty objective')
        if not isinstance(context, dict):
            raise ValueError('Subagent context must be an object')
        encoded_context = encode(context)
        if len(objective) > 65536 or len(encoded_context) > 2_000_000:
            raise ValueError('Subagent handoff exceeds its durable input limit')
        if deadline_at is not None and (not isinstance(deadline_at, (int, float)) or deadline_at <= 0):
            raise ValueError('Subagent deadline_at must be a positive timestamp or null')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            parent = db.execute('SELECT session_id FROM runs WHERE id=?', (rid,)).fetchone()
            if parent is None or parent['session_id'] != sid:
                raise ValueError('Parent task does not belong to this session')
            if db.execute('SELECT 1 FROM subagent_runs WHERE id=?', (child_id,)).fetchone():
                raise ValueError('Subagent ID already exists')
            now = time.time()
            db.execute('''INSERT INTO subagent_runs
                (id,session_id,parent_run_id,depth,objective,context,status,report,deadline_at,created,updated)
                VALUES(?,?,?,?,?,?,'running','{}',?,?,?)''',
                (child_id, sid, rid, 1, objective.strip(), encoded_context,
                 float(deadline_at) if deadline_at is not None else None, now, now))
        return self.get_subagent(child_id)

    def get_subagent(self, child_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute('SELECT * FROM subagent_runs WHERE id=?', (child_id,)).fetchone()
        return self._subagent_run(row)

    def subagents(self, sid: str, rid: str) -> list[dict]:
        """Return compact child rows for a parent task, without copying history."""
        with self.connect() as db:
            if db.execute('SELECT 1 FROM runs WHERE id=? AND session_id=?', (rid, sid)).fetchone() is None:
                raise ValueError('Parent task does not belong to this session')
            rows = db.execute('''SELECT * FROM subagent_runs
                WHERE session_id=? AND parent_run_id=? ORDER BY created,id''', (sid, rid)).fetchall()
        return [self._subagent_run(row) for row in rows]

    def subagent_message(self, child_id: str, message: dict) -> int:
        if (not isinstance(message, dict) or message.get('role') not in
                {'system', 'user', 'assistant', 'tool'}):
            raise ValueError('Invalid subagent message')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            child = db.execute('SELECT * FROM subagent_runs WHERE id=?', (child_id,)).fetchone()
            if child is None or child['status'] != 'running':
                raise ValueError('Subagent is missing or no longer running')
            now = time.time()
            row = db.execute('''INSERT INTO subagent_messages
                (subagent_id,session_id,parent_run_id,role,data,created)
                VALUES(?,?,?,?,?,?)''', (child_id, child['session_id'], child['parent_run_id'],
                                        message['role'], encode(message), now))
            db.execute('UPDATE subagent_runs SET updated=? WHERE id=?', (now, child_id))
            return int(row.lastrowid)

    def subagent_event(self, child_id: str, kind: str, data: Any) -> int:
        if not isinstance(kind, str) or not kind:
            raise ValueError('Subagent event kind must be nonempty text')
        with self.connect() as db:
            child = db.execute('SELECT * FROM subagent_runs WHERE id=?', (child_id,)).fetchone()
            if child is None:
                raise ValueError('Unknown subagent')
            now = time.time()
            row = db.execute('''INSERT INTO subagent_events
                (subagent_id,session_id,parent_run_id,kind,data,created)
                VALUES(?,?,?,?,?,?)''', (child_id, child['session_id'], child['parent_run_id'],
                                        kind, encode(data), now))
            db.execute('UPDATE subagent_runs SET updated=? WHERE id=?', (now, child_id))
            return int(row.lastrowid)

    def subagent_tool_outcome(self, child_id: str, call_id: str, name: str,
                              full_json: str, ok: bool) -> tuple[str, int]:
        """Archive a child tool result without writing to parent ``messages``."""
        if (not isinstance(call_id, str) or not call_id.startswith(child_id + ':')
                or not isinstance(name, str) or not name or not isinstance(full_json, str)
                or type(ok) is not bool):
            raise ValueError('Invalid subagent tool outcome')
        json.loads(full_json)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            child = db.execute('SELECT * FROM subagent_runs WHERE id=?', (child_id,)).fetchone()
            if child is None or child['status'] != 'running' or child['depth'] != 1:
                raise ValueError('Subagent is missing, terminal, or has invalid depth')
            old = db.execute('''SELECT * FROM subagent_tool_outcomes
                WHERE subagent_id=? AND call_id=?''', (child_id, call_id)).fetchone()
            if old is not None:
                return old['document_id'], old['message_id']
            did, now = uuid.uuid4().hex, time.time()
            db.execute('INSERT INTO documents(id,session_id,title,content,created) VALUES(?,?,?,?,?)',
                       (did, child['session_id'], 'Subagent tool ' + name, full_json, now))
            message = {'role': 'tool', 'tool_call_id': call_id, 'content': encode({
                'ok': ok, 'document_id': did,
                'message': ('Raw child tool outcome is archived for operator audit only. '
                            'A bounded model-facing projection follows before the next child turn.')})}
            row = db.execute('''INSERT INTO subagent_messages
                (subagent_id,session_id,parent_run_id,role,data,created)
                VALUES(?,?,?,?,?,?)''', (child_id, child['session_id'], child['parent_run_id'],
                                        'tool', encode(message), now))
            mid = int(row.lastrowid)
            db.execute('INSERT INTO subagent_tool_outcomes VALUES(?,?,?,?,?,?,?,?,?)',
                       (child_id, call_id, child['session_id'], child['parent_run_id'],
                        name, int(ok), did, mid, now))
            db.execute('UPDATE subagent_runs SET updated=? WHERE id=?', (now, child_id))
            return did, mid

    def update_subagent_tool_message(self, child_id: str, message_id: int, content: str):
        if not isinstance(content, str):
            raise ValueError('Subagent tool message content must be text')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('''SELECT m.data,t.document_id FROM subagent_messages m
                JOIN subagent_tool_outcomes t ON t.message_id=m.id AND t.subagent_id=m.subagent_id
                WHERE m.id=? AND m.subagent_id=? AND m.role='tool' ''',
                (message_id, child_id)).fetchone()
            if row is None:
                raise ValueError('Message is not an archived outcome for this subagent')
            data = json.loads(row['data'])
            data['content'] = (content if row['document_id'] in content else
                               content + '\n[Raw evidence document_id=' + row['document_id'] + ']')
            db.execute('UPDATE subagent_messages SET data=? WHERE id=?', (encode(data), message_id))

    def finish_subagent(self, child_id: str, status: str, report: dict) -> dict:
        if status not in {'completed', 'needs_attention', 'cancelled', 'timed_out', 'error'}:
            raise ValueError('Invalid terminal subagent status')
        if not isinstance(report, dict):
            raise ValueError('Subagent report must be an object')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            child = db.execute('SELECT * FROM subagent_runs WHERE id=?', (child_id,)).fetchone()
            if child is None:
                raise ValueError('Unknown subagent')
            if child['status'] != 'running':
                saved = json.loads(child['report'])
                if child['status'] != status or saved != report:
                    raise ValueError('Subagent already has a different terminal result')
                return self._subagent_run(child)
            now = time.time()
            db.execute('UPDATE subagent_runs SET status=?,report=?,updated=? WHERE id=?',
                       (status, encode(report), now, child_id))
        return self.get_subagent(child_id)

    def subagent_trace(self, sid: str, rid: str, child_id: str) -> dict:
        """Read a foldable operator trace; never used as parent model context."""
        with self.connect() as db:
            child = db.execute('''SELECT * FROM subagent_runs
                WHERE id=? AND session_id=? AND parent_run_id=?''', (child_id, sid, rid)).fetchone()
            if child is None:
                raise ValueError('Subagent does not belong to this parent task')
            messages = db.execute('''SELECT id,role,data,created FROM subagent_messages
                WHERE subagent_id=? ORDER BY id''', (child_id,)).fetchall()
            events = db.execute('''SELECT id,kind,data,created FROM subagent_events
                WHERE subagent_id=? ORDER BY id''', (child_id,)).fetchall()
            tools = db.execute('''SELECT call_id,name,ok,document_id,message_id,created
                FROM subagent_tool_outcomes WHERE subagent_id=? ORDER BY created,call_id''',
                (child_id,)).fetchall()
        return {
            'subagent': self._subagent_run(child),
            'messages': [{**dict(row), 'data': json.loads(row['data'])} for row in messages],
            'events': [{**dict(row), 'data': json.loads(row['data'])} for row in events],
            'tool_outcomes': [{**dict(row), 'ok': bool(row['ok'])} for row in tools],
        }

    def final_answer(self, sid: str, rid: str, review_id: str, answer: str) -> tuple[int, bool]:
        """Journal one reviewed final per task, atomically and idempotently.

        This does not finish the run: publication/reply receipts and cancellation
        still determine its terminal state in the caller.
        """
        if not isinstance(review_id, str) or not review_id or not isinstance(answer, str) or not answer.strip():
            raise ValueError('A nonempty review ID and final answer are required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM runs WHERE id=? AND session_id=?', (rid, sid)).fetchone() is None:
                raise ValueError('Run does not belong to this session')
            old = db.execute('''SELECT f.review_id,f.message_id,m.data FROM final_answers f
                JOIN messages m ON m.id=f.message_id AND m.session_id=f.session_id
                WHERE f.run_id=? AND f.session_id=?''', (rid, sid)).fetchone()
            if old:
                saved = json.loads(old['data'])
                if old['review_id'] != review_id or saved.get('content') != answer or saved.get('_final_review_id') != review_id:
                    raise ValueError('This task already has a different reviewed final answer')
                return old['message_id'], False
            # Also recognize an existing marked final written before this table's migration.
            for message in db.execute("SELECT id,data FROM messages WHERE run_id=? AND session_id=? AND role='assistant'", (rid, sid)):
                saved = json.loads(message['data'])
                if '_final_review_id' not in saved:
                    continue
                if saved['_final_review_id'] != review_id or saved.get('content') != answer:
                    raise ValueError('This task already has a different reviewed final answer')
                db.execute('INSERT INTO final_answers VALUES(?,?,?,?,?)', (rid, sid, review_id, message['id'], time.time()))
                return message['id'], False
            now = time.time()
            data = {'role': 'assistant', 'content': answer, '_final_review_id': review_id}
            row = db.execute('INSERT INTO messages(session_id,run_id,role,data,created) VALUES(?,?,?,?,?)',
                             (sid, rid, 'assistant', encode(data), now))
            mid = int(row.lastrowid)
            db.execute('INSERT INTO final_answers VALUES(?,?,?,?,?)', (rid, sid, review_id, mid, now))
            return mid, True

    def get_tool_outcome(self, sid: str, rid: str, call_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute('''SELECT t.*, d.content AS full_json FROM tool_outcomes t
                JOIN documents d ON d.id=t.document_id AND d.session_id=t.session_id
                WHERE t.session_id=? AND t.run_id=? AND t.call_id=?''', (sid, rid, call_id)).fetchone()
        return {**dict(row), 'ok': bool(row['ok'])} if row else None

    def update_tool_message(self, sid: str, message_id: int, content: str):
        """Update only the compact replay display, retaining raw document and ownership."""
        if not isinstance(content, str):
            raise ValueError('Tool message content must be text')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('''SELECT m.data,t.document_id FROM messages m JOIN tool_outcomes t ON t.message_id=m.id
                WHERE m.id=? AND m.session_id=? AND m.role='tool' AND t.session_id=?''', (message_id, sid, sid)).fetchone()
            if row is None:
                raise ValueError('Message is not an archived tool outcome belonging to this session')
            data = json.loads(row['data'])
            data['content'] = content if row['document_id'] in content else content + '\n[Original tool result document_id=' + row['document_id'] + ']'
            db.execute('UPDATE messages SET data=? WHERE id=?', (encode(data), message_id))

    def repair_tools(self, sid: str, rid: str):
        """Close interrupted call/result pairs before another model turn can replay them."""
        with self.connect() as db:
            rows = db.execute('SELECT data FROM messages WHERE session_id=? AND run_id=? ORDER BY id', (sid, rid)).fetchall()
            pending = {}
            for row in rows:
                message = json.loads(row['data'])
                if message['role'] == 'assistant':
                    for call in message.get('tool_calls', []):
                        pending[call['id']] = call
                elif message['role'] == 'tool':
                    pending.pop(message.get('tool_call_id'), None)
            for cid in pending:
                message = {'role': 'tool', 'tool_call_id': cid, 'content': encode({
                    'ok': False, 'error': 'Tool execution was interrupted. Its completion and side effects are unknown; inspect existing artifacts before retrying.'})}
                db.execute('INSERT INTO messages(session_id,run_id,role,data,created) VALUES(?,?,?,?,?)',
                           (sid, rid, 'tool', encode(message), time.time()))

    @staticmethod
    def _plan_item(row) -> dict:
        item = dict(row)
        item['id'] = item.pop('item_id')
        item['evidence_document_ids'] = json.loads(item.pop('evidence'))
        return item

    def task_plan(self, sid: str, rid: str) -> list[dict]:
        """Return the durable ordered plan owned by exactly one task."""
        with self.connect() as db:
            if db.execute('SELECT 1 FROM runs WHERE id=? AND session_id=?', (rid, sid)).fetchone() is None:
                raise ValueError('Task does not belong to this session')
            rows = db.execute('''SELECT item_id,ordinal,title,status,note,evidence,created,updated
                FROM task_plan_items WHERE run_id=? AND session_id=? ORDER BY ordinal,item_id''',
                (rid, sid)).fetchall()
        return [self._plan_item(row) for row in rows]

    @staticmethod
    def _validate_plan_items(items) -> list[tuple[str, str]]:
        if not isinstance(items, list) or not 1 <= len(items) <= 12:
            raise ValueError('Task plan must contain 1..12 ordered items')
        clean, seen = [], set()
        for item in items:
            if not isinstance(item, dict) or set(item) != {'id', 'title'}:
                raise ValueError('Each task-plan item must contain exactly id and title')
            item_id, title = item['id'], item['title']
            if (not isinstance(item_id, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,39}', item_id)
                    or item_id in seen):
                raise ValueError('Task-plan IDs must be unique lowercase identifiers')
            if not isinstance(title, str) or not title.strip() or len(title.strip()) > 160:
                raise ValueError('Task-plan titles must contain 1..160 characters')
            seen.add(item_id)
            clean.append((item_id, title.strip()))
        return clean

    def set_task_plan(self, sid: str, rid: str, items) -> list[dict]:
        """Create a plan once. Completed history cannot be replaced by a rewrite."""
        clean = self._validate_plan_items(items)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM runs WHERE id=? AND session_id=?', (rid, sid)).fetchone() is None:
                raise ValueError('Task does not belong to this session')
            if db.execute('SELECT 1 FROM task_plan_items WHERE run_id=?', (rid,)).fetchone():
                raise ValueError('Task plan already exists; update or add items instead of replacing history')
            now = time.time()
            for ordinal, (item_id, title) in enumerate(clean):
                db.execute('INSERT INTO task_plan_items VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (rid, sid, item_id, ordinal, title,
                     'in_progress' if ordinal == 0 else 'pending', '', '[]', now, now))
        return self.task_plan(sid, rid)

    def add_task_plan_items(self, sid: str, rid: str, items) -> list[dict]:
        """Append newly discovered work without renumbering or erasing prior items."""
        clean = self._validate_plan_items(items)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM runs WHERE id=? AND session_id=?', (rid, sid)).fetchone() is None:
                raise ValueError('Task does not belong to this session')
            count, ordinal = db.execute('SELECT COUNT(*),COALESCE(MAX(ordinal),-1) FROM task_plan_items WHERE run_id=?',
                                        (rid,)).fetchone()
            if count + len(clean) > 12:
                raise ValueError('Task plan cannot exceed 12 items')
            existing = {row[0] for row in db.execute('SELECT item_id FROM task_plan_items WHERE run_id=?', (rid,))}
            if existing.intersection(item_id for item_id, _ in clean):
                raise ValueError('New task-plan item ID already exists')
            now = time.time()
            has_active = db.execute("SELECT 1 FROM task_plan_items WHERE run_id=? AND status='in_progress'", (rid,)).fetchone()
            for offset, (item_id, title) in enumerate(clean, 1):
                status = 'in_progress' if not has_active and offset == 1 else 'pending'
                db.execute('INSERT INTO task_plan_items VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (rid, sid, item_id, ordinal + offset, title, status, '', '[]', now, now))
        return self.task_plan(sid, rid)

    def update_task_plan_item(self, sid: str, rid: str, item_id: str, status: str, *,
                              note: str = '', evidence_document_ids=None,
                              evidence_call_ids=None, next_id: str | None = None) -> list[dict]:
        """Atomically record evidence and advance one sequential complex-task plan."""
        if status not in {'in_progress', 'completed', 'blocked'}:
            raise ValueError('Task-plan status must be in_progress, completed, or blocked')
        if not isinstance(item_id, str) or not item_id:
            raise ValueError('Task-plan item ID is required')
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError('Task-plan note must be text up to 2000 characters')
        docs = list(evidence_document_ids or [])
        calls = list(evidence_call_ids or [])
        if (not all(isinstance(value, str) and value for value in docs + calls)
                or len(docs) + len(calls) > 16):
            raise ValueError('Task-plan evidence must contain at most 16 nonempty IDs')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('''SELECT * FROM task_plan_items
                WHERE run_id=? AND session_id=? AND item_id=?''', (rid, sid, item_id)).fetchone()
            if row is None:
                raise ValueError('Unknown task-plan item ID')
            for call_id in calls:
                outcome = db.execute('''SELECT document_id FROM tool_outcomes
                    WHERE run_id=? AND session_id=? AND call_id=?''', (rid, sid, call_id)).fetchone()
                if outcome is None:
                    # Qwen occasionally puts the returned document_id in the
                    # call-ID field. Accept it only when it is an actual
                    # document owned by this task session; arbitrary IDs and
                    # cross-session evidence remain rejected below.
                    if db.execute('SELECT 1 FROM documents WHERE id=? AND session_id=?',
                                  (call_id, sid)).fetchone() is None:
                        raise ValueError('Evidence ID is neither a completed call nor a document in this task: ' + call_id)
                    docs.append(call_id)
                else:
                    docs.append(outcome['document_id'])
            docs = list(dict.fromkeys(docs))
            for document_id in docs:
                if db.execute('SELECT 1 FROM documents WHERE id=? AND session_id=?',
                              (document_id, sid)).fetchone() is None:
                    raise ValueError('Evidence document does not belong to this task session: ' + document_id)
            previous = row['status']
            if previous == 'completed' and status != 'completed':
                raise ValueError('Completed task-plan items are immutable')
            evidence = list(dict.fromkeys(json.loads(row['evidence']) + docs))
            # Match OpenCode's todo semantics: the plan is durable navigation,
            # not a second evidence database or a tool-permission gate.  Tool
            # outcomes are already durably journaled, so attaching their IDs is
            # useful but optional.  Requiring one for every bookkeeping step
            # made the model re-read documents merely to close a todo.
            if status == 'in_progress':
                other = db.execute("SELECT item_id FROM task_plan_items WHERE run_id=? AND status='in_progress' AND item_id<>?",
                                   (rid, item_id)).fetchone()
                if other:
                    raise ValueError('Complete or block the current in-progress item before starting another')
            now = time.time()
            db.execute('UPDATE task_plan_items SET status=?,note=?,evidence=?,updated=? WHERE run_id=? AND item_id=?',
                       (status, note.strip(), encode(evidence), now, rid, item_id))
            if status == 'completed':
                candidate = None
                if next_id is not None:
                    candidate = db.execute('SELECT * FROM task_plan_items WHERE run_id=? AND item_id=?',
                                           (rid, next_id)).fetchone()
                    if candidate is None:
                        raise ValueError('Unknown next task-plan item ID')
                    if candidate['status'] != 'pending':
                        raise ValueError('next_id must identify a pending task-plan item')
                else:
                    candidate = db.execute("SELECT * FROM task_plan_items WHERE run_id=? AND status='pending' ORDER BY ordinal LIMIT 1",
                                           (rid,)).fetchone()
                if candidate is not None:
                    if db.execute("SELECT 1 FROM task_plan_items WHERE run_id=? AND status='in_progress'", (rid,)).fetchone():
                        raise ValueError('Another task-plan item is already in progress')
                    db.execute('UPDATE task_plan_items SET status=?,updated=? WHERE run_id=? AND item_id=?',
                               ('in_progress', now, rid, candidate['item_id']))
            elif next_id is not None:
                raise ValueError('next_id is valid only when completing an item')
        return self.task_plan(sid, rid)

    def event(self, sid: str, rid: str, kind: str, data: Any) -> int:
        with self.connect() as db:
            c = db.execute("INSERT INTO events(session_id,run_id,kind,data,created) VALUES(?,?,?,?,?)", (sid, rid, kind, encode(data), time.time()))
            db.execute("UPDATE sessions SET updated=? WHERE id=?", (time.time(), sid))
            return int(c.lastrowid)

    def events(self, sid: str, after: int = 0, run_id: str | None = None) -> list[dict]:
        with self.connect() as db:
            if run_id is not None and db.execute('SELECT 1 FROM runs WHERE id=? AND session_id=?', (run_id, sid)).fetchone() is None:
                raise ValueError('Task does not belong to this session')
            condition = ' AND run_id=?' if run_id else ''
            params = (sid, after, run_id) if run_id else (sid, after)
            rows = db.execute('SELECT * FROM events WHERE session_id=? AND id>?' + condition + ' ORDER BY id LIMIT 500', params).fetchall()
        return [{**dict(r), 'data': json.loads(r['data'])} for r in rows]

    def message(self, sid: str, rid: str, message: dict) -> int:
        # Reasoning stays in separate trace events; never replay it as assistant content.
        clean = {k: v for k, v in message.items() if k not in {'reasoning', 'reasoning_content'}}
        with self.connect() as db:
            c = db.execute("INSERT INTO messages(session_id,run_id,role,data,created) VALUES(?,?,?,?,?)", (sid, rid, clean['role'], encode(clean), time.time()))
            return int(c.lastrowid)

    def messages(self, sid: str, after: int = 0, *, run_id: str | None = None) -> list[dict]:
        with self.connect() as db:
            condition = ' AND run_id=?' if run_id is not None else ''
            parameters = (sid, after, run_id) if run_id is not None else (sid, after)
            rows = db.execute("SELECT id,data FROM messages WHERE session_id=? AND id>?" + condition + " ORDER BY id", parameters).fetchall()
        return [{'id': r['id'], 'message': json.loads(r['data'])} for r in rows]

    def checkpoint(self, sid: str, rid: str) -> dict:
        """A model may only replay a checkpoint belonging to its own task."""
        with self.connect() as db:
            row = db.execute('SELECT summary,compacted_until FROM run_checkpoints WHERE session_id=? AND run_id=?', (sid, rid)).fetchone()
            if row:
                return dict(row)
            legacy = db.execute('SELECT summary,compacted_until FROM sessions WHERE id=?', (sid,)).fetchone()
            # Preserve a legacy checkpoint only when its covered journal has
            # exactly this owner. Mixed-session summaries cannot be unmerged.
            if legacy and legacy['compacted_until']:
                owners = {r[0] for r in db.execute('SELECT DISTINCT run_id FROM messages WHERE session_id=? AND id<=?', (sid, legacy['compacted_until']))}
                if owners == {rid}:
                    return dict(legacy)
            return {'summary': '', 'compacted_until': 0}

    def compact(self, sid: str, summary: str, until: int, *, run_id: str | None = None):
        with self.connect() as db:
            if run_id is not None:
                if db.execute('SELECT 1 FROM runs WHERE id=? AND session_id=?', (run_id, sid)).fetchone() is None:
                    raise ValueError('Checkpoint does not belong to this task/session')
                db.execute('INSERT INTO run_checkpoints VALUES(?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET summary=excluded.summary,compacted_until=excluded.compacted_until,updated=excluded.updated',
                           (run_id, sid, summary, until, time.time()))
            # Keep the old display/inspection fields, never use them to grant a
            # different task access to this checkpoint.
            db.execute("UPDATE sessions SET summary=?,compacted_until=?,updated=? WHERE id=?", (summary, until, time.time(), sid))

    def document(self, sid: str, title: str, content: str) -> str:
        did = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?)", (did, sid, title, content, time.time()))
        return did

    def read_document(self, sid: str, did: str, offset: int = 0, length: int = 12000) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT title,content FROM documents WHERE id=? AND session_id=?", (did, sid)).fetchone()
        if row is None:
            raise ValueError('Document does not belong to this session')
        offset = max(0, int(offset))
        length = max(1, min(int(length), 20000))
        content = row['content']
        return {'id': did, 'title': row['title'], 'offset': offset, 'total_chars': len(content), 'text': content[offset:offset + length], 'has_more': offset + length < len(content)}

    def artifact(self, sid: str, path: str, mime_type: str, label: str = '') -> str:
        aid = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?)", (aid, sid, os.path.realpath(path), mime_type, label, time.time()))
        return aid

    def get_artifact(self, aid: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (aid,)).fetchone()
        return dict(row) if row else None
