"""Small, durable per-conversation Web tracker and task submitter."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import plar

from .sessiondb import SessionDB, encode
from .session_export import export_session_archive


_PHYSICSLAB_USER_ID = re.compile(r'[0-9a-fA-F]{24}')


def _public_user_profile(package, requested_user_id):
    """Project the public PhysicsLab identity fields used by the Web shell."""
    if not isinstance(package, dict):
        raise ValueError('物理实验室用户资料格式无效')
    raw_user = package.get('User') if isinstance(package.get('User'), dict) else {}
    raw_stats = package.get('Statistic') if isinstance(package.get('Statistic'), dict) else {}
    resolved = str(raw_user.get('ID') or '').strip()
    if resolved.casefold() != requested_user_id.casefold():
        raise ValueError('物理实验室用户 ID 与查询结果不一致')

    def text(value, limit):
        return plar.best_effort_extract_text(value).strip()[:limit] or None

    def integer(value):
        if isinstance(value, bool):
            return None
        try:
            return int(value) if value is not None and str(value).strip() else None
        except (TypeError, ValueError, OverflowError):
            return None

    return {
        'id': resolved,
        'nickname': text(raw_user.get('Nickname') or raw_user.get('Name'), 160),
        'signature': text(raw_user.get('Signature'), 1000),
        'level': integer(raw_user.get('Level')),
        'experience': integer(raw_user.get('Experience')),
        'stats': {
            'experiments': integer(raw_stats.get('ExperimentCount')),
            'comments': integer(raw_stats.get('CommentCount')),
            'followers': integer(raw_stats.get('FollowerCount')),
            'following': integer(raw_stats.get('FollowingCount')),
            'stars': integer(raw_stats.get('StarCount')),
            'supports': integer(raw_stats.get('SupportCount')),
        },
    }


def _public_subagent(row):
    """Return the bounded operator summary, never the child's private handoff/history."""
    keys = ('id', 'session_id', 'parent_run_id', 'depth', 'objective', 'status',
            'report', 'deadline_at', 'created', 'updated')
    return {key: row[key] for key in keys}


def _public_subagent_trace(trace):
    """Expose telemetry and evidence IDs without leaking child reasoning/messages."""
    return {
        'subagent': _public_subagent(trace['subagent']),
        'events': trace['events'],
        'tool_outcomes': trace['tool_outcomes'],
    }


class PersistentTaskQueue:
    """One durable bounded scheduler for all trusted request sources.

    Enqueue never calls a model or publishes. Workers claim atomically from
    SQLite, not from an in-memory list, so restarts preserve order and metadata.
    Different sessions can run in parallel; requests in one session are serial.
    """
    def __init__(self, database, agent, *, user=None, logger=None, on_result=None,
                 max_parallel_tasks=1):
        if (type(max_parallel_tasks) is not int or
                not 1 <= max_parallel_tasks <= 64):
            raise ValueError('max_parallel_tasks must be an integer in 1..64')
        self.database, self.agent, self.user = database, agent, user
        self.logger, self.on_result = logger, on_result
        self.max_parallel_tasks = max_parallel_tasks
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = None
        self.threads = []
        self.busy = threading.Event()
        self._busy_count = 0
        self._live_workers = 0
        self._state_lock = threading.Lock()
        # Make shutdown and the decision to claim the next durable task one
        # linearizable operation.  Without this lock a worker can observe an
        # unset stopping flag, lose the CPU to close(), and then turn a task
        # that should survive the restart as queued into an interrupted run.
        self._claim_lock = threading.Lock()
        self.lock_file = None

    def enqueue(self, session_id, original_user_request, **kwargs):
        if self.stopping.is_set():
            raise RuntimeError('Task worker is shutting down; request was not queued')
        # The incoming session is a UI/archive location, not permission to reuse
        # its model context. Only duplicate ingestion of the identical task ID
        # returns an existing receipt; it never requeues a finished task.
        task_id = kwargs.setdefault('task_id', uuid.uuid4().hex)
        existing = self.database.get_task(task_id)
        session_id = existing['session_id'] if existing else self.database.session(
            ('community-' if kwargs.get('source') == 'community' else 'task-') + task_id,
            title=kwargs.get('title') or original_user_request, source=kwargs.get('source', 'web'),
            owner_id=kwargs.get('owner_id') or '')
        rid = self.database.enqueue_task(session_id, original_user_request, **kwargs)
        if existing is None:
            self.database.event(session_id, rid, 'submitted', {'text': original_user_request, 'task_id': rid})
        self.wake.set()
        return rid

    def start(self):
        if self.threads:
            raise RuntimeError('Task workers already started')
        # Recovery is only safe after excluding another scheduler for this database.
        import fcntl
        lock_file = open(self.database.path + '.worker.lock', 'a')
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.database.recover()
        except BaseException:
            lock_file.close()
            raise
        self.lock_file = lock_file
        self.threads = [threading.Thread(
            target=self._loop, name=f'aurex-task-{index + 1}', daemon=True)
            for index in range(self.max_parallel_tasks)]
        self.thread = self.threads[0]
        self._live_workers = 0
        started = []
        try:
            for worker in self.threads:
                with self._state_lock:
                    self._live_workers += 1
                try:
                    worker.start()
                except BaseException:
                    with self._state_lock:
                        self._live_workers -= 1
                    raise
                started.append(worker)
        except BaseException:
            with self._claim_lock:
                self.stopping.set()
                self.wake.set()
            for worker in started:
                if worker.is_alive():
                    worker.join(5)
            with self._state_lock:
                # A worker that already claimed a task may still be finishing
                # its safe boundary.  It remains the lock owner and the last
                # worker's finally block releases the process-wide flock.
                if self._live_workers == 0 and self.lock_file:
                    self.lock_file.close()
                    self.lock_file = None
            self.threads = started
            self.thread = started[0] if started else None
            raise

    def wait_idle(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while self.busy.is_set() or self.database.has_pending_tasks():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def close(self, *, wait=False, timeout=None):
        # In-flight tools stop only at their safe boundary; do not kill external writes.
        with self._claim_lock:
            self.stopping.set()
            self.wake.set()
        if wait:
            deadline = None if timeout is None else time.monotonic() + max(0, timeout)
            for worker in self.threads:
                if worker is threading.current_thread():
                    continue
                remaining = None if deadline is None else max(0, deadline - time.monotonic())
                worker.join(remaining)

    def _busy_enter(self):
        with self._state_lock:
            self._busy_count += 1
            self.busy.set()

    def _busy_exit(self):
        with self._state_lock:
            self._busy_count -= 1
            if self._busy_count == 0:
                self.busy.clear()

    def run_next(self):
        with self._claim_lock:
            if self.stopping.is_set():
                return False
            task = self.database.claim_next_task(self.max_parallel_tasks)
            if task is None:
                return False
            self._busy_enter()
        sid, rid = task['session_id'], task['id']
        try:
            if self.database.cancel_requested(rid):
                self.database.finish_run(sid, rid, 'cancelled')
                return True
            result = self.agent.handle(user_text=task['prompt'], user=self.user, session_id=sid,
                                       run_id=rid, images=task['images'])
            # The agent owns the evidence-based terminal state, not this scheduler.
            actual = self.database.get_task(rid)
            if actual['status'] in {'queued', 'running', 'cancelling'}:
                self.database.finish_run(sid, rid, 'cancelled' if actual['cancel_requested'] else 'needs_attention')
                self.database.event(sid, rid, 'task_incomplete', {'message': 'Agent returned without a verified terminal task state.'})
            if self.on_result:
                try:
                    self.on_result(self.database.get_task(rid), result)
                except Exception as exc:
                    self.database.event(sid, rid, 'delivery_error', {'error': str(exc), 'message': 'Task result is saved; reply delivery failed. The task was not re-executed.'})
        except Exception as exc:
            self.database.finish_run(sid, rid, 'cancelled' if self.database.cancel_requested(rid) else 'error')
            self.database.event(sid, rid, 'error', {'error': str(exc), 'message': 'Task worker stopped this task; later queued requests remain available.'})
            if self.logger:
                self.logger.error('task %s failed: %s', rid, exc)
        finally:
            self._busy_exit()
            # A completed slot may unblock a queued sibling from the same
            # session; wake every scheduler worker promptly.
            self.wake.set()
        return True

    def _loop(self):
        try:
            while not self.stopping.is_set():
                try:
                    if self.run_next():
                        continue
                except Exception as exc:
                    if self.logger:
                        self.logger.error('task queue error: %s', exc)
                self.wake.wait(1)
                self.wake.clear()
        finally:
            with self._state_lock:
                self._live_workers -= 1
                if self._live_workers == 0 and self.lock_file:
                    self.lock_file.close()
                    self.lock_file = None


def serve(*, cfg, config_path, agent, user=None, hostname=None, port=None, poll=True, logger=None,
          on_task_result=None, enqueue_ready=None):
    database = SessionDB(cfg.resolve_path(cfg.tracking.database_path, config_path=config_path))
    cache = Path(cfg.resolve_path(cfg.storage.cache_dir, config_path=config_path)).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    retention_worker = None
    if cfg.storage.history_retention_enabled:
        from .history_retention import GIB, HistoryRetention, HistoryRetentionWorker
        configured_history = cfg.resolve_path(cfg.storage.history_dir, config_path=config_path)
        history_dir = configured_history or str(Path(database.path).parent / 'backups')
        retention_worker = HistoryRetentionWorker(HistoryRetention(
            history_dir,
            database_path=database.path,
            cache_dir=str(cache),
            compress_at_bytes=int(cfg.storage.history_compress_at_gib * GIB),
            delete_at_bytes=int(cfg.storage.history_delete_at_gib * GIB),
            min_age_sec=cfg.storage.history_min_age_sec,
            logger=logger,
        ), cfg.storage.history_check_interval_sec)
    admin_token = os.environ.get(cfg.tracking.token_env, '').strip()
    profile_cache = {}
    profile_gate = threading.Lock()
    gate = threading.Lock()
    export_gate = threading.Lock()
    export_ticket_gate = threading.Lock()
    export_tickets = {}
    export_ticket_ttl_sec = 10 * 60
    export_root = cache / 'session-exports'
    if export_root.is_symlink():
        raise ValueError('Session export directory must not be a symbolic link')
    export_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(export_root, 0o700)
    queue = PersistentTaskQueue(
        database, agent, user=user, logger=logger, on_result=on_task_result,
        max_parallel_tasks=cfg.tracking.max_parallel_tasks)
    if enqueue_ready:
        enqueue_ready(queue.enqueue)

    def cleanup_stale_export_directories():
        # Run only after queue.start() has acquired the process-wide database
        # lock.  A rejected second server must not remove the live server's
        # in-flight download.
        for stale in export_root.iterdir():
            if not stale.name.startswith('.session-export-'):
                continue
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)

    def remove_export_ticket(ticket):
        directory = Path(ticket['directory'])
        if directory.parent == export_root and directory.name.startswith('.session-export-'):
            shutil.rmtree(directory, ignore_errors=True)

    def cleanup_export_tickets(*, force=False):
        now = time.monotonic()
        expired = []
        with export_ticket_gate:
            for token, ticket in list(export_tickets.items()):
                if force or (ticket['expires_at'] <= now and not ticket['downloading']):
                    export_tickets.pop(token, None)
                    expired.append(ticket)
        for ticket in expired:
            remove_export_ticket(ticket)

    def user_profile(user_id, *, required=False):
        """Resolve public profile data without treating a public ID as authentication."""
        now = time.monotonic()
        with profile_gate:
            cached = profile_cache.get(user_id.casefold())
            if cached and now - cached[0] < 300:
                return cached[1]
        if user is None:
            if required:
                raise ValueError('当前服务未连接物理实验室，无法校验用户 ID')
            return {'id': user_id, 'nickname': None, 'signature': None, 'level': None,
                    'experience': None, 'stats': {}}
        try:
            profile = _public_user_profile(plar.get_user_by_id(user, user_id=user_id), user_id)
        except Exception as exc:
            if required:
                raise ValueError('无法读取该物理实验室用户 ID') from exc
            return {'id': user_id, 'nickname': None, 'signature': None, 'level': None,
                    'experience': None, 'stats': {}, 'unavailable': True}
        with profile_gate:
            if len(profile_cache) >= 256 and user_id.casefold() not in profile_cache:
                oldest = min(profile_cache, key=lambda key: profile_cache[key][0])
                profile_cache.pop(oldest, None)
            profile_cache[user_id.casefold()] = (now, profile)
        return profile

    def create_task(data, *, principal, session_id=None, source='web'):
        allowed = {'text', 'original_user_request', 'images', 'target'}
        if source == 'admin':
            allowed.update({'session_id', 'title', 'requester_user_id', 'explicit_publish_requested'})
        if set(data) - allowed:
            raise ValueError('Unknown or server-only task fields: ' + ', '.join(sorted(set(data) - allowed)))
        if 'text' in data and 'original_user_request' in data:
            raise ValueError('Use text or original_user_request, not both')
        original = data.get('original_user_request', data.get('text'))
        if not isinstance(original, str) or not original.strip() or len(original) > 500000:
            raise ValueError('Text must contain 1–500000 characters')
        target = data.get('target')
        prompt = original
        if target is not None:
            if not isinstance(target, dict) or set(target) != {'type', 'id'} or target.get('type') not in {'Experiment','Discussion','User'} or not isinstance(target.get('id'), str) or not target['id'].strip():
                raise ValueError('Invalid community target')
            prompt = 'CONTEXT_JSON:\n' + encode({'target': target}) + '\n\n' + original
        publish = data.get('explicit_publish_requested', False)
        if type(publish) is not bool:
            raise ValueError('explicit_publish_requested must be a boolean')
        requester = data.get('requester_user_id')
        if requester is not None and (not isinstance(requester, str) or not requester.strip() or len(requester) > 200):
            raise ValueError('requester_user_id must be a nonempty string or null')
        if 'title' in data and not isinstance(data['title'], str):
            raise ValueError('Title must be text')
        raw_images = data.get('images', [])
        if not isinstance(raw_images, list) or len(raw_images) > cfg.llm.max_images:
            raise ValueError('Too many images')
        paths = []
        if raw_images:
            from PIL import Image
            import io
            for encoded in raw_images:
                if not isinstance(encoded, str):
                    raise ValueError('Image must be encoded text')
                raw = base64.b64decode(encoded.split(',')[-1], validate=True)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError('Image exceeds 8 MiB')
                with Image.open(io.BytesIO(raw)) as image:
                    if image.width * image.height > 24000000:
                        raise ValueError('Image exceeds 24 megapixels')
                    file = cache / ('upload-' + uuid.uuid4().hex + '.png')
                    image.convert('RGB').save(file)
                    paths.append(str(file))
        sid = session_id or data.get('session_id')
        owner_id = principal['owner_id'] if principal['role'] == 'user' else ''
        if sid is not None and (not isinstance(sid, str) or not database.get(
                sid, owner_id=owner_id if principal['role'] == 'user' else None)):
            raise ValueError('Session not found')
        rid = queue.enqueue(sid, original, prompt=prompt, images=paths, source=source, target=target,
                            title=data.get('title'), requester_user_id=requester,
                            explicit_publish_requested=publish, owner_id=owner_id or None)
        task = database.get_task(rid)
        return {'session_id': task['session_id'], 'run_id': rid, 'task_id': rid, 'status': task['status'],
                'context_policy': 'fresh_session_per_request', 'previous_context_reused': False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # No prompt content, auth keys or query strings in access logs.
            if logger:
                logger.debug('web %s %s', self.command, urlparse(self.path).path)

        def cookies(self):
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get('Cookie', ''))
            except Exception:
                return {}
            return {key: morsel.value for key, morsel in cookie.items()}

        def principal(self):
            supplied = self.headers.get('Authorization', '').removeprefix('Bearer ')
            if not supplied:
                supplied = self.cookies().get('aurex_access', '')
            if admin_token and hmac.compare_digest(supplied, admin_token):
                return {'role': 'admin', 'owner_id': ''}
            cookies = self.cookies()
            user_secret = cookies.get('aurex_user', '')
            user_id = cookies.get('aurex_user_id', '')
            if (re.fullmatch(r'[A-Za-z0-9_-]{40,100}', user_secret)
                    and _PHYSICSLAB_USER_ID.fullmatch(user_id)):
                owner = hashlib.sha256((user_secret + '\0' + user_id.casefold()).encode()).hexdigest()
                return {'role': 'user', 'owner_id': owner, 'user_id': user_id.casefold()}
            return None

        @staticmethod
        def owns(principal, session):
            return bool(session) and (principal['role'] == 'admin' or session['owner_id'] == principal['owner_id'])

        def require_session(self, principal, sid):
            session = database.get(sid)
            return session if self.owns(principal, session) else None

        def send_cookies(self, values):
            for value in values:
                self.send_header('Set-Cookie', value + '; HttpOnly; SameSite=Strict; Path=/')

        def respond(self, value, status=200):
            body = encode(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def respond_with_cookies(self, value, cookies, status=200):
            body = encode(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_cookies(cookies)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 18 * 1024 * 1024:
                raise ValueError('Request must be between 1 byte and 18 MiB')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object')
            return data

        @staticmethod
        def export_payload(token, ticket):
            exported = ticket['export']
            return {
                'download_url': '/api/session-exports/' + token,
                'filename': exported.filename,
                'compressed': True,
                'format': 'sqlite3.zip',
                'captured_at': exported.captured_at,
                'session_status': exported.session_status,
                'sqlite_bytes': exported.sqlite_bytes,
                'archive_bytes': exported.archive_bytes,
                'estimated_logical_bytes': exported.estimated_logical_bytes,
            }

        def stream_session_export(self, principal, token):
            cleanup_export_tickets()
            with export_ticket_gate:
                ticket = export_tickets.get(token)
                if (ticket is None or ticket['principal_key'] != (
                        principal['role'], principal['owner_id'])):
                    ticket = None
            if ticket is None or not self.require_session(principal, ticket['session_id']):
                return self.respond({'error': 'Session export not found'}, 404)

            with export_ticket_gate:
                current = export_tickets.get(token)
                if current is not ticket:
                    return self.respond({'error': 'Session export not found'}, 404)
                if ticket['downloading']:
                    return self.respond({'error': 'Session export is already downloading'}, 409)
                ticket['downloading'] = True

            file = Path(ticket['export'].path).resolve()
            directory = Path(ticket['directory']).resolve()
            if (file.parent != directory or not file.is_file()
                    or not file.name.endswith('.sqlite3.zip')):
                with export_ticket_gate:
                    export_tickets.pop(token, None)
                remove_export_ticket(ticket)
                return self.respond({'error': 'Session export is unavailable'}, 404)

            completed = False
            try:
                size = file.stat().st_size
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition',
                    'attachment; filename="' + ticket['export'].filename + '"')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Length', str(size))
                self.end_headers()
                with file.open('rb') as source:
                    while block := source.read(1024 * 1024):
                        self.wfile.write(block)
                self.wfile.flush()
                completed = True
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                if logger:
                    logger.debug('session export download interrupted: %s', type(exc).__name__)
            finally:
                dispose = False
                with export_ticket_gate:
                    if export_tickets.get(token) is ticket:
                        if completed:
                            export_tickets.pop(token, None)
                            dispose = True
                        else:
                            ticket['downloading'] = False
                if dispose:
                    remove_export_ticket(ticket)

        def do_GET(self):
            route = urlparse(self.path)
            if route.path == '/health':
                return self.respond({'healthy': True, 'service': 'aurex3', 'model': cfg.llm.model})
            if route.path == '/':
                data = Path(__file__).with_name('tracking.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(data)
                return
            principal = self.principal()
            if not principal:
                return self.respond({'error': 'Access token required'}, 401)
            cleanup_export_tickets()
            export_match = re.fullmatch(
                r'/api/session-exports/([A-Za-z0-9_-]{40,100})', route.path)
            if export_match:
                return self.stream_session_export(principal, export_match.group(1))
            if route.path == '/api/me':
                return self.respond({'role': principal['role'], 'admin_available': bool(admin_token),
                    'user_id': principal.get('user_id'),
                    'profile': user_profile(principal['user_id']) if principal['role'] == 'user' else None,
                    'capabilities': {'create_admin_task': principal['role'] == 'admin',
                        'cancel_any_task': principal['role'] == 'admin',
                        'view_all_sessions': principal['role'] == 'admin'}})
            if route.path == '/api/sessions':
                return self.respond(database.list(owner_id=principal['owner_id'])
                    if principal['role'] == 'user' else database.list())
            if route.path == '/api/tasks':
                query = parse_qs(route.query)
                try:
                    sid_filter = query.get('session_id', [None])[0]
                    active_only = query.get('active', ['0'])[0] == '1'
                    if sid_filter and not self.require_session(principal, sid_filter):
                        return self.respond({'error': 'Task not found'}, 404)
                    if principal['role'] == 'user' and active_only and not sid_filter:
                        rows = database.tasks(status=query.get('status', [None])[0],
                            limit=int(query.get('limit', ['200'])[0]), active_only=True)
                    else:
                        rows = database.tasks(sid=sid_filter,
                            status=query.get('status', [None])[0], limit=int(query.get('limit', ['200'])[0]),
                            owner_id=principal['owner_id'] if principal['role'] == 'user' else None,
                            active_only=active_only)
                    # Keep FIFO polling lightweight; the full immutable request is at /api/tasks/:id.
                    visible = []
                    for position, row in enumerate(rows, 1):
                        session = database.get(row['session_id'])
                        mine = principal['role'] == 'admin' or session['owner_id'] == principal['owner_id']
                        if not mine:
                            visible.append({'id': 'private-' + str(position), 'session_id': '',
                                'title': '其他用户任务', 'status': row['status'], 'source': 'private',
                                'mine': False, 'queue_position': position, 'created': row['created'],
                                'updated': row['updated'], 'cancel_requested': False})
                            continue
                        item = {key: row[key] for key in ('id','session_id','title','status',
                            'source','requester_user_id','requester_nickname','target','explicit_publish_requested',
                            'cancel_requested','created','updated')}
                        item.update({'mine': True, 'queue_position': position})
                        visible.append(item)
                    return self.respond(visible)
                except ValueError as exc:
                    return self.respond({'error': str(exc)}, 400)
            if route.path.startswith('/api/tasks/'):
                task = database.get_task(route.path.rsplit('/', 1)[-1])
                if not task or not self.require_session(principal, task['session_id']):
                    return self.respond({'error': 'Task not found'}, 404)
                return self.respond(task)
            if route.path.startswith('/api/artifacts/'):
                artifact = database.get_artifact(route.path.rsplit('/', 1)[-1])
                if not artifact or not self.require_session(principal, artifact['session_id']):
                    return self.respond({'error': 'Artifact not found'}, 404)
                file = Path(artifact['path']).resolve()
                if not file.is_relative_to(cache) or not file.is_file():
                    return self.respond({'error': 'Artifact is unavailable'}, 404)
                if file.stat().st_size > 32 * 1024 * 1024:
                    return self.respond({'error': 'Artifact exceeds Web transfer limit'}, 413)
                self.send_response(200)
                mime = artifact['mime_type']
                self.send_header('Content-Type', mime)
                self.send_header('X-Content-Type-Options', 'nosniff')
                if not mime.startswith('image/'):
                    self.send_header('Content-Disposition', 'attachment')
                self.send_header('Content-Length', str(file.stat().st_size))
                self.end_headers()
                self.wfile.write(file.read_bytes())
                return
            parts = route.path.strip('/').split('/')
            if (len(parts) in {6, 7} and parts[:2] == ['api', 'sessions']
                    and parts[3] == 'tasks' and parts[5] == 'subagents'):
                sid, rid = parts[2], parts[4]
                task = database.get_task(rid)
                # Return the same result for an unknown task and a task owned by
                # another session.  A crafted URL must not disclose that the
                # foreign parent/child exists.
                if not self.require_session(principal, sid) or not task or task['session_id'] != sid:
                    return self.respond({'error': 'Task not found in session'}, 404)
                try:
                    if len(parts) == 6:
                        return self.respond([
                            _public_subagent(row) for row in database.subagents(sid, rid)
                        ])
                    return self.respond(_public_subagent_trace(
                        database.subagent_trace(sid, rid, parts[6])))
                except ValueError:
                    return self.respond({'error': 'Subagent not found in task'}, 404)
            if len(parts) >= 3 and parts[:2] == ['api', 'sessions']:
                sid = parts[2]
                if not self.require_session(principal, sid):
                    return self.respond({'error': 'Session not found'}, 404)
                if len(parts) == 3:
                    session = database.get(sid)
                    session.pop('owner_id', None)
                    return self.respond(session)
                query = parse_qs(route.query)
                try:
                    if parts[3] == 'events':
                        return self.respond(database.events(sid, int(query.get('after', ['0'])[0]), query.get('run_id', [None])[0]))
                    if parts[3] == 'documents' and len(parts) == 5:
                        return self.respond(database.read_document(sid, parts[4], int(query.get('offset', ['0'])[0])))
                except (ValueError, KeyError) as exc:
                    return self.respond({'error': str(exc)}, 400)
            return self.respond({'error': 'Not found'}, 404)

        def do_POST(self):
            # Reject browser cross-origin state changes even on a trusted local network.
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                return self.respond({'error': 'Cross-origin requests are not allowed'}, 403)
            try:
                data = self.read_json()
                path = urlparse(self.path).path
                if path == '/api/login':
                    mode = data.get('mode', 'admin')
                    if mode == 'user':
                        if set(data) != {'mode', 'user_id'}:
                            raise ValueError('用户模式需要且只接受物理实验室用户 ID')
                        user_id = str(data.get('user_id') or '').strip().casefold()
                        if not _PHYSICSLAB_USER_ID.fullmatch(user_id):
                            raise ValueError('物理实验室用户 ID 必须是 24 位十六进制字符')
                        profile = user_profile(user_id, required=True)
                        secret = self.cookies().get('aurex_user', '')
                        if not re.fullmatch(r'[A-Za-z0-9_-]{40,100}', secret):
                            secret = secrets.token_urlsafe(32)
                        return self.respond_with_cookies(
                            {'ok': True, 'role': 'user', 'user_id': user_id, 'profile': profile},
                            ['aurex_user=' + secret + '; Max-Age=2592000',
                             'aurex_user_id=' + user_id + '; Max-Age=2592000',
                             'aurex_access=; Max-Age=0'])
                    if mode != 'admin' or set(data) - {'mode', 'token'}:
                        raise ValueError('Invalid login mode')
                    if not admin_token:
                        return self.respond({'error': 'Administrator access is not configured'}, 503)
                    if not hmac.compare_digest(str(data.get('token', '')), admin_token):
                        return self.respond({'error': 'Incorrect access token'}, 401)
                    return self.respond_with_cookies({'ok': True, 'role': 'admin'},
                        ['aurex_access=' + admin_token])
                if path == '/api/logout':
                    return self.respond_with_cookies({'ok': True}, ['aurex_access=; Max-Age=0'])
                principal = self.principal()
                if not principal:
                    return self.respond({'error': 'Access token required'}, 401)
                if path == '/api/sessions':
                    sid = database.session(title=str(data.get('title') or 'New conversation'),
                        owner_id=principal['owner_id'] if principal['role'] == 'user' else '')
                    return self.respond({'id': sid}, 201)
                if path == '/api/tasks':
                    if principal['role'] != 'admin':
                        return self.respond({'error': 'Administrator access required'}, 403)
                    with gate:
                        result = create_task(data, principal=principal, source='admin')
                    return self.respond(result, 202)
                if path == '/api/requests':
                    with gate:
                        result = create_task(data, principal=principal, source='web')
                    return self.respond(result, 202)
                parts = path.strip('/').split('/')
                if len(parts) == 4 and parts[:2] == ['api', 'sessions'] and parts[3] == 'export':
                    if data:
                        raise ValueError('Session export accepts an empty JSON object')
                    sid = parts[2]
                    if not self.require_session(principal, sid):
                        return self.respond({'error': 'Session not found'}, 404)
                    cleanup_export_tickets()
                    principal_key = (principal['role'], principal['owner_id'])
                    with export_ticket_gate:
                        existing = next(((token, ticket) for token, ticket in export_tickets.items()
                            if ticket['session_id'] == sid
                            and ticket['principal_key'] == principal_key), None)
                    if existing:
                        if existing[1]['downloading']:
                            return self.respond({'error': 'Session export is already downloading'}, 409)
                        return self.respond(self.export_payload(*existing))
                    if not export_gate.acquire(blocking=False):
                        return self.respond({'error': 'Another session export is being prepared'}, 429)
                    directory = None
                    try:
                        directory = Path(tempfile.mkdtemp(
                            prefix='.session-export-', dir=export_root))
                        os.chmod(directory, 0o700)
                        exported = export_session_archive(
                            database.path, sid, str(directory),
                            include_private=principal['role'] == 'admin')
                        token = secrets.token_urlsafe(32)
                        ticket = {
                            'session_id': sid,
                            'principal_key': principal_key,
                            'directory': str(directory),
                            'export': exported,
                            'expires_at': time.monotonic() + export_ticket_ttl_sec,
                            'downloading': False,
                        }
                        with export_ticket_gate:
                            export_tickets[token] = ticket
                        directory = None
                        return self.respond(self.export_payload(token, ticket), 201)
                    except Exception as exc:
                        if logger:
                            logger.error('session export failed: %s', exc)
                        return self.respond({'error': 'Unable to create a verified session export'}, 500)
                    finally:
                        if directory is not None:
                            shutil.rmtree(directory, ignore_errors=True)
                        export_gate.release()
                if len(parts) == 4 and parts[:2] == ['api', 'tasks'] and parts[3] == 'cancel':
                    if data:
                        raise ValueError('Task cancellation accepts an empty JSON object')
                    task = database.get_task(parts[2])
                    if not task or not self.require_session(principal, task['session_id']):
                        return self.respond({'error': 'Task not found'}, 404)
                    result = database.request_cancel(task['session_id'], task['id'])
                    queue.wake.set()
                    return self.respond(result, 202 if result['status'] == 'cancelling' else 200)
                if len(parts) == 4 and parts[:2] == ['api', 'sessions'] and parts[3] == 'cancel':
                    if set(data) != {'run_id'}:
                        raise ValueError('Cancellation accepts only run_id')
                    with gate:
                        if not self.require_session(principal, parts[2]):
                            return self.respond({'error': 'Session not found'}, 404)
                        result = database.request_cancel(parts[2], data['run_id'])
                    queue.wake.set()
                    return self.respond(result, 202 if result['status'] == 'cancelling' else 200)
                if len(parts) == 4 and parts[:2] == ['api', 'sessions'] and parts[3] == 'messages':
                    sid = parts[2]
                    with gate:
                        if not self.require_session(principal, sid):
                            return self.respond({'error': 'Session not found'}, 404)
                        result = create_task(data, principal=principal, session_id=sid)
                    return self.respond(result, 202)
                return self.respond({'error': 'Not found'}, 404)
            except (ValueError, KeyError, OSError) as exc:
                return self.respond({'error': str(exc)}, 400)

    if poll and user is None:
        raise ValueError('Community polling requires PhysicsLab login')
    server = ThreadingHTTPServer((hostname or cfg.tracking.hostname, port or cfg.tracking.port), Handler)
    try:
        queue.start()
        cleanup_stale_export_directories()
        if retention_worker is not None:
            retention_worker.start()
        if poll:
            from .runloop import run_forever, normalize_targets, default_state_path
            bot = threading.Thread(target=run_forever, kwargs=dict(
                cfg=cfg, config_path=config_path, agent=agent, user=user,
                targets=normalize_targets(cfg.agent.targets), state_path=default_state_path(config_path),
                logger=logger, once=False, enqueue=queue.enqueue,
            ), name='aurex-community', daemon=True)
            bot.start()
        print(f'Aurex tracking: http://{server.server_address[0]}:{server.server_port}', flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        queue.close()
        if retention_worker is not None:
            retention_worker.close()
        cleanup_export_tickets(force=True)
