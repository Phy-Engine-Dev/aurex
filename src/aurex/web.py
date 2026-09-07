"""Small, durable per-conversation Web tracker and task submitter."""
from __future__ import annotations

import base64
import hmac
import json
import os
import threading
import time
import uuid
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .sessiondb import SessionDB, encode


class PersistentTaskQueue:
    """One durable FIFO for Web, administrator and trusted community requests.

    Enqueue never calls a model or publishes. A single worker claims from SQLite,
    not from an in-memory list, so restarts preserve queue order and metadata.
    """
    def __init__(self, database, agent, *, user=None, logger=None, on_result=None):
        self.database, self.agent, self.user = database, agent, user
        self.logger, self.on_result = logger, on_result
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = None
        self.busy = threading.Event()
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
            title=kwargs.get('title') or original_user_request, source=kwargs.get('source', 'web'))
        rid = self.database.enqueue_task(session_id, original_user_request, **kwargs)
        if existing is None:
            self.database.event(session_id, rid, 'submitted', {'text': original_user_request, 'task_id': rid})
        self.wake.set()
        return rid

    def start(self):
        if self.thread is not None:
            raise RuntimeError('Task worker already started')
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
        self.thread = threading.Thread(target=self._loop, name='aurex-task-fifo', daemon=True)
        self.thread.start()

    def wait_idle(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while self.busy.is_set() or self.database.has_pending_tasks():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def close(self, *, wait=False, timeout=None):
        # In-flight tools stop only at their safe boundary; do not kill external writes.
        self.stopping.set()
        self.wake.set()
        if wait and self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout)

    def run_next(self):
        if self.stopping.is_set():
            return False
        task = self.database.claim_next_task()
        if task is None:
            return False
        self.busy.set()
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
            self.busy.clear()
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
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None


def serve(*, cfg, config_path, agent, user=None, hostname=None, port=None, poll=True, logger=None,
          on_task_result=None, enqueue_ready=None):
    database = SessionDB(cfg.resolve_path(cfg.tracking.database_path, config_path=config_path))
    cache = Path(cfg.resolve_path(cfg.storage.cache_dir, config_path=config_path)).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    token = os.environ.get(cfg.tracking.token_env, '')
    gate = threading.Lock()
    queue = PersistentTaskQueue(database, agent, user=user, logger=logger, on_result=on_task_result)
    if enqueue_ready:
        enqueue_ready(queue.enqueue)

    def create_task(data, *, session_id=None, source='web'):
        allowed = {'text', 'original_user_request', 'images', 'target', 'title', 'requester_user_id', 'explicit_publish_requested'}
        if source == 'admin':
            allowed.add('session_id')
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
        if sid is not None and (not isinstance(sid, str) or not database.get(sid)):
            raise ValueError('Session not found')
        rid = queue.enqueue(sid, original, prompt=prompt, images=paths, source=source, target=target,
                            title=data.get('title'), requester_user_id=requester,
                            explicit_publish_requested=publish)
        task = database.get_task(rid)
        return {'session_id': task['session_id'], 'run_id': rid, 'task_id': rid, 'status': task['status'],
                'context_policy': 'fresh_session_per_request', 'previous_context_reused': False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # No prompt content, auth keys or query strings in access logs.
            if logger:
                logger.debug('web %s %s', self.command, urlparse(self.path).path)

        def auth(self):
            if not token:
                return True
            supplied = self.headers.get('Authorization', '').removeprefix('Bearer ')
            if not supplied:
                cookie = SimpleCookie()
                try:
                    cookie.load(self.headers.get('Cookie', ''))
                    supplied = cookie['aurex_access'].value if 'aurex_access' in cookie else ''
                except Exception:
                    supplied = ''
            return hmac.compare_digest(supplied, token)

        def respond(self, value, status=200):
            body = encode(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
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
            if not self.auth():
                return self.respond({'error': 'Access token required'}, 401)
            if route.path == '/api/sessions':
                return self.respond(database.list())
            if route.path == '/api/tasks':
                query = parse_qs(route.query)
                try:
                    rows = database.tasks(sid=query.get('session_id', [None])[0],
                        status=query.get('status', [None])[0], limit=int(query.get('limit', ['200'])[0]))
                    # Keep FIFO polling lightweight; the full immutable request is at /api/tasks/:id.
                    return self.respond([{key: row[key] for key in ('id','session_id','title','status',
                        'source','requester_user_id','requester_nickname','target','explicit_publish_requested',
                        'cancel_requested','created','updated')} for row in rows])
                except ValueError as exc:
                    return self.respond({'error': str(exc)}, 400)
            if route.path.startswith('/api/tasks/'):
                task = database.get_task(route.path.rsplit('/', 1)[-1])
                return self.respond(task if task else {'error': 'Task not found'}, 200 if task else 404)
            if route.path.startswith('/api/artifacts/'):
                artifact = database.get_artifact(route.path.rsplit('/', 1)[-1])
                if not artifact:
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
            if len(parts) >= 3 and parts[:2] == ['api', 'sessions']:
                sid = parts[2]
                if not database.get(sid):
                    return self.respond({'error': 'Session not found'}, 404)
                if len(parts) == 3:
                    return self.respond(database.get(sid))
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
                    if token and not hmac.compare_digest(str(data.get('token', '')), token):
                        return self.respond({'error': 'Incorrect access token'}, 401)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Set-Cookie', 'aurex_access=' + token + '; HttpOnly; SameSite=Strict; Path=/')
                    self.end_headers()
                    self.wfile.write(b'{"ok":true}')
                    return
                if not self.auth():
                    return self.respond({'error': 'Access token required'}, 401)
                if path == '/api/sessions':
                    sid = database.session(title=str(data.get('title') or 'New conversation'))
                    return self.respond({'id': sid}, 201)
                if path in ('/api/tasks', '/api/requests'):
                    # This deployment has one administrator access token, not per-user accounts.
                    with gate:
                        result = create_task(data, source='admin' if path == '/api/tasks' else 'web')
                    return self.respond(result, 202)
                parts = path.strip('/').split('/')
                if len(parts) == 4 and parts[:2] == ['api', 'tasks'] and parts[3] == 'cancel':
                    if data:
                        raise ValueError('Task cancellation accepts an empty JSON object')
                    task = database.get_task(parts[2])
                    if not task:
                        return self.respond({'error': 'Task not found'}, 404)
                    result = database.request_cancel(task['session_id'], task['id'])
                    queue.wake.set()
                    return self.respond(result, 202 if result['status'] == 'cancelling' else 200)
                if len(parts) == 4 and parts[:2] == ['api', 'sessions'] and parts[3] == 'cancel':
                    if set(data) != {'run_id'}:
                        raise ValueError('Cancellation accepts only run_id')
                    with gate:
                        result = database.request_cancel(parts[2], data['run_id'])
                    queue.wake.set()
                    return self.respond(result, 202 if result['status'] == 'cancelling' else 200)
                if len(parts) == 4 and parts[:2] == ['api', 'sessions'] and parts[3] == 'messages':
                    sid = parts[2]
                    with gate:
                        if not database.get(sid):
                            return self.respond({'error': 'Session not found'}, 404)
                        result = create_task(data, session_id=sid)
                    return self.respond(result, 202)
                return self.respond({'error': 'Not found'}, 404)
            except (ValueError, KeyError, OSError) as exc:
                return self.respond({'error': str(exc)}, 400)

    if poll and user is None:
        raise ValueError('Community polling requires PhysicsLab login')
    server = ThreadingHTTPServer((hostname or cfg.tracking.hostname, port or cfg.tracking.port), Handler)
    try:
        queue.start()
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
