"""Real local HTTP tests; never contact vLLM or execute a returned tool."""
from __future__ import annotations

import asyncio
import json
import select
import socket
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from aurex.vllm_client import ModelError, VLLMClient


class CancelledByUser(RuntimeError):
    pass


def packet(delta=None, finish=None):
    choice = {'delta': delta or {}}
    if finish:
        choice['finish_reason'] = finish
    return {'choices': [choice]}


class LocalServer:
    def __init__(self, scenario, *, health_status=200, health_hangs=False):
        self.scenario = scenario
        self.health_status = health_status
        self.health_hangs = health_hangs
        self.posts = []
        self.health_checks = 0
        self.post_seen = threading.Event()
        self.disconnected = threading.Event()
        self.stopping = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path != '/v1/models':
                    self.send_error(404)
                    return
                owner.health_checks += 1
                if owner.health_hangs:
                    while not owner.stopping.wait(0.02):
                        ready, _, _ = select.select([self.connection], [], [], 0)
                        if ready and self.connection.recv(1, socket.MSG_PEEK) == b'':
                            return
                    return
                body = b'{"data":[{"id":"test","max_model_len":90112}]}'
                self.send_response(owner.health_status)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_POST(self):
                if self.path != '/v1/chat/completions':
                    self.send_error(404)
                    return
                owner.posts.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                owner.post_seen.set()
                try:
                    owner.scenario(self, owner)
                except (BrokenPipeError, ConnectionResetError):
                    owner.disconnected.set()
                finally:
                    self.close_connection = True

            def sse_headers(self, status=200, length=None):
                self.send_response(status)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                if length is not None:
                    self.send_header('Content-Length', str(length))
                self.end_headers()
                self.wfile.flush()

            def event(self, data):
                text = data if isinstance(data, str) else json.dumps(data)
                self.wfile.write(('data: ' + text + '\n\n').encode())
                self.wfile.flush()

            def wait_disconnect(self):
                # Only the fake server observes its own socket. Production code
                # uses public async HTTP cancellation, not private socket access.
                while not owner.stopping.is_set():
                    ready, _, _ = select.select([self.connection], [], [], 0.02)
                    if ready and self.connection.recv(1, socket.MSG_PEEK) == b'':
                        owner.disconnected.set()
                        return

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': 0.02}, daemon=True)
        self.thread.start()

    def client(self):
        config = SimpleNamespace(base_url=f'http://127.0.0.1:{self.server.server_port}/v1',
                                 model='test', api_key_env='AUREX_TEST_UNUSED_KEY',
                                 context_length=90112, temperature=0.6,
                                 max_output_tokens=None, enable_thinking=False,
                                 reasoning_effort=None, timeout_sec=0.03)
        client = VLLMClient(config)
        client._TICK_INTERVAL = 0.02
        client._HEALTH_INTERVAL = 0.06
        client._HEALTH_TIMEOUT = 0.1
        client._CLEANUP_TIMEOUT = 1.0
        return client

    def close(self):
        self.stopping.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


@contextmanager
def serving(scenario, **kwargs):
    server = LocalServer(scenario, **kwargs)
    try:
        yield server
    finally:
        server.close()


class StreamLifecycleTests(unittest.TestCase):
    messages = [{'role': 'user', 'content': 'local fixture only'}]

    def tearDown(self):
        self.assertFalse([t for t in threading.enumerate() if t.name == 'aurex-vllm-stream'],
                         'stream worker leaked after return/cancellation')

    def test_buffered_tool_survives_silence_and_keeps_ticking(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'reasoning_content': 'brief'}))
            time.sleep(0.35)  # Far beyond this fixture's old 0.03s read timeout.
            h.event(packet({'tool_calls': [{'index': 0, 'id': 'call1',
                                            'function': {'name': 'inspect', 'arguments': '{"path":'}}]}))
            h.event(packet({'tool_calls': [{'index': 0, 'function': {'arguments': '"fixture"}'}}]}, 'tool_calls'))
            h.event({'choices': [], 'usage': {'completion_tokens': 12}})
            h.event('[DONE]')

        with serving(scenario) as server:
            ticks, deltas = [], []
            result = server.client().chat(self.messages, thinking=True,
                                          on_tick=lambda: ticks.append(time.monotonic()),
                                          on_delta=lambda kind, value: deltas.append((kind, value)))
            self.assertEqual(result.reasoning, 'brief')
            self.assertEqual(result.tool_calls[0]['function']['arguments'], '{"path":"fixture"}')
            self.assertEqual(result.usage, {'completion_tokens': 12})
            self.assertGreaterEqual(len(ticks), 6)
            self.assertGreaterEqual(server.health_checks, 2)
            self.assertEqual(len(server.posts), 1, 'health checks must not retry inference')
            self.assertEqual(deltas, [('reasoning', 'brief')])
            self.assertNotIn('max_tokens', server.posts[0])
            self.assertEqual(server.posts[0]['chat_template_kwargs'], {'enable_thinking': True})

    def test_cancel_before_request_does_not_start_inference(self):
        def never(h, _):
            self.fail('Cancelled request reached the server')
        with serving(never) as server:
            def cancel():
                raise CancelledByUser('stop')
            with self.assertRaises(CancelledByUser):
                server.client().chat(self.messages, on_tick=cancel)
            self.assertEqual(server.posts, [])

    def test_default_callback_is_per_call_and_explicit_callback_takes_priority(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'content': 'ok'}, 'stop'))
            h.event('[DONE]')
        with serving(scenario) as server:
            client = server.client()
            def cancel():
                raise CancelledByUser('stop')
            client.on_tick = cancel
            with self.assertRaises(CancelledByUser):
                client.chat(self.messages)
            self.assertEqual(server.posts, [])
            self.assertEqual(client.chat(self.messages, on_tick=lambda: None).content, 'ok')
            client.on_tick = lambda: None
            self.assertEqual(client.chat(self.messages).content, 'ok')
            self.assertEqual(len(server.posts), 2)

    def test_cancel_interrupts_silent_headers_and_body_and_closes_socket(self):
        for headers in (False, True):
            def scenario(h, _):
                if headers:
                    h.sse_headers()
                h.wait_disconnect()
            with self.subTest(headers=headers), serving(scenario) as server:
                def cancel():
                    if server.post_seen.is_set():
                        raise CancelledByUser('stop')
                started = time.monotonic()
                with self.assertRaises(CancelledByUser):
                    server.client().chat(self.messages, on_tick=cancel)
                self.assertLess(time.monotonic() - started, 1.0)
                self.assertTrue(server.disconnected.wait(0.5))
                self.assertEqual(len(server.posts), 1)

    def test_cancel_from_delta_callback_closes_socket(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'content': 'partial'}))
            h.wait_disconnect()
        with serving(scenario) as server:
            def cancel(kind, value):
                raise CancelledByUser('stop after delta')
            with self.assertRaises(CancelledByUser):
                server.client().chat(self.messages, on_delta=cancel)
            self.assertTrue(server.disconnected.wait(0.5))

    def test_cancel_during_a_hung_health_probe_still_closes_inference(self):
        def scenario(h, _):
            h.sse_headers()
            h.wait_disconnect()
        with serving(scenario, health_hangs=True) as server:
            client = server.client()
            client._HEALTH_TIMEOUT = 5.0
            def cancel():
                if server.health_checks:
                    raise CancelledByUser('stop during health probe')
            started = time.monotonic()
            with self.assertRaises(CancelledByUser):
                client.chat(self.messages, on_tick=cancel)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertTrue(server.disconnected.wait(0.5))

    def test_hung_health_probes_are_bounded_without_retrying_inference(self):
        def scenario(h, _):
            h.sse_headers()
            h.wait_disconnect()
        with serving(scenario, health_hangs=True) as server:
            with self.assertRaisesRegex(ModelError, 'health checks failed'):
                server.client().chat(self.messages)
            self.assertTrue(server.disconnected.wait(0.5))
            self.assertEqual(len(server.posts), 1)

    def test_clean_eof_without_finish_is_error_and_preserves_delivered_delta(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'content': 'not complete'}))
        with serving(scenario) as server:
            deltas = []
            with self.assertRaisesRegex(ModelError, 'incomplete'):
                server.client().chat(self.messages, on_delta=lambda *args: deltas.append(args))
            self.assertEqual(deltas, [('text', 'not complete')])

    def test_abrupt_http_disconnect_is_error(self):
        def scenario(h, _):
            h.sse_headers(length=10000)
            h.event(packet({'content': 'partial'}))
        with serving(scenario) as server:
            with self.assertRaisesRegex(ModelError, 'request failed'):
                server.client().chat(self.messages)

    def test_finish_without_done_is_not_success(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'content': 'apparently done'}, 'stop'))
        with serving(scenario) as server:
            with self.assertRaisesRegex(ModelError, r'without \[DONE\]'):
                server.client().chat(self.messages)

    def test_done_with_missing_finish_is_not_success(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'content': 'partial'}))
            h.event('[DONE]')
        with serving(scenario) as server:
            with self.assertRaisesRegex(ModelError, 'without a finish reason'):
                server.client().chat(self.messages)

    def test_complete_stop_closes_server_that_keeps_connection_open(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'content': 'ok'}, 'stop'))
            h.event('[DONE]')
            h.wait_disconnect()
        with serving(scenario) as server:
            client = server.client()
            client.config.reasoning_effort = 'medium'
            result = client.chat(self.messages, thinking=True, max_tokens=4096)
            self.assertEqual(result.content, 'ok')
            self.assertTrue(server.disconnected.wait(0.5))
            self.assertEqual(server.posts[0]['max_tokens'], 4096)
            self.assertEqual(server.posts[0]['chat_template_kwargs'],
                             {'enable_thinking': True, 'reasoning_effort': 'medium'})

    def test_healthy_active_stream_does_not_fail_on_unhealthy_probe_endpoint(self):
        def scenario(h, _):
            h.sse_headers()
            for _ in range(10):
                h.event(packet({'content': '.'}))
                time.sleep(0.025)
            h.event(packet({}, 'stop'))
            h.event('[DONE]')
        with serving(scenario, health_status=503) as server:
            self.assertEqual(server.client().chat(self.messages).content, '.' * 10)

    def test_silent_dead_service_health_failure_closes_inference(self):
        def scenario(h, _):
            h.sse_headers()
            h.wait_disconnect()
        with serving(scenario, health_status=503) as server:
            with self.assertRaisesRegex(ModelError, 'health checks failed'):
                server.client().chat(self.messages)
            self.assertGreaterEqual(server.health_checks, 3)
            self.assertEqual(len(server.posts), 1)
            self.assertTrue(server.disconnected.wait(0.5))

    def test_http_error_and_error_sse_are_errors(self):
        for kind in ('http', 'sse'):
            def scenario(h, _):
                h.sse_headers(status=503 if kind == 'http' else 200)
                h.event({'error': 'fixture error'})
            with self.subTest(kind=kind), serving(scenario) as server:
                with self.assertRaises(ModelError):
                    server.client().chat(self.messages)

    def test_invalid_or_incomplete_tool_call_cannot_be_success(self):
        cases = [('{"unfinished":', 'tool_calls'), ('[]', 'tool_calls'),
                 ('{}', 'stop'), ('{}', 'content_filter')]
        for arguments, finish in cases:
            def scenario(h, _):
                h.sse_headers()
                h.event(packet({'tool_calls': [{'index': 0, 'id': 'c',
                                                'function': {'name': 'publish', 'arguments': arguments}}]}, finish))
                h.event('[DONE]')
            with self.subTest(arguments=arguments, finish=finish), serving(scenario) as server:
                with self.assertRaises(ModelError):
                    server.client().chat(self.messages)

    def test_length_remains_explicit_incomplete_result_not_tool_completion(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'tool_calls': [{'index': 0, 'id': 'c',
                                            'function': {'name': 'inspect', 'arguments': '{"x":'}}]}, 'length'))
            h.event('[DONE]')
        with serving(scenario) as server:
            result = server.client().chat(self.messages)
            self.assertEqual(result.finish_reason, 'length')
            self.assertEqual(result.tool_calls[0]['function']['arguments'], '{"x":')

    def test_bad_json_closes_active_stream(self):
        def scenario(h, _):
            h.sse_headers()
            h.event('{invalid')
            h.wait_disconnect()
        with serving(scenario) as server:
            with self.assertRaises(ModelError):
                server.client().chat(self.messages)
            self.assertTrue(server.disconnected.wait(0.5))

    def test_sync_chat_can_be_called_from_an_existing_asyncio_loop(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'content': 'ok'}, 'stop'))
            h.event('[DONE]')
        with serving(scenario) as server:
            async def caller():
                return server.client().chat(self.messages)
            self.assertEqual(asyncio.run(caller()).content, 'ok')


if __name__ == '__main__':
    unittest.main()
