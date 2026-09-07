"""Offline generated-token observations: all transport and sockets are mocked."""
import hashlib
import io
import json
import socket
import unittest
from dataclasses import replace
from unittest.mock import patch

from aurex.config import ConfigError, LLMConfig, load_config
from aurex.vllm_client import (DegenerateGeneration, InvalidToolCall, ModelError,
                               VLLMClient, _TokenProgress)


def packet(delta=None, *, token_ids=None, finish=None, index=0, **extra):
    return b'data: ' + json.dumps({
        'choices': [{'index': index, 'delta': delta or {}, 'token_ids': token_ids,
                     'finish_reason': finish}], **extra}).encode()


TOOLS = [{'type': 'function', 'function': {'name': 'inspect', 'parameters': {'type': 'object'}}}]
VALID_TOOL = {'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function',
                            'function': {'name': 'inspect', 'arguments': '{"x":1}'}}]}


class TokenProgressTests(unittest.TestCase):
    def setUp(self):
        self.no_net = patch.object(socket.socket, 'connect', side_effect=AssertionError('NETWORK FORBIDDEN'))
        self.no_net.start()

    def tearDown(self):
        self.no_net.stop()

    def client(self, **options):
        config = replace(LLMConfig(), stream_token_progress=True, **options)
        return VLLMClient(config)

    def request(self, *, client=None, thinking=False, tools=TOOLS, lines=None, on_delta=None):
        client = client or self.client()
        sent = []
        lines = lines if lines is not None else [packet(VALID_TOOL, token_ids=[11, 12], finish='tool_calls'), b'data: [DONE]']
        def receive(payload, **_):
            sent.append(payload)
            yield from lines
        with patch.object(client, '_stream_lines', side_effect=receive):
            result = client.chat([{'role': 'user', 'content': 'Synthetic task'}],
                                 tools=tools, thinking=thinking, on_delta=on_delta)
        self.assertEqual(len(sent), 1, 'No automatic request retry')
        return result, sent[0]

    def test_enabled_only_no_thinking_with_tools(self):
        for enabled, thinking, tools, expected in [
            (False, False, TOOLS, False), (True, True, TOOLS, False),
            (True, False, [], False), (True, False, None, False),
            (True, False, TOOLS, True),
        ]:
            with self.subTest(enabled=enabled, thinking=thinking, tools=bool(tools)):
                cfg = replace(LLMConfig(), stream_token_progress=enabled)
                events = []
                _, payload = self.request(client=VLLMClient(cfg), thinking=thinking, tools=tools,
                                          on_delta=lambda *x: events.append(x))
                self.assertEqual(payload.get('return_token_ids', False), expected)
                self.assertEqual(bool(events), expected)
                self.assertNotIn('max_tokens', payload)

    def test_config_false_resolves_to_explicit_false_request(self):
        _, payload = self.request(client=self.client(enable_thinking=False), thinking=None)
        self.assertIs(payload['chat_template_kwargs']['enable_thinking'], False)
        self.assertTrue(payload['return_token_ids'])

    def test_default_config_and_strict_bool_loading(self):
        self.assertFalse(LLMConfig().stream_token_progress)
        for value in [False, True, 0, 1, 'false', None, [], {}]:
            with self.subTest(value=value), patch('builtins.open', return_value=io.StringIO(json.dumps({'llm': {'stream_token_progress': value}}))):
                if type(value) is bool:
                    self.assertIs(load_config('private-fixture.json').llm.stream_token_progress, value)
                else:
                    with self.assertRaisesRegex(ConfigError, 'stream_token_progress'):
                        load_config('private-fixture.json')

    def test_prompt_and_other_choice_ids_are_ignored(self):
        events = []
        lines = [packet(token_ids=[], prompt_token_ids=[900001] * 1000),
                 packet(token_ids=[800001] * 100, index=1),
                 packet(VALID_TOOL, token_ids=[11, 12], finish='tool_calls'), b'data: [DONE]']
        reply, _ = self.request(lines=lines, on_delta=lambda *x: events.append(x))
        report = json.loads(events[-1][1])
        self.assertEqual(report['generated_tokens'], 2)
        self.assertEqual(report['generated_sha256'], hashlib.sha256(b'\0\0\0\x0b\0\0\0\x0c').hexdigest())
        self.assertNotIn('900001', json.dumps(events))
        self.assertEqual(reply.tool_calls[0]['function']['arguments'], '{"x":1}')
        self.assertEqual(reply.reasoning, '')

    def test_periodic_pattern_counts_across_chunks_exactly(self):
        progress = _TokenProgress()
        progress.observe([10, 20, 30, 10])
        progress.observe([20, 30] + [10, 20, 30] * 18)
        result = json.loads(progress.take(final=True))
        self.assertEqual(result['generated_tokens'], 60)
        self.assertEqual(result['max_exact_repeat'], {'period_tokens': 3, 'copies': 20, 'span_tokens': 60})
        self.assertEqual(result['max_same_token_run'], 1)

    def test_constant_run_hits_only_the_extreme_per_response_fuse_and_memory_stays_bounded(self):
        progress = _TokenProgress()
        for _ in range(100):
            progress.observe([123456] * 100)
        result = json.loads(progress.take(final=True))
        self.assertEqual(result['max_same_token_run'], 10000)
        self.assertEqual(result['max_exact_repeat'], {'period_tokens': 1, 'copies': 10000, 'span_tokens': 10000})
        self.assertEqual(len(progress._history), 64)
        self.assertEqual(len(progress._matches), 65)
        self.assertNotIn('123456', json.dumps(result))
        self.assertEqual(progress.degenerate_reason(), 'same_token_run')

    def test_short_normal_repetition_does_not_trip_fuse(self):
        progress = _TokenProgress()
        progress.observe([7] * (_TokenProgress.SAME_TOKEN_FUSE - 1))
        self.assertIsNone(progress.degenerate_reason())
        progress.observe([8, 9] * 200)
        self.assertIsNone(progress.degenerate_reason())

    def test_broken_pattern_does_not_inflate_exact_span(self):
        progress = _TokenProgress()
        progress.observe([1, 2] * 4 + [9] + [1, 2] * 3)
        result = json.loads(progress.take(final=True))
        self.assertEqual(result['max_exact_repeat'], {'period_tokens': 2, 'copies': 4, 'span_tokens': 8})

    def test_period_coverage_is_explicit_not_a_task_limit(self):
        progress = _TokenProgress()
        for _ in range(100):
            progress.observe(list(range(64)))
        result = json.loads(progress.take(final=True))
        self.assertEqual(result['max_exact_repeat'], {'period_tokens': 64, 'copies': 100, 'span_tokens': 6400})
        other = _TokenProgress()
        other.observe(list(range(65)) * 3)
        result = json.loads(other.take(final=True))
        self.assertEqual(result['generated_tokens'], 195)
        self.assertEqual(result['observed_period_limit'], 64)
        self.assertEqual(result['max_exact_repeat']['span_tokens'], 0)

    def test_invalid_diagnostic_types_never_become_tokens_or_join_repeats(self):
        progress = _TokenProgress()
        progress.observe([5, 5, True, 5, 5, 1.5, '5', None, -1, 2**32, {'x': 5}])
        progress.observe('not a list')
        progress.observe(None)
        result = json.loads(progress.take(final=True))
        self.assertEqual(result['generated_tokens'], 4)
        self.assertEqual(result['max_same_token_run'], 2)
        self.assertEqual(result['invalid_token_entries'], 8)
        events = []
        reply, _ = self.request(lines=[packet(token_ids={'malformed': 'field'}),
                                       packet(VALID_TOOL, finish='tool_calls'), b'data: [DONE]'],
                                on_delta=lambda *x: events.append(x))
        self.assertEqual(reply.finish_reason, 'tool_calls')

    def test_five_second_throttle_and_final_flush(self):
        with patch('aurex.vllm_client.time.monotonic', return_value=100) as clock:
            progress = _TokenProgress()
            progress.observe([1])
            self.assertIsNone(progress.take())
            clock.return_value = 104.99
            self.assertIsNone(progress.take())
            clock.return_value = 105
            self.assertIsNotNone(progress.take())
            progress.observe([2])
            clock.return_value = 106
            self.assertIsNone(progress.take())
            self.assertEqual(json.loads(progress.take(final=True))['generated_tokens'], 2)
            self.assertIsNone(progress.take(final=True))

    def test_raw_reasoning_is_never_in_progress_event(self):
        events = []
        reply, _ = self.request(lines=[packet({'reasoning_content': 'PRIVATE_REASONING'}, token_ids=[101]),
                                       packet(VALID_TOOL, token_ids=[102], finish='tool_calls'), b'data: [DONE]'],
                                on_delta=lambda *x: events.append(x))
        self.assertEqual(reply.reasoning, 'PRIVATE_REASONING', 'Existing ModelReply semantics remain unchanged')
        progress = [text for kind, text in events if kind == 'progress']
        self.assertEqual(len(progress), 1)
        self.assertNotIn('PRIVATE_REASONING', progress[0])
        self.assertNotIn('token_ids', progress[0])

    def test_invalid_tool_validation_still_fails_closed(self):
        malformed = {'tool_calls': [{'index': 0, 'id': 'call-1', 'function': {'name': 'inspect', 'arguments': '{'}}]}
        with self.assertRaises(InvalidToolCall):
            self.request(lines=[packet(malformed, token_ids=[1, 1], finish='tool_calls'), b'data: [DONE]'])

    def test_length_and_missing_done_preserve_existing_behavior(self):
        reply, _ = self.request(lines=[packet(token_ids=[1, 2], finish='length'), b'data: [DONE]'])
        self.assertEqual(reply.finish_reason, 'length')
        with self.assertRaisesRegex(ModelError, 'without.*DONE'):
            self.request(lines=[packet(VALID_TOOL, token_ids=[1, 2], finish='tool_calls')])

    def test_progress_callback_cancellation_closes_iterator_without_retry(self):
        client, calls, closed = self.client(), [], []
        def receive(payload, **_):
            calls.append(payload)
            try:
                yield packet(token_ids=[1, 1])
                yield packet(VALID_TOOL, finish='tool_calls')
                yield b'data: [DONE]'
            finally:
                closed.append(True)
        def cancel(kind, _):
            if kind == 'progress':
                raise RuntimeError('operator cancellation')
        with patch.object(client, '_stream_lines', side_effect=receive), patch.object(_TokenProgress, 'INTERVAL_SECONDS', 0):
            with self.assertRaisesRegex(RuntimeError, 'operator cancellation'):
                client.chat([{'role': 'user', 'content': 'synthetic'}], tools=TOOLS, thinking=False, on_delta=cancel)
        self.assertEqual((len(calls), closed), (1, [True]))

    def test_extreme_repetition_aborts_one_stream_and_closes_it_without_retry(self):
        client, calls, closed = self.client(), [], []
        def receive(payload, **_):
            calls.append(payload)
            try:
                yield packet(token_ids=[777] * _TokenProgress.SAME_TOKEN_FUSE)
                yield packet(VALID_TOOL, finish='tool_calls')
                yield b'data: [DONE]'
            finally:
                closed.append(True)
        with patch.object(client, '_stream_lines', side_effect=receive):
            with self.assertRaises(DegenerateGeneration) as caught:
                client.chat([{'role': 'user', 'content': 'synthetic'}],
                            tools=TOOLS, thinking=False)
        self.assertEqual((len(calls), closed), (1, [True]))
        self.assertEqual(caught.exception.progress['reason'], 'same_token_run')
        self.assertEqual(caught.exception.reply.finish_reason, 'repetition_guard')
        self.assertEqual(caught.exception.reply.tool_calls, [])


if __name__ == '__main__':
    unittest.main()
