"""Actual local SSE streams; no production model or returned tool is executed."""
import threading
import unittest

from aurex.vllm_client import InvalidToolCall, ModelError, ModelReply
from test_stream_lifecycle import CancelledByUser, packet, serving


def tool_delta(index=0, *, cid='call-1', name='inspect', arguments='{}'):
    call = {'index': index, 'function': {'arguments': arguments}}
    if cid is not None:
        call['id'] = cid
    if name is not None:
        call['function']['name'] = name
    return {'tool_calls': [call]}


class InvalidToolTransportTests(unittest.TestCase):
    messages = [{'role': 'user', 'content': 'Private local fixture'}]

    def tearDown(self):
        self.assertFalse([t for t in threading.enumerate() if t.name == 'aurex-vllm-stream'])

    def completed(self, deltas, *, finish='tool_calls'):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'content': 'RAW_PRIVATE_CONTENT', 'reasoning_content': 'RAW_PRIVATE_REASONING'}))
            for delta in deltas:
                h.event(packet(delta))
            h.event(packet({}, finish))
            h.event({'choices': [], 'usage': {'prompt_tokens': 9, 'completion_tokens': 13}})
            h.event('[DONE]')
        return scenario

    def invalid(self, deltas, *, finish='tool_calls'):
        with serving(self.completed(deltas, finish=finish)) as server:
            with self.assertRaises(InvalidToolCall) as caught:
                server.client().chat(self.messages)
            error = caught.exception
            self.assertIsInstance(error, ModelError)
            self.assertIsInstance(error.reply, ModelReply)
            self.assertEqual(error.reply.content, 'RAW_PRIVATE_CONTENT')
            self.assertEqual(error.reply.reasoning, 'RAW_PRIVATE_REASONING')
            self.assertEqual(error.reply.usage, {'prompt_tokens': 9, 'completion_tokens': 13})
            self.assertEqual(error.reply.finish_reason, finish)
            self.assertEqual(str(error), error.diagnostic)
            self.assertNotIn('RAW_PRIVATE', str(error))
            self.assertEqual(len(server.posts), 1, 'Invalid batches never cause transport to retry inference')
            return error

    def test_valid_partial_argument_and_name_fragments_remain_valid(self):
        deltas = [tool_delta(name='in', arguments='{"path":'),
                  tool_delta(cid=None, name='spect', arguments='"RAW_PRIVATE_ARGUMENT"}')]
        with serving(self.completed(deltas)) as server:
            result = server.client().chat(self.messages)
            self.assertEqual(result.tool_calls, [{'id': 'call-1', 'type': 'function', 'function': {
                'name': 'inspect', 'arguments': '{"path":"RAW_PRIVATE_ARGUMENT"}'}}])
            self.assertEqual(result.content, 'RAW_PRIVATE_CONTENT')
            self.assertEqual(result.reasoning, 'RAW_PRIVATE_REASONING')
            self.assertEqual(len(server.posts), 1)

    def test_complete_sse_with_malformed_json_retains_raw_arguments_privately(self):
        raw = '{"private": "RAW_PRIVATE_ARGUMENT",'
        error = self.invalid([tool_delta(arguments=raw)])
        self.assertEqual(error.reply.tool_calls[0]['function']['arguments'], raw)
        self.assertIn('incomplete tool arguments', error.diagnostic)

    def test_json_nonobjects_are_invalid_complete_responses(self):
        for raw in ('[]', 'null', 'true', '7', '"RAW_PRIVATE_ARGUMENT"'):
            with self.subTest(raw=raw):
                error = self.invalid([tool_delta(arguments=raw)])
                self.assertEqual(error.reply.tool_calls[0]['function']['arguments'], raw)
                self.assertIn('JSON object', error.diagnostic)

    def test_invalid_second_call_preserves_entire_unexecuted_batch(self):
        error = self.invalid([tool_delta(arguments='{"path":"valid-but-unexecuted"}'),
                              tool_delta(1, cid='call-2', arguments='{"RAW_PRIVATE_ARGUMENT":')])
        self.assertEqual(len(error.reply.tool_calls), 2)
        self.assertEqual(error.reply.tool_calls[0]['function']['arguments'], '{"path":"valid-but-unexecuted"}')
        self.assertEqual(error.reply.tool_calls[1]['function']['arguments'], '{"RAW_PRIVATE_ARGUMENT":')
        self.assertEqual([c['id'] for c in error.reply.tool_calls], ['call-1', 'call-2'])

    def test_repeated_function_name_with_distinct_call_ids_is_valid(self):
        deltas = [tool_delta(arguments='{"offset":0}'),
                  tool_delta(1, cid='call-2', arguments='{"offset":8}')]
        with serving(self.completed(deltas)) as server:
            result = server.client().chat(self.messages)
            self.assertEqual([c['id'] for c in result.tool_calls], ['call-1', 'call-2'])
            self.assertEqual([c['function']['name'] for c in result.tool_calls], ['inspect', 'inspect'])
            self.assertEqual(len(server.posts), 1)

    def test_missing_duplicate_and_blank_identity_are_invalid(self):
        for deltas in ([tool_delta(cid=None)], [tool_delta(name=None)], [tool_delta(cid=' ')],
                       [tool_delta(name=' ')], [tool_delta(cid=['not-a-string'])],
                       [tool_delta(), tool_delta(1)]):
            with self.subTest(deltas=deltas):
                error = self.invalid(deltas)
                self.assertIn('identity', error.diagnostic)
                self.assertEqual(len(error.reply.tool_calls), len(deltas))

    def test_finish_mismatch_preserves_calls_and_empty_batch(self):
        with_calls = self.invalid([tool_delta()], finish='stop')
        self.assertEqual(len(with_calls.reply.tool_calls), 1)
        without_calls = self.invalid([], finish='tool_calls')
        self.assertEqual(without_calls.reply.tool_calls, [])
        self.assertIn('disagree', without_calls.diagnostic)

    def test_length_with_invalid_partial_call_still_returns_modelreply(self):
        with serving(self.completed([tool_delta(cid=None, name=None, arguments='{"partial":')], finish='length')) as server:
            result = server.client().chat(self.messages)
            self.assertIsInstance(result, ModelReply)
            self.assertEqual(result.finish_reason, 'length')
            self.assertEqual(result.tool_calls[0]['function']['arguments'], '{"partial":')
            self.assertEqual(len(server.posts), 1)

    def test_missing_done_missing_finish_and_abnormal_finish_stay_fatal(self):
        for finish, done in (('tool_calls', False), (None, True), ('content_filter', True)):
            def scenario(h, _):
                h.sse_headers()
                h.event(packet(tool_delta(arguments='{"partial":'), finish))
                if done:
                    h.event('[DONE]')
            with self.subTest(finish=finish, done=done), serving(scenario) as server:
                with self.assertRaises(ModelError) as caught:
                    server.client().chat(self.messages)
                self.assertNotIsInstance(caught.exception, InvalidToolCall)
                self.assertEqual(len(server.posts), 1)

    def test_truncated_http_body_and_malformed_sse_are_not_complete_invalid_tools(self):
        for truncated in (True, False):
            def scenario(h, _):
                h.sse_headers(length=10000 if truncated else None)
                if truncated:
                    h.event(packet(tool_delta(arguments='{"partial":')))
                else:
                    h.event('{not-json')
                    h.event('[DONE]')
            with self.subTest(truncated=truncated), serving(scenario) as server:
                with self.assertRaises(ModelError) as caught:
                    server.client().chat(self.messages)
                self.assertNotIsInstance(caught.exception, InvalidToolCall)
                self.assertEqual(len(server.posts), 1)

    def test_cancellation_after_content_stays_cancellation_and_does_not_retry(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'content': 'unfinished'}))
            h.wait_disconnect()
        with serving(scenario) as server:
            def cancel(*_):
                raise CancelledByUser('stop')
            with self.assertRaises(CancelledByUser):
                server.client().chat(self.messages, on_delta=cancel)
            self.assertEqual(len(server.posts), 1)
            self.assertTrue(server.disconnected.wait(.5))


if __name__ == '__main__':
    unittest.main()
