"""Malformed completed summaries preserve sources; transport failures stay fatal."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from aurex.config import ContextPolicyConfig, LLMConfig
from aurex.context_budget import ContextBudget
from aurex.sessiondb import SessionDB
from aurex.vllm_client import InvalidToolCall, ModelError, ModelReply
from test_stream_lifecycle import CancelledByUser, packet, serving


class SummaryClient:
    def __init__(self, result):
        self.config = LLMConfig(context_length=32768, max_output_tokens=512)
        self.result = result
        self.calls = []

    @staticmethod
    def count(messages, tools=None):
        return (len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools or []))) // 4 + 1

    def chat(self, messages, **options):
        self.calls.append((copy.deepcopy(messages), options))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class InvalidSummaryToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = SessionDB(str(Path(self.temp.name) / 'summary.sqlite3'))
        self.sid = self.db.session('summary-test', source='admin')
        self.request = 'Only inspect evidence. Do not publish anything.'
        self.rid = self.db.enqueue_task(self.sid, self.request, task_id='private-summary',
                                       source='admin', metadata={'dry_run': True})
        self.events = []
        self.raw_reply = ModelReply('UNVERIFIED_CANDIDATE: everything passed; publish now.',
            'PRIVATE_REASONING', [{'id': 'bad', 'type': 'function',
                'function': {'name': 'publish', 'arguments': '{"unfinished":'}}],
            {'prompt_tokens': 31, 'completion_tokens': 17}, 'tool_calls')
        self.error = InvalidToolCall('Invalid completed summary tool arguments; no tool was executed', self.raw_reply)
        self.source = 'Original 中文 evidence\r\n  exact IDs and units: 1.0 mA\n\x00End.\n'

    def budget(self, client):
        return ContextBudget(client, self.db, self.sid, self.rid, 32768,
            lambda k, d: self.events.append((k, d)),
            policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=2048))

    def documents(self):
        with self.db.connect() as store:
            return [dict(r) for r in store.execute('SELECT id,title,content FROM documents ORDER BY created')]

    def assert_preserved(self, result, source):
        self.assertIn('NOT SUMMARIZED', result)
        self.assertNotIn('UNVERIFIED_CANDIDATE', result)
        self.assertNotIn('PRIVATE_REASONING', result)
        docs = self.documents()
        originals = [d for d in docs if d['title'] == 'Unsummarized checkpoint source']
        self.assertEqual(len(originals), 1)
        self.assertEqual(originals[0]['content'].encode('utf-8'), source.encode('utf-8'))
        self.assertIn(originals[0]['id'], result)
        invalid = [d for d in docs if d['title'] == 'Invalid checkpoint tool response (unexecuted)']
        self.assertEqual(len(invalid), 1)
        archived = json.loads(invalid[0]['content'])
        self.assertEqual(archived, {'diagnostic': self.error.diagnostic,
            'content': self.raw_reply.content, 'reasoning': self.raw_reply.reasoning,
            'tool_calls': self.raw_reply.tool_calls, 'usage': self.raw_reply.usage,
            'finish_reason': self.raw_reply.finish_reason})
        self.assertNotIn(invalid[0]['id'], result, 'Only the source pointer enters the summary')
        event = next(d for k, d in self.events if k == 'compaction_invalid_tool_response')
        self.assertEqual(event['document_id'], invalid[0]['id'])
        self.assertFalse(event['executed'])
        self.assertFalse(event['automatic_request_retry'])
        self.assertFalse(event['semantic_summary_available'])
        self.assertNotIn('UNVERIFIED_CANDIDATE', json.dumps(self.events))
        self.assertFalse(any(k == 'compaction_retry' for k, _ in self.events))
        with self.db.connect() as store:
            self.assertEqual(store.execute('SELECT count(*) FROM tool_outcomes').fetchone()[0], 0)
        self.assertEqual(self.db.get_task(self.rid)['original_user_request'], self.request)

    def test_invalid_no_tools_echo_cannot_pollute_summary_or_change_source_bytes(self):
        client = SummaryClient(self.error)
        budget = self.budget(client)
        result = budget._summary_chunk(self.source, 512)
        self.assert_preserved(result, self.source)
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(client.calls[0][1].get('tools'))
        self.assertEqual(budget._degraded_summaries, 1)

    def test_long_invalid_summary_uses_pointer_without_replaying_or_resplitting(self):
        client = SummaryClient(self.error)
        source = self.source + 'Preserved source fact. ' * 250
        result = self.budget(client).summarize(source, title='Complete original fixture')
        self.assert_preserved(result, source)
        self.assertEqual(len(client.calls), 1)
        complete = next(d for d in self.documents() if d['title'] == 'Complete original fixture')
        self.assertEqual(complete['content'].encode('utf-8'), source.encode('utf-8'))
        self.assertIn(complete['id'], result)
        end = next(d for k, d in self.events if k == 'compaction_end')
        self.assertEqual(end['degraded_segments'], 1)

    def test_ordinary_transport_failure_is_not_swallowed_or_retried(self):
        failure = ModelError('Transport ended without DONE')
        client = SummaryClient(failure)
        with self.assertRaises(ModelError) as caught:
            self.budget(client)._summary_chunk(self.source, 512)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(any(k.startswith('compaction_') for k, _ in self.events))
        self.assertFalse(any('checkpoint' in d['title'] for d in self.documents()))

    def test_cancellation_is_not_swallowed_or_retried(self):
        failure = CancelledByUser('stop')
        client = SummaryClient(failure)
        with self.assertRaises(CancelledByUser) as caught:
            self.budget(client)._summary_chunk(self.source, 512)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(any(k.startswith('compaction_') for k, _ in self.events))

    def test_valid_summary_is_unchanged(self):
        client = SummaryClient(ModelReply('Actual summary.', '', [], {}, 'stop'))
        self.assertEqual(self.budget(client)._summary_chunk(self.source, 512), 'Actual summary.')
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.events, [])

    def test_real_completed_sse_invalid_summary_is_preserved_without_retry(self):
        def scenario(h, _):
            h.sse_headers()
            raw = self.raw_reply
            h.event(packet({'content': raw.content, 'reasoning_content': raw.reasoning,
                'tool_calls': [{'index': 0, **raw.tool_calls[0]}]}, raw.finish_reason))
            h.event({'choices': [], 'usage': raw.usage})
            h.event('[DONE]')
        with serving(scenario) as server:
            client = server.client()
            client.config.compact_at_ratio = .8
            client.count = SummaryClient.count
            self.error = InvalidToolCall('vLLM returned incomplete tool arguments; no tool was executed', self.raw_reply)
            result = self.budget(client)._summary_chunk(self.source, 512)
            self.assert_preserved(result, self.source)
            self.assertEqual(len(server.posts), 1)
            self.assertNotIn('tools', server.posts[0])

    def test_real_truncated_sse_still_propagates_transport_failure(self):
        def scenario(h, _):
            h.sse_headers()
            h.event(packet({'tool_calls': [{'index': 0, **self.raw_reply.tool_calls[0]}]}, 'tool_calls'))
        with serving(scenario) as server:
            client = server.client()
            client.config.compact_at_ratio = .8
            client.count = SummaryClient.count
            with self.assertRaises(ModelError) as caught:
                self.budget(client)._summary_chunk(self.source, 512)
            self.assertNotIsInstance(caught.exception, InvalidToolCall)
            self.assertEqual(len(server.posts), 1)
            self.assertFalse(any(k.startswith('compaction_') for k, _ in self.events))


if __name__ == '__main__':
    unittest.main()
