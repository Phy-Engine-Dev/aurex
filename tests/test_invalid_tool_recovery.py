"""Invalid terminal tool responses cannot execute a prefix or terminate a task."""
import json
from pathlib import Path
import unittest
from unittest import mock

import test_aurex_v3 as support
from aurex.task_reply import FINAL_SYSTEM
from aurex.tools.registry import ToolRegistry, ToolSpec
from aurex.vllm_client import InvalidToolCall, ModelError


def call(cid, name='measure', arguments='{}'):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}


class InvalidToolRecoveryTests(unittest.TestCase):
    setUp = support.SessionAgentTests.setUp
    agent = support.SessionAgentTests.agent

    def arrange(self, actions, tools=None, reviews=None, review_error=None):
        agent, fake = self.agent([], tools)
        normal_chat = fake.chat
        actions = iter(actions)
        errors = [review_error] if review_error else []
        fake.final_reviews = iter(reviews or [{'outcome': 'completed', 'answer': 'Measured 5 V.'}])

        def chat(messages, **options):
            if messages[0].get('content') == FINAL_SYSTEM:
                if errors:
                    fake.requests.append((messages, options))
                    raise errors.pop()
                return normal_chat(messages, **options)
            fake.requests.append((messages, options))
            value = next(actions)
            if isinstance(value, Exception):
                raise value
            fake.last_output = value
            if options.get('on_delta') and value.content:
                options['on_delta']('text', value.content)
            return value

        fake.chat = chat
        return agent, fake

    def tools(self):
        tools = ToolRegistry()
        execute = mock.Mock(return_value={'voltage': 5})
        forbidden = mock.Mock(side_effect=AssertionError('Invalid batch prefix executed'))
        tools.register(ToolSpec('measure', 'Measure', {'type': 'object'}, execute))
        tools.register(ToolSpec('external_write', 'Write', {'type': 'object'}, forbidden))
        return tools, execute, forbidden

    def test_malformed_second_call_rejects_valid_prefix_and_reviews_real_evidence(self):
        tools, execute, forbidden = self.tools()
        invalid = support.reply('UNVERIFIED_CANDIDATE', calls=[call('prefix', 'external_write'),
            call('invalid', arguments='{"bad":')], finish='tool_calls')
        agent, fake = self.arrange([
            InvalidToolCall('Incomplete JSON object', invalid),
            support.reply('', calls=[call('corrected')], finish='tool_calls'),
            support.reply('Measured 5 V.'),
        ], tools, [{'outcome': 'continue', 'answer': 'Submit the focused measurement with complete parameters.'},
                   {'outcome': 'completed', 'answer': 'Measured 5 V.'}])
        result = agent.handle(user_text='Measure voltage')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(execute.call_count, 1)
        forbidden.assert_not_called()
        events = agent.db.events(result['session_id'])
        invalid_event = next(e for e in events if e['kind'] == 'invalid_tool_response')
        self.assertFalse(invalid_event['data']['executed'])
        self.assertEqual(sum(e['kind'] == 'answer' for e in events), 1)
        self.assertEqual(len(agent.db.tasks(result['session_id'])), 1)
        with agent.db.connect() as db:
            raw = json.loads(db.execute('SELECT content FROM documents WHERE id=?',
                (invalid_event['data']['document_id'],)).fetchone()[0])
        self.assertEqual(raw['tool_calls'], invalid.tool_calls)
        self.assertEqual(raw['content'], 'UNVERIFIED_CANDIDATE')
        for messages, _ in fake.requests:
            if messages[0]['content'] == FINAL_SYSTEM:
                self.assertNotIn('UNVERIFIED_CANDIDATE', json.dumps(messages))
        self.assertTrue(fake.requests[2][1]['tools'])

    def test_reused_call_id_is_not_executed_twice_or_reset_by_review(self):
        tools, execute, forbidden = self.tools()
        agent, fake = self.arrange([
            support.reply('', calls=[call('same')], finish='tool_calls'),
            support.reply('', calls=[call('same', 'external_write')], finish='tool_calls'),
            support.reply('', calls=[call('new')], finish='tool_calls'),
            support.reply('Measured 5 V.'),
        ], tools, [{'outcome': 'continue', 'answer': 'Use a fresh call identity for the next required measurement.'},
                   {'outcome': 'completed', 'answer': 'Measured 5 V.'}])
        result = agent.handle(user_text='Measure voltage twice')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(execute.call_count, 2)
        forbidden.assert_not_called()
        self.assertEqual(sum(e['kind'] == 'invalid_tool_response' for e in agent.db.events(result['session_id'])), 1)

    def test_length_truncated_tool_batch_retries_in_same_agent_before_execution(self):
        tools, execute, forbidden = self.tools()
        partial = support.reply('', calls=[call('cut', 'external_write', '{"rows":[')], finish='length')
        agent, fake = self.arrange([
            partial, support.reply('', calls=[call('small')], finish='tool_calls'), support.reply('Measured 5 V.'),
        ], tools, [{'outcome': 'continue', 'answer': 'Use one focused measurement rather than the unfinished batch.'},
                   {'outcome': 'completed', 'answer': 'Measured 5 V.'}])
        result = agent.handle(user_text='Measure voltage')
        self.assertEqual(result['status'], 'completed')
        forbidden.assert_not_called()
        execute.assert_called_once()
        self.assertNotEqual(fake.requests[1][0][0]['content'], FINAL_SYSTEM)
        self.assertTrue(fake.requests[1][1]['tools'])
        self.assertTrue(fake.requests[2][1]['tools'])
        self.assertFalse(any(messages[0].get('content') == FINAL_SYSTEM
                             for messages, _ in fake.requests))
        self.assertEqual(sum(e['kind'] == 'generation_continuation' for e in agent.db.events(result['session_id'])), 1)

    def test_clarification_invalid_response_never_enables_tools(self):
        tools, execute, forbidden = self.tools()
        invalid = support.reply('WRONG_REFERENT', calls=[call('bad', arguments='[')], finish='tool_calls')
        agent, fake = self.arrange([InvalidToolCall('Arguments are not an object', invalid)], tools,
            [{'outcome': 'completed', 'answer': '你具体指的是哪条内容？'}])
        with mock.patch('aurex.community_context.resolve_wall_reference', return_value={
                'requires_reference_clarification': True, 'reason_code': 'missing_reference'}):
            result = agent.handle(user_text='这是啥？')
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(all(not options['tools'] for _, options in fake.requests))
        execute.assert_not_called()
        forbidden.assert_not_called()

    def test_direct_answer_is_not_reopened_by_an_independent_reviewer(self):
        invalid = support.reply('UNVERIFIED_REVIEW', calls=[call('bad', arguments='[')], finish='tool_calls')
        agent, fake = self.arrange([support.reply('Candidate one')],
            review_error=InvalidToolCall('Malformed review tool arguments', invalid))
        result = agent.handle(user_text='Explain the supplied measurement')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['answer'], 'Candidate one')
        events = agent.db.events(result['session_id'])
        self.assertEqual(sum(e['kind'] == 'answer' for e in events), 1)
        self.assertEqual(sum(e['kind'] == 'task_continues' for e in events), 0)
        self.assertEqual(len(fake.requests), 1)
        self.assertNotEqual(fake.requests[0][0][0].get('content'), FINAL_SYSTEM)
        self.assertNotIn('UNVERIFIED_REVIEW', json.dumps(fake.requests[0][0]))

    def test_transport_failure_is_not_silently_reissued_as_format_recovery(self):
        agent, fake = self.arrange([ModelError('Connection failed')])
        with self.assertRaisesRegex(ModelError, 'Connection failed'):
            agent.handle(user_text='Measure voltage')
        self.assertEqual(len(fake.requests), 1)
        task = agent.db.tasks()[0]
        self.assertEqual(task['status'], 'error')
        self.assertFalse(any(e['kind'] == 'invalid_tool_response' for e in agent.db.events(task['session_id'])))

    def test_complete_experiment_summary_is_archived_without_flooding_the_next_prompt(self):
        description = '真实作者介绍。' * 12000 + 'ARCHIVED_TAIL_ONLY'
        full = Path(self.temp.name) / 'full-description.txt'
        full.write_text(description, encoding='utf-8')
        summary = Path(self.temp.name) / 'full-summary.json'
        summary.write_text(json.dumps({'Subject': 'Source title', 'Description': [description]}), encoding='utf-8')
        tools = ToolRegistry()
        tools.register(ToolSpec('plar_get_experiment_file', 'Read source', {'type': 'object'},
            lambda rt, args: {'source_summary': {'title': 'Source title', 'description_preview': description[:100]},
                             'full_description_path': str(full), 'full_summary_path': str(summary)}))
        agent, fake = self.arrange([
            support.reply('', calls=[call('source', 'plar_get_experiment_file')], finish='tool_calls'),
            support.reply('Read source introduction.'),
        ], tools)
        result = agent.handle(user_text='Introduce the experiment')
        self.assertEqual(result['status'], 'completed')
        with agent.db.connect() as db:
            rows = db.execute('SELECT title,content FROM documents WHERE session_id=?',
                              (result['session_id'],)).fetchall()
        source = next(row[1] for row in rows if row[0] == 'plar_get_experiment_file: full full_description_path')
        self.assertEqual(source.encode('utf-8'), description.encode('utf-8'))
        presented = json.dumps(fake.requests[1][0], ensure_ascii=False)
        self.assertIn('full_description_path', presented)
        self.assertNotIn('read_context', presented)
        self.assertIn('audit_only_no_reread', presented)
        self.assertNotIn('ARCHIVED_TAIL_ONLY', presented)


if __name__ == '__main__':
    unittest.main()
