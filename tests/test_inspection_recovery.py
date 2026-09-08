"""Inspection recovery groups actual failures without limiting the task."""
import json
import unittest
from unittest import mock

import test_aurex_v3 as support
from aurex.session_agent import _inspection_failure_key
from aurex.task_reply import FINAL_SYSTEM
from aurex.tools.registry import ToolError, ToolRegistry, ToolSpec


NOT_FOUND = 'C++ circuit renderer failed: No components match focus_ids/query\n'


def call(cid, query, *, focus=False):
    arguments = {'path': 'original.sav', 'limit': 4,
                 **({'focus_ids': [query]} if focus else {'query': query})}
    return support.reply('', calls=[{'id': cid, 'type': 'function', 'function': {
        'name': 'circuit_inspect', 'arguments': json.dumps(arguments)}}], finish='tool_calls')


class InspectionRecoveryTests(unittest.TestCase):
    setUp = support.SessionAgentTests.setUp
    agent = support.SessionAgentTests.agent

    def fixture(self, outputs, effects):
        registry = ToolRegistry()
        execute = mock.Mock(side_effect=effects)
        registry.register(ToolSpec('circuit_inspect', 'Inspect actual components', {'type': 'object'}, execute))
        agent, fake = self.agent(outputs, registry)
        return agent, fake, execute

    @staticmethod
    def recoveries(agent, result):
        return [e for e in agent.db.events(result['session_id']) if e['kind'] == 'loop_recovery']

    def test_different_guessed_queries_are_recorded_without_disabling_tools(self):
        outputs = [call(str(i), f'm000{8+i:02x}') for i in range(3)]
        outputs += [call('unexecuted-echo', 'm0000b'), call('actual', 'real-original-id'),
                    support.reply('Found the original component; its function was not tested.')]
        agent, fake, execute = self.fixture(
            outputs, [ToolError(NOT_FOUND)] * 4 + [{'components': [{'id': 'real-original-id'}]}])
        fake.final_reviews = iter([
            {'outcome': 'completed', 'answer': 'Found the original component; its function was not tested.'},
        ])
        result = agent.handle(user_text='Inspect the original circuit, no blind enumeration or publishing')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([c.args[1]['query'] for c in execute.call_args_list],
                         ['m00008', 'm00009', 'm0000a', 'm0000b', 'real-original-id'])
        self.assertTrue(fake.requests[3][1]['tools'])
        recovery_context = json.dumps(fake.requests[3][0], ensure_ascii=False)
        for text in ('停止盲目枚举', '真实存在的精确ID', '未命中不代表电路设计错误', '不是任何新的测量结果'):
            self.assertIn(text, recovery_context)
        self.assertGreaterEqual(len(self.recoveries(agent, result)), 1)
        self.assertFalse(agent.db.get_tool_outcome(
            result['session_id'], result['task_id'], 'unexecuted-echo')['ok'])
        for i in range(3):
            self.assertIsNotNone(agent.db.get_tool_outcome(result['session_id'], result['task_id'], str(i)))

    def test_no_tool_candidate_finishes_without_hidden_review_continuation(self):
        # Three failed lookups produce one same-agent navigation notice. Once
        # that execution agent returns a final answer, no independent reviewer
        # may say "continue" and consume the unused scripted calls.
        outputs = [call(f'bad-{i}', f'm000{i+8:02x}', focus=i % 2 == 0)
                   for i in range(3)]
        outputs.append(support.reply('Only component lookup was attempted, not simulation.'))
        agent, fake, execute = self.fixture(outputs, [ToolError(NOT_FOUND)] * 3)
        result = agent.handle(user_text='Inspect the requested component')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['answer'], 'Only component lookup was attempted, not simulation.')
        self.assertEqual(execute.call_count, 3)
        self.assertEqual(len(self.recoveries(agent, result)), 1)
        self.assertEqual(len(agent.db.tasks(result['session_id'])), 1)
        for i in range(3):
            self.assertIsNotNone(agent.db.get_tool_outcome(result['session_id'], result['task_id'], f'bad-{i}'))
        self.assertFalse(any(messages[0].get('content') == FINAL_SYSTEM
                             for messages, _ in fake.requests))
        self.assertEqual(sum(event['kind'] == 'answer'
                             for event in agent.db.events(result['session_id'])), 1)

    def test_successful_inspection_resets_the_failure_streak(self):
        outputs = [call(str(i), f'id-{i}') for i in range(5)] + [support.reply('Inspected available evidence.')]
        agent, fake, execute = self.fixture(outputs, [ToolError(NOT_FOUND)] * 2
            + [{'components': [{'id': 'id-2'}]}] + [ToolError(NOT_FOUND)] * 2)
        result = agent.handle(user_text='Inspect')
        self.assertEqual(execute.call_count, 5)
        self.assertEqual(self.recoveries(agent, result), [])
        self.assertTrue(all(o['tools'] for m, o in fake.requests if m[0]['content'] != FINAL_SYSTEM))

    def test_different_real_diagnostics_do_not_inherit_missing_id_streak(self):
        outputs = [call(str(i), f'id-{i}') for i in range(5)] + [support.reply('Different checks failed; no measurements.')]
        agent, _, execute = self.fixture(outputs, [ToolError(NOT_FOUND)] * 2
            + [ToolError('Expected an existing .sav artifact')] + [ToolError(NOT_FOUND)] * 2)
        result = agent.handle(user_text='Inspect')
        self.assertEqual(execute.call_count, 5)
        self.assertEqual(self.recoveries(agent, result), [])

    def test_exception_type_is_part_of_the_failure_group(self):
        outputs = [call(str(i), f'id-{i}') for i in range(5)] + [support.reply('Checks were inspected.')]
        agent, _, execute = self.fixture(outputs, [ToolError(NOT_FOUND)] * 2
            + [ValueError(NOT_FOUND)] + [ToolError(NOT_FOUND)] * 2)
        result = agent.handle(user_text='Inspect')
        self.assertEqual(execute.call_count, 5)
        self.assertEqual(self.recoveries(agent, result), [])

    def test_same_structural_error_groups_despite_different_queries_without_new_measurements(self):
        outputs = [call(str(i), f'id-{i}') for i in range(3)]
        outputs += [support.reply('Input artifact unavailable; no circuit measurement was made.')]
        agent, fake, execute = self.fixture(outputs, [ToolError('Expected an existing .sav artifact')] * 3)
        result = agent.handle(user_text='Inspect')
        self.assertEqual(execute.call_count, 3)
        self.assertEqual(len(self.recoveries(agent, result)), 1)
        self.assertTrue(fake.requests[3][1]['tools'])
        self.assertEqual(result['answer'], 'Input artifact unavailable; no circuit measurement was made.')

    def test_clarification_never_enables_inspection_or_hidden_continue(self):
        agent, fake, execute = self.fixture([call('echo-1', 'guessed-1'), call('echo-2', 'guessed-2')],
            [AssertionError('Clarification must not inspect')])
        fake.final_reviews = iter([
            {'outcome': 'continue', 'answer': '仅澄清当前指代，不得恢复调查。'},
            {'outcome': 'completed', 'answer': '你指的是哪条内容？'},
        ])
        with mock.patch('aurex.community_context.resolve_wall_reference', return_value={
            'requires_reference_clarification': True, 'reason_code': 'missing_deictic_referent'}):
            result = agent.handle(user_text='这是啥')
        self.assertIn('具体指代对象尚不明确', result['answer'])
        self.assertIn('请指出要查询的实验、讨论、评论或用户', result['answer'])
        execute.assert_not_called()
        self.assertTrue(all(not options['tools'] for _, options in fake.requests))
        self.assertFalse(any(messages[0].get('content') == FINAL_SYSTEM
                             for messages, _ in fake.requests))


class InspectionFailureKeyTests(unittest.TestCase):
    def test_selector_diagnostics_group_without_quoted_ids_but_preserve_tool_and_type(self):
        first = _inspection_failure_key('circuit_inspect', {'type': 'ToolError', 'error': NOT_FOUND})
        for error in ('No components match focus_ids/query: m00008', "Unknown component 'm00022'"):
            self.assertEqual(first, _inspection_failure_key('circuit_inspect', {'type': 'ToolError', 'error': error}))
        self.assertEqual(first, ('circuit_inspect', 'ToolError', 'selection_not_found'))
        self.assertNotEqual(first, _inspection_failure_key('circuit_inspect', {'type': 'ValueError', 'error': NOT_FOUND}))
        self.assertIsNone(_inspection_failure_key('circuit_analyze', {'type': 'ToolError', 'error': NOT_FOUND}))

    def test_unrelated_diagnostic_text_is_not_aggressively_normalized(self):
        def key(text):
            return _inspection_failure_key('circuit_inspect', {'type': 'ToolError', 'error': text})
        self.assertNotEqual(key('Missing mapped pins on x'), key('Missing mapped pins on y'))
        self.assertNotEqual(key('Missing mapped pins on x'), key('Missing mapped pins on X'))
        self.assertNotEqual(key('Missing mapped pins on a  b'), key('Missing mapped pins on a b'))
        self.assertNotEqual(key('Invalid JSON artifact'), key('Missing renderer'))
        self.assertNotEqual(key("Unknown component type 'x'"), key("Unknown component type 'y'"))


if __name__ == '__main__':
    unittest.main()
