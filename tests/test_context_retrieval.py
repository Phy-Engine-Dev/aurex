import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aurex.context_retrieval import read_context
from aurex.sessiondb import SessionDB
from aurex.config import LLMConfig
from aurex.vllm_client import VLLMClient
from aurex.tools.registry import ToolRegistry, ToolSpec
import test_aurex_v3 as support


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = SessionDB(str(Path(temp.name) / 'db.sqlite3'))
        self.sid = self.db.session('source', source='admin')
        self.raw = json.dumps({'data': {'ports': [{'id': '原输入', 'state': 0}, {'id': '输出', 'state': 2}],
            'a/b': {'~key': [False, None, 0]}}}, ensure_ascii=False)
        self.did = self.db.document(self.sid, 'Exact source', self.raw)

    def test_exact_subtree_zero_unknown_and_escaped_keys(self):
        for pointer, expected in (('/data/ports/0', {'id': '原输入', 'state': 0}),
                                  ('/data/ports/1/state', 2), ('/data/a~1b/~0key', [False, None, 0])):
            result = read_context(self.db, self.sid, {'document_id': self.did, 'json_pointer': pointer})
            self.assertEqual(json.loads(result['text']), expected)
            self.assertEqual(result['document_sha256'], hashlib.sha256(self.raw.encode()).hexdigest())
            self.assertTrue(result['recorded_not_resimulated'])
        self.assertEqual(self.db.read_document(self.sid, self.did)['text'], self.raw)

    def test_subtree_paging_is_not_full_document_offset(self):
        text = json.dumps(json.loads(self.raw)['data']['ports'], ensure_ascii=False, separators=(',', ':'))
        pieces = []
        for offset in range(0, len(text), 7):
            result = read_context(self.db, self.sid, {'document_id': self.did, 'json_pointer': '/data/ports',
                'offset': offset, 'length': 7})
            pieces.append(result['text'])
            self.assertEqual(result['total_chars'], len(text))
            self.assertIn('selected subtree', result['offset_scope'])
        self.assertEqual(''.join(pieces), text)

    def test_cross_session_and_bad_pointers_never_fall_back_to_another_value(self):
        other = self.db.session('other', source='admin')
        with self.assertRaises(ValueError):
            read_context(self.db, other, {'document_id': self.did, 'json_pointer': '/data'})
        for pointer in ('data', '/data/absent', '/data/ports/-1', '/data/ports/01', '/data/a~2b', '/data/ports/999999999999999999999999'):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                read_context(self.db, self.sid, {'document_id': self.did, 'json_pointer': pointer})
        plain = self.db.document(self.sid, 'Plain', 'not JSON')
        with self.assertRaisesRegex(ValueError, 'not JSON'):
            read_context(self.db, self.sid, {'document_id': plain, 'json_pointer': ''})
        self.assertEqual(read_context(self.db, self.sid, {'document_id': plain})['text'], 'not JSON')
        ambiguous = self.db.document(self.sid, 'Ambiguous', '{"state":0,"state":1}')
        with self.assertRaisesRegex(ValueError, 'duplicate keys'):
            read_context(self.db, self.sid, {'document_id': ambiguous, 'json_pointer': '/state'})

    def test_typo_is_rejected_instead_of_silently_expanding_default_page(self):
        with self.assertRaisesRegex(ValueError, r'length \(not limit\)'):
            read_context(self.db, self.sid, {'document_id': self.did, 'limit': 20000})

    def test_json_field_search_finds_nested_records_without_text_scanning(self):
        source = {'wrapper': {'components': [
            {'id': 'c1', 'ref': 'C1', 'type': 'Basic Capacitor',
             'properties': {'电容': 5.0, '频率': 10.0}},
            {'id': 'c5', 'ref': 'C5', 'type': 'Square Source',
             'properties': {'频率': 10.0, '占空比': .25}},
            {'id': 'c24', 'ref': 'C24', 'type': 'Square Source',
             'properties': {'频率': 20.0}},
        ]}}
        did = self.db.document(self.sid, 'circuit_query_many: full netlist_path',
                               json.dumps(source, ensure_ascii=False))
        result = read_context(self.db, self.sid, {'document_id': did, 'json_search': {
            'field': 'type', 'match': 'exact', 'value': 'Square Source',
            'fields': ['id', 'ref', 'type', 'properties'], 'limit': 1}})
        data = json.loads(result['text'])
        self.assertEqual(data['total_matches'], 2)
        self.assertEqual(data['rows'][0]['json_pointer'], '/wrapper/components/1')
        self.assertEqual(data['rows'][0]['value']['id'], 'c5')
        self.assertTrue(data['has_more_rows'])
        by_frequency = read_context(self.db, self.sid, {'document_id': did, 'json_search': {
            'field': 'properties.频率', 'match': 'exact', 'value': 10,
            'fields': ['id', 'ref', 'type', 'properties']}})
        rows = json.loads(by_frequency['text'])['rows']
        self.assertEqual([row['value']['id'] for row in rows], ['c1', 'c5'])

    def test_json_field_search_is_strict_and_bounded(self):
        cases = (
            {'json_search': {}},
            {'json_search': {'field': 'type', 'fields': ['id']}},
            {'json_search': {'field': 'type', 'match': 'contains', 'value': 10, 'fields': ['id']}},
            {'json_search': {'field': 'type', 'match': 'exists', 'value': None, 'fields': ['id']}},
            {'json_search': {'field': 'type', 'match': 'bad', 'value': 'x', 'fields': ['id']}},
            {'json_search': {'field': 'type', 'value': 'x', 'fields': [], 'limit': 33}},
        )
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                read_context(self.db, self.sid, {'document_id': self.did, **extra})
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            read_context(self.db, self.sid, {'document_id': self.did, 'json_pointer': '/data',
                'json_search': {'field': 'id', 'value': 'x', 'fields': ['id']}})

    def test_recorded_read_result_cannot_become_a_nested_source(self):
        result_id = self.db.document(self.sid, 'Tool read_context', json.dumps({'ok': True, 'data': {
            'id': self.did, 'json_pointer': '/data/ports', 'offset': 10,
            'text': 'abcdef', 'total_chars': 100, 'has_more': True,
        }}))
        with self.assertRaises(ValueError) as caught:
            read_context(self.db, self.sid, {'document_id': result_id, 'offset': 16, 'length': 20})
        message = str(caught.exception)
        self.assertIn('derived Tool read_context outcome', message)
        self.assertIn('diagnostics-only', message)
        self.assertIn(self.did, message)
        self.assertIn('"offset":16', message)
        self.assertNotIn('abcdef', message)


class NoToolsTransportTests(unittest.TestCase):
    def test_exact_tokenizer_cache_never_caches_fallback_or_mutable_image_URL(self):
        client = VLLMClient(LLMConfig())
        response = mock.Mock(ok=True)
        response.json.return_value = {'count': 123}
        messages = [{'role': 'user', 'content': '原始请求'}]
        with mock.patch.object(client.http, 'post', return_value=response) as post:
            self.assertEqual(client.count(messages), 123)
            self.assertEqual(client.count(copy.deepcopy(messages)), 123)
            self.assertEqual(post.call_count, 1)
            client.count([{'role': 'user', 'content': '不同请求'}])
            self.assertEqual(post.call_count, 2)
            image_messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://example.test/mutable.png'}}]}]
            client.count(image_messages)
            client.count(image_messages)
            self.assertEqual(post.call_count, 4)
            client._count_cache.clear()
            response.ok = False
            self.assertNotEqual(client.count(messages), 123)
            response.ok = True
            self.assertEqual(client.count(messages), 123)
            self.assertEqual(post.call_count, 6)

    def test_tokenizer_cache_expires_and_capacity_refresh_clears_it(self):
        client = VLLMClient(LLMConfig())
        response = mock.Mock(ok=True)
        response.json.return_value = {'count': 7, 'data': [{'id': client.config.model, 'max_model_len': 65536}]}
        with mock.patch.object(client.http, 'post', return_value=response) as post, \
             mock.patch.object(client.http, 'get', return_value=response), \
             mock.patch('aurex.vllm_client.time.monotonic', return_value=1) as clock:
            client.count([])
            client.count([])
            self.assertEqual(post.call_count, 1)
            clock.return_value = 32
            client.count([])
            self.assertEqual(post.call_count, 2)
            client.capacity()
            client.count([])
            self.assertEqual(post.call_count, 3)

    def test_summary_and_review_explicitly_disable_tool_choice(self):
        client = VLLMClient(LLMConfig())
        for tools in (None, [], [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}]):
            stream = iter([b'data: {"choices":[{"delta":{"content":"Done"},"finish_reason":"stop"}]}', b'data: [DONE]'])
            with self.subTest(tools=tools), mock.patch.object(client, '_stream_lines', return_value=stream) as request:
                client.chat([{'role': 'user', 'content': 'Summarize'}], tools=tools, thinking=False)
            payload = request.call_args.args[0]
            self.assertEqual(payload['tool_choice'], 'auto' if tools else 'none')
            self.assertEqual(payload['chat_template_kwargs']['enable_thinking'], False)


class RepeatedReadTests(unittest.TestCase):
    setUp = support.SessionAgentTests.setUp
    agent = support.SessionAgentTests.agent

    def test_failed_reads_still_enter_recovery_instead_of_looping_silently(self):
        tools = ToolRegistry()
        execute = mock.Mock(side_effect=ValueError('Missing recorded path'))
        tools.register(ToolSpec('circuit_read_trace', 'Read', {'type': 'object'}, execute))
        outputs = [support.reply('', calls=[{'id': str(i), 'type': 'function', 'function': {
            'name': 'circuit_read_trace', 'arguments': '{"path":"missing"}'}}], finish='tool_calls') for i in range(3)]
        outputs.append(support.reply('The selected artifact was not found; no measured result.'))
        agent, fake = self.agent(outputs, tools)
        result = agent.handle(user_text='Read this result')
        events = agent.db.events(result['session_id'])
        self.assertEqual(sum(event['kind'] == 'loop_recovery' for event in events), 1)
        self.assertEqual(sum(event['kind'] == 'retrieval_review' for event in events), 0)
        self.assertTrue(fake.requests[3][1]['tools'])

    def test_resume_does_not_duplicate_request_or_first_thinking_or_step_number(self):
        agent, fake = self.agent([support.reply('Existing probe failed; no CPU instruction was verified.')])
        sid = agent.db.session('resume-test', source='admin')
        request = 'Continue the CPU verification; no community publishing'
        rid = agent.db.enqueue_task(sid, request, source='admin', metadata={'dry_run': True})
        agent.db.message(sid, rid, {'role': 'user', 'content': request})
        agent.db.event(sid, rid, 'model_start', {'step': 4, 'thinking': True})
        agent.db.event(sid, rid, 'model_end', {'step': 4})
        agent.db.run_status(rid, 'interrupted')
        agent.handle(user_text=request, session_id=sid, run_id=rid)
        with agent.db.connect() as store:
            users = store.execute("SELECT COUNT(*) FROM messages WHERE run_id=? AND role='user'", (rid,)).fetchone()[0]
        self.assertEqual(users, 1)
        self.assertFalse(fake.requests[0][1]['thinking'])
        events = agent.db.events(sid)
        self.assertEqual([e['data']['step'] for e in events if e['kind'] == 'model_start' and type(e['data'].get('step')) is int], [4, 5])
        self.assertEqual(sum(e['kind'] == 'resumed' for e in events), 1)

    def test_community_resume_uses_persisted_reference_without_enrichment_or_resummary(self):
        agent, fake = self.agent([support.reply('请明确你指的是哪个实验。')])
        sid = agent.db.session('community-resume-test', source='community')
        target = {'type': 'User', 'id': 'a' * 24}
        request = 'CONTEXT_JSON: ' + json.dumps({'target': target}) + '\n这是啥？'
        rid = agent.db.enqueue_task(sid, request, source='community', target=target,
            requester_user_id='b' * 24, requester_nickname='原提问者', metadata={'dry_run': True})
        agent.db.message(sid, rid, {'role': 'user', 'content': request})
        agent.db.event(sid, rid, 'task_context_bound', {'reference_resolution': {
            'requires_reference_clarification': True, 'reason_code': 'original_missing_reference'}})
        agent.db.event(sid, rid, 'model_start', {'step': 0, 'thinking': True})
        agent.db.event(sid, rid, 'model_end', {'step': 0})
        agent.db.run_status(rid, 'interrupted')
        with mock.patch('aurex.community_context.build_mention_context', side_effect=AssertionError('Must not reload community')), \
             mock.patch('aurex.community_context.resolve_wall_reference', side_effect=AssertionError('Must not re-resolve')):
            agent.handle(user_text=request, session_id=sid, run_id=rid)
        self.assertFalse(fake.requests[0][1]['tools'])
        self.assertFalse(fake.requests[0][1]['thinking'])
        self.assertIn('original_missing_reference', json.dumps(fake.requests[0][0]))

    def test_alternating_identical_recorded_trace_reads_are_allowed(self):
        tools = ToolRegistry()
        execute = mock.Mock(side_effect=lambda rt, args: {'state_path': args['path'],
            'recorded_not_resimulated': True, 'points': [[1.0, 0.0]]})
        tools.register(ToolSpec('circuit_read_trace', 'Read', {'type': 'object'}, execute))
        outputs = []
        for index, path in enumerate(('a', 'b', 'a')):
            outputs.append(support.reply('', calls=[{'id': str(index), 'type': 'function', 'function': {
                'name': 'circuit_read_trace', 'arguments': json.dumps({'path': path})}}], finish='tool_calls'))
        outputs.append(support.reply('Read all three requested trace snapshots; no new simulation was performed.'))
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': 'Read all three requested trace snapshots; no new simulation was performed.'}])
        result = agent.handle(user_text='Check the measured results, without publishing')
        self.assertEqual(execute.call_count, 3)
        events = agent.db.events(result['session_id'])
        self.assertEqual(sum(event['kind'] == 'retrieval_review' for event in events), 0)
        self.assertEqual(sum(event['kind'] == 'loop_recovery' for event in events), 0)
        self.assertTrue(fake.requests[3][1]['tools'])
        self.assertTrue(all(not opts['thinking'] for _, opts in fake.requests[1:4]))
        repeated = agent.db.get_tool_outcome(result['session_id'], result['task_id'], '2')
        self.assertTrue(repeated['ok'])
        self.assertTrue(any(event['kind'] == 'repeated_evidence_call' and event['data']['executed']
                            for event in events))
        self.assertEqual(len(agent.db.tasks(result['session_id'])), 1)

    def test_identical_narration_with_different_tools_gets_same_agent_notice(self):
        tools = ToolRegistry()
        execute = mock.Mock(side_effect=lambda rt, args: {'measured': args['point']})
        tools.register(ToolSpec('probe', 'Probe', {'type': 'object', 'properties': {
            'point': {'type': 'integer'}}, 'required': ['point']}, execute))
        narration = ('I need to inspect more inputs before I can test the CPU. '
                     'The baseline is not a functional test, so I will trace another port before deciding the stimulus. '
                     'I still need to understand the same input mapping.')
        outputs = [support.reply(narration, calls=[{'id': 'p'+str(i), 'type':'function',
            'function': {'name':'probe','arguments':json.dumps({'point':i})}}], finish='tool_calls')
            for i in range(3)]
        outputs.append(support.reply('Three different points were measured; this is the bounded result.'))
        agent, fake = self.agent(outputs, tools)
        result = agent.handle(user_text='Verify representative CPU behavior')
        self.assertEqual(result['status'],'completed');self.assertEqual(execute.call_count,3)
        self.assertTrue(fake.requests[3][1]['tools'])
        events=agent.db.events(result['session_id'])
        self.assertEqual(sum(e['kind']=='assistant_repetition_notice' for e in events),1)
        self.assertTrue(any(e['kind']=='loop_recovery' for e in events))
        from aurex.task_reply import FINAL_SYSTEM
        self.assertFalse(any(messages[0].get('content') == FINAL_SYSTEM
                             for messages, _ in fake.requests))


if __name__ == '__main__':
    unittest.main()
