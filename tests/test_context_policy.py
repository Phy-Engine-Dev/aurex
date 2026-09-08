from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from aurex.config import ConfigError, ContextPolicyConfig, LLMConfig, load_config, parse_context_policy, save_config
from aurex.context_budget import ContextBudget
from aurex.sessiondb import SessionDB


class Client:
    def __init__(self, config=None, *, content='Measured facts and unfinished work retained.', finish='stop'):
        self.config = config or LLMConfig(context_length=8192, max_output_tokens=512)
        self.calls = []
        self.content, self.finish = content, finish

    def count(self, messages, tools=None):
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) + len(json.dumps(tools or []))

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        return SimpleNamespace(content=self.content, finish_reason=self.finish, tool_calls=[])


def call(cid):
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': cid, 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]}


class ContextPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = SessionDB(str(Path(self.temp.name) / 'sessions.sqlite3'))
        self.sid = self.db.session()
        self.db.begin(self.sid, 'Current isolated request', 'run')
        self.events = []
        self.client = Client()

    def budget(self, **values):
        policy = ContextPolicyConfig(safety_tokens=128, summary_max_tokens=512, **values)
        return ContextBudget(self.client, self.db, self.sid, 'run', 8192,
                             lambda kind, data: self.events.append((kind, data)), policy=policy)

    def add(self, role, content='', **fields):
        return self.db.message(self.sid, 'run', {'role': role, 'content': content, **fields})

    def test_configuration_profiles_and_roundtrip_preserve_defaults_and_selection(self):
        path = Path(self.temp.name) / 'config.json'
        path.write_text(json.dumps({'context': {'profile': 'long', 'profiles': {
            'long': {'prune': True, 'retain_recent_turns': 3}}}}))
        cfg = load_config(str(path))
        self.assertIsNone(cfg.llm.max_output_tokens)
        self.assertIsNone(cfg.llm.reasoning_effort)
        self.assertTrue(cfg.context.resolved().prune)
        self.assertEqual(cfg.context.resolved().retain_recent_turns, 3)
        self.assertFalse(cfg.context.prune)
        save_config(cfg, str(path))
        self.assertEqual(load_config(str(path)).context, cfg.context)

    def test_config_rejects_typo_nan_boolean_integer_and_recursive_profile(self):
        for raw in [{'auto_compact': 'false'}, {'safety_tokens': True}, {'compact_at_ratio': float('nan')},
                    {'tool_output_tokens': 0}, {'unknown': 1}, {'profile': 'missing'},
                    {'profiles': {'bad': {'profiles': {}}}}, {'summary_max_tokens': 1}]:
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                parse_context_policy(raw)

    def test_effort_and_large_optional_output_are_not_task_limits(self):
        path = Path(self.temp.name) / 'config.json'
        path.write_text(json.dumps({'llm': {'context_length': 65536, 'max_output_tokens': 40000,
                                          'reasoning_effort': 'medium'}}))
        self.assertEqual(load_config(str(path)).llm.max_output_tokens, 40000)
        path.write_text(json.dumps({'llm': {'reasoning_effort': 'high'}}))
        with self.assertRaises(ConfigError):
            load_config(str(path))

    def test_capacity_never_expands_to_configured_window_or_profile(self):
        client = Client(LLMConfig(context_length=32768, max_output_tokens=None))
        client.output_reserve_tokens = lambda capacity: capacity // 4
        budget = ContextBudget(client, self.db, self.sid, 'run', 8192, lambda *x: None,
                               policy=ContextPolicyConfig(safety_tokens=128))
        self.assertEqual(budget.capacity, 8192)
        self.assertEqual(budget.output_reserve, 2048)
        self.assertEqual(budget.usable, 6016)
        with self.assertRaisesRegex(RuntimeError, 'actual model'):
            ContextBudget(client, self.db, self.sid, 'run', 8192, lambda *x: None,
                          policy=parse_context_policy({'profile': 'too_big', 'profiles': {
                              'too_big': {'reserved_output_tokens': 9000}}}))

    def test_disabling_auto_preserves_full_document_even_above_soft_threshold(self):
        source = 'whole untrusted source ' * 200
        budget = self.budget(auto_compact=False, document_budget_ratio=.1)
        self.assertEqual(budget.document('source', source), source)
        self.assertFalse(self.client.calls)

    def test_disabling_auto_overflow_archives_complete_original_without_summary(self):
        source = 'HEAD' + 'x' * 10000 + 'TAIL'
        with self.assertRaisesRegex(RuntimeError, 'automatic compaction disabled'):
            self.budget(auto_compact=False).document('source', source)
        with self.db.connect() as con:
            self.assertEqual(con.execute('SELECT content FROM documents').fetchone()[0], source)
        self.assertFalse(self.client.calls)

    def test_disabling_auto_never_deletes_or_advances_messages_on_overflow(self):
        self.add('user', 'source ' * 1800)
        before = self.db.messages(self.sid)
        with self.assertRaisesRegex(RuntimeError, 'auto_compact disabled'):
            self.budget(auto_compact=False).messages('system', [])
        self.assertEqual(self.db.messages(self.sid), before)
        self.assertEqual(self.db.get(self.sid)['compacted_until'], 0)
        self.assertFalse(self.client.calls)

    def test_compaction_keeps_complete_tool_pair_and_all_original_rows(self):
        self.add('user', 'old request ' * 300)
        old_end = self.add('assistant', 'old completion ' * 100)
        self.add('user', 'Latest user constraint')
        self.db.message(self.sid, 'run', call('current'))
        self.add('tool', 'actual measurement', tool_call_id='current')
        before = self.db.messages(self.sid)
        result = self.budget(compact_at_ratio=.5).messages('system', [{'function': {'name': 'read'}}])
        self.assertEqual(self.db.get(self.sid)['compacted_until'], old_end)
        self.assertEqual(self.db.messages(self.sid), before)
        self.assertEqual(result[-2]['tool_calls'][0]['id'], 'current')
        self.assertEqual(result[-1]['tool_call_id'], 'current')
        self.assertIn('Latest user constraint', json.dumps(result))
        self.assertTrue(self.client.calls)
        self.assertTrue(all(not kw['thinking'] for _, kw in self.client.calls))

    def test_truncated_summary_keeps_sources_and_recovers_without_accepting_partial_text(self):
        self.client.finish = 'length'
        self.client.content = 'INCOMPLETE_UNVERIFIED_RESULT'
        self.add('user', 'old ' * 1500)
        self.add('assistant', 'old')
        self.add('user', 'new')
        before = self.db.messages(self.sid)
        result = self.budget(compact_at_ratio=.5).messages('system', [])
        self.assertEqual(self.db.messages(self.sid), before)
        self.assertIn('NOT SUMMARIZED', json.dumps(result))
        self.assertNotIn('INCOMPLETE_UNVERIFIED_RESULT', json.dumps(result))
        self.assertTrue(any(k == 'compaction_recovered' for k, _ in self.events))
        self.assertGreater(self.db.get(self.sid)['compacted_until'], 0)

    def test_nonreducing_candidate_falls_back_to_lossless_source_reference(self):
        self.client.content = 's' * 7000
        self.add('user', 'x' * 6200)
        self.add('assistant', 'done')
        self.add('user', 'y' * 2000)
        result = self.budget().messages('system', [])
        self.assertIn('NOT SUMMARIZED', json.dumps(result))
        self.assertIn('y' * 2000, json.dumps(result))
        self.assertNotIn('s' * 7000, json.dumps(result))
        self.assertGreater(self.db.get(self.sid)['compacted_until'], 0)

    def test_valid_chunk_handoffs_reduce_by_tokens_even_below_byte_chunk_size(self):
        budget = self.budget()
        def summarize(messages, **kwargs):
            self.client.calls.append((copy.deepcopy(messages), kwargs))
            # Three valid individual summaries fit the byte chunk size but
            # their combined token count exceeds summary_max_tokens=512.
            text = 'verified voltage 3 V; ' + 's' * 320 if len(self.client.calls) <= 3 else 'Verified voltage 3 V; next verify the load.'
            return SimpleNamespace(content=text, finish_reason='stop', tool_calls=[])
        self.client.chat = summarize
        result = budget.summarize('original facts ' * 600, title='Conversation checkpoint')
        self.assertEqual(len(self.client.calls), 4)
        self.assertTrue(all(call[1]['max_tokens'] == 512 for call in self.client.calls))
        self.assertIn('Verified voltage 3 V', result)
        self.assertNotIn('NOT SUMMARIZED', result)
        self.assertTrue(any(kind == 'compaction_reduce' for kind, _ in self.events))
        self.assertFalse(any(kind == 'compaction_recovered' for kind, _ in self.events))

    def test_summary_cancellation_or_transport_error_is_never_retried(self):
        from aurex.session_agent import RunCancelled
        from aurex.vllm_client import ModelError
        for exception in [RunCancelled('cancelled'), ModelError('disconnected')]:
            calls = []
            def fail(*args, **kwargs):
                calls.append(args)
                raise exception
            self.client.chat = fail
            with self.assertRaises(type(exception)):
                self.budget()._summary_chunk('source ' * 500, 512)
            self.assertEqual(len(calls), 1)

    def test_reduce_archive_wrapper_does_not_discard_valid_near_budget_handoff(self):
        budget = self.budget()
        overhead = self.client.count([{'role': 'user', 'content': ''}])
        near_budget = 'v' * (512 - overhead)
        self.assertEqual(self.client.count([{'role': 'user', 'content': near_budget}]), 512)
        def summarize(messages, **kwargs):
            self.client.calls.append((copy.deepcopy(messages), kwargs))
            return SimpleNamespace(content='s' * 350 if len(self.client.calls) <= 3 else near_budget,
                                   finish_reason='stop', tool_calls=[])
        self.client.chat = summarize
        result = budget.summarize('original facts ' * 600, title='Conversation checkpoint')
        self.assertEqual(len(self.client.calls), 4)
        self.assertIn(near_budget, result)
        self.assertEqual(result.count('[Complete original archived for operator audit'), 1)
        self.assertNotIn('NOT SUMMARIZED', result)
        self.assertFalse(any(kind == 'compaction_recovered' for kind, _ in self.events))

    def test_incomplete_summary_does_not_retry_or_accept_partial_output(self):
        calls = []
        def summarize(messages, **kwargs):
            calls.append((messages, kwargs))
            return SimpleNamespace(content='INCOMPLETE SEMANTIC CLAIM', finish_reason='length', tool_calls=[])
        self.client.chat = summarize
        result = self.budget()._summary_chunk('x' * 4000, 512)
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(options['max_tokens'] == 512 for _, options in calls))
        self.assertEqual([len(msgs[-1]['content']) for msgs, _ in calls], [4000])
        self.assertIn('NOT SUMMARIZED', result)
        self.assertNotIn('INCOMPLETE SEMANTIC CLAIM', result)
        incomplete = next(data for kind, data in self.events if kind == 'compaction_incomplete')
        self.assertFalse(incomplete['automatic_retry'])
        self.assertFalse(incomplete['partial_output_accepted'])

    def test_large_configured_summary_uses_short_semantic_handoff_allowance(self):
        from aurex.context_budget import SEMANTIC_SUMMARY_MAX_TOKENS, SUMMARY_PROMPT
        client = Client(LLMConfig(context_length=32768, max_output_tokens=512))
        budget = ContextBudget(client, self.db, self.sid, 'run', 32768, lambda *x: None,
                               policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=4096))
        result = budget.summarize('one bounded completed fact ' * 300, title='checkpoint')
        self.assertTrue(result)
        self.assertEqual(SEMANTIC_SUMMARY_MAX_TOKENS, 2048)
        self.assertTrue(all(options['max_tokens'] == 2048 for _, options in client.calls))
        self.assertIn('under 1200 tokens', SUMMARY_PROMPT)

    def test_pruning_is_opt_in_archived_and_keeps_result_ids_with_auto_disabled(self):
        self.add('user', 'read data')
        self.db.message(self.sid, 'run', call('old'))
        self.add('tool', 'old evidence ' * 600, tool_call_id='old')
        self.db.message(self.sid, 'run', call('new'))
        self.add('tool', 'fresh evidence', tool_call_id='new')
        before = self.db.messages(self.sid)
        result = self.budget(auto_compact=False, prune=True, prune_keep_tool_results=1).messages('system', [])
        old = next(m for m in result if m.get('tool_call_id') == 'old')
        new = next(m for m in result if m.get('tool_call_id') == 'new')
        self.assertNotIn('read_context', old['content'])
        self.assertEqual(json.loads(old['content'])['retrieval'], 'audit_only_no_reread')
        self.assertEqual(new['content'], 'fresh evidence')
        self.assertEqual(self.db.messages(self.sid), before)
        self.assertFalse(self.client.calls)

    def test_retains_configured_recent_turns_and_excludes_attachments_as_boundaries(self):
        self.add('user', 'first ' * 1100)
        self.add('assistant', 'old answer')
        self.add('user', 'second request')
        self.add('user', 'image text', _attachment=True)
        self.add('assistant', 'second answer')
        self.add('user', 'third request')
        budget = self.budget(retain_recent_turns=2, compact_at_ratio=.5)
        self.assertEqual(budget._cut(self.db.messages(self.sid)), 2)
        messages = budget.messages('system', [])
        self.assertIn('second request', json.dumps(messages))
        self.assertIn('third request', json.dumps(messages))
        self.assertFalse(any('_attachment' in m for m in messages))

    def test_pending_multi_tool_group_has_no_cut_inside_pair(self):
        self.add('user', 'request')
        c = call('one')
        c['tool_calls'].append({**call('two')['tool_calls'][0]})
        self.db.message(self.sid, 'run', c)
        self.add('tool', 'one done', tool_call_id='one')
        self.add('user', 'attachment', _attachment=True)
        self.add('tool', 'two done', tool_call_id='two')
        self.add('assistant', 'next')
        self.assertEqual(self.budget()._boundaries(self.db.messages(self.sid)), [0, 1, 5])

    def test_image_pruning_explicit_and_does_not_mutate_originals(self):
        images = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': f'image{i}'}} for i in range(3)]}]
        before = copy.deepcopy(images)
        with self.assertRaisesRegex(RuntimeError, 'prune_images is disabled'):
            self.budget(prune_images=False).limit_images(images)
        result = self.budget().limit_images(images)
        self.assertEqual(images, before)
        self.assertEqual(sum(p['type'] == 'image_url' for p in result[0]['content']), 2)
        self.assertTrue(any(k == 'context_images_pruned' for k, _ in self.events))

    def test_only_current_task_explicit_images_reach_count_and_requests(self):
        image = lambda url: {'type': 'image_url', 'image_url': {'url': url}}
        self.add('user', [image('old-auto-image')])
        self.add('user', [image('old-explicit-image')], _image_requested_by='old-task', _attachment=True)
        self.add('user', [image('current-explicit-image')], _image_requested_by='run', _attachment=True)
        before = self.db.messages(self.sid)
        counted = []
        original_count = self.client.count
        def count(messages, tools=None):
            counted.append(copy.deepcopy(messages))
            return original_count(messages, tools)
        self.client.count = count
        budget = self.budget()
        budget.image_request_scope = 'run'
        messages = budget.messages('system', [])
        urls = [part['image_url']['url'] for message in messages if isinstance(message.get('content'), list)
                for part in message['content'] if part.get('type') == 'image_url']
        self.assertEqual(urls, ['current-explicit-image'])
        self.assertNotIn('old-auto-image', json.dumps(counted))
        self.assertNotIn('old-explicit-image', json.dumps(counted))
        self.assertEqual(self.db.messages(self.sid), before)
        self.assertFalse(any('_image_requested_by' in message for message in messages))

    def test_tool_output_policy_and_summary_thinking_are_independent(self):
        self.client.config = replace(self.client.config, enable_thinking=False, max_output_tokens=None)
        budget = self.budget(tool_output_tokens=256, summary_thinking=True)
        self.assertEqual(budget.document('ordinary source', 'x' * 400), 'x' * 400)
        compact = budget.document('tool', 'x' * 400, kind='tool')
        self.assertNotIn('read_context', compact)
        self.assertEqual(json.loads(compact)['retrieval'], 'audit_only_no_reread')
        self.assertTrue(all(kw['thinking'] for _, kw in self.client.calls))
        self.assertTrue(all(kw['max_tokens'] == 512 for _, kw in self.client.calls))
        self.assertFalse(self.client.config.enable_thinking)

    def test_large_read_projection_keeps_source_cursor_and_demotes_tool_receipt_id(self):
        source_id = self.db.document(self.sid, 'Original source', 'z' * 3000)
        data = {'id': source_id, 'title': 'Original source', 'json_pointer': '/data/ports',
                'document_sha256': 'a' * 64, 'offset': 100, 'total_chars': 3000,
                'text': 'z' * 1200, 'has_more': True, 'recorded_not_resimulated': True}
        full = json.dumps({'ok': True, 'data': data})
        result_id, _ = self.db.tool_outcome(self.sid, 'run', 'context-page', 'read_context', full, True)
        output = self.budget().tool_document('Tool read_context', 'large presentation\n' + 'x' * 5000,
            document_id=result_id, tool_name='read_context',
            tool_args={'document_id': source_id, 'json_pointer': '/data/ports', 'offset': 100, 'length': 1200},
            tool_data=data)
        projected = json.loads(output)
        self.assertEqual(projected['source_document_id'], source_id)
        self.assertEqual(projected['tool_result_document_id'], result_id)
        self.assertEqual(projected['tool_result_document_id_usage'], 'operator_audit_only')
        self.assertNotIn('document_id', projected)
        self.assertNotIn('continue_source_only', projected)
        self.assertEqual(projected['retrieval'],
                         'legacy_archived_page_not_available_as_an_agent_tool')

    def test_interface_projection_preserves_exact_connectivity_fields(self):
        port = {'id': 'input-uuid', 'ref': 'C152', 'label': '', 'direction': 'input',
                'node': 'N610', 'node_connection_count': 5,
                'connected_to_other_components': True, 'logic': 0, 'logic_text': 'L',
                'logic_source': 'saved input setting, not a new solve', 'large_optional': 'x' * 6000}
        full = json.dumps({'ok': True, 'data': {'interface_only': True, 'ports': [port]}})
        result_id, _ = self.db.tool_outcome(self.sid, 'run', 'interface-page', 'circuit_inspect', full, True)
        client = Client(LLMConfig(context_length=8192, max_output_tokens=512))
        budget = ContextBudget(client, self.db, self.sid, 'run', 8192, lambda *x: None,
            policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=4096))
        projected = json.loads(budget.tool_document('Tool circuit_inspect', full,
            document_id=result_id, tool_name='circuit_inspect'))
        row = projected['sections']['/data/ports']['rows'][0]['value']
        for field in ('id', 'ref', 'direction', 'node', 'node_connection_count',
                      'connected_to_other_components', 'logic', 'logic_text', 'logic_source'):
            self.assertEqual(row[field], port[field])

        evidence = budget._evidence_capsule([{
            'name': 'circuit_inspect', 'ok': True, 'call_id': 'interface-evidence',
            'document_id': result_id, 'message_id': 12, 'arguments': {'interface_only': True},
            'arguments_sha256': 'b' * 64, 'result': {'ok': True, 'data': {
                'ports': [port], 'total_inputs': 1, 'total_outputs': 0, 'total_ports': 1}}}], 12)
        saved = evidence['interface_records'][0]['returned_ports'][0]
        for field in ('id', 'ref', 'direction', 'node', 'node_connection_count',
                      'connected_to_other_components', 'logic', 'logic_text', 'logic_source'):
            self.assertEqual(saved[field], port[field])
        self.assertIn('bit significance', evidence['interface_records'][0]['scope'])

    def test_circuit_projection_keeps_control_contract_without_renderer_payload(self):
        data = {
            'controls_only': True,
            'controls': [{'id': 'SW', 'kind': 'spst', 'value_name': 'closed',
                          'current': 0, 'allowed': [0, 1],
                          'source_model_id': 'Simple Switch',
                          'primitive_component_ids': ['SW']}],
            'circuit_path': '/immutable/design.circuit.json',
            'artifact': {'netlist_path': '/immutable/netlist.json',
                         'camera_path': '/immutable/camera-view.json'},
            'camera': {'source': 'not-rendered', 'image_generated': False,
                       'position': [1, 2, 3], 'warnings': ['with_image=false']},
            'netlist': {'nodes': [{'id': 'N1', 'connections': [
                {'component': 'SW', 'pin': 0}], 'total_connections': 1}],
                        'components': [{'id': 'SW', 'ref': 'C1',
                                        'type': 'Simple Switch',
                                        'position': [.2, .3, 0],
                                        'rotation': [0, 0, 180],
                                        'pins': [{'pin': 0, 'node': 'N1'}]}]},
        }
        full = json.dumps({'ok': True, 'data': data}, ensure_ascii=False)
        result_id, _ = self.db.tool_outcome(self.sid, 'run', 'controls-spatial',
                                            'circuit_inspect', full, True)
        client = Client(LLMConfig(context_length=8192, max_output_tokens=512))
        budget = ContextBudget(client, self.db, self.sid, 'run', 8192,
                               lambda *x: None,
                               policy=ContextPolicyConfig(safety_tokens=128,
                                                          summary_max_tokens=4096))
        projected = json.loads(budget.tool_document(
            'Tool circuit_inspect', full, document_id=result_id,
            tool_name='circuit_inspect'))
        row = projected['sections']['/data/controls']['rows'][0]['value']
        self.assertEqual(row['id'], 'SW')
        self.assertEqual(row['allowed'], [0, 1])
        self.assertEqual(projected['fields']['/data/circuit_path'], data['circuit_path'])
        self.assertNotIn('/data/camera', projected['fields'])
        self.assertNotIn('/data/netlist/nodes', projected['sections'])
        self.assertNotIn('/data/netlist/components', projected['sections'])
        self.assertNotIn('/data/artifact', projected['fields'])
        self.assertEqual(projected['retrieval'], 'audit_only_no_reread')
        self.assertFalse('full_presentation_document_id' in projected)

    def test_exact_node_projection_keeps_page_cursor_and_selected_components(self):
        components = [{'id': f'dff-{i}', 'ref': f'C{148 + i}', 'type': 'D Flipflop',
                       'pins': [{'pin': 0, 'node': f'N{i}'}, {'pin': 3, 'node': 'N24'}],
                       'selection_role': 'primary'} for i in range(8)]
        data = {'node_query': {'node': 'N24', 'exact': True, 'match_count': 40,
                               'offset': 0, 'limit': 8, 'requested_limit': 8, 'next_offset': 8},
                'pagination': {'query': 'N24', 'match_count': 40, 'offset': 0,
                               'limit': 8, 'next_offset': 8},
                'netlist': {'components': components, 'scope': 'exact saved-node matches'},
                'untrusted_optional': 'x' * 8000}
        full = json.dumps({'ok': True, 'data': data})
        result_id, _ = self.db.tool_outcome(self.sid, 'run', 'node-page', 'circuit_inspect', full, True)
        client = Client(LLMConfig(context_length=8192, max_output_tokens=512))
        budget = ContextBudget(client, self.db, self.sid, 'run', 8192, lambda *x: None,
            policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=4096))
        projected = json.loads(budget.tool_document('Tool circuit_inspect', full,
            document_id=result_id, tool_name='circuit_inspect'))
        self.assertEqual(projected['fields']['/data/node_query']['next_offset'], 8)
        self.assertEqual(projected['fields']['/data/pagination']['next_offset'], 8)
        rows = projected['sections']['/data/components']
        self.assertEqual(rows['shown_rows'], 8)
        self.assertEqual([row['value']['ref'] for row in rows['rows']], [row['ref'] for row in components])

    def test_batch_circuit_projection_preserves_exact_selected_fields_byte_for_byte(self):
        results = [{
            'query': f'C{i}', 'ok': True, 'component_ids': [f'uuid-{i}'],
            'components': [{'id': f'uuid-{i}', 'ref': f'C{i}',
                            'type': 'Logic Output', 'label': f'OUT{i}',
                            'pins': [{'pin': 0, 'node': f'N{i}',
                                      'total_connections': 2, 'connected': True}],
                            'properties': {'高电平': 3.0}}],
            'nodes': [], 'match_count': 1, 'has_more': False,
        } for i in range(1, 7)]
        data = {'batch': True, 'query_count': len(results),
                'selected_fields': ['identity', 'pins', 'properties.高电平'],
                'results': results}
        full = json.dumps({'ok': True, 'data': data})
        result_id, _ = self.db.tool_outcome(
            self.sid, 'run', 'batch-circuit', 'circuit_query_many', full, True)
        budget = ContextBudget(Client(LLMConfig(context_length=8192, max_output_tokens=512)),
            self.db, self.sid, 'run', 8192, lambda *x: None,
            policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=4096))
        projected = json.loads(budget.tool_document('Tool circuit_query_many', full,
            document_id=result_id, tool_name='circuit_query_many'))
        self.assertEqual(projected, json.loads(full))
        rows = projected['data']['results']
        self.assertEqual([row['query'] for row in rows], [f'C{i}' for i in range(1, 7)])
        self.assertEqual([row['components'][0]['id'] for row in rows],
                         [f'uuid-{i}' for i in range(1, 7)])
        self.assertTrue(all(row['components'][0]['properties'] == {'高电平': 3.0}
                            for row in rows))
        self.assertTrue(all('低电平' not in row['components'][0]['properties']
                            for row in rows))
        self.assertTrue(all(row['components'][0]['pins'][0]['node'] == f'N{i}'
                            for i, row in enumerate(rows, 1)))

    def test_circuit_projection_keeps_compact_measured_trace_summary_before_large_rows(self):
        trace_summary = {
            'sample_count': 200, 'time_start_s': 1.5e-5, 'time_end_s': .003,
            'node_ranges_v': {
                'vin': {'min_v': -.01, 'max_v': .01, 'peak_to_peak_v': .02,
                        'first_v': .001, 'last_v': 0.0, 'mean_v': 0.0},
                'vout': {'min_v': -.91, 'max_v': .87, 'peak_to_peak_v': 1.78,
                         'first_v': -.17, 'last_v': -.01, 'mean_v': 0.0},
            },
            'scope': 'Actual recorded sample points only.',
        }
        data = {'numerical_verification': {'verified': True,
                    'functional_verification': False, 'trace_summary': trace_summary},
                'state_path': '/tmp/snapshot.pe-state.json',
                'netlist': {'components': [
                    {'id': f'R{i}', 'type': 'Resistor', 'properties': {'blob': 'x' * 400}}
                    for i in range(40)]}}
        full = json.dumps({'ok': True, 'data': data})
        result_id, _ = self.db.tool_outcome(self.sid, 'run', 'analog-tr', 'circuit_analyze', full, True)
        budget = ContextBudget(Client(LLMConfig(context_length=8192, max_output_tokens=512)),
            self.db, self.sid, 'run', 8192, lambda *x: None,
            policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=2048))
        projected = json.loads(budget.tool_document('Tool circuit_analyze', full,
            document_id=result_id, tool_name='circuit_analyze'))
        kept = projected['fields']['/data/numerical_verification']['trace_summary']
        self.assertEqual(kept, trace_summary)
        self.assertEqual(kept['node_ranges_v']['vout']['peak_to_peak_v'], 1.78)

    def test_circuit_projection_keeps_complete_component_manifest(self):
        manifest = [{'id': f'C{i}', 'type': 'resistor'} for i in range(20)]
        data = {'component_manifest': manifest, 'circuit_path': '/tmp/design.circuit.json',
                'netlist': {'components': [
                    {'id': f'C{i}', 'type': 'Resistor', 'properties': {'blob': 'x' * 600}}
                    for i in range(20)]}}
        full = json.dumps({'ok': True, 'data': data})
        result_id, _ = self.db.tool_outcome(self.sid, 'run', 'create', 'circuit_create', full, True)
        budget = ContextBudget(Client(LLMConfig(context_length=8192, max_output_tokens=512)),
            self.db, self.sid, 'run', 8192, lambda *x: None,
            policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=2048))
        projected = json.loads(budget.tool_document('Tool circuit_create', full,
            document_id=result_id, tool_name='circuit_create'))
        self.assertEqual(projected['fields']['/data/component_manifest'], manifest)

    def test_circuit_projection_keeps_native_edit_manifest(self):
        manifest = [{
            'id': 'R1', 'type': 'resistor', 'nodes': ['in', 'gnd'],
            'params': {'r': 1000.0}, 'position': [0.2, 0.0, 0.0],
            'rotation': [0.0, 0.0, 180.0], 'pin_labels': ['1', '2'],
        }]
        data = {'native_component_manifest': manifest,
                'circuit_path': '/tmp/design.circuit.json',
                'netlist': {'components': [
                    {'id': 'R1', 'type': 'Resistor',
                     'properties': {'电阻': 1000, 'blob': 'x' * 1200}}
                ]}}
        full = json.dumps({'ok': True, 'data': data})
        result_id, _ = self.db.tool_outcome(self.sid, 'run', 'native-manifest',
                                             'circuit_edit', full, True)
        budget = ContextBudget(self.client, self.db, self.sid, 'run', 8192,
                               lambda *x: None,
                               policy=ContextPolicyConfig(safety_tokens=128,
                                                          summary_max_tokens=2048))
        projected = json.loads(budget.tool_document('Tool circuit_edit', full,
                                                     document_id=result_id,
                                                     tool_name='circuit_edit'))
        if '/data/native_component_manifest' in projected['fields']:
            self.assertEqual(projected['fields']['/data/native_component_manifest'], manifest)
        else:
            section = projected['sections']['/data/native_component_manifest']
            self.assertEqual(section['rows'][0]['value'], manifest[0])
            self.assertTrue(section['edit_contract'])

    def test_checkpoint_machine_evidence_keeps_exact_node_pages_and_exceptional_pin(self):
        pages = [
            (0, 8, [{'id': 'dff-148', 'ref': 'C148', 'type': 'D Flipflop',
                      'pins': [{'pin': 0, 'node': 'N1'}, {'pin': 3, 'node': 'N24'}],
                      'selection_role': 'primary'}]),
            (16, 24, [{'id': 'and-497', 'ref': 'C497', 'type': 'And Gate',
                        'pins': [{'pin': 0, 'node': 'N4'}, {'pin': 1, 'node': 'N5'},
                                 {'pin': 2, 'node': 'N24', 'label': 'out'}],
                        'selection_role': 'primary'}]),
        ]
        until = 0
        for offset, next_offset, components in pages:
            cid = f'node-page-{offset}'
            args = {'path': '/tmp/original.sav', 'query': 'N24', 'offset': offset, 'limit': 8}
            self.db.message(self.sid, 'run', {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': cid, 'type': 'function', 'function': {
                    'name': 'circuit_inspect', 'arguments': json.dumps(args)}}]})
            data = {'node_query': {'node': 'N24', 'exact': True, 'match_count': 40,
                                   'offset': offset, 'limit': 8, 'requested_limit': 8,
                                   'next_offset': next_offset},
                    'netlist': {'components': components, 'scope': 'exact saved-node matches'}}
            _, until = self.db.tool_outcome(self.sid, 'run', cid, 'circuit_inspect',
                                             json.dumps({'ok': True, 'data': data}), True)

        component_args = {'path': '/tmp/original.sav', 'query': 'C152', 'limit': 8}
        self.db.message(self.sid, 'run', {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'component-C152', 'type': 'function', 'function': {
                'name': 'circuit_inspect', 'arguments': json.dumps(component_args)}}]})
        component_data = {'pagination': {'match_count': 1, 'offset': 0, 'limit': 8,
                                         'next_offset': None, 'total_matches': 1},
                          'netlist': {'components': [{
                              'id': 'input-152', 'ref': 'C152', 'type': 'Logic Input',
                              'label': '', 'selection_role': 'primary',
                              'pin_semantics_source': 'Aurex faithful PLSAV-to-Phy-Engine import mapping',
                              'pins': [{'pin': 0, 'node': 'N610', 'label': 'out'}],
                              'properties': {'开关': 0.0, '高电平': 3.0}}]}}
        _, until = self.db.tool_outcome(self.sid, 'run', 'component-C152', 'circuit_inspect',
                                         json.dumps({'ok': True, 'data': component_data}), True)

        budget = self.budget()
        journal = budget._tool_index(until, 4096)
        payload = json.loads(journal.split('\n', 1)[1])
        refs = payload['machine_evidence']['connectivity_refs']
        self.assertTrue(any(row['query'] == 'C152' for row in refs))
        full = budget._index_cache[3]['connectivity_records']
        self.assertEqual({(row['offset'], row['next_offset']) for row in full if row['query'] == 'N24'},
                         {(0, 8), (16, 24)})
        exceptional = next(row for row in full if row.get('offset') == 16)['components'][0]
        self.assertEqual((exceptional['ref'], exceptional['type']), ('C497', 'And Gate'))
        self.assertEqual(exceptional['matching_pins'], [{'pin': 2, 'node': 'N24', 'label': 'out'}])
        self.assertEqual(full[1]['classification'],
                         'exact_saved_node_connectivity_page_not_signal_role_inference')
        self.assertIn('not inferred', full[1]['scope'])
        component = next(row for row in full if row['query'] == 'C152')
        self.assertEqual(component['components'][0]['pins'], [{'pin': 0, 'node': 'N610', 'label': 'out'}])
        self.assertEqual(component['components'][0]['properties']['开关'], 0.0)

    def test_large_interface_uses_complete_compact_exact_port_index(self):
        ports = [{'id': f'{i:032x}', 'ref': f'C{i + 1}', 'label': '',
                  'direction': 'input' if i < 45 else 'output', 'node': f'N{i}',
                  'node_connection_count': 2, 'connected_to_other_components': True,
                  'logic': 0, 'logic_text': 'L',
                  'logic_source': 'saved input setting, not a new solve' if i < 45 else 'saved output statistic, not a new solve',
                  'position': [i, 0, 0], 'rotation': [0, 0, 0]} for i in range(61)]
        full = json.dumps({'ok': True, 'data': {'interface_only': True, 'ports': ports,
            'total_inputs': 45, 'total_outputs': 16, 'total_ports': 61}})
        result_id, _ = self.db.tool_outcome(self.sid, 'run', 'large-interface', 'circuit_inspect', full, True)
        client = Client(LLMConfig(context_length=8192, max_output_tokens=512))
        client.count = lambda messages, tools=None: len(json.dumps([messages, tools or []], ensure_ascii=False).encode()) // 3 + 1
        budget = ContextBudget(client, self.db, self.sid, 'run', 8192, lambda *x: None,
            policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=4096))
        projected = json.loads(budget.tool_document('Tool circuit_inspect', full,
            document_id=result_id, tool_name='circuit_inspect'))
        section = projected['sections']['/data/ports']
        self.assertEqual(section['shown_rows'], 61)
        self.assertEqual(section['omitted_rows'], 0)
        self.assertTrue(section['complete_interface_index'])
        self.assertEqual(section['columns'][0:6], ['id', 'ref', 'label', 'direction', 'node', 'node_connection_count'])
        self.assertEqual(section['rows'][44][0:6], [ports[44][field] for field in section['columns'][0:6]])

    def test_active_request_is_pinned_after_long_single_turn_compaction(self):
        self.add('user', 'original request ' * 350)
        self.db.message(self.sid, 'run', call('old'))
        self.add('tool', 'measured old data ' * 90, tool_call_id='old')
        self.db.message(self.sid, 'run', call('latest'))
        self.add('tool', 'latest data', tool_call_id='latest')
        budget = self.budget(compact_at_ratio=.5)
        budget.active_request = 'Exact objective: preserve experiment IDs and prove the requested behavior, not a different task.'
        result = budget.messages('system', [])
        self.assertGreater(self.db.get(self.sid)['compacted_until'], 0)
        self.assertTrue(any(m['role'] == 'user' and budget.active_request in m.get('content', '') for m in result))
        self.assertTrue(any('Current task original request' in m.get('content', '') for m in budget.messages('system', [])))


if __name__ == '__main__':
    unittest.main()
