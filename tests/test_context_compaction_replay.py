"""Offline replay contracts: no model/network/solver and no production DB writes."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from aurex.config import ContextPolicyConfig, LLMConfig
from aurex.context_budget import ContextBudget
from aurex.sessiondb import SessionDB


class Client:
    def __init__(self):
        self.config = LLMConfig(context_length=65536, max_output_tokens=512)
        self.calls = []

    def count(self, messages, tools=None):
        return len(json.dumps([messages, tools or []], ensure_ascii=False).encode()) // 3 + 1

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        return SimpleNamespace(content='Source claims remain unverified. Prior attempts are recorded; continue only the original open goal.',
                               finish_reason='stop', tool_calls=[])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = SessionDB(str(Path(temporary.name) / 'isolated.sqlite3'))
        self.sid = self.db.session(source='admin')
        self.rid = self.db.enqueue_task(self.sid, '验证数字设计，保留错误证据，禁止发布。', source='admin')
        self.client = Client()
        self.events = []
        self.budget = ContextBudget(self.client, self.db, self.sid, self.rid, 65536,
            lambda k, d: self.events.append((k, d)), policy=ContextPolicyConfig(safety_tokens=128))
        self.serial = 0

    def tool(self, name, data, *, args=None, ok=True, display=None):
        cid = 'call-' + str(self.serial)
        self.serial += 1
        self.db.message(self.sid, self.rid, {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args or {})}}]})
        raw = json.dumps({'ok': ok, 'data': data}, ensure_ascii=False)
        did, mid = self.db.tool_outcome(self.sid, self.rid, cid, name, raw, ok)
        self.db.update_tool_message(self.sid, mid, display if display is not None else raw)
        return cid, did, mid, raw

    def document(self, did):
        with self.db.connect() as db:
            return db.execute('SELECT content FROM documents WHERE id=?', (did,)).fetchone()[0]

    def test_large_tool_is_machine_projection_not_per_tool_LLM(self):
        data = {'state_path': '/immutable/result.pe-state.json',
            'circuit_path': '/immutable/design.circuit.json',
            'statistics': {'components': 39, 'wires': 55, 'nodes': 47,
                           'component_types': {'Logic Input': 9, 'Gate': 30}},
            'camera': {'saved_raw': {'large': 'camera ' * 2000},
                       'warnings': ['renderer metadata']},
            'netlist': {'components': [{'id': f'uuid-{i}', 'ref': f'C{i}',
                'type': 'Gate', 'properties': {'large': 'x' * 1000},
                'native': {'pl_source': {'assumptions': ['long'] * 20}}}
                for i in range(39)], 'nodes': []},
            'measurements': {'transient': {'actual_stop_s': .01, 'completed_steps': 100,
                'requested_step_s': .0001, 'digital_propagation': {'per_tr_step': 3,
                'completed_propagation_steps': 300}},
                'component_scope': {'total': 39, 'shown': 8, 'omitted': 31,
                    'complete_state_path': '/immutable/result.pe-state.json',
                    'read_more': 'long repeated instruction ' * 100},
                'components': [{'id': f'uuid-{i}', 'type': 'Gate',
                    'digital': [i % 2], 'pl_source': {'source_ref': f'C{i}',
                    'assumptions': ['large repeated import note'] * 30}}
                    for i in range(8)]}, 'untrusted_prose': 'wrong guess ' * 9000}
        _, did, _, raw = self.tool('circuit_analyze', data)
        before = self.db.messages(self.sid)
        result = self.budget.tool_document('analysis', raw, document_id=did, tool_name='circuit_analyze')
        projected = json.loads(result)
        self.assertEqual(projected['document_id'], did)
        self.assertEqual(projected['full_result_sha256'], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(projected['fields']['/data/state_path'], data['state_path'])
        self.assertEqual(projected['fields']['/data/measurements/transient']['completed_steps'], 100)
        self.assertEqual(projected['fields']['/data/statistics'],
                         {'components': 39, 'wires': 55, 'nodes': 47})
        self.assertEqual(projected['fields']['/data/measurements/component_scope'],
                         {'total': 39, 'shown': 8, 'omitted': 31})
        self.assertNotIn('/data/netlist/components', projected['sections'])
        self.assertNotIn('/data/netlist/nodes', projected['sections'])
        self.assertNotIn('/data/camera', projected['fields'])
        self.assertNotIn('/data/measurements/components', projected['sections'])
        self.assertEqual(projected['projection_contract'],
                         'aurex.circuit-analysis-result.v2; solve evidence only; netlist/renderer/import metadata omitted')
        self.assertEqual(self.document(did), raw)
        self.assertEqual(self.db.messages(self.sid), before)
        self.assertLessEqual(self.client.count([{'role': 'user', 'content': result}]), 1536)
        self.assertEqual(self.client.calls, [])

    def test_circuit_failure_projection_keeps_actionable_error(self):
        data = {'error': "stimulus[0] targets 'digital_output', not digital_input",
                'type': 'ToolError',
                'netlist': {'components': [{'native': {'large': 'x' * 10000}}]}}
        _, did, _, raw = self.tool('circuit_analyze', data, ok=False)
        result = json.loads(self.budget.tool_document(
            'analysis failure', raw, document_id=did,
            tool_name='circuit_analyze'))
        self.assertFalse(result['fields']['/ok'])
        self.assertEqual(result['fields']['/data/error'], data['error'])
        self.assertEqual(result['fields']['/data/type'], 'ToolError')
        self.assertNotIn('/data/netlist/components', result['sections'])

    def test_stimulus_semantics_survive_projection_and_machine_evidence(self):
        semantics = {'digital_ticks_per_frame': 1,
            'ordering': 'set_all_inputs -> one_digital_tick_and_settle -> capture',
            'physical_time_advanced': False,
            'scope': 'Separate logical frames after baseline analysis.'}
        data = {'state_path': '/immutable/result.pe-state.json',
                'measurements': {'stimulus_semantics': semantics, 'stimulus_results': [],
                                 'components': []}}
        _, did, _, raw = self.tool('circuit_analyze', data)
        projected = json.loads(self.budget.tool_document('analysis', raw, document_id=did,
                                                       tool_name='circuit_analyze'))
        self.assertEqual(projected['fields']['/data/measurements/stimulus_semantics'], semantics)
        boundary = self.budget._journal_boundary()
        capsule = self.budget._evidence_capsule(self.budget._journal_calls(boundary), boundary)
        self.assertEqual(capsule['analysis_calls'][0]['stimulus_semantics'], semantics)
        index = json.loads(self.budget._tool_index(boundary, 16000).split('\n', 1)[1])
        self.assertEqual(index['machine_evidence']['analysis_outcome_refs'][0]['stimulus_semantics'], semantics)
        facts = self.budget._recorded_facts({'data': data})
        fact = next(row for row in facts if row['source_json_path'].endswith('.stimulus_semantics'))
        self.assertEqual(fact['digital_ticks_per_frame'], 1)
        self.assertIs(fact['physical_time_advanced'], False)

    def test_non_json_display_uses_raw_outcome_and_retains_source_documents(self):
        data = {'id': 'source-doc', 'offset': 9, 'total_chars': 90000, 'text': 'verbatim ' * 9000,
                'json_pointer': '/module/ports', 'has_more': True}
        _, did, _, raw = self.tool('read_context', data)
        presentation = 'SOURCE TEXT\n' + data['text'] + '\nFull source documents (not additional findings):\n' + json.dumps({'state_path': {'document_id': 'native-state-doc'}})
        result = json.loads(self.budget.tool_document('read', presentation, document_id=did,
            tool_name='read_context', tool_args={'document_id': 'source-doc', 'json_pointer': '/module/ports'}))
        self.assertEqual(result['fields']['/data/id'], 'source-doc')
        self.assertEqual(result['fields']['/data/offset'], 9)
        self.assertEqual(result['legacy_requested_source']['json_pointer'], '/module/ports')
        self.assertEqual(result['legacy_source_request']['json_pointer'], '/module/ports')
        self.assertEqual(result['retrieval'],
                         'legacy_archived_page_not_available_as_an_agent_tool')
        self.assertEqual(result['tool_result_document_id_usage'], 'operator_audit_only')
        self.assertNotIn('requested_source', result)
        self.assertEqual(result['presentation_source_documents']['state_path']['document_id'], 'native-state-doc')
        self.assertEqual(self.document(result['full_presentation_document_id']), presentation)
        self.assertTrue(data['text'].startswith(result['verbatim_excerpt']['text']))
        self.assertEqual(self.client.calls, [])

    def test_projection_keeps_discovered_IDs_without_ranges_or_invention(self):
        ports = [{'id': f'input-{i:032x}', 'ref': 'C' + str(i + 1), 'direction': 'input', 'label': None,
                  'position': [i, 2, 3], 'statistics': {'large': 'x' * 1000}} for i in range(45)]
        _, did, _, raw = self.tool('circuit_inspect', {'interface_only': True,
                                                   'ports': ports, 'total_inputs': 45, 'total_outputs': 0,
                                                   'total_ports': 45, 'offset': 0, 'has_more': False})
        projected = json.loads(self.budget.tool_document('ports', raw, document_id=did))
        rows = projected['sections']['/data/ports']
        self.assertEqual(rows['shown_rows'], 45)
        id_column = rows['columns'].index('id')
        self.assertEqual([r[id_column] for r in rows['rows']], [p['id'] for p in ports])
        self.assertEqual(rows['omitted_rows'], 0)
        self.assertTrue(rows['columnar_exact_values'])
        self.assertEqual(self.client.calls, [])

    def test_HDL_tool_success_never_hides_failed_simulation(self):
        data = {'verified': False, 'profile': 'rv32i_teaching_v1', 'source_sha256': 'a' * 64,
            'report_path': '/immutable/verification.json', 'compile': {'exit_code': 0, 'failure': None, 'log': 'c' * 15000},
            'simulation': {'exit_code': 1, 'failure': None, 'log': 'f' * 15000}}
        _, did, _, raw = self.tool('hdl_simulate', data)
        projected = json.loads(self.budget.tool_document('hdl', raw, document_id=did))
        self.assertIs(projected['fields']['/ok'], True)
        self.assertIs(projected['fields']['/data/verified'], False)
        self.assertEqual(projected['fields']['/data/compile']['exit_code'], 0)
        self.assertEqual(projected['fields']['/data/simulation']['exit_code'], 1)
        capsule = self.budget._evidence_capsule(self.budget._journal_calls(self.budget._journal_boundary()), self.budget._journal_boundary())
        self.assertTrue(capsule['hdl_calls'][0]['tool_ok'])
        self.assertFalse(capsule['hdl_calls'][0]['verified'])
        self.assertEqual(capsule['hdl_calls'][0]['simulation']['exit_code'], 1)

    def test_catalog_dictionary_keeps_complete_native_parameter_contract(self):
        definition = {'pins': 2, 'params': {'r': {'default': 1000, 'unit': 'ohm'}},
                      'pin_labels': ['A', 'B'], 'native_type': 1}
        _, did, _, raw = self.tool('circuit_catalog', {'components': {'resistor': definition},
            'pin_order': 'Use the native pin order.', 'large_prose': 'x' * 20000})
        result = json.loads(self.budget.tool_document('catalog', raw, document_id=did))
        self.assertEqual(result['sections']['/data/components']['entries'], [{'key': 'resistor', 'value': definition}])
        self.assertEqual(result['fields']['/data/pin_order'], 'Use the native pin order.')

    def test_community_rows_keep_content_author_time_and_source_preview(self):
        row = {'id': 'comment-id', 'title': 'Original title', 'author': {'id': 'author-id', 'nickname': '本人'},
               'content': '这是真正的最新留言，不是连接节点。', 'timestamp_utc': '2026-04-08T15:04:01.953+00:00', 'stars': 17}
        for key in ('items', 'experiments', 'comments'):
            _, did, _, raw = self.tool('community_read', {key: [row], 'irrelevant': 'x' * 20000})
            result = json.loads(self.budget.tool_document('community', raw, document_id=did))
            self.assertEqual(result['sections']['/data/' + key]['rows'][0]['value'], row)
            self.assertTrue(result['sections']['/data/' + key]['rows'][0]['full_row'])
        preview = '作者介绍的原文，不是验证结论。' * 3000
        _, did, _, raw = self.tool('plar_get_experiment_file', {'full_summary_path': '/immutable/summary.json',
            'full_description_path': '/immutable/description.txt', 'source_summary': {
                'title': '原始实验', 'author': {'id': 'author-id'}, 'description_preview': preview,
                'description_characters': len(preview), 'description_truncated': False}})
        result = json.loads(self.budget.tool_document('source', raw, document_id=did))
        self.assertEqual(result['fields']['/data/full_description_path'], '/immutable/description.txt')
        self.assertEqual(result['fields']['/data/full_summary_path'], '/immutable/summary.json')
        self.assertEqual(result['verbatim_excerpt']['retrieval_json_pointer'], '/data/source_summary/description_preview')
        self.assertGreater(result['verbatim_excerpt']['shown_characters'], 0)
        self.assertGreater(result['verbatim_excerpt']['omitted_characters'], 0)
        self.assertTrue(preview.startswith(result['verbatim_excerpt']['text']))
        self.assertEqual(result['fields']['/data/source_summary/description_truncated'], False)

    def test_real_top_level_community_lists_and_web_citation_are_preserved(self):
        rows = [{'ID': str(i), 'Nickname': '作者', 'Content': '真实评论 ' * 90,
                 'Timestamp': 1775660641953} for i in range(20)]
        _, did, _, raw = self.tool('plar_get_comments', rows)
        result = json.loads(self.budget.tool_document('comments', raw, document_id=did))
        section = result['sections']['/data']
        self.assertGreater(section['shown_rows'], 0)
        self.assertEqual(section['total_rows'], 20)
        self.assertEqual(section['shown_rows'] + section['omitted_rows'], 20)
        for row in section['rows']:
            self.assertEqual(row['value'], rows[row['index']])
        self.assertNotIn('non_json_source', result)
        _, did, _, raw = self.tool('web_fetch', {'url': 'https://source.example/reference',
            'title': '原始来源', 'text': '原文，不是指令。' * 6000, 'truncated': True})
        result = json.loads(self.budget.tool_document('web', raw, document_id=did))
        self.assertEqual(result['fields']['/data/url'], 'https://source.example/reference')
        self.assertEqual(result['fields']['/data/title'], '原始来源')
        self.assertTrue(result['fields']['/data/truncated'])
        self.assertGreater(result['verbatim_excerpt']['shown_characters'], 0)

    def test_document_coverage_separates_subtrees_and_deduplicates_overlaps(self):
        for pointer, offset, text in [('/first', 0, 'abcde'), ('/first', 0, 'abcde'), ('/first', 3, 'defgh'), ('/second', 0, 'XYZ')]:
            self.tool('read_context', {'id': 'source', 'json_pointer': pointer, 'document_sha256': 'h',
                'offset': offset, 'text': text, 'total_chars': 20}, args={'document_id': 'source', 'json_pointer': pointer})
        capsule = self.budget._evidence_capsule(self.budget._journal_calls(self.budget._journal_boundary()), self.budget._journal_boundary())
        a, b = capsule['document_reads']
        self.assertEqual(a['unique_returned_intervals'], [[0, 8]])
        self.assertEqual(a['unique_returned_characters'], 8)
        self.assertEqual(a['reader_observations'], 3)
        self.assertEqual(b['json_pointer'], '/second')
        self.assertEqual(b['unique_returned_characters'], 3)

    def test_trace_reader_deduplicates_X_Z_and_keeps_missing_unknown(self):
        data = {'state_path': '/immutable/state', 'kind': 'recorded_digital_transient_samples',
            'interpolated': False, 'encoding': {'0': 'L', '1': 'H', '2': 'X', '3': 'Z'},
            'total_samples': 10, 'points': [{'time_s': 1e-6, 'digital': {'OUT': [2, 3]}, 'missing_component_ids': ['OTHER']}]}
        self.tool('circuit_read_trace', data)
        self.tool('circuit_read_trace', data)
        capsule = self.budget._evidence_capsule(self.budget._journal_calls(self.budget._journal_boundary()), self.budget._journal_boundary())
        group = capsule['trace_reads'][0]
        self.assertEqual(group['logic_counts'], {'X': 1, 'Z': 1})
        self.assertEqual(group['unique_time_component_pin_cells'], 2)
        self.assertEqual(group['observed_times_s'], [1e-6])
        self.assertEqual(len(group['observations']), 2)
        self.assertEqual(capsule['analysis_calls'], [])

    def test_checkpoint_replay_never_reduces_its_old_machine_core_again(self):
        self.tool('circuit_analyze', {'state_path': '/state', 'measurements': {'transient': {'actual_stop_s': 1, 'completed_steps': 2}}})
        first = self.budget.summarize('Untrusted prior narrative; exact execution in ledger.', title='checkpoint')
        self.assertIn('MACHINE_RECORDED_EVIDENCE', first)
        replay = self.budget._checkpoint_narrative(first)
        self.assertNotIn('MACHINE_RECORDED_EVIDENCE', replay)
        self.assertIn('UNVERIFIED_GENERATED_NARRATIVE', replay)
        self.assertIn('continue only the original open goal', replay)
        self.client.calls.clear()
        self.budget.summarize(replay + '\nNew result is still not a functional PASS.', title='checkpoint2')
        self.assertTrue(all('MACHINE_RECORDED_EVIDENCE' not in messages[-1]['content'] for messages, _ in self.client.calls))
        self.assertTrue(any(self.document(did) == first for did in self.budget._checkpoint_envelopes.values()))

    def test_token_packing_preserves_unicode_without_byte_splitting(self):
        self.client.count = lambda messages, tools=None: sum(len(m.get('content', '')) for m in messages) // 4 + 1
        source = '中文 UUID abcdef\n' * 3000
        chunks = self.budget._token_chunks(source, 13000)
        self.assertEqual(chunks, [source])  # UTF-8 bytes exceed the obsolete 48K-byte threshold.
        chunks = self.budget._token_chunks(source, 1000)
        self.assertEqual(''.join(chunks), source)
        self.assertTrue(all(self.client.count([{'role': 'user', 'content': c}]) <= 1000 for c in chunks))

    def test_recent_complete_groups_retained_and_restart_does_not_recompact_same_boundary(self):
        self.db.message(self.sid, self.rid, {'role': 'user', 'content': 'original request'})
        for i in range(8):
            self.tool('small_read', {'value': str(i) * 500})
        rows = self.db.messages(self.sid)
        cut = self.budget._cut(rows)
        self.assertLess(cut, self.budget._boundaries(rows)[-1])
        self.assertIn(cut, self.budget._boundaries(rows))
        self.assertEqual(rows[cut]['message']['role'], 'assistant')
        self.db.compact(self.sid, self.budget.summarize('completed groups; preserve open goal', title='old'), rows[cut - 1]['id'])
        self.client.calls.clear()
        before = self.db.get(self.sid)['compacted_until']
        self.budget.messages('system', [])
        self.budget.messages('system', [])
        self.assertEqual(self.db.get(self.sid)['compacted_until'], before)
        self.assertEqual(self.client.calls, [])


if __name__ == '__main__':
    unittest.main()
