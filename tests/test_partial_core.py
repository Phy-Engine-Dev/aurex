"""Offline shape regressions; never call a model, a circuit tool or production DB."""
import copy
import json
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
        self.requests = []

    def count(self, messages, tools=None):
        # Deliberately conservative deterministic allocation, not a claim about
        # the production tokenizer. Its real count is audited separately.
        return len(json.dumps([messages, tools or []], ensure_ascii=False).encode()) // 3 + 1

    def chat(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), kwargs))
        return SimpleNamespace(content='Unverified semantic handoff.', finish_reason='stop', tool_calls=[])


class PartialCoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = SessionDB(str(Path(temp.name) / 'test.sqlite3'))
        self.sid = self.db.session('isolated', source='admin')
        self.rid = self.db.enqueue_task(self.sid, '介绍原作品并抽样；不外发。', source='admin')
        self.client = Client()
        self.budget = ContextBudget(self.client, self.db, self.sid, self.rid, 65536, lambda *args: None,
            policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=4096))
        self.serial = 0

    def tool(self, name, data, *, ok=True, args=None):
        cid = f'call_{self.serial:024x}'
        self.serial += 1
        call = {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args or {})}}
        self.db.message(self.sid, self.rid, {'role': 'assistant', 'content': '', 'tool_calls': [call]})
        doc, mid = self.db.tool_outcome(self.sid, self.rid, cid, name,
            json.dumps({'ok': ok, 'data': data}, ensure_ascii=False), ok)
        return cid, doc, mid

    def source(self, title='原实验标题', original=(631, 1238)):
        self.tool('plar_get_experiment_file', {'summary_id': 'd' * 24, 'elements': original[0], 'wires': original[1],
            'sav_path': '/archive/experiment-' + 'a' * 180 + '.sav', 'sha256': 'f' * 64,
            'source_summary': {'title': title, 'author': {'id': 'e' * 24, 'nickname': '真实作者'},
                'description_preview': '原文声称目标功能，尚需验证。' * 600,
                'description_characters': len('原文声称目标功能，尚需验证。' * 600),
                'description_truncated': False}})

    def analysis(self, index, *, ok=True, stimulus=False, counts=(768, 1375), types=None):
        path = '/archive/circuits/task/' + f'revision-{index:032x}' + '/snapshot.pe-state.json'
        data = {'state_path': path, 'statistics': {'components': counts[0], 'wires': counts[1], 'nodes': 1087},
            'measurements': {'transient': {'actual_stop_s': 1e-5, 'requested_stop_s': 1e-5,
                'requested_step_s': 1e-6, 'completed_steps': 10, 'sample_count': 10}}}
        if stimulus:
            data['measurements']['stimulus_scope'] = {'total_steps': 10, 'shown_steps': 0, 'omitted_steps': 10}
            data['stimulus_input_format'] = {'format': 'stimulus_table', 'input_count': 8, 'frame_count': 10,
                'native_frame_step_s': 1e-8, 'recorded_frame_count': 10}
        if types:
            data['statistics']['component_types'] = types
        if not ok:
            data = {'error': 'Validation refused; no solver execution.'}
        result = self.tool('circuit_analyze', data, ok=ok,
            args={'path': '/archive/original.sav', 'analysis': 'tr', 'tr_step': 1e-6, 'tr_stop': 1e-5, 'case': index})
        return result, path

    def reader(self, path, *, n=1):
        for page in range(n):
            self.tool('circuit_read_stimulus', {'state_path': path, 'recorded_not_resimulated': True,
                'encoding': {'0': 'L', '1': 'H', '2': 'X', '3': 'Z'}, 'total_steps': 10,
                'steps': [{'step': 0, 'digital': {'output-exact': [2, 3]}}]}, args={'path': path, 'offset': page})

    def preview(self, limit=2048):
        text = self.budget._tool_index(self.budget._journal_boundary(), limit)
        payload = json.loads(text.split('\n', 1)[1])
        with self.db.connect() as store:
            full = json.loads(store.execute('SELECT content FROM documents WHERE id=?', (payload['document_id'],)).fetchone()[0])
        self.assertLessEqual(self.client.count([{'role': 'user', 'content': text}]), limit)
        return payload['machine_evidence'], full, payload

    def test_second_cpu_checkpoint_shape_preserves_partial_core_not_global_fallback(self):
        # Shape/count categories from the second real CPU checkpoint. IDs and
        # paths are synthetic; these values are test data, never model hints.
        original_types = {'And Gate': 158, 'D Flipflop': 137, 'Multiplier': 92,
            'Or Gate': 74, 'No Gate': 65, 'Logic Input': 45, 'Xor Gate': 19,
            'Logic Output': 16, 'Full Adder': 11, 'Xnor Gate': 6, 'Yes Gate': 4,
            'Nor Gate': 2, 'Counter': 1, 'Half Adder': 1}
        imported_types = {'No Gate': 202, 'And Gate': 158, 'Phy-Engine digital_dff': 137,
            'Phy-Engine digital_mul2': 92, 'Or Gate': 74, 'Logic Input': 45,
            'Xor Gate': 19, 'Logic Output': 16, 'Phy-Engine digital_full_adder': 11,
            'Xnor Gate': 6, 'Yes Gate': 4, 'Nor Gate': 2, 'Phy-Engine digital_counter4': 1,
            'Phy-Engine digital_half_adder': 1}
        self.source()
        for page in range(4):
            self.tool('circuit_inspect', {'statistics': {'components': 631, 'wires': 1238,
                'nodes': 1087, 'component_types': original_types}}, args={'path': '/original.sav', 'offset': page})
        for page in range(7):
            self.tool('circuit_inspect', {'error': 'Input validation failed'}, ok=False, args={'offset': 99 + page})
        for page in range(5):
            self.tool('read_context', {'document_id': f'{page:032x}', 'offset': page})
        ports = [{'id': f'input-{i:032x}', 'direction': 'input'} for i in range(45)]
        self.tool('circuit_inspect', {'ports': ports, 'total_inputs': 45, 'total_outputs': 16,
            'total_ports': 61, 'offset': 0, 'has_more': False})
        for index in range(2):
            self.analysis(index, types=imported_types)
        _, state = self.analysis(2, stimulus=True, types=imported_types)
        self.analysis(3, ok=False)
        self.tool('circuit_read_stimulus', {'state_path': state, 'recorded_not_resimulated': True,
            'encoding': {'0': 'L', '1': 'H', '2': 'X', '3': 'Z'}, 'total_steps': 10,
            'steps': [{'step': step, 'digital': {f'{pin:032x}': [int(pin == 0 and step % 2 == 1)]
                for pin in range(8)}} for step in range(10)]})
        core, full, payload = self.preview()
        self.assertEqual(core['tool_totals']['circuit_analyze'], {'completed_outcomes': 4, 'ok': 3, 'error': 1})
        self.assertIsNone(core['server_task_binding']['requester_user_id'])
        self.assertTrue(core['source_facts'][0]['retrieved_complete'])
        self.assertTrue(core['source_facts'][0]['index_excerpt_truncated'])
        self.assertTrue(any((s.get('elements'), s.get('wires')) == (631, 1238) for s in core['structure_counts']))
        self.assertTrue(any(s.get('counts', {}).get('components') == 768 and s['counts']['wires'] == 1375 for s in core['structure_counts']))
        for group in core['structure_counts']:
            self.assertNotIn('record_index', group)  # Group ordinal is not an artifact record index.
            for index in group['structure_record_indices']:
                record = full['machine_evidence']['structure_records'][index]
                for field in ('classification', 'elements', 'wires', 'counts'):
                    if field in group:
                        self.assertEqual(record[field], group[field])
        self.assertTrue(core['interface_sets'])
        refs = core['analysis_outcome_refs']
        self.assertTrue(any(r['tool_ok'] is False for r in refs))
        self.assertTrue(any(r.get('transient', {}).get('actual_stop_s') == 1e-5 and
            r.get('stimulus_input_format', {}).get('native_frame_step_s') == 1e-8 for r in refs))
        self.assertEqual(core['recorded_state_read_refs'][0]['logic_counts'], {'L': 75, 'H': 5})
        self.assertEqual(full['machine_evidence']['interface_records'][0]['returned_ports'], ports)
        self.assertNotIn('input-000', json.dumps(core))
        self.assertEqual(core['families']['analysis_outcome_refs']['complete_document_id'], payload['document_id'])

    def test_twelve_and_one_hundred_twenty_eight_analyses_keep_total_and_paged_refs(self):
        for total in (12, 128):
            with self.subTest(total=total):
                # Separate task scope prevents accumulated counts across cases.
                self.rid = self.db.enqueue_task(self.sid, 'New bounded fixture', source='admin')
                self.budget = ContextBudget(self.client, self.db, self.sid, self.rid, 65536, lambda *args: None,
                    policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=4096))
                self.source(original=(17, 39))
                successful = 0
                for index in range(total):
                    ok = index % 7 != 0
                    successful += ok
                    _, path = self.analysis(index, ok=ok, stimulus=ok)
                    if ok:
                        self.reader(path)
                before = self.db.messages(self.sid)
                core, full, payload = self.preview()
                totals = core['tool_totals']['circuit_analyze']
                self.assertEqual((totals['completed_outcomes'], totals['ok'], totals['error']), (total, successful, total - successful))
                self.assertEqual(len(full['machine_evidence']['analysis_calls']), total)
                family = core['families']['analysis_outcome_refs']
                self.assertGreater(family['shown'], 0)
                self.assertGreater(family['omitted'], 0)
                self.assertEqual(family['shown'] + family['omitted'], total)
                self.assertEqual(family['complete_document_id'], payload['document_id'])
                self.assertTrue(any(r['tool_ok'] for r in core['analysis_outcome_refs']))
                self.assertTrue(any(not r['tool_ok'] for r in core['analysis_outcome_refs']))
                self.assertEqual(core['source_facts'][0]['summary_id'], 'd' * 24)
                for ref in core['analysis_outcome_refs']:
                    original = full['machine_evidence']['analysis_calls'][ref['record_index']]
                    self.assertEqual(ref['call_id'], original['call_id'])
                    self.assertEqual(ref['result_document_id'], original['result_document_id'])
                self.assertEqual(before, self.db.messages(self.sid))
                self.assertEqual(self.client.requests, [])

    def test_huge_optional_title_does_not_remove_source_completeness_or_counts(self):
        self.source(title='Untrusted title ' * 1000, original=(19, 43))
        self.analysis(0)
        core, full, _ = self.preview()
        source = core['source_facts'][0]
        self.assertEqual(source['detail_level'], 'basic')
        self.assertTrue(source['retrieved_complete'])
        self.assertNotIn('title', source)
        self.assertEqual(len(full['machine_evidence']['source_records'][0]['title']), len('Untrusted title ' * 1000))
        self.assertEqual(core['structure_counts'][0]['elements'], 19)

    def test_each_family_reports_exact_omissions_and_empty_is_not_unexecuted_history(self):
        self.source()
        for i in range(20):
            self.analysis(i, ok=i % 2 == 0)
        core, _, payload = self.preview()
        for name, meta in core['families'].items():
            self.assertEqual(meta['shown'], len(core[name]))
            self.assertEqual(meta['total'], meta['shown'] + meta['omitted'])
            self.assertEqual(meta['complete_document_id'], payload['document_id'])
            self.assertTrue(meta['location'].startswith('machine_evidence.'))
            self.assertLessEqual(meta['expanded'], meta['shown'])
        for name in ('source_facts', 'structure_counts', 'analysis_outcome_refs'):
            self.assertIn(name, core['families'])
        self.assertEqual(core['empty_families'],
                         ['recorded_state_read_refs', 'interface_sets'])
        for name in core['empty_families']:
            self.assertNotIn(name, core['families'])
            self.assertNotIn(name, core)
        self.assertEqual(core['tool_totals']['circuit_analyze']['completed_outcomes'], 20)
        self.budget.summarize('Retain the actual goal and open issues. ' * 2000, title='checkpoint')
        self.assertTrue(all(options['max_tokens'] >= 2048 for _, options in self.client.requests))


if __name__ == '__main__':
    unittest.main()
