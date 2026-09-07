"""Summary constraints come from the actual local journal, never a live model."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from aurex.config import ContextPolicyConfig, LLMConfig
from aurex.context_budget import ContextBudget, text_of
from aurex.sessiondb import SessionDB


class Client:
    def __init__(self):
        self.config = LLMConfig(context_length=32768, max_output_tokens=512)
        self.calls = []
        self.handler = None

    def count(self, messages, tools=None):
        return (len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools or []))) // 4 + 1

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        text = self.handler(messages) if self.handler else 'Only the reported local evidence is summarized.'
        return SimpleNamespace(content=text, finish_reason='stop', tool_calls=[])


def block(messages, prefix):
    found = [m['content'] for m in messages if isinstance(m.get('content'), str) and m['content'].startswith(prefix)]
    return json.loads(found[0].split('\n', 1)[1]) if found else None


class SummaryEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = SessionDB(str(Path(temporary.name) / 'isolated.sqlite3'))
        self.sid = self.db.session('session', source='admin')
        self.rid = self.db.enqueue_task(self.sid, '介绍原实验并做少量样例；不发布。', task_id='task', source='admin')
        self.client = Client()
        self.events = []
        self.budget = ContextBudget(self.client, self.db, self.sid, self.rid, 32768,
            lambda k, d: self.events.append((k, d)), policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=2048))

    def tool(self, cid, name, data, *, ok=True, rid=None, args=None):
        rid = rid or self.rid
        call = {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args or {})}}
        self.db.message(self.sid, rid, {'role': 'assistant', 'content': '', 'tool_calls': [call]})
        return self.db.tool_outcome(self.sid, rid, cid, name, json.dumps({'ok': ok, 'data': data}, ensure_ascii=False), ok)

    def source(self):
        return self.tool('download', 'plar_get_experiment_file', {
            'summary_id': 'a' * 24, 'sav_path': '/private/original.sav',
            'full_summary_path': '/private/summary.json', 'full_description_path': '/private/description.txt',
            'source_summary': {'title': '原实验乘法器', 'author': {'id': 'b' * 24, 'nickname': '原作者',
                'source_field': 'Summary.User'}, 'description_preview': '作者声称其功能是四位乘法；这是原文，不是测试结论。',
                'description_characters': 42, 'description_truncated': False, 'untrusted_reference': True}})

    def recorded(self):
        return self.tool('stimulus-page', 'circuit_read_stimulus', {
            'state_path': '/private/actual.pe-state.json', 'recorded_not_resimulated': True,
            'steps': [{'step': 4, 'digital': {'Q': [1, 0, 1]}}, {'step': 5, 'digital': {'Q': [0, 1, 0]}}],
            'offset': 4, 'total_steps': 8, 'has_more': True,
        }, args={'path': '/private/actual.pe-state.json', 'offset': 4, 'limit': 2})

    def test_text_of_never_adds_empty_calls_to_a_real_stimulus_result(self):
        message = {'role': 'tool', 'tool_call_id': 'stimulus-page', 'content': '{"steps": [{"step": 4}]}'}
        before = copy.deepcopy(message)
        for candidate in (message, {**message, 'tool_calls': []}):
            text = text_of(candidate)
            self.assertNotIn('[]', text)
            self.assertNotIn('ACTUAL_TOOL_CALLS', text)
            self.assertNotIn('read_context', text)
            self.assertIn('stimulus-page', text)
            self.assertIn('"step": 4', text)
        self.assertEqual(message, before)
        call = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'stimulus-page', 'type': 'function',
                'function': {'name': 'circuit_read_stimulus', 'arguments': '{"offset":4}'}}]}
        serialized = text_of(call)
        self.assertIn('ACTUAL_TOOL_CALLS:', serialized)
        self.assertIn('circuit_read_stimulus', serialized)
        self.assertEqual(text_of({'role': 'user', 'content': 'hello'}), 'user: hello')

    def test_source_summary_and_recorded_counts_stay_durable_without_chunk_ledger_injection(self):
        source_did, _ = self.source()
        step_did, until = self.recorded()
        self.budget._summary_chunk('This late slice only mentions an unrelated failed URL.', 2048)
        self.assertIsNone(block(self.client.calls[0][0], 'DETERMINISTIC TOOL JOURNAL'))
        record = json.loads(self.budget._tool_index(until, 4096).split('\n', 1)[1])
        self.assertEqual(record['run_id'], self.rid)
        self.assertEqual(record['session_id'], self.sid)
        self.assertEqual(record['snapshot_until_message_id'], until)
        entries = {e['name']: e for e in record['entries']}
        source = entries['plar_get_experiment_file']
        stimulus = entries['circuit_read_stimulus']
        self.assertEqual(source['document_id'], source_did)
        self.assertEqual(stimulus['document_id'], step_did)
        self.assertEqual(stimulus['last_call_id'], 'stimulus-page')
        self.assertEqual(stimulus['result_message_id'], until)
        facts = {f['source_json_path']: f for f in source['reported_facts']}
        self.assertEqual(facts['$.data.source_summary']['title'], '原实验乘法器')
        self.assertIs(facts['$.data.source_summary']['description_preview_present'], True)
        self.assertEqual(facts['$.data.source_summary.author']['id'], 'b' * 24)
        self.assertEqual(facts['$.data.source_summary.author']['nickname'], '原作者')
        actual = next(f for f in stimulus['reported_facts'] if 'total_steps' in f)
        self.assertEqual(actual['total_steps'], 8)
        self.assertEqual(actual['returned_steps_count'], 2)
        self.assertEqual((actual['returned_first_step'], actual['returned_last_step']), (4, 5))
        self.assertIs(actual['recorded_not_resimulated'], True)
        self.assertNotIn('digital', json.dumps(stimulus['reported_facts']))
        self.assertNotIn('verified', actual)
        self.assertIn('NOT a verified experiment pass', record['note'])
        with self.db.connect() as store:
            archived = json.loads(store.execute('SELECT content FROM documents WHERE id=?', (record['document_id'],)).fetchone()[0])
        self.assertEqual(archived['run_id'], self.rid)
        self.assertEqual({e['document_id'] for e in archived['entries']}, {source_did, step_did})

    def test_chunk_reductions_keep_scope_without_reinjecting_the_whole_journal(self):
        source_did, _ = self.source()
        step_did, until = self.recorded()
        seen = []
        def summary(messages):
            scope = block(messages, 'SOURCE_SLICE_REFERENCE')
            ledger = block(messages, 'DETERMINISTIC TOOL JOURNAL')
            seen.append((scope, ledger, messages[-1]['content']))
            self.assertIs(scope['local_slice_only'], True)
            self.assertIs(scope['absence_from_slice_is_not_nonexecution'], True)
            self.assertIn('Absence from this slice is not evidence of absence', messages[0]['content'])
            self.assertIn('a later slice or generated summary must not overwrite them', messages[0]['content'])
            self.assertIsNone(ledger)
            # Deliberately force a reduction. Its source is generated text, but
            # the reducer must still see the same original execution records.
            return 'Observed source and recorded page only. ' + ('s' * 3500 if len(seen) <= 4 else '')
        self.client.handler = summary
        # Actual token packing no longer splits merely because UTF-8 bytes
        # cross 32K; keep this fixture genuinely multi-chunk/reducing.
        text = 'EARLY: original introduction was fetched; two stored stimulus rows were read.\n' + 'later local material\n' * 12000
        self.budget.summarize(text, title='Conversation checkpoint')
        self.assertGreater(len(seen), 4)
        self.assertTrue(any(s['source_reference']['reduction_depth'] > 0 for s, _, _ in seen))
        late = [(s, t) for s, _, t in seen if s['source_reference']['chunk_index'] > 1]
        self.assertTrue(late)
        self.assertTrue(any('EARLY:' not in t for _, t in late))
        originals = {s['source_reference']['original_document_id'] for s, _, _ in seen}
        self.assertEqual(len(originals), 1)
        with self.db.connect() as store:
            self.assertEqual(store.execute('SELECT content FROM documents WHERE id=?', (originals.pop(),)).fetchone()[0], text)
        durable = json.loads(self.budget._tool_index(until, 4096).split('\n', 1)[1])
        self.assertEqual({e['document_id'] for e in durable['entries']}, {source_did, step_did})

    def test_lossy_final_summary_cannot_overwrite_separate_deterministic_memory(self):
        self.source()
        _, until = self.recorded()
        # A prompt is not a mathematical guarantee of model compliance. Even
        # a bad model summary must not delete or replace the genuine journal.
        self.db.compact(self.sid, 'No introduction was fetched and no stimulus was read.', until)
        before = self.db.messages(self.sid)
        messages = self.budget.messages('system', [])
        ledger = block(messages, 'DETERMINISTIC TOOL JOURNAL')
        self.assertEqual({e['name'] for e in ledger['entries']}, {'plar_get_experiment_file', 'circuit_read_stimulus'})
        self.assertIn('原实验乘法器', json.dumps(ledger, ensure_ascii=False))
        self.assertIn('returned_steps_count', json.dumps(ledger))
        self.assertEqual(before, self.db.messages(self.sid))
        self.assertEqual(self.client.calls, [])

    def test_other_run_and_pending_calls_do_not_enter_chunk_or_task_journal(self):
        other = self.db.enqueue_task(self.sid, 'Other task', task_id='other-task', source='admin')
        self.tool('shared', 'circuit_read_stimulus', {'total_steps': 999999}, rid=other)
        did, _ = self.recorded()
        self.db.message(self.sid, self.rid, {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'pending', 'type': 'function', 'function': {'name': 'circuit_analyze', 'arguments': '{}'}}]})
        self.budget._summary_chunk('Local slice', 2048)
        self.assertIsNone(block(self.client.calls[0][0], 'DETERMINISTIC TOOL JOURNAL'))
        ledger = json.loads(self.budget._tool_index(self.budget._journal_boundary(), 2048).split('\n', 1)[1])
        self.assertEqual(len(ledger['entries']), 1)
        self.assertEqual(ledger['entries'][0]['document_id'], did)
        self.assertNotIn('999999', json.dumps(ledger))
        self.assertNotIn('pending', json.dumps(ledger))

    def test_sample_metadata_is_bounded_and_does_not_copy_waveforms_or_fabricate_missing_zero(self):
        rows = [{'time_s': i / 10, 'components': [{'id': 'HUGE_WAVEFORM', 'pin_voltage': [i]}]} for i in range(201)]
        facts = ContextBudget._recorded_facts({'data': {'measurement_source': 'fresh solve', 'measurements': {
            'transient': {'actual_stop_s': .5, 'requested_stop_s': 1, 'requested_step_s': .1,
                'completed_steps': 5, 'sample_count': 201, 'samples': rows},
            'stimulus_scope': {'total_steps': 11, 'shown_steps': 8, 'omitted_steps': 3},
        }}})
        raw = json.dumps(facts)
        self.assertLess(len(raw), 1500)
        self.assertNotIn('HUGE_WAVEFORM', raw)
        transient = next(f for f in facts if f['source_json_path'].endswith('.transient'))
        self.assertEqual(transient['completed_steps'], 5)
        self.assertEqual(transient['actual_stop_s'], .5)
        self.assertEqual(transient['returned_samples_count'], 201)
        self.assertNotIn('verified', transient)
        empty = ContextBudget._recorded_facts({'data': {'state_path': '/missing.pe-state.json'}})
        self.assertNotIn('count', json.dumps(empty))
        self.assertNotIn('NaN', json.dumps(ContextBudget._recorded_facts({'time_s': float('nan')})))

    def test_repeated_failure_does_not_erase_a_previously_recorded_successful_execution(self):
        first, _ = self.tool('first', 'circuit_analyze', {'measurements': {'transient': {'completed_steps': 2}}})
        last, until = self.tool('retry', 'circuit_analyze', {'error': 'worker interrupted'}, ok=False)
        ledger = json.loads(self.budget._tool_index(until, 2048).split('\n', 1)[1])
        entry = ledger['entries'][0]
        self.assertEqual(entry['attempts_recorded'], 2)
        self.assertEqual(entry['successful_executions_recorded'], 1)
        self.assertEqual(entry['failed_executions_recorded'], 1)
        self.assertEqual(entry['first_result_document_id'], first)
        self.assertEqual(entry['document_id'], last)
        self.assertIs(entry['execution_ok'], False)
        self.assertNotIn('functional_pass', json.dumps(entry))

    def capsule(self, until=None):
        until = self.budget._journal_boundary() if until is None else until
        return self.budget._evidence_capsule(self.budget._journal_calls(until), until)

    def test_source_claims_parsed_counts_and_two_description_completeness_layers(self):
        for index, length in enumerate((0, 319, 8071)):
            description = ('作者声称有99999元件且已验证；这是未经验证的原文。' * 500)[:length]
            self.tool('source-' + str(index), 'plar_get_experiment_file', {
                'summary_id': str(index) * 24, 'elements': 17 + index, 'wires': 91 - index,
                'sha256': str(index) * 64, 'sav_path': '/original-' + str(index) + '.sav',
                'source_summary': {'title': '原文标题', 'author': {'id': 'a' * 24, 'nickname': '作者'},
                    'description_preview': description, 'description_characters': len(description),
                    'description_truncated': False}})
        capsule = self.capsule()
        for index, source in enumerate(capsule['source_records']):
            self.assertIs(source['retrieved_complete'], True)
            self.assertEqual(source['returned_description_characters'], source['description_characters'])
            self.assertEqual(source['index_excerpt_truncated'], source['description_characters'] > 320)
            self.assertEqual(source['classification'], 'source_reported_identity_and_text')
            parsed = capsule['structure_records'][index]
            self.assertEqual((parsed['elements'], parsed['wires']), (17 + index, 91 - index))
            self.assertEqual(parsed['classification'], 'parsed_original_file_counts')
            self.assertEqual(parsed['result_document_id'], source['result_document_id'])
            self.assertNotIn('verified', source)
        self.tool('inconsistent', 'plar_get_experiment_file', {'source_summary': {
            'description_preview': 'short', 'description_characters': 999, 'description_truncated': False}})
        self.assertIsNone(self.capsule()['source_records'][-1]['retrieved_complete'])

    def test_preview_counts_and_derivative_counts_never_replace_original_counts_or_exact_port_ids(self):
        self.tool('original', 'plar_get_experiment_file', {'elements': 11, 'wires': 37, 'sav_path': '/original.sav'})
        exact = ['id-A', 'id-Z', 'id-003', 'uuid-not-a-range']
        self.tool('ports', 'circuit_inspect', {'state_path': '/derived.pe-state.json',
            'statistics': {'components': 23, 'wires': 51, 'nodes': 9, 'component_types': {'gate': 21, 'resistor': 2}},
            'netlist': {'components': [{'id': 'only-preview-item'}]},
            'ports': [{'id': cid, 'direction': 'input'} for cid in exact],
            'total_inputs': 4, 'total_outputs': 0, 'offset': 0, 'has_more': False}, args={'path': '/derived.pe-state.json'})
        cap = self.capsule()
        self.assertEqual(cap['structure_records'][0]['elements'], 11)
        self.assertEqual(cap['structure_records'][1]['counts']['components'], 23)
        self.assertEqual(cap['structure_records'][1]['component_types'], {'gate': 21, 'resistor': 2})
        self.assertEqual([p['id'] for p in cap['interface_records'][0]['returned_ports']], exact)
        self.assertNotIn('only-preview-item', json.dumps(cap))

    def test_all_analysis_totals_and_artifact_bindings_survive_many_reader_pages(self):
        analyses = []
        for index in range(3):
            did, _ = self.tool('solve-' + str(index), 'circuit_analyze', {
                'state_path': f'/immutable/state-{index}.pe-state.json',
                'measurements': {'stimulus_scope': {'total_steps': index + 2}}},
                args={'path': '/original.sav', 'digital_clock_ticks': index + 7})
            analyses.append(did)
        self.tool('solve-error', 'circuit_analyze', {'error': 'validation refused'}, ok=False)
        for index in range(9):
            self.tool('reader-' + str(index), 'circuit_read_stimulus', {
                'state_path': '/immutable/state-2.pe-state.json', 'recorded_not_resimulated': True,
                'encoding': {'0': 'L', '1': 'H', '2': 'X', '3': 'Z'}, 'total_steps': 4,
                'steps': [{'step': 1, 'digital': {'exact-O': [2, 0]}}]}, args={'offset': index})
        until = self.budget._journal_boundary()
        cap = self.capsule()
        self.assertEqual(cap['tool_totals']['circuit_analyze'], {'completed_outcomes': 4, 'ok': 3, 'error': 1})
        self.assertEqual([a['result_document_id'] for a in cap['analysis_calls'][:3]], analyses)
        self.assertEqual([a['requested']['digital_clock_ticks'] for a in cap['analysis_calls'][:3]], [7, 8, 9])
        self.assertEqual([a['stimulus_scope']['total_steps'] for a in cap['analysis_calls'][:3]], [2, 3, 4])
        reader = cap['recorded_state_reads'][0]
        self.assertEqual(reader['producer_calls'], [{'call_id': 'solve-2', 'result_document_id': analyses[-1]}])
        self.assertEqual(len(reader['observations']), 9)
        self.assertEqual(reader['unique_step_component_pairs'], 1)
        self.assertEqual(reader['unique_pin_observations'], 2)
        self.assertEqual(reader['logic_counts'], {'X': 1, 'L': 1})
        self.assertEqual(reader['per_step_logic_counts'], {'1': {'X': 1, 'L': 1}})
        preview = json.loads(self.budget._tool_index(until, 1024).split('\n', 1)[1])
        self.assertEqual(preview['machine_evidence']['tool_totals'], cap['tool_totals'])
        with self.db.connect() as store:
            archived = json.loads(store.execute('SELECT content FROM documents WHERE id=?', (preview['document_id'],)).fetchone()[0])
        self.assertEqual(archived['machine_evidence'], cap)

    def test_reader_unknown_encoding_and_conflicting_observations_fail_closed(self):
        for suffix, encoding, value in [('a', {'0': 'L', '2': 'X'}, 2), ('b', {'0': 'H', '2': 'Z'}, 2)]:
            self.tool(suffix, 'circuit_read_stimulus', {'state_path': '/state', 'recorded_not_resimulated': True,
                'encoding': encoding, 'steps': [{'step': 0, 'digital': {'O': [value]}}]})
        self.tool('missing-encoding', 'circuit_read_stimulus', {'state_path': '/missing-encoding',
            'recorded_not_resimulated': True, 'steps': [{'step': 0, 'digital': {'O': [2]}}]})
        for group in self.capsule()['recorded_state_reads']:
            self.assertEqual(group['encoding_status'], 'unknown_or_conflicting')
            self.assertIsNone(group['logic_counts'])
            self.assertEqual(group['producer_binding'], 'unknown')
        self.tool('conflict', 'circuit_read_stimulus', {'state_path': '/state', 'recorded_not_resimulated': True,
            'encoding': {'0': 'L', '2': 'X'}, 'steps': [{'step': 0, 'digital': {'O': [0]}}]})
        self.assertEqual(self.capsule()['recorded_state_reads'][0]['conflicting_pin_observations'], 1)

    def test_final_handoff_has_one_non_reduced_machine_core_even_with_wrong_narrative(self):
        self.tool('real-solve', 'circuit_analyze', {'state_path': '/real', 'measurements': {'stimulus_scope': {'total_steps': 3}}})
        self.client.handler = lambda messages: 'No analysis was performed; every observed output is H.'
        result = self.budget.summarize('Original task history. ' * 2000, title='checkpoint')
        self.assertEqual(result.count('MACHINE_RECORDED_EVIDENCE ('), 1)
        self.assertEqual(result.count('UNVERIFIED_GENERATED_NARRATIVE ('), 1)
        self.assertIn('No analysis was performed', result)  # Explicitly untrusted, not accepted as machine truth.
        core = json.loads(result.split('MACHINE_RECORDED_EVIDENCE ', 1)[1].split('\n', 1)[1].split('\n\n', 1)[0])
        self.assertEqual(core['machine_evidence']['tool_totals']['circuit_analyze']['ok'], 1)
        self.assertLessEqual(self.client.count([{'role': 'user', 'content': result}]), self.budget.policy.summary_max_tokens)
        for messages, _ in self.client.calls:
            self.assertNotIn('MACHINE_RECORDED_EVIDENCE (program-built', messages[-1]['content'])
        before = self.db.messages(self.sid)
        self.assertEqual(len(self.capsule()['analysis_calls']), 1)
        self.assertEqual(self.db.messages(self.sid), before)

    def test_reduce_input_never_contains_machine_capsule_and_snapshot_remains_fixed(self):
        self.recorded()
        seen = []
        def reply(messages):
            scope = block(messages, 'SOURCE_SLICE_REFERENCE')
            seen.append(scope)
            self.assertNotIn('MACHINE_RECORDED_EVIDENCE (program-built', messages[-1]['content'])
            return ('local words ' * 120 if scope['source_reference']['reduction_depth'] == 0 else 'Short narrative.')
        self.client.handler = reply
        result = self.budget.summarize('raw source ' * 24000, title='checkpoint')
        self.assertTrue(any(s['source_reference']['reduction_depth'] > 0 for s in seen))
        self.assertEqual(len({s['journal_snapshot_until_message_id'] for s in seen}), 1)
        self.assertEqual(result.count('MACHINE_RECORDED_EVIDENCE (program-built'), 1)
        self.assertEqual(len({s['source_reference']['original_document_id'] for s in seen}), 1)

    def test_tight_excerpt_allocation_keeps_exact_totals_and_full_core_pointer(self):
        for i in range(20):
            self.tool('tool-' + str(i), 'tool_kind_' + str(i), {'arbitrary': 'not interpreted'})
        until = self.budget._journal_boundary()
        # This is an excerpt allocation, not permission to erase recorded work.
        # The irreducible scoped ledger can exceed it, but the physical window
        # is still enforced by its caller.
        text = self.budget._tool_index(until, 128)
        payload = json.loads(text.split('\n', 1)[1])
        self.assertEqual(payload['snapshot_until_message_id'], until)
        self.assertEqual(payload['run_id'], self.rid)
        self.assertEqual(len(payload['machine_evidence']['tool_totals']), 20)
        self.assertTrue(payload['machine_evidence']['details_omitted'])
        self.assertEqual(payload['document_id'], payload['machine_evidence']['complete_machine_evidence_document_id'])
        self.assertLess(self.client.count([{'role': 'user', 'content': text}]), self.budget.usable)
        full = [{'role': 'system', 'content': 'x' * (self.budget.usable * 4)}]
        self.assertEqual(self.budget._with_tool_index(full, [], until), full)

    def test_no_task_outcomes_never_claims_the_whole_session_has_no_history(self):
        other = self.db.enqueue_task(self.sid, 'Other', task_id='other', source='admin')
        self.tool('prior', 'circuit_analyze', {'measurements': {'stimulus_scope': {'total_steps': 99}}}, rid=other)
        cap = self.capsule()
        self.assertEqual(cap['tool_totals'], {})
        self.assertIn('not the whole session', cap['scope'])
        self.assertEqual(self.budget._tool_index(self.budget._journal_boundary(), 2048), '')

    def test_admin_identity_time_domains_and_narrative_minimum_survive_core(self):
        self.tool('timed', 'circuit_analyze', {'state_path': '/timed-state', 'measurements': {
            'transient': {'actual_stop_s': 2e-6, 'completed_steps': 200},
            'stimulus_scope': {'total_steps': 4}},
            'stimulus_input_format': {'native_frame_step_s': 1e-8, 'recorded_frame_count': 4}},
            args={'digital_clock_ticks': 17, 'stimulus_table': {'inputs': ['C1'], 'vectors': [[0], [1], [0], [1]]}})
        cap = self.capsule()
        self.assertIsNone(cap['server_task_binding']['requester_user_id'])
        self.assertEqual(cap['server_task_binding']['source'], 'admin')
        self.assertEqual(cap['analysis_calls'][0]['transient']['actual_stop_s'], 2e-6)
        self.assertEqual(cap['analysis_calls'][0]['stimulus_input_format']['native_frame_step_s'], 1e-8)
        self.assertNotIn('C1', json.dumps(cap))  # Input identity is not a clock designation.
        self.budget.summarize('Only the journal records these independent domains. ' * 1500, title='checkpoint')
        self.assertTrue(all(options['max_tokens'] >= self.budget.policy.summary_max_tokens // 2
                            for _, options in self.client.calls))


if __name__ == '__main__':
    unittest.main()
