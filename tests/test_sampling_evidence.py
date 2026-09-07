"""Offline sampling/provenance contracts: no artifacts, network or generation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aurex.config import ContextPolicyConfig
from aurex.context_budget import ContextBudget, SUMMARY_PROMPT
from aurex.sessiondb import SessionDB
from test_context_compaction_replay import Client


class SamplingEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = SessionDB(str(Path(temporary.name) / 'evidence.sqlite3'))
        self.sid = self.db.session(source='admin')
        self.rid = self.db.enqueue_task(self.sid, '做少量测试，不要假称全部验证', source='admin')
        self.client = Client()
        self.budget = ContextBudget(self.client, self.db, self.sid, self.rid, 65536,
            lambda *args: None, policy=ContextPolicyConfig(safety_tokens=128))
        self.serial = 0

    def tool(self, name, data, args=None, ok=True):
        cid = 'sampling-' + str(self.serial)
        self.serial += 1
        self.db.message(self.sid, self.rid, {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args or {})}}]})
        raw = json.dumps({'ok': ok, 'data': data}, ensure_ascii=False)
        did, mid = self.db.tool_outcome(self.sid, self.rid, cid, name, raw, ok)
        return did, mid, raw

    def capsule(self):
        until = self.budget._journal_boundary()
        with patch.object(Path, 'open', side_effect=AssertionError('Do not open state artifacts')):
            return self.budget._evidence_capsule(self.budget._journal_calls(until), until)

    def recorded_data(self):
        # Actual prior failure shape: all-state samples exist, short tool
        # preview contains only input rows. No output-reader call occurred.
        return {'state_path': '/must-not-open/snapshot.pe-state.json',
            'measurements': {'transient': {'actual_stop_s': 1e-5, 'requested_step_s': 1e-6,
                'requested_stop_s': 1e-5, 'completed_steps': 10, 'sample_count': 10,
                'sample_every': 1, 'trace_access': {'kind': 'recorded_digital_solver_samples',
                    'location': 'measurements.transient.samples', 'state_path': '/must-not-open/snapshot.pe-state.json'}},
                'components': [{'id': 'input-' + str(n), 'type': 'digital_input', 'digital': [0]} for n in range(8)],
                'component_scope': {'total': 768, 'shown': 8, 'omitted': 760,
                    'complete_state_path': '/must-not-open/snapshot.pe-state.json'}}}

    def test_recorded_is_not_previewed_or_compared(self):
        self.tool('circuit_analyze', self.recorded_data())
        evidence = self.capsule()
        scope = evidence['analysis_calls'][0]['sampling_coverage']
        self.assertEqual(scope['archive_samples_reported'], 10)
        self.assertEqual(scope['returned_component_rows'], 8)
        self.assertEqual(scope['component_preview']['total'], 768)
        self.assertEqual(scope['component_preview']['omitted'], 760)
        self.assertEqual(scope['trace_access']['location'], 'measurements.transient.samples')
        self.assertNotIn('returned_transient_sample_rows', scope)
        self.assertTrue(scope['comparison_not_inferred_from_sampling'])
        self.assertEqual(evidence['trace_reads'], [])
        self.assertEqual(evidence['tool_totals']['circuit_analyze']['completed_outcomes'], 1)
        self.assertEqual(self.client.calls, [])

    def test_basic_core_preserves_sample_count_and_preview_scope(self):
        self.tool('circuit_analyze', self.recorded_data())
        index = json.loads(self.budget._tool_index(self.budget._journal_boundary(), 4096).split('\n', 1)[1])
        core = index['machine_evidence']
        self.assertEqual(core['families']['analysis_outcome_refs']['shown'], 1)
        sample = core['analysis_outcome_refs'][0]['sampling_coverage']
        self.assertEqual(sample['archive_samples_reported'], 10)
        self.assertEqual(sample['component_preview']['shown'], 8)
        self.assertIn('expected-result comparison are distinct', core['interpretation'])
        self.assertLessEqual(self.client.count([{'role':'user', 'content':self.budget._tool_index(self.budget._journal_boundary(), 4096)}]), 4096)

    def test_only_state_path_is_unknown_not_zero(self):
        self.tool('circuit_analyze', {'state_path': '/must-not-open/huge-state.json'})
        scope = self.capsule()['analysis_calls'][0]['sampling_coverage']
        self.assertEqual(scope['archive_sampling_status'], 'unknown_from_tool_result')
        self.assertNotIn('archive_samples_reported', scope)
        self.assertNotIn('returned_component_rows', scope)
        self.assertIn('archive sampling coverage is unknown, not zero', SUMMARY_PROMPT)

    def test_projection_retains_original_component_scope(self):
        data = self.recorded_data() | {'long_log': 'unrelated ' * 8000}
        did, _, raw = self.tool('circuit_analyze', data)
        projected = json.loads(self.budget.tool_document('analysis', raw, document_id=did))
        self.assertEqual(projected['fields']['/data/measurements/component_scope'], data['measurements']['component_scope'])
        self.assertEqual(projected['fields']['/data/measurements/transient']['sample_count'], 10)

    def test_source_title_hash_declared_type_and_raw_tool_are_distinct(self):
        draft_raw = json.dumps({'schema':'test.design-note.v1', 'kind':'draft', 'body':'未验证猜测'})
        draft = self.db.document(self.sid, '设计草稿，不是验证表', draft_raw)
        plain_raw = '|元件|末态|\n|R1|尚未测量|'
        plain = self.db.document(self.sid, 'analysis-table.zh.md', plain_raw)
        original, _, raw_tool = self.tool('circuit_analyze', {'state_path':'/state.json'})
        for did in (draft, plain, original):
            self.tool('read_context', self.db.read_document(self.sid, did, 0, 8), {'document_id':did})
        reads = {g['source_document_id']: g for g in self.capsule()['document_reads']}
        first = reads[draft]['source_metadata']
        self.assertEqual(first['title'], '设计草稿，不是验证表')
        self.assertEqual(first['content_sha256'], hashlib.sha256(draft_raw.encode()).hexdigest())
        self.assertEqual(first['declared_fields_untrusted'], {'schema':'test.design-note.v1','kind':'draft'})
        self.assertEqual(first['storage_kind'], 'archived_document')
        self.assertNotIn('declared_fields_untrusted', reads[plain]['source_metadata'])
        self.assertEqual(reads[plain]['source_metadata']['storage_kind'], 'archived_document')
        self.assertEqual(reads[original]['source_metadata']['storage_kind'], 'recorded_tool_outcome')
        self.assertEqual(reads[original]['source_metadata']['producer_tool'], 'circuit_analyze')
        self.assertEqual(reads[original]['source_metadata']['content_sha256'], hashlib.sha256(raw_tool.encode()).hexdigest())
        self.assertEqual(reads[draft]['unique_returned_characters'], 8)

    def test_source_lookup_cannot_cross_session_or_infer_from_title(self):
        other = self.db.session(source='web')
        did = self.db.document(other, 'secret validation report', '{"kind":"verified"}')
        self.tool('read_context', {'id':did,'title':'spoofed','text':'quoted','offset':0}, {'document_id':did})
        meta = self.capsule()['document_reads'][0]['source_metadata']
        self.assertEqual(meta, {'lookup':'unavailable_in_current_session'})

    def test_reported_source_hash_mismatch_remains_explicit(self):
        did = self.db.document(self.sid, 'source', '{"type":"design_draft","data":[]}')
        self.tool('read_context', {'id':did,'json_pointer':'/data','document_sha256':'0'*64,
            'text':'[]','offset':0,'total_chars':2}, {'document_id':did,'json_pointer':'/data'})
        entry = self.capsule()['document_reads'][0]
        self.assertFalse(entry['reported_hash_matches_source'])
        self.assertEqual(entry['document_sha256'], '0'*64)
        self.assertNotEqual(entry['source_metadata']['content_sha256'], '0'*64)
        self.assertEqual(entry['source_metadata']['declared_fields_untrusted'], {'type':'design_draft'})

    def test_failed_HDL_keeps_exact_source_documents_and_visible_retrieval(self):
        sources = [{'name':f'part_{i}.sv', 'document_id':f'{i:032x}', 'sha256':f'{i:064x}',
            'characters':1000+i, 'retrieval':'read_context', 'source_kind':'submitted_hdl_source',
            'note':'Stored source is not verification evidence. ' * 60} for i in range(16)]
        data = {'verified':False, 'profile':'custom', 'source_sha256':'b'*64,
            'compile':{'exit_code':0,'failure':None}, 'simulation':{'exit_code':1,'failure':None},
            'source_documents':sources}
        did, _, raw = self.tool('hdl_simulate', data)
        projected = json.loads(self.budget.tool_document('HDL result', raw, document_id=did))
        section = projected['sections']['/data/source_documents']
        self.assertEqual(section['total_rows'], 16)
        self.assertEqual(section['shown_rows'] + section['omitted_rows'], 16)
        self.assertGreater(section['omitted_rows'], 0)
        self.assertEqual(section['retrieval_json_pointer'], '/data/source_documents')
        for row in section['rows']:
            self.assertEqual(row['value'], sources[row['index']])
        self.assertIs(projected['fields']['/data/verified'], False)
        self.assertEqual(projected['fields']['/data/simulation']['exit_code'], 1)
        self.assertLessEqual(self.client.count([{'role':'user','content':json.dumps(projected, ensure_ascii=False, separators=(',',':'))}]), 4096)
        until = self.budget._journal_boundary()
        index = json.loads(self.budget._tool_index(until, 2048).split('\n',1)[1])
        ref = index['machine_evidence']['hdl_outcome_refs'][0]
        self.assertEqual(ref['source_document_count'], 16)
        self.assertEqual(ref['source_documents_json_pointer'], '/machine_evidence/hdl_calls/0/source_documents')
        self.assertFalse(ref['verified'])
        self.assertEqual(ref['source_documents_omitted'], 16)
        with self.db.connect() as store:
            archive = json.loads(store.execute('SELECT content FROM documents WHERE id=?',(index['document_id'],)).fetchone()[0])
        self.assertEqual(archive['machine_evidence']['hdl_calls'][0]['source_documents'], sources)
        self.assertEqual(self.client.calls, [])

    def test_HDL_source_index_survives_other_large_optional_metadata(self):
        source = {'name':'dut.sv','document_id':'f'*32,'sha256':'a'*64,'source_kind':'submitted_hdl_source'}
        data = {'verified':False,'source_documents':[source], 'checks':{'untrusted_text':'x'*11000},
            'simulation':{'exit_code':1,'failure':'timeout'},'long_log':'x'*15000}
        did, _, raw = self.tool('hdl_simulate', data)
        text = self.budget.tool_document('HDL result', raw, document_id=did)
        projected = json.loads(text)
        self.assertIn('/data/source_documents', projected['sections'])
        self.assertEqual(projected['sections']['/data/source_documents']['total_rows'], 1)
        self.assertIs(projected['fields']['/data/verified'], False)
        self.assertEqual(projected['fields']['/data/simulation']['failure'], 'timeout')
        self.assertLessEqual(self.client.count([{'role':'user','content':text}]), 4096)


if __name__ == '__main__':
    unittest.main()
