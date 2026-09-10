import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from aurex.config import ContextPolicyConfig, LLMConfig
from aurex.context_budget import ContextBudget
from aurex.sessiondb import SessionDB


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        db = SessionDB(str(Path(temp.name) / 'test.sqlite'))
        sid = db.session(source='admin')
        rid = db.enqueue_task(sid, 'diagnose', source='admin')
        self.db, self.sid, self.rid = db, sid, rid
        client = SimpleNamespace(config=LLMConfig(context_length=65536, max_output_tokens=512),
            count=lambda messages, tools=None: len(json.dumps(messages, ensure_ascii=False)) // 3 + 1)
        self.budget = ContextBudget(client, db, sid, rid, 65536, lambda *args: None,
                                    policy=ContextPolicyConfig(safety_tokens=128))

    def test_large_settle_failure_survives_optional_measurements(self):
        data = {'execution_status': 'failed', 'waveform_valid': False,
                'digital_settle': {'settled': False, 'reason': 'DIGITAL_NOT_SETTLED',
                                  'processed_events': 100000, 'pending_count': 533,
                                  'pending': [{'node': f'N{i}', 'events': i} for i in range(533)]},
                'coverage': {'completed': 0, 'requested': 1000},
                'measurements': {'components': [{'id': str(i), 'digital': [2]} for i in range(500)]}}
        raw = json.dumps({'ok': False, 'data': data})
        output = json.loads(self.budget.tool_document('analysis', raw, tool_name='circuit_analyze'))
        fields = output['fields']
        self.assertEqual(fields['/data/execution_status'], 'failed')
        self.assertFalse(fields['/data/waveform_valid'])
        self.assertEqual(fields['/data/digital_settle']['pending_count'], 533)
        self.assertEqual(fields['/data/digital_settle']['pending']['omitted'], 529)
        self.assertEqual(fields['/data/coverage']['completed'], 0)

    def test_diagnose_preserves_fitting_producer_payload(self):
        data = {'verdict': 'INCONCLUSIVE', 'execution': {'completed': False},
                'coverage': {'assertions': 5, 'evaluated': 0},
                'findings': {'total': 1, 'shown': 1, 'rows': [{'node': 'N2', 'kind': 'multiple_drivers'}]}}
        raw = json.dumps({'ok': True, 'data': data})
        result = self.budget.tool_document('diagnose', raw, tool_name='circuit_diagnose')
        self.assertEqual(result, raw)

    def test_diagnose_smaller_budget_keeps_verdict_and_reports_omission(self):
        data = {'verdict': 'INCONCLUSIVE', 'execution': {'completed': False},
                'coverage': {'assertions': 5, 'evaluated': 0},
                'findings': {'total': 50, 'shown': 50, 'has_more': False,
                             'rows': [{'node': f'N{i}', 'kind': 'multiple_drivers', 'detail': 'x' * 300} for i in range(50)]}}
        result = json.loads(self.budget.tool_document('diagnose', json.dumps({'ok': True, 'data': data}),
            tool_name='circuit_diagnose', _token_limit=600))['data']
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertFalse(result['execution']['completed'])
        self.assertEqual(result['coverage']['evaluated'], 0)
        self.assertFalse(result['findings']['has_more'])
        self.assertGreater(result['findings']['projection_omitted'], 0)

    def test_run_contract_32_assertions_and_preflight_fit_without_losing_verdict(self):
        assertions = [{'component': f'OUT-{index}', 'pin': 0, 'expected': index & 1,
                       'actual': index & 1, 'status': 'pass', 'detail': 'a' * 320}
                      for index in range(32)]
        data = {
            'verdict': 'INCONCLUSIVE',
            'failure_class': 'invalid_netlist_or_drive_contract',
            'execution': {'started': False, 'completed': False},
            'coverage': {'assertions': 32, 'evaluated': 0},
            'source_sha256': 'a' * 64,
            'native_spec_sha256': 'b' * 64,
            'contract_sha256': 'c' * 64,
            'assertions': {'total': 32, 'shown': 32, 'offset': 0,
                           'has_more': False, 'rows': assertions},
            'preflight': {
                'finding_counts': {'multiple_drivers': 1},
                'blocking_finding_counts': {'multiple_drivers': 1},
                'findings': {'total': 1, 'shown': 1, 'offset': 0,
                             'has_more': False, 'rows': [{
                                 'node': 'N77', 'kind': 'multiple_drivers',
                                 'drive_policy': 'multiple_active_output_pins',
                                 'detail': 'f' * 320}]},
            },
        }
        output = json.loads(self.budget.tool_document(
            'contract', json.dumps({'ok': True, 'data': data}),
            tool_name='circuit_diagnose'))['data']
        for key in ('verdict', 'failure_class', 'execution', 'coverage',
                    'source_sha256', 'native_spec_sha256', 'contract_sha256'):
            self.assertEqual(output[key], data[key])
        self.assertGreater(output['assertions'].get('projection_omitted', 0), 0)
        self.assertEqual(output['preflight']['finding_counts'], {'multiple_drivers': 1})
        self.assertEqual(output['preflight']['findings']['rows'][0]['node'], 'N77')

    def test_run_contract_hash_binding_survives_checkpoint_index(self):
        data = {'verdict': 'INCONCLUSIVE',
                'failure_class': 'invalid_netlist_or_drive_contract',
                'target_relevance_evaluated': True,
                'observation_targets': ['C11', 'C12'],
                'finding_scope': 'backward dependency cone of requested observations',
                'next_action': {'action': 'stop', 'reason': 'target cone is blocked'},
                'expected_source': 'integer square-root specification',
                'contract_sha256': 'c' * 64,
                'native_spec_sha256': 'n' * 64,
                'replay_source_sha256': 'r' * 64,
                'execution': {'started': False},
                'coverage': {'assertions': 10, 'evaluated': 0},
                'preflight': {'finding_counts': {'multiple_drivers': 1},
                              'blocking_finding_counts': {'multiple_drivers': 1}}}
        arguments = {'path': '/immutable.circuit.json', 'mode': 'run_contract',
                     'contract': {'expected_source': data['expected_source']}}
        self.db.message(self.sid, self.rid, {'role': 'assistant', 'content': '',
            'tool_calls': [{'id': 'contract-call', 'type': 'function',
                'function': {'name': 'circuit_diagnose',
                             'arguments': json.dumps(arguments)}}]})
        _, until = self.db.tool_outcome(
            self.sid, self.rid, 'contract-call', 'circuit_diagnose',
            json.dumps({'ok': True, 'data': data}), True)
        journal = self.budget._tool_index(until, 3072)
        evidence = json.loads(journal.split('\n', 1)[1])['machine_evidence']
        ref = evidence['diagnostic_refs'][0]
        for key in ('expected_source', 'contract_sha256', 'native_spec_sha256',
                    'replay_source_sha256', 'target_relevance_evaluated',
                    'observation_targets', 'finding_scope', 'next_action'):
            self.assertEqual(ref[key], data[key])

    def test_source_last_page_is_not_projection_complete(self):
        data = {'interface_only': True, 'total_ports': 61, 'total_inputs': 61,
                'total_outputs': 0, 'offset': 0, 'has_more': False,
                'ports': [{'id': f'id{i}-' + 'x' * 128, 'ref': f'C{i}', 'label': 'long label' * 12,
                           'node': f'N{i}', 'direction': 'input', 'logic': 2} for i in range(61)]}
        output = json.loads(self.budget.tool_document('inspect', json.dumps({'ok': True, 'data': data}),
                                                     tool_name='circuit_inspect'))
        self.assertFalse(output['fields']['/data/has_more'])
        section = output['sections']['/data/ports']
        self.assertTrue(section['projection_truncated'])
        self.assertEqual(section['omitted_rows'], 61 - section['shown_rows'])
        self.assertIn('source paging only', output['pagination_semantics'])


if __name__ == '__main__':
    unittest.main()
