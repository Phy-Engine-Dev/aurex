from __future__ import annotations

import json
import copy
import os
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aurex.circuit_diagnostics import (CircuitGraph, circuit_diagnostics,
                                       normalize_run_contract, evaluate_run_contract)
from aurex.config import AurexConfig
from aurex.tools.circuits import circuit_diagnose
from aurex.tools.registry import ToolError, ToolRuntime


def component(cid, kind, *nodes, ref=None):
    row = {'id': cid, 'type': 'digital_' + kind, 'nodes': list(nodes), 'params': {}}
    if ref:
        row['pl_source'] = {'source_ref': ref}
    return row


def cpu_conflict():
    return {'components': [component('input', 'input', 'one'),
                           component('and', 'and', 'one', 'one', 'N2', ref='C136'),
                           component('ff', 'dff', 'N2', 'N676', 'N2', ref='C450'),
                           component('not', 'not', 'one', 'unrelated'),
                           component('out', 'output', 'N2')]}


class DiagnosticsTests(unittest.TestCase):
    def healthy_spec(self):
        return {'components': [component('in', 'input', 'a'), component('out', 'not', 'a', 'b')]}

    def contract(self):
        return {'expected_source': 'Boolean NOT truth table',
                'expected': [{'component': 'out', 'pin': 1, 'equals': 0, 'frame': 0}]}

    def test_contract_pass_fail_and_missing_settle(self):
        spec, contract = self.healthy_spec(), self.contract()
        measurements = {'execution_status': 'completed', 'digital_settle': {'settled': True}}
        sample = [{'digital': {'out': [1, 0]}, 'digital_settled': True}]
        self.assertEqual(evaluate_run_contract(spec, contract, measurements, stimulus=sample)['verdict'], 'PASS')
        sample[0]['digital']['out'][1] = 1
        self.assertEqual(evaluate_run_contract(spec, contract, measurements, stimulus=sample)['verdict'], 'FAIL')
        sample[0].pop('digital_settled')
        self.assertEqual(evaluate_run_contract(spec, contract, measurements, stimulus=sample)['verdict'], 'INCONCLUSIVE')
        self.assertEqual(evaluate_run_contract(spec, contract, {}, stimulus=sample)['verdict'], 'INCONCLUSIVE')

    def test_contract_unknown_is_not_a_functional_failure(self):
        result = evaluate_run_contract(self.healthy_spec(), self.contract(),
            {'execution_status': 'completed', 'digital_settle': {'settled': True}},
            stimulus=[{'digital_settled': True, 'digital': {'out': [1, 2]}}])
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertEqual(result['failure_class'], 'insufficient_observation')

    def test_contract_clock_and_bus_order_are_explicit(self):
        spec = {'components': [component('clk', 'input', 'c'), component('hi', 'input', 'h'), component('lo', 'input', 'l'),
                               component('out', 'not', 'c', 'o')]}
        contract = self.contract() | {'frame_count': 4,
            'clocks': [{'input': 'clk', 'period_frames': 2, 'high_frames': 1, 'phase_frames': 1}],
            'buses': [{'inputs_msb_first': ['hi', 'lo'], 'values': [0, 1, 2, 3]}]}
        table = normalize_run_contract(spec, contract)
        self.assertEqual(table, {'inputs': ['clk', 'hi', 'lo'], 'vectors': [[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]]})

    def test_contract_requires_independent_expected_and_valid_digital_values(self):
        for value in (2, 3, True):
            contract = self.contract()
            contract['expected'][0]['equals'] = value
            with self.assertRaises(ValueError):
                normalize_run_contract(self.healthy_spec(), contract)
        contract = self.contract()
        contract.pop('expected_source')
        with self.assertRaises(ValueError):
            normalize_run_contract(self.healthy_spec(), contract)
        contract = self.contract()
        contract['expected'][0]['pin'] = 0
        with self.assertRaisesRegex(ValueError, 'output pin'):
            normalize_run_contract(self.healthy_spec(), contract)

    def test_contract_accepts_logic_output_probe_but_not_logic_input(self):
        spec = {'components': [component('in', 'input', 'signal'),
                               component('probe', 'output', 'signal')]}
        contract = {'expected_source': 'independent expected probe value',
                    'expected': [{'component': 'probe', 'pin': 0,
                                  'equals': 1, 'frame': 0}]}
        self.assertIsNone(normalize_run_contract(spec, contract))
        contract['expected'][0]['component'] = 'in'
        with self.assertRaisesRegex(ValueError, 'output pin or Logic Output/Display probe'):
            normalize_run_contract(spec, contract)

    def test_contract_analog_missing_sample_and_spec_failure_are_distinct(self):
        spec = {'components': [{'id': 'v', 'type': 'vdc', 'nodes': ['a', 'gnd']}]}
        contract = {'expected_source': '5 V ideal source',
                    'expected': [{'node': 'a', 'equals': 5, 'tolerance': .01, 'time_s': .2}]}
        measurements = {'execution_status': 'completed', 'transient': {'actual_stop_s': .2, 'requested_stop_s': .2}}
        missing = evaluate_run_contract(spec, contract, measurements)
        self.assertEqual(missing['failure_class'], 'insufficient_observation')
        failed = evaluate_run_contract(spec, contract, measurements,
            points=[{'time_s': .2, 'components': [{'nodes': ['a'], 'voltage': [4]}]}])
        self.assertEqual(failed['failure_class'], 'specification_assertion_failed')
        measurements['execution_status'] = 'failed'
        result = evaluate_run_contract(spec, contract, measurements,
            points=[{'time_s': .2, 'components': [{'nodes': ['a'], 'voltage': [5]}]}])
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')

    def test_cpu_conflict_and_undriven_clock_are_exact(self):
        result = circuit_diagnostics(cpu_conflict())
        self.assertFalse(result['target_relevance_evaluated'])
        self.assertEqual(result['next_action']['mode'], 'slice')
        rows = result['findings']['rows']
        conflict = next(row for row in rows if row['kind'] == 'multiple_drivers')
        self.assertEqual(conflict['node'], 'N2')
        self.assertEqual([(p['source_ref'], p['label']) for p in conflict['evidence']['drivers']['pins']],
                         [('C136', 'out'), ('C450', 'q')])
        clock = next(row for row in rows if row['kind'] == 'undriven_clock')
        self.assertEqual(clock['node'], 'N676')
        self.assertEqual(clock['evidence']['drivers']['total'], 0)
        self.assertNotIn('combinational_cycle', result['finding_counts'])

    def test_target_slice_blocker_is_formal_inconclusive_without_execution(self):
        spec = cpu_conflict()
        targets = ['out'] * 10
        result = circuit_diagnostics(spec, mode='slice', targets=targets)
        self.assertTrue(result['target_relevance_evaluated'])
        self.assertEqual(result['observation_targets'], targets)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertEqual(result['failure_class'], 'invalid_netlist_or_drive_contract')
        self.assertEqual(result['execution'], {'started': False})
        self.assertEqual(result['blocking_finding_counts'], {
            'multiple_active_output_pins': 1, 'undriven_clock': 1})

    def test_undriven_model_defaults_are_info_and_not_contract_blockers(self):
        spec = {'components': [
            component('counter_clock', 'input', 'counter_clk'),
            component('counter', 'counter4', 'cq3', 'cq2', 'cq1', 'cq0',
                      'counter_clk', 'counter_enable'),
            component('random_clock', 'input', 'random_clk'),
            component('random', 'random4', 'rq3', 'rq2', 'rq1', 'rq0',
                      'random_clk', 'random_reset_n'),
        ]}
        result = circuit_diagnostics(spec)
        rows = {row['node']: row for row in result['findings']['rows']}
        self.assertEqual(result['finding_counts'], {'undriven_with_model_default': 2})
        self.assertEqual(rows['counter_enable']['severity'], 'info')
        self.assertEqual(rows['counter_enable']['evidence']['loads']['pins'][0]['model_default'], {
            'unconnected_state': 'Z', 'effective_value': 'HIGH', 'meaning': 'ENABLED'})
        self.assertEqual(rows['random_reset_n']['evidence']['loads']['pins'][0]['model_default'], {
            'unconnected_state': 'Z', 'effective_value': 'HIGH', 'meaning': 'NOT_RESET'})
        self.assertTrue(rows['random_reset_n']['evidence']['all_loads_have_model_defaults'])

        contract = {'expected_source': 'counter initial state',
                    'expected': [{'component': 'counter', 'pin': 3, 'equals': 0, 'frame': 0}]}
        verified = evaluate_run_contract(spec, contract,
            {'execution_status': 'completed', 'digital_settle': {'settled': True}},
            stimulus=[{'digital_settled': True, 'digital': {'counter': [0, 0, 0, 0]}}])
        self.assertEqual(verified['verdict'], 'PASS')
        self.assertEqual(verified['preflight_blockers']['total'], 0)

    def test_real_undriven_clock_and_data_remain_structural_findings(self):
        spec = {'components': [
            component('counter', 'counter4', 'q3', 'q2', 'q1', 'q0',
                      'clock_open', 'enable_open'),
            component('gate', 'not', 'data_open', 'gate_out'),
        ]}
        result = circuit_diagnostics(spec)
        rows = {row['node']: row for row in result['findings']['rows']}
        self.assertEqual(rows['clock_open']['kind'], 'undriven_clock')
        self.assertEqual(rows['data_open']['kind'], 'undriven_net')
        self.assertEqual(rows['enable_open']['kind'], 'undriven_with_model_default')

    def test_target_slice_includes_clock_excludes_sequential_data_and_unrelated(self):
        spec = {'components': [component('data', 'not', 'a', 'd'),
                               component('clock', 'not', 'b', 'clk'),
                               component('ff', 'dff', 'd', 'clk', 'q'),
                               component('out', 'output', 'q')]}
        result = circuit_diagnostics(spec, mode='slice', targets=['out'])
        self.assertEqual({row['id'] for row in result['slice']['rows']}, {'ff', 'clock'})
        boundary = result['boundaries']['rows'][0]
        self.assertEqual(boundary['data_nodes'], ['d'])
        self.assertEqual(boundary['controls'], {'clk': 'clk'})
        self.assertEqual(result['target_scope']['unresolved_targets'], [])

    def test_source_ref_resolves_target_and_unknown_target_is_not_guessed(self):
        result = circuit_diagnostics(cpu_conflict(), targets=['C450', 'missing'])
        self.assertEqual(result['target_scope']['unresolved_targets'], ['missing'])
        self.assertEqual(result['finding_counts']['multiple_drivers'], 1)

    def test_analog_attached_net_is_not_declared_undriven(self):
        spec = {'components': [component('out', 'output', 'a'),
                               {'id': 'v', 'type': 'vdc', 'nodes': ['a', 'gnd']}]}
        result = circuit_diagnostics(spec)
        self.assertEqual(result['finding_counts'], {})

    def test_ground_aliases_share_native_ground_semantics_without_mutation(self):
        for name in ('gnd', 'ground', '0', 'GROUND', 'GND'):
            spec = {'components': [component('gate', 'not', name, 'q')]}
            before = copy.deepcopy(spec)
            result = circuit_diagnostics(spec, mode='slice', targets=['q'])
            self.assertEqual(result['finding_counts'], {}, name)
            self.assertEqual(result['slice']['rows'][0]['nodes'], ['gnd', 'q'])
            self.assertEqual(spec, before)
        graph = CircuitGraph({'components': [component('a', 'input', '0'),
                                            component('b', 'not', 'GROUND', 'q')]})
        self.assertEqual(graph.nodes, {'gnd', 'q'})
        self.assertEqual(graph.targets(['ground'])[0], {'gnd'})

    def test_tri_state_multiple_driver_needs_resolution_not_automatic_failure(self):
        spec = {'components': [component('a', 'tri', 'i', 'en1', 'out'),
                               component('b', 'tri', 'i', 'en2', 'out')]}
        diagnostics = circuit_diagnostics(spec)
        row = diagnostics['findings']['rows'][0]
        self.assertEqual(row['drive_policy'], 'requires_resolution')
        self.assertEqual(row['status'], 'observed')
        self.assertEqual(diagnostics['blocking_finding_counts'], {})

        contract = {'expected_source': 'resolved tri-state bus expectation',
                    'expected': [{'component': 'a', 'pin': 2,
                                  'equals': 1, 'frame': 0}]}
        verified = evaluate_run_contract(spec, contract,
            {'execution_status': 'completed',
             'digital_settle': {'settled': True}},
            stimulus=[{'digital_settled': True,
                       'digital': {'a': [1, 1, 1]}}])
        self.assertEqual(verified['verdict'], 'PASS')
        self.assertEqual(verified['preflight_blockers']['total'], 0)

    def test_contract_blockers_are_limited_to_observation_cone(self):
        spec = {'components': [
            component('wanted-in', 'input', 'wanted-a'),
            component('wanted-out', 'not', 'wanted-a', 'wanted-b'),
            component('unrelated', 'dff', 'u-data', 'u-clock', 'u-q'),
        ]}
        contract = {'expected_source': 'NOT gate truth table',
                    'expected': [{'component': 'wanted-out', 'pin': 1,
                                  'equals': 0, 'frame': 0}]}
        diagnostics = circuit_diagnostics(
            spec, targets=['wanted-out'], depth=len(spec['components']))
        self.assertEqual(diagnostics['blocking_finding_counts'], {})
        self.assertEqual(diagnostics['global_finding_counts']['undriven_clock'], 1)
        result = evaluate_run_contract(spec, contract,
            {'execution_status': 'completed',
             'digital_settle': {'settled': True}},
            stimulus=[{'digital_settled': True,
                       'digital': {'wanted-out': [1, 0]}}])
        self.assertEqual(result['verdict'], 'PASS')

    def test_comb_cycle_vs_registered_feedback(self):
        spec = {'components': [component('a', 'not', 'x', 'y'), component('b', 'not', 'y', 'x')]}
        result = circuit_diagnostics(spec)
        self.assertEqual(result['finding_counts']['combinational_cycle'], 1)
        self.assertEqual(result['findings']['rows'][0]['nodes'], ['x', 'y'])

    def test_large_graph_is_iterative_and_payload_is_bounded(self):
        spec = {'components': [component(f'g{i}', 'not', f'n{i}', f'n{i+1}') for i in range(6000)]}
        result = circuit_diagnostics(spec, mode='slice', targets=['n6000'], depth=32)
        self.assertEqual(result['target_scope']['component_count'], 33)
        self.assertEqual(result['target_scope']['frontier_count'], 1)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 7000)

    def test_presentation_truncation_does_not_falsify_source_paging(self):
        spec = {'components': [component('g' + str(i) + 'a' * 90, 'not', f'n{i}', f'o{i}') for i in range(16)]}
        result = circuit_diagnostics(spec, limit=16, max_characters=2200)
        self.assertFalse(result['findings']['has_more'])
        self.assertGreater(result['findings']['projection_omitted'], 0)
        self.assertEqual(result['findings']['next_offset'], result['findings']['shown'])
        self.assertLessEqual(len(json.dumps(result, separators=(',', ':'))), 2200)

    def test_diagnose_preserves_unsettled_evidence_and_never_claims_pass(self):
        result = circuit_diagnostics(cpu_conflict(), mode='diagnose', measurements={
            'execution': {'completed': False, 'reason': 'DIGITAL_NOT_SETTLED', 'pending_nodes': list(range(500))},
            'transient': {'actual_stop_s': .2, 'requested_stop_s': 1., 'completed_steps': 200}})
        self.assertFalse(result['execution']['execution']['completed'])
        self.assertEqual(result['execution']['execution']['pending_nodes']['omitted'], 496)
        self.assertEqual(result['timeline']['actual_stop_s'], .2)
        self.assertEqual(result['conclusion'], 'INCONCLUSIVE')

    def test_public_tool_native_spec_is_readonly_and_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.circuit.json'
            spec = cpu_conflict()
            spec['components'][0]['params'] = {'state': 1}
            spec['schema'] = 'aurex.circuit.v1'
            original = json.dumps(spec)
            path.write_text(original)
            runtime = ToolRuntime('diagnostics', 'zh', '', AurexConfig(), folder)
            result = circuit_diagnose(runtime, {'path': str(path), 'mode': 'preflight'})
            self.assertEqual(result['finding_counts']['multiple_drivers'], 1)
            self.assertEqual(path.read_text(), original)

    def test_public_run_contract_replays_instead_of_trusting_forged_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.pe-state.json'
            spec = self.healthy_spec()
            spec['components'][0]['params'] = {'state': 1}
            snapshot = {'schema': 'aurex.pe-state.v1', 'spec': spec, 'scene': {},
                'measurements': {'execution_status': 'completed', 'digital_settle': {'settled': True},
                    'stimulus_results': [{'step': 0, 'digital_settled': True, 'digital': {'out': [1, 0]}}]}}
            raw = json.dumps(snapshot)
            path.write_text(raw)
            runtime = ToolRuntime('diagnostics', 'zh', '', AurexConfig(), folder)
            fresh = {'execution_status': 'completed', 'digital_settle': {'settled': True},
                     'components': [], 'stimulus_results': [
                         {'step': 0, 'digital_settled': True, 'digital': {'out': [0, 1]}}]}
            fresh['state'] = {'schema': 'aurex.pe-state.v1', 'spec': spec,
                              'measurements': copy.deepcopy(fresh)}
            with patch('aurex.tools.circuits.pe_simulate', return_value=fresh) as simulate:
                result = circuit_diagnose(runtime, {'path': str(path), 'mode': 'run_contract', 'contract': self.contract()})
            self.assertEqual(result['verdict'], 'FAIL')
            simulate.assert_called_once()
            self.assertEqual(path.read_text(), raw)
            self.assertEqual(result['coverage']['fail'], 1)
            self.assertTrue(result['execution']['replayed_complete_spec'])
            self.assertNotEqual(result['state_path'], str(path))
            self.assertEqual(result['expected_source'], self.contract()['expected_source'])
            self.assertRegex(result['contract_sha256'], r'^[0-9a-f]{64}$')
            self.assertRegex(result['native_spec_sha256'], r'^[0-9a-f]{64}$')
            self.assertRegex(result['replay_source_sha256'], r'^[0-9a-f]{64}$')
            self.assertRegex(result['result_state_sha256'], r'^[0-9a-f]{64}$')
            self.assertNotIn('source_sha256', result)

    def test_public_run_contract_missing_spec_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'missing.pe-state.json'
            path.write_text(json.dumps({'schema': 'aurex.pe-state.v1', 'scene': {},
                                       'measurements': {'execution_status': 'completed'}}))
            runtime = ToolRuntime('diagnostics', 'zh', '', AurexConfig(), folder)
            with patch('aurex.tools.circuits.pe_simulate') as simulate:
                result = circuit_diagnose(runtime, {'path': str(path), 'mode': 'run_contract', 'contract': self.contract()})
            self.assertEqual(result['verdict'], 'INCONCLUSIVE')
            self.assertEqual(result['failure_class'], 'replay_source_unavailable')
            self.assertEqual(result['expected_source'], self.contract()['expected_source'])
            self.assertRegex(result['contract_sha256'], r'^[0-9a-f]{64}$')
            simulate.assert_not_called()

    def test_public_run_contract_preflight_prevents_invalid_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.circuit.json'
            spec = cpu_conflict()
            spec['components'][0]['params'] = {'state': 1}
            spec['schema'] = 'aurex.circuit.v1'
            path.write_text(json.dumps(spec))
            runtime = ToolRuntime('diagnostics', 'zh', '', AurexConfig(), folder)
            contract = {'expected_source': 'expected q from external specification',
                        'expected': [{'component': 'ff', 'pin': 2, 'equals': 1, 'frame': 0}]}
            with patch('aurex.tools.circuits.pe_simulate') as simulate:
                result = circuit_diagnose(runtime, {'path': str(path), 'mode': 'run_contract', 'contract': contract})
            self.assertEqual(result['verdict'], 'INCONCLUSIVE')
            self.assertEqual(result['failure_class'], 'invalid_netlist_or_drive_contract')
            self.assertEqual(result['expected_source'], contract['expected_source'])
            self.assertRegex(result['contract_sha256'], r'^[0-9a-f]{64}$')
            self.assertRegex(result['native_spec_sha256'], r'^[0-9a-f]{64}$')
            self.assertRegex(result['replay_source_sha256'], r'^[0-9a-f]{64}$')
            simulate.assert_not_called()

    def test_public_explicit_stimulus_table_cannot_bypass_validator(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.circuit.json'
            spec = self.healthy_spec()
            spec['schema'] = 'aurex.circuit.v1'
            spec['components'][0]['params'] = {'state': 0}
            path.write_text(json.dumps(spec))
            runtime = ToolRuntime('diagnostics', 'zh', '', AurexConfig(), folder)
            invalid = [({'inputs': ['in'], 'vectors': [[value]]})
                       for value in (True, False, 'X', 'Z', 4, -1)]
            invalid += [{'inputs': ['missing'], 'vectors': [[0]]},
                        {'inputs': ['out'], 'vectors': [[0]]},
                        {'inputs': ['in'], 'vectors': [[0, 1]]}]
            with patch('aurex.tools.circuits.pe_simulate') as simulate:
                for table in invalid:
                    with self.subTest(table=table), self.assertRaises(ToolError):
                        circuit_diagnose(runtime, {'path': str(path), 'mode': 'run_contract',
                            'contract': self.contract() | {'stimulus_table': table}})
            simulate.assert_not_called()

    def test_public_analog_numerical_failure_is_not_specification_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.circuit.json'
            path.write_text(json.dumps({'schema': 'aurex.circuit.v1', 'components': [
                {'id': 'v', 'type': 'vdc', 'nodes': ['a', 'gnd'], 'params': {'v': 5}}]}))
            runtime = ToolRuntime('diagnostics', 'zh', '', AurexConfig(), folder)
            contract = {'expected_source': 'Ideal source voltage',
                        'expected': [{'node': 'a', 'equals': 5, 'tolerance': .001, 'time_s': .1}]}
            with patch('aurex.tools.circuits.pe_simulate', side_effect=ToolError('singular matrix: convergence failed')):
                result = circuit_diagnose(runtime, {'path': str(path), 'mode': 'run_contract', 'contract': contract})
            self.assertEqual(result['verdict'], 'INCONCLUSIVE')
            self.assertEqual(result['failure_class'], 'numerical_nonconvergence')
            self.assertEqual(result['coverage']['evaluated'], 0)
            self.assertEqual(result['expected_source'], contract['expected_source'])
            self.assertRegex(result['contract_sha256'], r'^[0-9a-f]{64}$')
            self.assertRegex(result['native_spec_sha256'], r'^[0-9a-f]{64}$')


@unittest.skipUnless(os.environ.get('AUREX_PHY_ENGINE_BUILD'), 'native library required')
class NativeContractTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        build = Path(os.environ['AUREX_PHY_ENGINE_BUILD']).resolve()
        config = AurexConfig()
        config = replace(config, phy_engine=replace(config.phy_engine,
            cmake_build_dir=str(build), verilog2plsav_path=str(build / 'verilog2plsav'),
            phyengine_lib_path=os.environ.get('AUREX_PHY_ENGINE_TEST_LIB', str(build / 'libphyengine.so'))))
        self.runtime = ToolRuntime('contract-native', 'zh', '', config, temp.name)

    def test_native_digital_contract_and_saved_recheck(self):
        spec = {'schema': 'aurex.circuit.v1', 'components': [
            component('in', 'input', 'a'), component('out', 'not', 'a', 'b'),
            component('probe', 'output', 'b')]}
        spec['components'][0]['params'] = {'state': 0}
        source = self.folder / 'not.circuit.json'
        source.write_text(json.dumps(spec))
        contract = {'expected_source': 'NOT truth table, two input values',
                    'stimulus_table': {'inputs': ['in'], 'vectors': [[0], [1]]},
                    'expected': [{'component': 'out', 'pin': 1, 'equals': 1, 'frame': 0},
                                 {'component': 'out', 'pin': 1, 'equals': 0, 'frame': 1}]}
        result = circuit_diagnose(self.runtime, {'path': str(source), 'mode': 'run_contract', 'contract': contract})
        self.assertEqual(result['verdict'], 'PASS', result)
        self.assertEqual(result['coverage']['pass'], 2)
        self.assertNotIn('stimulus_frame_s', result['time_semantics'])
        self.assertIn('logical stimulus frame', result['time_semantics']['stimulus_frame_unit'])
        contract.pop('stimulus_table')
        again = circuit_diagnose(self.runtime, {'path': result['state_path'], 'mode': 'run_contract', 'contract': contract})
        self.assertEqual(again['verdict'], 'PASS')
        self.assertNotEqual(again['state_path'], result['state_path'])
        self.assertTrue(again['execution']['replayed_complete_spec'])

    def test_native_analog_contract_with_exact_time(self):
        spec = {'schema': 'aurex.circuit.v1', 'analysis': 'tr', 'tr_step': .001,
                'tr_stop': .002, 'tr_sample_every': 1, 'components': [
                    {'id': 'v', 'type': 'vdc', 'nodes': ['a', 'gnd'], 'params': {'v': 5}},
                    {'id': 'r', 'type': 'resistor', 'nodes': ['a', 'gnd'], 'params': {'r': 10}}]}
        source = self.folder / 'source.circuit.json'
        source.write_text(json.dumps(spec))
        contract = {'expected_source': 'Ideal 5 V source across 10 ohm resistor',
                    'expected': [{'node': 'a', 'equals': 5, 'tolerance': 1e-8, 'time_s': .002}]}
        result = circuit_diagnose(self.runtime, {'path': str(source), 'mode': 'run_contract', 'contract': contract})
        self.assertEqual(result['verdict'], 'PASS', result)
        self.assertEqual(result['coverage']['recorded_time_samples'], 2)


if __name__ == '__main__':
    unittest.main()
