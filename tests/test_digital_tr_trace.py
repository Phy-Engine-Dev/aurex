"""Timestamped digital TR observations, separate from legacy stimulus frames."""
from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import jsonschema

from aurex.config import AurexConfig
from aurex.phy_engine.ffi import _Circuit, PhyEngineError
from aurex.tools.circuits import (_trace_sample_index_guide, circuit_analyze,
                                  circuit_create, circuit_read_trace,
                                  register_circuit_tools)
from aurex.tools.registry import ToolError, ToolRegistry, ToolRuntime


def design():
    return {"components": [
        {"id": "IN", "type": "digital_input", "nodes": ["a"], "params": {"state": 0}},
        {"id": "NOT", "type": "digital_not", "nodes": ["a", "b"]},
        {"id": "OUT", "type": "digital_output", "nodes": ["b"]},
    ]}


def recorded():
    spec = design()
    return {"schema": "aurex.pe-state.v1", "spec": spec, "measurements": {
        "transient": {"actual_stop_s": .003, "samples": [
            {"time_s": (i + 1) / 1000, "completed_steps": i + 1,
             "components": [{"id": "IN", "digital": [0]}, {"id": "NOT", "digital": [0, value]},
                            {"id": "OUT", "digital": [value]}]}
            for i, value in enumerate((1, 2, 3))]},
        "stimulus_results": [{"step": 0, "digital": {"OUT": [0]}}]}}


class DigitalTraceReaderTests(unittest.TestCase):
    def read(self, value, **args):
        with patch('aurex.tools.circuits._input', return_value=(Path('/recorded.pe-state.json'), value)), \
             patch('aurex.tools.circuits.pe_simulate', side_effect=AssertionError('read cannot run solver')):
            return circuit_read_trace(None, {'path': '/recorded.pe-state.json', **args})

    def test_legacy_page_keeps_real_times_X_Z_and_warns_not_new_stimulus(self):
        value = recorded()
        before = copy.deepcopy(value)
        result = self.read(value, component_ids=['OUT'], offset=1, limit=1)
        self.assertEqual(result['points'], [{'sample_index': 1, 'time_s': .002, 'completed_steps': 2,
            'digital': {'OUT': [2]}, 'missing_component_ids': []}])
        self.assertEqual(result['selection_mode'], 'page')
        self.assertEqual(result['total_samples'], 3)
        self.assertTrue(result['has_more'])
        self.assertFalse(result['digital_propagation']['verified_per_step'])
        self.assertIsNotNone(result['warning'])
        self.assertEqual(self.read(value, component_ids=['OUT'], offset=2)['points'][0]['digital']['OUT'], [3])
        self.assertEqual(value, before)

    def test_verified_policy_preserved_and_offset_beyond_end_is_empty(self):
        value = recorded()
        value['measurements']['transient']['completed_steps'] = 3
        value['measurements']['transient']['digital_propagation'] = {
            'version': 1, 'verified_per_step': True, 'policy': 'once_after_each_native_solve_before_sampling',
            'per_tr_step': 1, 'completed_propagation_steps': 3}
        result = self.read(value, offset=99)
        self.assertEqual(result['points'], [])
        self.assertIsNone(result['warning'])
        self.assertFalse(result['has_more'])

    def test_invalid_policy_is_not_certified_and_invalid_times_are_rejected(self):
        for override in ({'version': 0}, {'version': 2}, {'version': True},
                         {'completed_propagation_steps': 999}, {'per_tr_step': True},
                         {'policy': 'some_other_method'}, {'per_tr_step': 3},
                         {'configured_version': 2, 'count_origin': 'native_counter'}):
            value = recorded()
            value['measurements']['transient']['completed_steps'] = 3
            value['measurements']['transient']['digital_propagation'] = {
                'version': 1, 'verified_per_step': True, 'per_tr_step': 1,
                'policy': 'once_after_each_native_solve_before_sampling',
                'completed_propagation_steps': 3, **override}
            with self.subTest(override=override):
                self.assertFalse(self.read(value)['digital_propagation']['verified_per_step'])
        for timestamp in (float('nan'), float('inf'), -.1, True, .001):
            value = recorded()
            value['measurements']['transient']['samples'][1]['time_s'] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ToolError):
                self.read(value, offset=99)

    def test_per_step_count_validation_is_strict(self):
        for count in (0, 65, -1, True, 1.0, '3', None):
            with self.subTest(count=count), self.assertRaises(PhyEngineError):
                _Circuit._validate_digital_steps(count)
        for count in (1, 3, 64):
            _Circuit._validate_digital_steps(count)

    def test_unknown_configured_ABI_is_rejected_before_execution(self):
        obj = object.__new__(_Circuit)
        entry = lambda *args: self.fail('Unknown ABI must not execute')
        for version in (None, 0, 2):
            dll = SimpleNamespace(circuit_run_transient_trace_configured=entry)
            if version is not None:
                dll.circuit_transient_digital_propagation_configured_version = lambda: version
            obj.lib = SimpleNamespace(_dll=dll)
            with self.subTest(version=version), self.assertRaisesRegex(PhyEngineError, 'Unsupported configured'):
                obj._configured_transient_entry('circuit_run_transient_trace_configured')

    def test_missing_sample_is_not_zero_and_invalid_states_are_rejected(self):
        value = recorded()
        value['measurements']['transient']['samples'][0]['components'].pop()
        point = self.read(value, component_ids=['OUT'], limit=1)['points'][0]
        self.assertEqual(point['digital'], {})
        self.assertEqual(point['missing_component_ids'], ['OUT'])
        for invalid in ([True], [1.0], [4], [], [1, 0]):
            value = recorded()
            value['measurements']['transient']['samples'][0]['components'][-1]['digital'] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ToolError):
                self.read(value, component_ids=['OUT'])

    def test_exact_digital_ids_pagination_and_conflicting_selector_validation(self):
        for args in ({'component_ids': ['unknown']}, {'component_ids': ['OUT', 'OUT']},
                     {'component_ids': None}, {'component_ids': [True]}, {'component_ids': []},
                     {'component_ids': ['OUT'], 'nodes': ['b']}, {'nodes': ['b']},
                     {'offset': -1}, {'offset': True}, {'limit': 65}, {'limit': False}):
            with self.subTest(args=args), self.assertRaises(ToolError):
                self.read(recorded(), **args)
        registry = ToolRegistry()
        register_circuit_tools(registry)
        schema = registry.get('circuit_read_trace').parameters
        jsonschema.validate({'path': 'x', 'component_ids': ['OUT']}, schema)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({'path': 'x', 'nodes': ['b'], 'component_ids': ['OUT']}, schema)

    def test_sparse_sample_indices_select_exact_frames_without_pagination(self):
        result = self.read(recorded(), component_ids=['OUT'], sample_indices=[0, 2])
        self.assertEqual(result['selection_mode'], 'sample_indices')
        self.assertEqual(result['sample_indices'], [0, 2])
        self.assertEqual([point['sample_index'] for point in result['points']], [0, 2])
        self.assertEqual([point['digital']['OUT'] for point in result['points']], [[1], [3]])
        self.assertFalse(result['has_more'])
        self.assertEqual(result['total_samples'], 3)
        for args in ({'sample_indices': []}, {'sample_indices': [0, 0]},
                     {'sample_indices': [-1]}, {'sample_indices': [True]},
                     {'sample_indices': [3]}, {'sample_indices': [0], 'offset': 0},
                     {'sample_indices': [0], 'limit': 1}):
            with self.subTest(args=args), self.assertRaises(ToolError):
                self.read(recorded(), component_ids=['OUT'], **args)
        registry = ToolRegistry()
        register_circuit_tools(registry)
        schema = registry.get('circuit_read_trace').parameters
        jsonschema.validate({'path': 'x', 'component_ids': ['OUT'], 'sample_indices': [0, 2]}, schema)

    def test_sparse_sample_indices_work_for_analog_trace(self):
        value = recorded()
        value['spec']['components'].append(
            {'id': 'R', 'type': 'resistor', 'nodes': ['v', 'gnd'], 'params': {'r': 10}})
        for index, point in enumerate(value['measurements']['transient']['samples']):
            for component in point['components']:
                component['nodes'] = next(
                    row['nodes'] for row in value['spec']['components'] if row['id'] == component['id'])
            point['components'].append(
                {'id': 'R', 'nodes': ['v', 'gnd'], 'voltage': [index + 1.0, 0.0]})
        result = self.read(value, nodes=['v'], sample_indices=[2, 0])
        self.assertEqual(result['selection_mode'], 'sample_indices')
        self.assertEqual(result['sample_indices'], [2, 0])
        self.assertEqual(result['selected_sample_indices'], [2, 0])
        self.assertEqual(result['points'], [[.003, 3.0], [.001, 1.0]])

    def test_trace_index_guide_maps_interactions_to_recorded_frames_not_solver_steps(self):
        points = [
            {'time_s': value, 'completed_steps': int(round(value * 1000))}
            for value in (.05, .10, .15, .20, .25, .30, .80, .85, 1.0)
        ]
        guide = _trace_sample_index_guide(
            points, [{'time_s': .2, 'set': {'S': 1}}, {'time_s': .8, 'set': {'S': 0}}])
        self.assertEqual(guide['zero_based_index_range'], [0, 8])
        self.assertEqual(guide['interaction_boundaries'], [
            {'interaction_time_s': .2, 'before_index': 2, 'at_or_after_index': 3, 'after_index': 4},
            {'interaction_time_s': .8, 'before_index': 5, 'at_or_after_index': 6, 'after_index': 7},
        ])
        self.assertEqual([row['sample_index'] for row in guide['suggested_samples']],
                         [0, 2, 3, 4, 5, 6, 7, 8])
        self.assertIn('not solver-step numbers', guide['usage'])

    def test_disjoint_mixed_state_selects_digital_ids_without_fabricated_voltages(self):
        value = recorded()
        value['spec']['components'].append({'id': 'R', 'type': 'resistor', 'nodes': ['v', 'gnd'], 'params': {'r': 10}})
        for point in value['measurements']['transient']['samples']:
            point['components'].append({'id': 'R', 'nodes': ['v', 'gnd'], 'voltage': [5, 0], 'digital': [2, 2]})
        self.assertEqual(self.read(value, component_ids=['OUT'])['points'][0]['digital'], {'OUT': [1]})
        with self.assertRaises(ToolError):
            self.read(value, component_ids=['R'])

    def test_no_TR_samples_does_not_substitute_stimulus(self):
        value = recorded()
        value['measurements']['transient'].pop('samples')
        with self.assertRaisesRegex(ToolError, 'separately recorded stimulus'):
            self.read(value)

    def test_capability_is_read_from_library_not_assumed_from_python_version(self):
        obj = object.__new__(_Circuit)
        obj.lib = SimpleNamespace(_dll=SimpleNamespace())
        self.assertFalse(obj.transient_digital_policy(10)['verified_per_step'])
        self.assertIsNone(obj.transient_digital_policy(10)['completed_propagation_steps'])
        obj.lib._dll.circuit_transient_digital_propagation_version = lambda: 1
        self.assertEqual(obj.transient_digital_policy(10)['completed_propagation_steps'], 10)
        obj.lib._dll.circuit_transient_digital_propagation_version = lambda: 2
        self.assertFalse(obj.transient_digital_policy(10)['verified_per_step'])


@unittest.skipUnless(os.environ.get('AUREX_PHY_ENGINE_BUILD'), 'native build not configured')
class DigitalTraceNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        build = Path(os.environ['AUREX_PHY_ENGINE_BUILD']).resolve()
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(build),
            verilog2plsav_path=str(build / 'verilog2plsav'), phyengine_lib_path=str(build / 'libphyengine.so')))
        self.rt = ToolRuntime('digital-tr', 'zh', str(Path(self.temp.name) / 'config.json'), cfg, self.temp.name)

    def run_trace(self, **args):
        return circuit_analyze(self.rt, {'spec': design(), 'analysis': 'tr', 'tr_step': 1e-6,
            'tr_stop': 1e-5, 'tr_sample_every': 1, **args})

    def test_fresh_not_trace_is_propagated_without_implicit_extra_tick(self):
        result = self.run_trace()
        trace = result['measurements']['transient']
        self.assertTrue(trace['digital_propagation']['verified_per_step'])
        self.assertEqual(trace['digital_propagation']['completed_propagation_steps'], 10)
        self.assertEqual(trace['post_trace_digital_ticks']['count'], 0)
        self.assertFalse(trace['post_trace_digital_ticks']['explicit'])
        before = Path(result['state_path']).read_bytes()
        with patch('aurex.tools.circuits.pe_simulate', side_effect=AssertionError('read cannot solve')):
            page = circuit_read_trace(self.rt, {'path': result['state_path'], 'component_ids': ['IN', 'OUT']})
        self.assertEqual(len(page['points']), 10)
        self.assertTrue(all(p['digital'] == {'IN': [0], 'OUT': [1]} for p in page['points']))
        self.assertEqual([p['completed_steps'] for p in page['points']], list(range(1, 11)))
        self.assertEqual(Path(result['state_path']).read_bytes(), before)

    def test_explicit_extra_ticks_remain_explicit_and_do_not_change_TR_history(self):
        result = self.run_trace(digital_clock_ticks=2)
        self.assertEqual(result['measurements']['transient']['post_trace_digital_ticks']['count'], 2)
        self.assertTrue(result['measurements']['transient']['post_trace_digital_ticks']['explicit'])
        self.assertEqual(result['measurements']['transient']['sample_count'], 10)

    def test_configured_counts_are_native_and_independent_of_sample_every(self):
        for count in (1, 3, 64):
            with self.subTest(count=count):
                result = self.run_trace(digital_steps_per_tr_step=count, tr_sample_every=3)
                trace = result['measurements']['transient']
                policy = trace['digital_propagation']
                self.assertEqual(policy['per_tr_step'], count)
                self.assertEqual(policy['completed_propagation_steps'], 10 * count)
                self.assertEqual(policy['count_origin'], 'native_counter')
                self.assertEqual(trace['sample_count'], 4)
                page = circuit_read_trace(self.rt, {'path': result['state_path'], 'component_ids': ['OUT']})
                self.assertTrue(page['digital_propagation']['verified_per_step'])
                self.assertEqual([p['completed_steps'] for p in page['points']], [3, 6, 9, 10])
                self.assertTrue(all(p['digital']['OUT'] == [1] for p in page['points']))
        result = circuit_analyze(self.rt, {'spec': design(), 'analysis': 'tr',
            'tr_step': 1e-6, 'tr_stop': 1e-5, 'digital_steps_per_tr_step': 3})
        self.assertEqual(result['measurements']['transient']['digital_propagation']['completed_propagation_steps'], 30)

    def test_bad_counts_or_wrong_analysis_are_rejected(self):
        for count in (0, 65, True, 1.0, '3', None):
            with self.subTest(count=count), self.assertRaises(ToolError):
                self.run_trace(digital_steps_per_tr_step=count)
        with self.assertRaisesRegex(ToolError, 'only to analysis=tr'):
            self.run_trace(analysis='op', digital_steps_per_tr_step=3)

    def test_plsav_roundtrip_uses_the_same_propagated_TR_path(self):
        created = circuit_create(self.rt, {'spec': design()})
        result = circuit_analyze(self.rt, {'path': created['sav_path'], 'analysis': 'tr',
            'tr_step': 1e-6, 'tr_stop': 1e-5, 'tr_sample_every': 1})
        page = circuit_read_trace(self.rt, {'path': result['state_path'], 'component_ids': ['OUT']})
        self.assertTrue(all(p['digital']['OUT'] == [1] for p in page['points']))
        self.assertTrue(page['digital_propagation']['verified_per_step'])

    def test_disjoint_and_coupled_analog_digital_TR_are_both_supported(self):
        spec = design()
        spec['components'] += [
            {'id': 'V', 'type': 'vdc', 'nodes': ['v', 'gnd'], 'params': {'v': 5}},
            {'id': 'R', 'type': 'resistor', 'nodes': ['v', 'gnd'], 'params': {'r': 10}},
        ]
        result = self.run_trace(spec=spec)
        digital = circuit_read_trace(self.rt, {'path': result['state_path'], 'component_ids': ['OUT']})
        analog = circuit_read_trace(self.rt, {'path': result['state_path'], 'nodes': ['v']})
        self.assertTrue(all(p['digital']['OUT'] == [1] for p in digital['points']))
        self.assertTrue(all(p[1] == 5 for p in analog['points']))
        self.assertNotIn('available_nodes', analog)
        self.assertGreaterEqual(analog['available_node_count'], 1)
        # Couple a native digital input to an analog load without adding a
        # redundant ideal VDC on that node. The digital output must propagate
        # through NOT while the mixed node is solved at the input's saved Hl.
        spec['components'][0]['nodes'] = ['mixed']
        spec['components'][1]['nodes'][0] = 'mixed'
        spec['components'].append(
            {'id': 'MIXLOAD', 'type': 'resistor', 'nodes': ['mixed', 'gnd'], 'params': {'r': 10}})
        coupled = self.run_trace(spec=spec)
        mixed_voltage = circuit_read_trace(
            self.rt, {'path': coupled['state_path'], 'nodes': ['mixed']})
        mixed_digital = circuit_read_trace(
            self.rt, {'path': coupled['state_path'], 'component_ids': ['OUT']})
        self.assertTrue(all(abs(point[1] - 0.0) < 1e-12 for point in mixed_voltage['points']))
        self.assertTrue(all(point['digital']['OUT'] == [1] for point in mixed_digital['points']))
        self.assertIn('mixed_signal_scope', coupled['measurements'])

    def test_native_cpp_step_edge_delay_and_mixed_boundary_contracts(self):
        compiler = shutil.which('clang++-21') or shutil.which('clang++-22')
        if not compiler:
            self.skipTest('Clang 21/22 required for native C++ contract harness')
        repo = Path(__file__).resolve().parents[1]
        source_root = Path(os.environ.get('AUREX_PHY_ENGINE_SOURCE', str(repo / 'third-parties/Phy-Engine')))
        executable = Path(self.temp.name) / 'native-tr-contracts'
        compile_result = subprocess.run([compiler, '-std=c++20', '-O0', '-Wno-braced-scalar-init',
            '-I' + str(source_root / 'include'), str(repo / 'tests/native/test_transient_digital.cpp'),
            '-ldl', '-o', str(executable)], capture_output=True, text=True, timeout=90)
        self.assertEqual(compile_result.returncode, 0, compile_result.stdout + compile_result.stderr)
        run = subprocess.run([str(executable), str(Path(os.environ['AUREX_PHY_ENGINE_BUILD']).resolve() / 'libphyengine.so')],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
