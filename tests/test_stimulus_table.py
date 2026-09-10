"""Compact stimulus is only a syntax change; no model or community calls."""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import jsonschema

from aurex.config import AurexConfig
from aurex.tools.circuits import _expand_stimulus_table, circuit_analyze, circuit_read_stimulus, normalize_spec, register_circuit_tools
from aurex.tools.registry import ToolError, ToolRegistry, ToolRuntime


A, B, KEEP = '0dbcaca2ef8f4c638962d809cae564a4', '64c5b605551651dfd94c1017-input-b', 'unselected-input'


def design():
    return {'components': [
        {'id': A, 'type': 'digital_input', 'nodes': ['a'], 'params': {'state': 0}, 'label': 'B-display-only'},
        {'id': B, 'type': 'digital_input', 'nodes': ['b'], 'params': {'state': 0}, 'label': 'A-display-only'},
        {'id': KEEP, 'type': 'digital_input', 'nodes': ['held'], 'params': {'state': 1}},
        {'id': 'AND', 'type': 'digital_and', 'nodes': ['a', 'b', 'q']},
        {'id': 'OUT', 'type': 'digital_output', 'nodes': ['q']},
        {'id': 'KEEP-OUT', 'type': 'digital_output', 'nodes': ['held']},
    ]}


class StopBeforeNative(Exception):
    pass


class StimulusTableTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.runtime = ToolRuntime('stimulus-test', 'zh', str(self.folder / 'cfg.json'), AurexConfig(), str(self.folder))
        registry = ToolRegistry()
        register_circuit_tools(registry)
        self.schema = registry.get('circuit_analyze').parameters
        self.reader_schema = registry.get('circuit_read_stimulus').parameters

    def captured(self, args):
        saved = []
        def capture(runtime, payload):
            saved.append(copy.deepcopy(payload['spec']))
            raise StopBeforeNative()
        with patch('aurex.tools.circuits.pe_simulate', side_effect=capture):
            with self.assertRaises(StopBeforeNative):
                circuit_analyze(self.runtime, args)
        return saved[0]

    def test_exact_column_order_four_states_and_unselected_state_are_preserved(self):
        spec = normalize_spec(design())
        table = {'inputs': [B, A], 'vectors': [[0, 1], [2, 3], [1, 0]]}
        before = copy.deepcopy((spec, table))
        expected = [{'set': {B: 0, A: 1}}, {'set': {B: 2, A: 3}}, {'set': {B: 1, A: 0}}]
        self.assertEqual(_expand_stimulus_table(spec, table), expected)
        self.assertEqual((spec, table), before)
        args = {'spec': design(), 'stimulus_table': table, 'digital_clock_ticks': 7, 'analysis': 'tr',
                'tr_step': 3e-8, 'tr_stop': 9e-8}
        original = copy.deepcopy(args)
        actual = self.captured(args)
        self.assertEqual(actual['stimulus'], expected)
        self.assertEqual(list(actual['stimulus'][0]['set']), [B, A])
        self.assertEqual(actual['digital_clock_ticks'], 7)
        self.assertEqual((actual['tr_step'], actual['tr_stop']), (3e-8, 9e-8))
        self.assertEqual(next(c for c in actual['components'] if c['id'] == KEEP)['params']['state'], 1)
        self.assertTrue(all(KEEP not in frame['set'] for frame in actual['stimulus']))
        self.assertNotIn('stimulus_table', actual)
        self.assertEqual(args, original)

    def test_unknown_output_internal_gate_and_label_are_rejected_before_simulation(self):
        for cid in ('unknown', 'OUT', 'AND', 'A-display-only'):
            with self.subTest(cid=cid), patch('aurex.tools.circuits.pe_simulate') as solve:
                with self.assertRaises(ToolError):
                    circuit_analyze(self.runtime, {'spec': design(), 'stimulus_table': {'inputs': [cid], 'vectors': [[1]]}})
                solve.assert_not_called()
        with self.assertRaisesRegex(ToolError, 'not digital_input'):
            _expand_stimulus_table(normalize_spec(design()), {'inputs': ['OUT'], 'vectors': [[1]]})

    def test_table_values_are_strict_integers_not_boolean_float_or_coercible_text(self):
        for state in (True, False, 1.0, '1', None, -1, 4, float('nan'), {}, []):
            with self.subTest(state=state), self.assertRaisesRegex(ToolError, 'vectors\[0\]\[0\]'):
                _expand_stimulus_table(normalize_spec(design()), {'inputs': [A], 'vectors': [[state]]})

    def test_shape_unique_ids_and_existing_frame_bound_are_checked_without_padding(self):
        invalid = [None, {}, {'inputs': [A], 'vectors': [[1]], 'clock': 'guess'},
            {'inputs': [], 'vectors': [[]]}, {'inputs': [A, A], 'vectors': [[0, 1]]},
            {'inputs': [True], 'vectors': [[0]]}, {'inputs': [A], 'vectors': []},
            {'inputs': [A, B], 'vectors': [[1]]}, {'inputs': [A], 'vectors': [[0, 1]]},
            {'inputs': [A], 'vectors': [None]}, {'inputs': [A], 'vectors': [[0]] * 129}]
        for table in invalid:
            with self.subTest(table=table), self.assertRaises(ToolError):
                _expand_stimulus_table(normalize_spec(design()), table)
        self.assertEqual(len(_expand_stimulus_table(normalize_spec(design()), {'inputs': [A], 'vectors': [[0]] * 128})), 128)

    def test_both_explicit_formats_and_nested_table_are_not_silently_chosen(self):
        table = {'inputs': [A], 'vectors': [[1]]}
        for args in ({'spec': design(), 'stimulus': [], 'stimulus_table': table},
                     {'spec': {**design(), 'stimulus': []}, 'stimulus_table': table},
                     {'spec': {**design(), 'stimulus_table': table}},
                     {'spec': {**design(), 'stimulus_table': table}, 'stimulus_table': table}):
            with self.subTest(args=args), patch('aurex.tools.circuits.pe_simulate') as solve:
                with self.assertRaises(ToolError):
                    circuit_analyze(self.runtime, args)
                solve.assert_not_called()

    def test_explicit_new_table_overrides_only_path_loaded_old_sequence(self):
        source = self.folder / 'original.circuit.json'
        saved = normalize_spec({**design(), 'stimulus': [{'set': {A: 0, KEEP: 0}}]})
        source.write_text(json.dumps(saved))
        before = source.read_bytes()
        actual = self.captured({'path': str(source), 'stimulus_table': {'inputs': [A], 'vectors': [[1]]}})
        self.assertEqual(actual['stimulus'], [{'set': {A: 1}}])
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(next(c for c in actual['components'] if c['id'] == KEEP)['params']['state'], 1)

    def test_legacy_sequence_and_hold_frames_are_unchanged(self):
        legacy = [{'set': {A: 0}}, {}, {'set': {}}, {'set': {B: 1}}, {'set': {A: 2}}]
        args = {'spec': design(), 'stimulus': legacy}
        jsonschema.validate(args, self.schema)
        self.assertEqual(self.captured(args)['stimulus'], legacy)
        self.assertNotIn('stimulus', self.captured({'spec': design()}))

    def test_json_schema_exposes_typed_set_values_and_rejects_top_level_conflict(self):
        props = self.schema['properties']
        self.assertEqual(props['stimulus']['items']['properties']['set']['additionalProperties'],
                         {'type': 'integer', 'enum': [0, 1, 2, 3]})
        self.assertIn('one digital tick/settle', props['stimulus']['description'])
        self.assertIn('without advancing physical time', props['stimulus']['description'])
        self.assertIn('one digital tick/settle', props['stimulus_table']['description'])
        self.assertIn('without advancing physical time', props['stimulus_table']['description'])
        self.assertNotIn('10ns/frame', props['stimulus']['description'])
        self.assertNotIn('10ns/frame', props['stimulus_table']['description'])
        table = {'inputs': [A, B], 'vectors': [[0, 1], [2, 3]]}
        jsonschema.validate({'stimulus_table': table}, self.schema)
        for invalid in ({'stimulus': [{'set': {A: True}}]}, {'stimulus': [{'set': {A: '1'}}]},
                        {'stimulus': [{A: 1}]}, {'stimulus': [], 'stimulus_table': table}):
            with self.subTest(invalid=invalid), self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(invalid, self.schema)

    def test_stimulus_reader_can_keep_an_ordinary_truth_table_in_one_call(self):
        component_ids = [f'component-{index}' for index in range(11)]
        jsonschema.validate({'path': '/tmp/state.pe-state.json', 'component_ids': component_ids,
                             'offset': 0, 'limit': 8}, self.reader_schema)
        self.assertEqual(self.reader_schema['properties']['component_ids']['maxItems'], 24)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({'path': '/tmp/state.pe-state.json',
                                 'component_ids': [f'component-{index}' for index in range(25)]},
                                self.reader_schema)

    def test_reader_keeps_default_small_and_emits_exact_single_call_logic_summary(self):
        input_refs = ('C1', 'C3', 'C2')
        output_refs = ('C9', 'C10', 'C13', 'C16', 'C19', 'C24', 'C25', 'C26')
        components = [
            {'id': 'input-' + ref, 'type': 'digital_input', 'nodes': ['n-' + ref],
             'params': {'state': 0}, 'pl_source': {'source_ref': ref}}
            for ref in input_refs
        ] + [
            {'id': 'output-' + ref, 'type': 'digital_output', 'nodes': ['n-' + ref],
             'pl_source': {'source_ref': ref}}
            for ref in output_refs
        ]
        expected_outputs = ('C9', 'C10', 'C13', 'C16', 'C19', 'C24', 'C25', 'C26')
        rows = []
        for step, output_ref in enumerate(expected_outputs):
            digital = {component['id']: [0] for component in components}
            for bit, ref in enumerate(input_refs):
                digital['input-' + ref] = [(step >> (2 - bit)) & 1]
            digital['output-' + output_ref] = [1]
            row = {'step': step, 'digital': digital,
                   'inputs': {'input-' + ref: digital['input-' + ref][0] for ref in input_refs}}
            if step != 1:  # A missing marker is unknown, not an implicit failure.
                row['digital_settled'] = step != 2
            rows.append(row)
        # Missing is distinct from L, while X and Z retain their exact states.
        exceptional = {component['id']: [0] for component in components}
        exceptional['input-C1'] = [2]
        exceptional['input-C3'] = [3]
        exceptional['output-C9'] = [2]
        exceptional['output-C10'] = [3]
        exceptional.pop('input-C2')
        exceptional.pop('output-C13')
        rows.append({'step': 8, 'digital': exceptional, 'inputs': {}, 'digital_settled': True})
        state = self.folder / 'truth.pe-state.json'
        state.write_text(json.dumps({'schema': 'aurex.pe-state.v1', 'spec': {'components': components},
                                     'scene': {}, 'measurements': {'stimulus_results': rows}}))

        default = circuit_read_stimulus(self.runtime, {'path': str(state), 'limit': 1})
        self.assertEqual(len(default['components']), 8)
        self.assertTrue(default['logic_summary']['all_rows_settled'])
        self.assertEqual(default['steps'][0]['digital_settled'], True)
        first_two = circuit_read_stimulus(self.runtime, {'path': str(state), 'limit': 2})
        self.assertIsNone(first_two['logic_summary']['all_rows_settled'])
        self.assertEqual([row['digital_settled'] for row in first_two['steps']], [True, None])
        selected = [component['id'] for component in components]
        result = circuit_read_stimulus(self.runtime, {'path': str(state),
                                                       'component_ids': selected, 'limit': 9})
        self.assertEqual([component['id'] for component in result['components']], selected)
        self.assertEqual([component['column'] for component in result['components']], list(range(11)))
        self.assertEqual([component['source_ref'] for component in result['components']],
                         list(input_refs + output_refs))
        summary = result['logic_summary']
        self.assertFalse(summary['all_rows_settled'])
        self.assertEqual(summary['settlement_scope'], 'returned_rows_only')
        self.assertEqual([row['digital_settled'] for row in summary['rows'][:3]],
                         [True, None, False])
        self.assertNotIn('after each stored stimulus step settled', summary['scope'])
        self.assertEqual([column['source_ref'] for column in summary['input_columns']], list(input_refs))
        self.assertEqual([column['source_ref'] for column in summary['output_columns']], list(output_refs))
        self.assertEqual(summary['rows'][5]['input_vector'], [1, 0, 1])
        self.assertEqual(summary['rows'][5]['output_vector'], [0, 0, 0, 0, 0, 1, 0, 0])
        self.assertEqual(summary['rows'][5]['high_outputs'],
                         [{'id': 'output-C24', 'source_ref': 'C24'}])
        self.assertEqual(summary['rows'][6]['input_vector'], [1, 1, 0])
        self.assertEqual(summary['rows'][6]['high_outputs'],
                         [{'id': 'output-C25', 'source_ref': 'C25'}])
        last = summary['rows'][8]
        self.assertEqual(last['input_vector'], [2, 3, None])
        self.assertEqual(last['output_vector'][:4], [2, 3, None, 0])
        self.assertEqual({item['state'] for item in last['unknown_or_high_impedance']}, {2, 3})
        self.assertEqual(set(last['missing_component_ids']), {'input-C2', 'output-C13'})
        self.assertEqual(result['steps'][5]['digital'], rows[5]['digital'])

    def test_mixed_circuit_keeps_existing_digital_only_constraint(self):
        spec = design()
        spec['components'].append({'id': 'R', 'type': 'resistor', 'nodes': ['a', 'gnd'], 'params': {'r': 10}})
        with patch('aurex.tools.circuits.pe_simulate') as solve, self.assertRaisesRegex(ToolError, 'digital-only'):
            circuit_analyze(self.runtime, {'spec': spec, 'stimulus_table': {'inputs': [A], 'vectors': [[1]]}})
        solve.assert_not_called()

    def test_fixture_encoding_is_smaller_without_claiming_model_or_runtime_speed(self):
        ids = [hashlib.sha256(str(i).encode()).hexdigest()[:32] for i in range(8)]
        vectors = [[(row >> bit) & 1 for bit in range(8)] for row in range(16)]
        legacy = {'stimulus': [{'set': dict(zip(ids, row))} for row in vectors]}
        compact = {'stimulus_table': {'inputs': ids, 'vectors': vectors}}
        size = lambda value: len(json.dumps(value, separators=(',', ':')).encode())
        self.assertLess(size(compact), size(legacy) / 4)
        self.assertEqual([dict(zip(ids, row)) for row in compact['stimulus_table']['vectors']], [row['set'] for row in legacy['stimulus']])


@unittest.skipUnless(os.environ.get('AUREX_PHY_ENGINE_BUILD'), 'set private native build path for real solver equivalence')
class StimulusTableNativeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        build = Path(os.environ['AUREX_PHY_ENGINE_BUILD'])
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(build),
            verilog2plsav_path=str(build / 'verilog2plsav'), phyengine_lib_path=str(build / 'libphyengine.so'), run_timeout_sec=30))
        self.runtime = ToolRuntime('table-native', 'zh', str(self.folder / 'cfg.json'), cfg, str(self.folder))
        network = patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden'))
        network.start()
        self.addCleanup(network.stop)

    def test_actual_native_sequence_matches_legacy_and_only_archives_expanded_uuid_maps(self):
        source = self.folder / 'original.circuit.json'
        source.write_text(json.dumps(normalize_spec(design())))
        original = source.read_bytes()
        table = {'inputs': [B, A], 'vectors': [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [3, 1]]}
        common = {'path': str(source), 'analysis': 'tr', 'tr_step': 1e-8, 'tr_stop': 1e-8, 'digital_clock_ticks': 1}
        compact = circuit_analyze(self.runtime, {**common, 'stimulus_table': table})
        legacy = circuit_analyze(self.runtime, {**common, 'stimulus': _expand_stimulus_table(normalize_spec(design()), table)})
        a = json.loads(Path(compact['state_path']).read_text())
        b = json.loads(Path(legacy['state_path']).read_text())
        self.assertEqual(a['spec'], b['spec'])
        self.assertEqual(a['measurements']['stimulus_results'], b['measurements']['stimulus_results'])
        steps = a['measurements']['stimulus_results']
        self.assertEqual([row['step'] for row in steps], list(range(6)))
        self.assertEqual([row['digital']['OUT'][0] for row in steps[:5]], [0, 0, 0, 1, 2])
        self.assertTrue(all(row['digital'][KEEP] == [1] and row['digital']['KEEP-OUT'] == [1] for row in steps))
        self.assertEqual(source.read_bytes(), original)
        self.assertNotIn('stimulus_results', compact['measurements'])
        self.assertEqual(compact['measurements']['stimulus_scope']['total_steps'], 6)
        self.assertEqual(compact['measurements']['stimulus_scope']['shown_steps'], 0)
        self.assertEqual(compact['stimulus_input_format']['native_frame_step_s'], 1e-8)
        self.assertEqual(compact['stimulus_input_format']['recorded_frame_count'], len(steps))
        self.assertNotIn(A, json.dumps(compact['stimulus_input_format']))
        self.assertEqual(compact['images'], [])
        page = circuit_read_stimulus(self.runtime, {'path': compact['state_path'], 'component_ids': ['OUT', KEEP], 'offset': 3, 'limit': 2})
        self.assertIs(page['recorded_not_resimulated'], True)
        self.assertEqual([row['step'] for row in page['steps']], [3, 4])
        self.assertEqual([row['digital']['OUT'][0] for row in page['steps']], [1, 2])


if __name__ == '__main__':
    unittest.main()
