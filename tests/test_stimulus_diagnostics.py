"""Actionable errors for long digital tasks; never repair identities silently."""
import copy
import unittest
from unittest.mock import patch

from aurex.tools.phy_engine import _simulate_spec
from aurex.tools.registry import ToolError


class StimulusDiagnosticsTests(unittest.TestCase):
    def spec(self):
        return {'components': [
            {'id': '0dbcaca2ef8f4c638962d809cae564a4', 'type': 'digital_input', 'nodes': ['a'], 'params': {'state': 0}},
            {'id': 'output', 'type': 'digital_output', 'nodes': ['a'], 'params': {}},
        ]}

    def fail_before_native(self, spec):
        original = copy.deepcopy(spec)
        with patch('aurex.tools.phy_engine.load_library', side_effect=AssertionError('Native must not be loaded')) as load:
            with self.assertRaises(ToolError) as ctx:
                _simulate_spec(spec, '/does/not/exist.so')
            load.assert_not_called()
        self.assertEqual(spec, original)
        return str(ctx.exception)

    def test_real_one_character_typo_identifies_exact_candidate_and_frame(self):
        wrong = '0dbcaca2ef8f4c638962d09cae564a4'
        spec = self.spec()
        spec['stimulus'] = [{'set': {}}, {'set': {wrong: 1}}]
        error = self.fail_before_native(spec)
        self.assertIn('stimulus[1].set', error)
        self.assertIn(wrong, error)
        self.assertIn(spec['components'][0]['id'], error)
        self.assertIn('No ID was automatically substituted', error)

    def test_output_id_explains_wrong_component_type(self):
        spec = {**self.spec(), 'stimulus': [{'set': {'output': 1}}]}
        error = self.fail_before_native(spec)
        self.assertIn('output', error)
        self.assertIn('digital_output', error)
        self.assertIn('not digital_input', error)

    def test_invalid_state_identifies_exact_input(self):
        for value in ('1', True, 7, None):
            spec = self.spec()
            cid = spec['components'][0]['id']
            spec['stimulus'] = [{'set': {cid: value}}]
            with self.subTest(value=value):
                error = self.fail_before_native(spec)
                self.assertIn(cid, error)
                self.assertIn('state must be', error)

    def test_invalid_vector_points_to_frame(self):
        for value in (None, 'input=1', {'set': []}):
            spec = {**self.spec(), 'stimulus': [{}, value]}
            with self.subTest(value=value):
                self.assertIn('stimulus[1]', self.fail_before_native(spec))


if __name__ == '__main__':
    unittest.main()
