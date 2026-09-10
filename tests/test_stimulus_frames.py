"""Native command boundary: exactly one sequential advance per input frame."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

from aurex.config import AurexConfig
from aurex.tools.phy_engine import pe_simulate
from aurex.tools.registry import ToolRuntime


@unittest.skipUnless(os.environ.get('AUREX_PHY_ENGINE_BUILD'), 'native build not configured')
class StimulusFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        build = Path(os.environ['AUREX_PHY_ENGINE_BUILD'])
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(build),
            verilog2plsav_path=str(build / 'verilog2plsav'), phyengine_lib_path=str(build / 'libphyengine.so')))
        self.runtime = ToolRuntime('stimulus-regression', 'zh', str(Path(self.temp.name) / 'config.json'), cfg, self.temp.name)

    def test_dff_samples_edges_once_and_holds_between_edges(self):
        vectors = [{'set': {'D': d, 'CLK': clk}} for d, clk in
                   [(1, 0), (1, 1), (0, 1), (0, 0), (0, 1), (1, 1)]]
        result = pe_simulate(self.runtime, {'spec': {'analysis': 'dc', 'components': [
            {'id': 'D', 'type': 'digital_input', 'nodes': ['d'], 'params': {'state': 0}},
            {'id': 'CLK', 'type': 'digital_input', 'nodes': ['clk'], 'params': {'state': 0}},
            {'id': 'FF', 'type': 'digital_dff', 'nodes': ['d', 'clk', 'q']},
            {'id': 'Q', 'type': 'digital_output', 'nodes': ['q']},
        ], 'stimulus': vectors}})
        frames = result['stimulus_results']
        # PE DFF starts at zero; held-high clock is not an extra edge, even
        # when D changes within the next logical frame.
        self.assertEqual([frame['digital']['Q'][0] for frame in frames], [0, 1, 1, 1, 0, 0])
        self.assertTrue(all(frame['digital_settled'] for frame in frames))
        self.assertEqual(result['digital_settle']['attempted_ticks'], 1 + len(vectors))
        self.assertEqual(result['digital_settle']['settled_ticks'], 1 + len(vectors))
        self.assertEqual(result['stimulus_semantics']['digital_ticks_per_frame'], 1)
        self.assertFalse(result['stimulus_semantics']['physical_time_advanced'])

    def test_combinational_truth_table_settles_in_same_frame(self):
        result = pe_simulate(self.runtime, {'spec': {'analysis': 'dc', 'components': [
            {'id': 'A', 'type': 'digital_input', 'nodes': ['a'], 'params': {'state': 0}},
            {'id': 'B', 'type': 'digital_input', 'nodes': ['b'], 'params': {'state': 0}},
            {'id': 'G', 'type': 'digital_and', 'nodes': ['a', 'b', 'q']},
            {'id': 'Q', 'type': 'digital_output', 'nodes': ['q']},
        ], 'stimulus': [{'set': {'A': a, 'B': b}} for a, b in [(0,0),(0,1),(1,0),(1,1),(1,2)]]}})
        self.assertEqual([frame['digital']['Q'][0] for frame in result['stimulus_results']], [0,0,0,1,2])
        self.assertEqual(result['digital_settle']['attempted_ticks'], 6)
