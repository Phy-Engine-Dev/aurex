import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from pe_builder import build_circuit, parse_pe_sim_spec  # type: ignore


class TestPEBuilder(unittest.TestCase):
    def test_parse_requires_ground_and_source(self):
        obj = {
            "analysis": {"type": "dc", "ac_omega_rad_s": None, "tr_t_step_s": None, "tr_t_stop_s": None},
            "components": [
                {"id": "R1", "type": "resistor", "nodes": ["n1", "n2"], "params": {"r_ohm": 10}},
            ],
            "probes": [],
        }
        with self.assertRaises(Exception):
            parse_pe_sim_spec(obj, max_components=10, max_probes=10)

    def test_build_simple_series(self):
        obj = {
            "analysis": {"type": "dc", "ac_omega_rad_s": None, "tr_t_step_s": None, "tr_t_stop_s": None},
            "components": [
                {"id": "V1", "type": "vdc", "nodes": ["n1", "gnd"], "params": {"v_v": 5}},
                {"id": "R1", "type": "resistor", "nodes": ["n1", "n2"], "params": {"r_ohm": 4}},
                {"id": "R2", "type": "resistor", "nodes": ["n2", "gnd"], "params": {"r_ohm": 4}},
            ],
            "probes": [{"kind": "node_voltage", "target": "n2"}],
        }
        spec = parse_pe_sim_spec(obj, max_components=10, max_probes=10)
        built = build_circuit(spec)
        self.assertEqual(built.element_codes[0], 0)
        self.assertIn("V1", built.element_index_by_id)
        self.assertIn("R2", built.element_index_by_id)
        # wires are quads
        self.assertEqual(len(built.wires) % 4, 0)
        # must include a non-trivial gnd connection
        self.assertIn("gnd", built.node_to_pin)


if __name__ == "__main__":
    unittest.main()
