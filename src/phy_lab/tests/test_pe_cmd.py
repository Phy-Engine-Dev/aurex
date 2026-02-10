import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from pe_cmd import PEScriptError, parse_pe_script_to_spec_obj


class TestPEScript(unittest.TestCase):
    def test_parse_add_and_probe(self):
        script = "\n".join(
            [
                "ANALYSIS dc",
                "ADD V1 vdc vin gnd v=5",
                "ADD R1 r vin vout r=1000",
                "ADD R2 r vout gnd r=2000",
                "PROBE NODE vout",
                "PROBE I R1",
                "RUN",
            ]
        )
        obj = parse_pe_script_to_spec_obj(script, max_components=10, max_probes=10)
        self.assertEqual(obj["analysis"]["type"], "dc")
        self.assertEqual(len(obj["components"]), 3)
        self.assertEqual(obj["components"][0]["id"], "R1")  # sorted by id
        self.assertEqual(obj["components"][2]["id"], "V1")
        self.assertEqual(len(obj["probes"]), 2)
        self.assertEqual(obj["probes"][0]["kind"], "node_voltage")

    def test_wire_overrides_component_nodes(self):
        script = "\n".join(
            [
                "ADD V1 vdc x y v=5",
                "WIRE vin V1.0",
                "WIRE gnd V1.1",
            ]
        )
        obj = parse_pe_script_to_spec_obj(script)
        comps = {c["id"]: c for c in obj["components"]}
        self.assertEqual(comps["V1"]["nodes"], ["vin", "gnd"])

    def test_rejects_unknown_command(self):
        with self.assertRaises(PEScriptError):
            parse_pe_script_to_spec_obj("HACK rm -rf /")

    def test_parses_engineering_suffixes(self):
        script = "\n".join(
            [
                "ANALYSIS tr",
                "SET TR 1ms 10ms",
                "ADD V1 vdc vin gnd v=5V",
                "ADD R1 r vin n1 r=1k",
                "ADD C1 c n1 gnd c=100nF",
                "PROBE NODE n1",
            ]
        )
        obj = parse_pe_script_to_spec_obj(script, max_components=10, max_probes=10)
        self.assertEqual(obj["analysis"]["type"], "tr")
        self.assertAlmostEqual(obj["analysis"]["tr_t_step_s"], 1e-3)
        self.assertAlmostEqual(obj["analysis"]["tr_t_stop_s"], 1e-2)
        comps = {c["id"]: c for c in obj["components"]}
        self.assertAlmostEqual(comps["R1"]["params"]["r_ohm"], 1000.0)
        self.assertAlmostEqual(comps["C1"]["params"]["c_f"], 100e-9)

    def test_parse_digital_clk_ticks_and_digital_probe(self):
        script = "\n".join(
            [
                "ANALYSIS dc",
                "SET DIGITAL_CLK_TICKS 3",
                "ADD IN1 digital_input a state=1",
                "ADD N1 digital_not a y",
                "ADD OUT1 digital_output y",
                "PROBE DNODE y",
            ]
        )
        obj = parse_pe_script_to_spec_obj(script, max_components=10, max_probes=10)
        self.assertEqual(obj["analysis"]["digital_clk_ticks"], 3)
        comps = {c["id"]: c for c in obj["components"]}
        self.assertEqual(comps["IN1"]["nodes"], ["a"])
        self.assertEqual(comps["IN1"]["params"]["state"], 1.0)
        self.assertEqual(comps["N1"]["nodes"], ["a", "y"])
        self.assertEqual(comps["OUT1"]["nodes"], ["y"])
        self.assertEqual(obj["probes"][0]["kind"], "node_digital")
        self.assertEqual(obj["probes"][0]["target"], "y")

    def test_parse_multi_pin_and_wire_override(self):
        script = "\n".join(
            [
                "ADD V1 vdc n1 gnd v=5",
                "ADD G1 vccs x0 x1 x2 x3 g=0.001",
                "WIRE n2 G1.0",
                "WIRE gnd G1.1 G1.3",
                "WIRE n1 G1.2",
            ]
        )
        obj = parse_pe_script_to_spec_obj(script, max_components=10, max_probes=10)
        comps = {c["id"]: c for c in obj["components"]}
        self.assertEqual(comps["G1"]["nodes"], ["n2", "gnd", "n1", "gnd"])
        self.assertAlmostEqual(comps["G1"]["params"]["g"], 0.001)
