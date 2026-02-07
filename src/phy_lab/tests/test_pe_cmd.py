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

