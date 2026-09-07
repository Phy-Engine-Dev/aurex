import copy
import json
import math
import os
import subprocess
import sys
import unittest

from aurex.tools.plar_power_import import import_element
from aurex.tools.registry import ToolError


def fixture(kind="Transformer", props=None):
    if props is None:
        props = {"输入电压": 200, "输出电压": 7, "额定功率": 20, "耦合系数": 1}
    el = {"id": "original", "type": kind, "properties": props, "statistics": {"电压1": 156.72628784179688},
          "pins": [{"pin": p, "node": "N" + str(p)} for p in (2, 0, 3, 1)],
          "position": [.1, .2, .3], "rotation": [1, 2, 3], "label": "变压器"}
    return el, {"components": [el], "wires": [{"Source": "original", "SourcePin": 0, "Target": "other", "TargetPin": 0}]}


class PowerImportTests(unittest.TestCase):
    def test_unknown_untouched(self):
        self.assertIsNone(import_element({"type": "Basic Capacitor"}, scene={}))
        self.assertIsNone(import_element({"type": "Rectifier"}, scene={}))  # owned by semiconductor importer

    def test_transformer_ratio_and_original_pair_order(self):
        el, scene = fixture()
        result = import_element(el, scene=scene)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["nodes"], ["N0", "N1", "N2", "N3"])
        self.assertEqual(result[0]["params"], {"ratio": 200 / 7})
        self.assertEqual(result[0]["type"], "transformer")

    def test_no_nominal_source_inserted_and_rating_retained_not_clamped(self):
        el, scene = fixture()
        result = import_element(el, scene=scene)[0]
        self.assertEqual(result["pl_source"]["raw_properties"]["额定功率"], 20)
        self.assertNotIn("v", result["params"])
        self.assertTrue(any("not silently converted" in s for s in result["pl_source"]["assumptions"]))

    def test_mutual_inductors_si_and_coupling(self):
        for k in (0, 0.63, 1):
            el, scene = fixture("Mutual Inductor", {"电感1": 4, "电感2": .001, "耦合系数": k})
            result = import_element(el, scene=scene)[0]
            self.assertEqual(result["type"], "coupled_inductors")
            self.assertEqual(result["params"], {"l1": 4, "l2": .001, "k": k})

    def test_no_source_mutation_and_full_metadata(self):
        el, scene = fixture()
        before = copy.deepcopy(scene)
        result = import_element(el, scene=scene)[0]
        for key in ("id", "position", "rotation", "label"):
            self.assertEqual(result[key], el[key])
        source = result["pl_source"]
        self.assertEqual(source["parent_identifier"], el["id"])
        self.assertFalse(source["numerical_equivalence_to_original"])
        self.assertFalse(source["is_helper"])
        self.assertEqual(source["raw_statistics"], el["statistics"])
        result["position"][0] = 77
        source["raw_properties"]["额定功率"] = 44
        self.assertEqual(scene, before)

    def test_unwired_pin_retains_original_floating_node(self):
        el, scene = fixture()
        result = import_element(el, scene=scene)[0]
        self.assertEqual(result["pl_source"]["implicit_references"], [])
        self.assertEqual([p["pl_pin"] for p in result["pl_source"]["pin_mapping"] if p["externally_wired"]], [0])
        self.assertNotIn("gnd", result["nodes"])

    def test_missing_or_invalid_parameters_not_defaulted(self):
        for key in ("输入电压", "输出电压", "额定功率", "耦合系数"):
            for value in (None, "1", True, math.inf, math.nan, 10**1000):
                el, scene = fixture()
                el["properties"][key] = value
                with self.subTest(key=key, value=str(value)[:20]), self.assertRaises(ToolError):
                    import_element(el, scene=scene)

    def test_transformer_nonideal_coupling_uses_explicit_inductance_normalization(self):
        for value in (0, .63, .999):
            el, scene = fixture()
            el["properties"]["耦合系数"] = value
            imported = import_element(el, scene=scene)[0]
            self.assertEqual(imported["type"], "coupled_inductors")
            self.assertEqual(imported["params"]["k"], value)
            self.assertEqual(imported["params"]["l1"], 1)
            self.assertAlmostEqual(imported["params"]["l2"], (7 / 200) ** 2)
            self.assertIn("nonideal_transformer_normalization",
                          imported["pl_source"]["engineering_defaults"])
        for value in (-1, 2):
            el, scene = fixture()
            el["properties"]["耦合系数"] = value
            with self.assertRaises(ToolError):
                import_element(el, scene=scene)

    def test_invalid_ratios_and_inductances_fail(self):
        for key in ("输入电压", "输出电压", "额定功率"):
            for value in (0, -1):
                el, scene = fixture()
                el["properties"][key] = value
                with self.assertRaises(ToolError):
                    import_element(el, scene=scene)
        for l1, l2, k in ((0, 1, 1), (-1, 1, 1), (1, 1, -1), (1, 1, 1.01), (1e300, 1e300, 1), (1e-300, 1e-300, 1)):
            el, scene = fixture("Mutual Inductor", {"电感1": l1, "电感2": l2, "耦合系数": k})
            with self.assertRaises(ToolError):
                import_element(el, scene=scene)

    def test_winding_loss_fields_become_explicit_series_resistors(self):
        for field in ("内阻", "内阻1", "线圈电阻2", "初级电阻", "次级电阻"):
            el, scene = fixture()
            el["properties"][field] = .5
            result = import_element(el, scene=scene)
            helpers = [row for row in result if row["type"] == "resistor"]
            self.assertEqual(len(helpers), 1)
            self.assertEqual(helpers[0]["params"]["r"], .5)
            self.assertEqual(helpers[0]["pl_source"]["combined_saved_fields"], {field: .5})
            self.assertNotEqual(result[0]["nodes"], ["N0", "N1", "N2", "N3"])

    def test_broken_state_routes_to_complete_sav_damage_adapter(self):
        el, scene = fixture()
        el["IsBroken"] = True
        with self.assertRaisesRegex(ToolError, "broken-device"):
            import_element(el, scene=scene)

    def test_missing_pose_pin_duplicate_scene_and_malformed_wire_fail(self):
        for mutation in (lambda e, s: e.pop("position"), lambda e, s: e["pins"].pop(),
                         lambda e, s: e["pins"][0].update(pin=True),
                         lambda e, s: s["components"].append(copy.deepcopy(e)),
                         lambda e, s: s["wires"][0].update(SourcePin=20)):
            el, scene = fixture()
            mutation(el, scene)
            with self.assertRaises(ToolError):
                import_element(el, scene=scene)

    def test_center_tap_declares_full_secondary_convention_and_reorders_only_native_pins(self):
        el, scene = fixture()
        el["type"] = "Tapped Transformer"
        el["pins"].append({"pin": 4, "node": "center"})
        result = import_element(el, scene=scene)[0]
        self.assertEqual(result["type"], "transformer_center_tap")
        self.assertEqual(result["nodes"], ["N0", "N1", "N2", "center", "N3"])
        self.assertEqual(result["params"], {"ratio": 200 / 7})
        self.assertEqual(result["pl_source"]["primitive_pin_mapping"], [0, 1, 2, 4, 3])
        self.assertEqual([p["pl_pin"] for p in result["pl_source"]["pin_mapping"]], [0, 1, 2, 3, 4])
        self.assertEqual(result["pl_source"]["engineering_defaults"]["output_voltage_definition"], "full_secondary_end_to_end")
        self.assertTrue(any("not independently verified" in s for s in result["pl_source"]["assumptions"]))

    def relay_fixture(self):
        el, scene = fixture("Relay Component", {"线圈电感": .000001, "线圈电阻": 20,
            "接通电流": .019999999552965164, "额定电流": 1, "开关": 0, "锁定": 1})
        el["pins"].append({"pin": 4, "node": "N4"})
        return el, scene

    def test_relay_preserves_five_pins_real_coil_and_explicit_assumptions(self):
        el, scene = self.relay_fixture()
        result = import_element(el, scene=scene)[0]
        self.assertEqual(result["type"], "relay_current_spdt")
        self.assertEqual(result["nodes"], ["N0", "N1", "N2", "N3", "N4"])
        self.assertEqual(result["params"]["l"], .000001)
        self.assertEqual(result["params"]["r"], 20)
        self.assertEqual(result["params"]["i_pull"], el["properties"]["接通电流"])
        self.assertEqual(result["params"]["i_drop"], .8 * el["properties"]["接通电流"])
        self.assertIn("engineering_defaults", result["pl_source"])
        self.assertFalse(result["pl_source"]["numerical_equivalence_to_original"])
        self.assertEqual(result["pl_source"]["raw_properties"], el["properties"])

    def test_relay_saved_contact_resistance_is_used_directly(self):
        el, scene = self.relay_fixture()
        el["properties"]["接触电阻"] = .125
        result = import_element(el, scene=scene)[0]
        self.assertEqual(result["params"]["r_on"], .125)
        self.assertEqual(result["pl_source"]["engineering_defaults"]
                         ["contact_closed_resistance_ohm"], .125)

    def test_relay_invalid_contracts_not_coerced(self):
        for key, value in (("线圈电感", -1), ("线圈电阻", 0), ("接通电流", 0),
                           ("额定电流", 0), ("开关", 2), ("开关", True), ("线圈电阻", "20")):
            el, scene = self.relay_fixture()
            el["properties"][key] = value
            with self.assertRaises(ToolError):
                import_element(el, scene=scene)

    def test_relay_does_not_reuse_four_pin_unloaded_family(self):
        el, scene = self.relay_fixture()
        el["pins"].pop()
        with self.assertRaisesRegex(ToolError, "all 5 original pins"):
            import_element(el, scene=scene)


@unittest.skipUnless(os.environ.get("AUREX_TEST_PHYENGINE_LIB"), "optional native fixture library not configured")
class NativePowerTests(unittest.TestCase):
    def solve(self, components, **settings):
        payload = {"spec": {"analysis": "dc", "components": components, **settings}, "lib_path": os.environ["AUREX_TEST_PHYENGINE_LIB"]}
        process = subprocess.run([sys.executable, "-m", "aurex.phy_engine.worker"], input=json.dumps(payload),
                                 text=True, capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def test_transformer_real_dc_ratio_current_and_power_conservation(self):
        el, scene = fixture()
        for pin in el["pins"]:
            if pin["pin"] in (1, 3):
                pin["node"] = "gnd"
        result = self.solve(import_element(el, scene=scene) + [
            {"id": "V", "type": "vdc", "nodes": ["N0", "gnd"], "params": {"v": 100}},
            {"id": "R", "type": "resistor", "nodes": ["N2", "gnd"], "params": {"r": 100}},
        ])
        rows = {r["id"]: r for r in result["components"]}
        self.assertAlmostEqual(rows["R"]["voltage"][0], 3.5, places=9)
        ip, isec = rows["original"]["current"]
        self.assertAlmostEqual(ip * 100 + isec * 3.5, 0, places=10)
        self.assertAlmostEqual(isec, -3.5 / 100, places=10)

    def test_mutual_real_ac_matches_independent_complex_z_equations(self):
        l1, l2, k, omega, load = .02, .01, .5, 314.1592653589793, 10
        el, scene = fixture("Mutual Inductor", {"电感1": l1, "电感2": l2, "耦合系数": k})
        for pin in el["pins"]:
            if pin["pin"] in (1, 3):
                pin["node"] = "gnd"
        result = self.solve(import_element(el, scene=scene) + [
            {"id": "V", "type": "vac", "nodes": ["N0", "gnd"], "params": {"vp": 1, "freq_hz": 50}},
            {"id": "R", "type": "resistor", "nodes": ["N2", "gnd"], "params": {"r": load}},
        ], analysis="ac", ac_omega=omega)
        rows = {r["id"]: r for r in result["components"]}
        vp = complex(rows["original"]["voltage"][0], rows["original"]["voltage_imag"][0])
        m = k * math.sqrt(l1 * l2)
        z1, z2, zm = 1j * omega * l1, 1j * omega * l2, 1j * omega * m
        expected_i1 = vp / (z1 - zm * zm / (load + z2))
        expected_i2 = -zm * expected_i1 / (load + z2)
        actual_i1 = complex(rows["original"]["current"][0], rows["original"]["current_imag"][0])
        actual_i2 = complex(rows["original"]["current"][1], rows["original"]["current_imag"][1])
        self.assertLess(abs(actual_i1 - expected_i1), 1e-9)
        self.assertLess(abs(actual_i2 - expected_i2), 1e-9)


if __name__ == "__main__":
    unittest.main()
