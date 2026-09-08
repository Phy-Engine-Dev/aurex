import copy
import math
import json
import os
import subprocess
import sys
import unittest

from aurex.tools.plar_passive_import import import_element
from aurex.tools.registry import ToolError


def fixture(kind, count, props, cid="original"):
    element = {"id": cid, "type": kind, "properties": props, "statistics": {"电流": 0.123},
               "pins": [{"pin": p, "node": f"N{p}"} for p in reversed(range(count))],
               "position": [0.4, 0.8, -0.1], "rotation": [20, 30, 40], "label": "原始标签"}
    # Deliberately wire only pin 0. Other schema pins must not become ground.
    scene = {"components": [element], "nodes": [],
             "wires": [{"Source": cid, "SourcePin": 0, "Target": "external", "TargetPin": 0}]}
    return element, scene


def connected_pairs(items):
    return {frozenset(c["nodes"]) for c in items if c["type"] == "switch" and c["params"]["closed"] == 1}


class PassiveImportTests(unittest.TestCase):
    def test_unrecognized_type_returns_none_without_inspecting_its_fields(self):
        self.assertIsNone(import_element({"type": "Multimeter"}, scene={}))
        self.assertIsNone(import_element({"type": "555 Timer"}, scene={}))

    def test_push_uses_current_not_default_state(self):
        for current, default in ((0, 1), (1, 0)):
            element, scene = fixture("Push Switch", 2, {"开关": current, "默认开关": default})
            result = import_element(element, scene=scene)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["params"], {"closed": current, "r_closed": 1e-9})
            self.assertEqual(result[0]["pl_source"]["raw_properties"]["默认开关"], default)
            self.assertEqual(result[0]["interaction"]["control_id"], "original")
            self.assertTrue(result[0]["interaction"]["momentary"])

    def test_simple_and_air_switch_are_controllable_analog_contacts(self):
        for kind in ("Simple Switch", "Air Switch"):
            element, scene = fixture(kind, 2, {"开关": 1})
            result = import_element(element, scene=scene)
            self.assertEqual((result[0]["type"], result[0]["params"]),
                             ("switch", {"closed": 1, "r_closed": 1e-9}))
            self.assertEqual(result[0]["interaction"]["kind"], "spst")
            self.assertFalse(result[0]["interaction"]["momentary"])

    def test_spdt_all_three_states_match_off_left_right_contract(self):
        expected = {0: set(), 1: {frozenset(["N0", "N1"])}, 2: {frozenset(["N1", "N2"])}}
        for state in range(3):
            element, scene = fixture("SPDT Switch", 3, {"开关": float(state)})
            result = import_element(element, scene=scene)
            self.assertEqual(len(result), 2)
            self.assertEqual(connected_pairs(result), expected[state])
            self.assertEqual({n for c in result for n in c["nodes"]}, {"N0", "N1", "N2"})

    def test_dpdt_poles_remain_separate_and_all_states_are_explicit(self):
        for state in range(3):
            element, scene = fixture("DPDT Switch", 6, {"开关": state})
            result = import_element(element, scene=scene)
            expected = set() if state == 0 else {
                frozenset(["N1", "N0" if state == 1 else "N2"]),
                frozenset(["N4", "N3" if state == 1 else "N5"]),
            }
            self.assertEqual(connected_pairs(result), expected)
            self.assertEqual(len(result), 4)
            self.assertTrue(all(set(c["nodes"]) <= {"N0", "N1", "N2"} or
                                set(c["nodes"]) <= {"N3", "N4", "N5"} for c in result))

    def test_all_original_identity_pose_properties_are_preserved_without_mutation(self):
        element, scene = fixture("DPDT Switch", 6, {"开关": 2, "锁定": 1, "editor": "untrusted text"})
        before = copy.deepcopy(scene)
        result = import_element(element, scene=scene)
        self.assertEqual(scene, before)
        self.assertEqual(result[0]["id"], element["id"])
        self.assertEqual(result[0]["label"], element["label"])
        self.assertTrue(all("label" not in item for item in result[1:]))
        for item in result:
            self.assertEqual(item["position"], element["position"])
            self.assertEqual(item["rotation"], element["rotation"])
            self.assertEqual(item["pl_source"]["raw_properties"], element["properties"])
            self.assertEqual(item["pl_source"]["raw_statistics"], element["statistics"])
            self.assertEqual(item["pl_source"]["implicit_references"], [])
        result[0]["pl_source"]["raw_properties"]["开关"] = 0
        self.assertEqual(scene, before)

    def test_schema_pin_is_not_treated_as_wired_or_implicitly_grounded(self):
        element, scene = fixture("SPDT Switch", 3, {"开关": 1})
        result = import_element(element, scene=scene)
        mapping = result[0]["pl_source"]["pin_mapping"]
        self.assertEqual([p["pl_pin"] for p in mapping if p["externally_wired"]], [0])
        self.assertFalse(any(n == "gnd" for c in result for n in c["nodes"]))

    def test_helper_ids_stable_unique_bounded_and_avoid_original_scene_ids(self):
        element, scene = fixture("DPDT Switch", 6, {"开关": 1}, cid="x" * 128)
        first = import_element(element, scene=scene)
        self.assertEqual(first, import_element(element, scene=scene))
        collision = first[1]["id"]
        scene["components"].append({"id": collision})
        second = import_element(element, scene=scene)
        self.assertNotIn(collision, [c["id"] for c in second])
        self.assertEqual(second, import_element(element, scene=scene))
        self.assertEqual(len({c["id"] for c in second}), 4)
        self.assertTrue(all(len(c["id"]) <= 128 for c in second))

    def test_separate_original_components_cannot_share_helpers(self):
        a, sa = fixture("DPDT Switch", 6, {"开关": 1}, cid="a")
        b, sb = fixture("DPDT Switch", 6, {"开关": 1}, cid="b")
        scene = {"components": [a, b], "wires": sa["wires"] + sb["wires"]}
        ids = [c["id"] for el in (a, b) for c in import_element(el, scene=scene)]
        self.assertEqual(len(ids), len(set(ids)))

    def test_resistance_box_uses_actual_value_not_editor_range(self):
        element, scene = fixture("Resistance Box", 2, {"电阻": 12345.67890123, "最大电阻": 10000, "最小电阻": 0.1})
        result = import_element(element, scene=scene)
        self.assertEqual(result[0]["params"], {"r": 12345.67890123})
        self.assertEqual(result[0]["type"], "resistor")
        # Independent electrical check for the mapped resistance, not an app run.
        self.assertAlmostEqual(3.0 / result[0]["params"]["r"], 3.0 / element["properties"]["电阻"])

    def test_slide_rheostat_preserves_four_terminal_wiper_topology(self):
        element, scene = fixture("Slide Rheostat", 4,
            {"额定电阻": 10, "滑块位置": .4, "电阻1": 4, "电阻2": 6})
        result = import_element(element, scene=scene)
        self.assertEqual([(c["type"], c["nodes"], c["params"]) for c in result], [
            ("resistor", ["N0", "N2"], {"r": 4.0}),
            ("resistor", ["N1", "N3"], {"r": 6.0}),
            ("switch", ["N2", "N3"], {"closed": 1}),
        ])
        self.assertEqual({c["interaction"]["control_id"] for c in result}, {"original"})
        self.assertEqual({c["interaction"]["role"] for c in result},
                         {"segment_left", "segment_right", "wiper_link"})

    def test_slide_rheostat_rejects_inconsistent_saved_segments(self):
        element, scene = fixture("Slide Rheostat", 4,
            {"额定电阻": 10, "滑块位置": .4, "电阻1": 8, "电阻2": 2})
        with self.assertRaisesRegex(ToolError, "disagree"):
            import_element(element, scene=scene)

    def test_zero_ohm_box_uses_contact_not_illegal_zero_resistor_or_voltage_source(self):
        element, scene = fixture("Resistance Box", 2, {"电阻": 0})
        result = import_element(element, scene=scene)
        self.assertEqual((result[0]["type"], result[0]["params"]), ("switch", {"closed": 1}))

    def test_unknown_switch_values_fail_without_coercion(self):
        for kind, count, key in (("Push Switch", 2, "开关"), ("Push Switch", 2, "默认开关"),
                                 ("SPDT Switch", 3, "开关"), ("DPDT Switch", 6, "开关")):
            for value in (None, True, "1", -1, 3, 0.5, float("nan"), float("inf")):
                with self.subTest(kind=kind, key=key, value=value):
                    props = {"开关": 0, key: value}
                    element, scene = fixture(kind, count, props)
                    with self.assertRaises(ToolError):
                        import_element(element, scene=scene)

    def test_invalid_resistance_does_not_become_a_default(self):
        for value in (None, "10", True, -1, math.nan, math.inf, 10**1000):
            element, scene = fixture("Resistance Box", 2, {"电阻": value})
            with self.subTest(value=str(value)[:30]), self.assertRaises(ToolError):
                import_element(element, scene=scene)

    def test_missing_duplicate_or_extra_pins_are_rejected(self):
        for mutate in (lambda e: e["pins"].pop(), lambda e: e["pins"].append(e["pins"][0]),
                       lambda e: e["pins"][0].update(pin=True), lambda e: e["pins"][0].update(node=None)):
            element, scene = fixture("SPDT Switch", 3, {"开关": 1})
            mutate(element)
            with self.assertRaises(ToolError):
                import_element(element, scene=scene)

    def test_missing_source_pose_is_not_silently_generated(self):
        element, scene = fixture("Push Switch", 2, {"开关": 1})
        del element["position"]
        with self.assertRaisesRegex(ToolError, "original finite position"):
            import_element(element, scene=scene)

    def test_huge_nonfinite_pose_and_malformed_scene_fail_as_tool_errors(self):
        element, scene = fixture("Push Switch", 2, {"开关": 1})
        element["position"][0] = 10**1000
        with self.assertRaisesRegex(ToolError, "original finite position"):
            import_element(element, scene=scene)
        element["position"][0] = 0
        scene["components"].append(None)
        with self.assertRaisesRegex(ToolError, "malformed original scene component"):
            import_element(element, scene=scene)

    def test_duplicate_scene_id_and_invalid_wire_reference_are_rejected(self):
        element, scene = fixture("Push Switch", 2, {"开关": 1})
        scene["components"].append(copy.deepcopy(element))
        with self.assertRaisesRegex(ToolError, "unique original IDs"):
            import_element(element, scene=scene)
        scene["components"].pop()
        scene["wires"][0]["SourcePin"] = 8
        with self.assertRaisesRegex(ToolError, "invalid original pin"):
            import_element(element, scene=scene)

    def test_nonzero_contact_resistance_and_broken_state_are_not_discarded(self):
        for extras in ({"内阻": 0.5}, {"接触电阻": 0.5}):
            element, scene = fixture("Push Switch", 2, {"开关": 1, **extras})
            with self.assertRaisesRegex(ToolError, "cannot be discarded"):
                import_element(element, scene=scene)
        element, scene = fixture("Push Switch", 2, {"开关": 1})
        element["IsBroken"] = True
        with self.assertRaisesRegex(ToolError, "broken-device"):
            import_element(element, scene=scene)

    def test_resistance_law_routes_to_complete_device_importer(self):
        self.assertIsNone(import_element({"id": "raw", "type": "Resistance Law"}, scene={}))


@unittest.skipUnless(os.environ.get("AUREX_TEST_PHYENGINE_LIB"), "optional native fixture library not configured")
class NativePassiveImportTests(unittest.TestCase):
    def run_spec(self, spec):
        payload = {"spec": spec, "lib_path": os.environ["AUREX_TEST_PHYENGINE_LIB"]}
        process = subprocess.run([sys.executable, "-m", "aurex.phy_engine.worker"],
            input=json.dumps(payload), text=True, capture_output=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def solve(self, components):
        return {row["id"]: row for row in self.run_spec(
            {"analysis": "dc", "components": components})["components"]}

    def test_actual_native_switched_loads_for_all_snapshot_states(self):
        for kind, count, states in (("Push Switch", 2, range(2)), ("SPDT Switch", 3, range(3)),
                                    ("DPDT Switch", 6, range(3))):
            for state in states:
                with self.subTest(kind=kind, state=state):
                    element, scene = fixture(kind, count, {"开关": state})
                    components = import_element(element, scene=scene)
                    if kind == "Push Switch":
                        sources, loads = [("N0", 5)], [("N1", 5 if state else 0)]
                    else:
                        sources = [("N1", 5)]
                        loads = [("N0", 5 if state == 1 else 0), ("N2", 5 if state == 2 else 0)]
                        if kind == "DPDT Switch":
                            sources += [("N4", 9)]
                            loads += [("N3", 9 if state == 1 else 0), ("N5", 9 if state == 2 else 0)]
                    for index, (node, volts) in enumerate(sources):
                        components.append({"id": f"V{index}", "type": "vdc", "nodes": [node, "gnd"], "params": {"v": volts}})
                    for index, (node, _) in enumerate(loads):
                        components.append({"id": f"R{index}", "type": "resistor", "nodes": [node, "gnd"], "params": {"r": 1000}})
                    result = self.solve(components)
                    for index, (_, expected) in enumerate(loads):
                        self.assertAlmostEqual(result[f"R{index}"]["voltage"][0], expected, delta=1e-6)

    def test_actual_native_resistance_box_divider_and_zero_ohm(self):
        for ohms, expected in ((1000, 2.5), (0, 5)):
            with self.subTest(ohms=ohms):
                element, scene = fixture("Resistance Box", 2, {"电阻": ohms})
                result = self.solve(import_element(element, scene=scene) + [
                    {"id": "V", "type": "vdc", "nodes": ["N0", "gnd"], "params": {"v": 5}},
                    {"id": "R", "type": "resistor", "nodes": ["N1", "gnd"], "params": {"r": 1000}},
                ])
                self.assertAlmostEqual(result["R"]["voltage"][0], expected, delta=1e-8)

    def test_finite_switch_contacts_keep_parallel_sources_solvable(self):
        # Regression for real PhysicsLab comparator buses: equal/unequal ideal
        # sources may meet through closed contacts, but the contacts themselves
        # are finite and must not become contradictory ideal MNA constraints.
        result = self.solve([
            {"id": "VH", "type": "vdc", "nodes": ["hi", "gnd"], "params": {"v": 5}},
            {"id": "VL", "type": "vdc", "nodes": ["lo", "gnd"], "params": {"v": 0}},
            {"id": "SH", "type": "switch", "nodes": ["hi", "bus"],
             "params": {"closed": 1, "r_closed": 1e-9}},
            {"id": "SL", "type": "switch", "nodes": ["lo", "bus"],
             "params": {"closed": 1, "r_closed": 1e-9}},
            {"id": "R", "type": "resistor", "nodes": ["bus", "gnd"], "params": {"r": 1000}},
        ])
        self.assertAlmostEqual(result["R"]["voltage"][0], 2.5, delta=1e-6)

    def test_immediate_logic_output_decodes_analog_operating_point(self):
        result = self.run_spec({"analysis": "dc", "digital_clock_ticks": 1, "components": [
            {"id": "V", "type": "vdc", "nodes": ["sense", "gnd"], "params": {"v": 5}},
            {"id": "O", "type": "digital_output", "nodes": ["sense"],
             "params": {"low_v": 0, "high_v": 3, "setup_time_s": 0, "hold_time_s": 0}},
        ]})
        output = next(row for row in result["components"] if row["id"] == "O")
        self.assertEqual(output["model_digital_state"]["value"], 1)

    def test_distinct_digital_drivers_still_report_real_contention(self):
        payload = {"spec": {"analysis": "dc", "components": [
            {"id": "L", "type": "digital_input", "nodes": ["bus"], "params": {"state": 0}},
            {"id": "H", "type": "digital_input", "nodes": ["bus"], "params": {"state": 1}},
            {"id": "R", "type": "resistor", "nodes": ["bus", "gnd"], "params": {"r": 1000}},
        ]}, "lib_path": os.environ["AUREX_TEST_PHYENGINE_LIB"]}
        process = subprocess.run([sys.executable, "-m", "aurex.phy_engine.worker"],
            input=json.dumps(payload), text=True, capture_output=True, timeout=15)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("rc=6", process.stderr + process.stdout)

    def test_actual_native_slide_rheostat_snapshot_divider(self):
        element, scene = fixture("Slide Rheostat", 4,
            {"额定电阻": 10, "滑块位置": .4, "电阻1": 4, "电阻2": 6})
        result = self.solve(import_element(element, scene=scene) + [
            {"id": "V", "type": "vdc", "nodes": ["N0", "gnd"], "params": {"v": 5}},
            {"id": "V0", "type": "vdc", "nodes": ["N1", "gnd"], "params": {"v": 0}},
        ])
        self.assertAlmostEqual(result["original"]["voltage"][1], 3, delta=1e-8)

    def test_slide_rheostat_moves_during_one_actual_transient(self):
        element, scene = fixture("Slide Rheostat", 4,
            {"额定电阻": 10, "滑块位置": .4, "电阻1": 4, "电阻2": 6})
        components = import_element(element, scene=scene) + [
            {"id": "V", "type": "vdc", "nodes": ["N0", "gnd"], "params": {"v": 5}},
            {"id": "V0", "type": "vdc", "nodes": ["N1", "gnd"], "params": {"v": 0}},
        ]
        result = self.run_spec({"analysis": "tr", "tr_step": .1, "tr_stop": .3,
            "tr_sample_every": 1, "components": components,
            "tr_interactions": [
                {"time_s": .1, "set": {"original": .2}},
                {"time_s": .2, "set": {"original": .8}},
            ]})
        wiper = [next(c for c in point["components"] if c["id"] == "original")["voltage"][1]
                 for point in result["transient"]["samples"]]
        self.assertAlmostEqual(wiper[0], 4, delta=1e-7)
        self.assertAlmostEqual(wiper[1], 1, delta=1e-7)
        self.assertAlmostEqual(wiper[2], 1, delta=1e-7)
        self.assertEqual(result["interaction_states"]["original"], .8)

    def test_spdt_selector_changes_branch_during_one_actual_transient(self):
        element, scene = fixture("SPDT Switch", 3, {"开关": 0})
        components = import_element(element, scene=scene) + [
            {"id": "V", "type": "vdc", "nodes": ["N1", "gnd"], "params": {"v": 5}},
            {"id": "RL", "type": "resistor", "nodes": ["N0", "gnd"], "params": {"r": 1000}},
            {"id": "RR", "type": "resistor", "nodes": ["N2", "gnd"], "params": {"r": 1000}},
        ]
        result = self.run_spec({"analysis": "tr", "tr_step": .1, "tr_stop": .3,
            "tr_sample_every": 1, "components": components,
            "tr_interactions": [
                {"time_s": .1, "set": {"original": 1}},
                {"time_s": .2, "set": {"original": 2}},
            ]})
        branches = []
        for point in result["transient"]["samples"]:
            by_id = {c["id"]: c for c in point["components"]}
            branches.append((by_id["RL"]["voltage"][0], by_id["RR"]["voltage"][0]))
        self.assertAlmostEqual(branches[0][0], 5, delta=1e-8)
        self.assertAlmostEqual(branches[0][1], 0, delta=1e-6)
        self.assertAlmostEqual(branches[1][0], 0, delta=1e-6)
        self.assertAlmostEqual(branches[1][1], 5, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
