"""Source/passive import tests with independent analytic native-load oracles."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from aurex.phy_engine.catalog import COMPONENTS
from aurex.phy_engine.ffi import _Lib
from aurex.tools.registry import ToolError

_MODULE = Path(__file__).resolve().parents[1] / "src/aurex/tools/plar_source_import.py"
_SPEC = importlib.util.spec_from_file_location("aurex.tools.plar_source_import", _MODULE)
source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(source)


def element(kind="Battery Source", **properties):
    defaults = {
        "Battery Source": {"电压": 5, "内阻": 2},
        "Current Source": {"电流": .01, "内阻": 1000},
        "Basic Capacitor": {"电容": 1e-4, "内阻": 100, "理想模式": 0, "耐压": 16},
        "Basic Inductor": {"电感": 1, "内阻": 100, "理想模式": 0, "额定电流": 1},
        "Sinewave Source": {"电压": 3, "偏移": 2, "频率": 5, "占空比": .5, "内阻": 100},
        "Square Source": {"电压": 2, "偏移": 3, "频率": 10, "占空比": .25, "内阻": 100},
        "Sawtooth Source": {"电压": 3, "偏移": 2, "频率": 10, "占空比": .5, "内阻": 100},
        "Pulse Source": {"电压": 3, "偏移": 2, "频率": 10, "占空比": .1, "内阻": 100},
        "Student Source": {"交流电压": 10, "直流电压": 3, "频率": 50, "开关": 1},
    }
    return {"id": "original-element", "type": kind, "label": "原标签",
        "properties": {**defaults.get(kind, {}), **properties},
        "statistics": {"电流": 99, "电压": -222},
        "pins": ([{"pin": 0, "node": "dc+"}, {"pin": 1, "node": "dc-"},
                  {"pin": 2, "node": "ac+"}, {"pin": 3, "node": "ac-"}] if kind == "Student Source" else
                 [{"pin": 0, "node": "red"}, {"pin": 1, "node": "gnd"}]),
        "position": [.1, -.2, .03], "rotation": [12, 25, 37], "pin_count_known": True}


def scene_for(el):
    return {"components": [el], "wires": [{"Source": el["id"], "SourcePin": 0,
        "Target": el["id"], "TargetPin": 0}]}


def convert(el):
    return source.import_element(el, scene=scene_for(el))


class ImportTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_other_groups_untouched_and_incomplete_recognized_models_blocked(self):
        for kind in ("Resistor", "Triangle Source", "Transformer", "555 Timer"):
            self.assertIsNone(convert(element(kind)))
        for kind in ("Student Source", "Pulse Source"):
            el = element(kind)
            el["properties"] = {}
            with self.assertRaises(ToolError):
                convert(el)

    def test_student_independent_channels_rms_conversion_and_no_ground_invention(self):
        result = convert(element("Student Source"))
        self.assertEqual([r["type"] for r in result], ["vdc", "switch", "vac", "switch"])
        self.assertEqual([r["params"]["r_closed"] for r in result if r["type"] == "switch"],
                         [1e-9, 1e-9])
        self.assertEqual(result[0]["params"], {"v": 3})
        self.assertEqual(result[2]["params"], {"vp": 10 * math.sqrt(2), "freq_hz": 50, "phase_deg": 0})
        self.assertEqual(result[0]["nodes"][1], "dc-")
        self.assertEqual(result[2]["nodes"][1], "ac-")
        self.assertEqual(result[1]["nodes"][1], "dc+")
        self.assertEqual(result[3]["nodes"][1], "ac+")
        self.assertFalse(set(result[0]["nodes"] + result[1]["nodes"]) & set(result[2]["nodes"] + result[3]["nodes"]))
        self.assertNotIn("gnd", [n for r in result for n in r["nodes"]])
        self.assertEqual([p["pl_pin"] for p in result[0]["pl_source"]["pin_mapping"]], [0, 1, 2, 3])

    def test_student_off_uses_open_contacts_not_zero_voltage_short(self):
        result = convert(element("Student Source", **{"开关": 0}))
        self.assertEqual([r["params"]["closed"] for r in result if r["type"] == "switch"], [0, 0])
        self.assertEqual(result[0]["params"]["v"], 3)
        self.assertEqual(result[2]["params"]["vp"], 10 * math.sqrt(2))
        self.assertTrue(any("0 V source short" in s for s in result[0]["pl_source"]["assumptions"]))

    def test_student_missing_internal_resistance_is_explicit_ideal_convention(self):
        original = element("Student Source")
        result = convert(original)
        source = result[0]["pl_source"]
        self.assertNotIn("内阻", source["raw_properties"])
        self.assertEqual(source["engineering_defaults"]["output_series_resistance_ohm_per_channel"], 0)
        self.assertTrue(any("SDK has no" in s for s in source["assumptions"]))
        resistive = convert(element("Student Source", **{"内阻": 4}))
        self.assertEqual(len(resistive), 6)
        self.assertEqual([r["params"] for r in resistive if r["type"] == "resistor"], [{"r": 4}, {"r": 4}])

    def test_student_contract_validation_does_not_coerce_switch_or_timing(self):
        for props in ({"开关": True}, {"开关": "1"}, {"开关": 2}, {"交流电压": -1},
                      {"频率": 0}, {"频率": 1e308}, {"频率": 1e-320}, {"交流电压": 1.7e308}):
            with self.subTest(props=props), self.assertRaises(ToolError):
                convert(element("Student Source", **props))
        el = element("Student Source")
        el["pins"].pop()
        with self.assertRaisesRegex(ToolError, "all 4"):
            convert(el)

    def test_student_provenance_immutable_and_only_primary_has_original_label(self):
        el = element("Student Source")
        scene = scene_for(el)
        before = copy.deepcopy(scene)
        result = source.import_element(el, scene=scene)
        self.assertEqual(scene, before)
        self.assertEqual(result[0]["id"], el["id"])
        for index, row in enumerate(result):
            self.assertEqual(row["pl_source"]["raw_properties"], el["properties"])
            self.assertEqual(row["pl_source"]["parent_identifier"], el["id"])
            self.assertEqual("label" in row, index == 0)
        result[0]["pl_source"]["engineering_defaults"]["dc_terminals"]["positive"] = 3
        self.assertEqual(result[2]["pl_source"]["engineering_defaults"]["dc_terminals"]["positive"], 0)

    def test_student_ac_only_wiring_does_not_claim_unwired_dc_was_connected(self):
        el = element("Student Source")
        scene = {"components": [el], "wires": [
            {"Source": el["id"], "SourcePin": 2, "Target": "transformer", "TargetPin": 0},
            {"Source": el["id"], "SourcePin": 3, "Target": "transformer", "TargetPin": 1}]}
        result = source.import_element(el, scene=scene)
        self.assertEqual([p["pl_pin"] for p in result[0]["pl_source"]["pin_mapping"] if p["externally_wired"]], [2, 3])
        self.assertNotIn("gnd", [n for r in result for n in r["nodes"]])
        disconnected = result[0]["pl_source"]["inactive_unconnected_subchannel"]
        self.assertEqual(len(disconnected), 1)
        self.assertEqual(disconnected[0]["channel"], "dc")
        self.assertEqual(disconnected[0]["pl_pins"], [0, 1])
        self.assertEqual(disconnected[0]["pin_ids"], [el["id"] + ":0", el["id"] + ":1"])
        self.assertTrue(disconnected[0]["retained_in_native_spec"])

    def test_pulse_uses_native_finite_width_spike_with_explicit_ramps(self):
        result = convert(element("Pulse Source"))
        self.assertEqual(result[0]["type"], "pulse")
        self.assertEqual(result[0]["params"], {"high_v": 5, "low_v": -1, "freq_hz": 10,
            "duty": .1, "phase_rad": 0, "rise_s": .005, "fall_s": .005})
        self.assertEqual(result[1]["params"], {"r": 100})
        self.assertEqual(result[0]["pl_source"]["engineering_defaults"]["pulse_shape"], "symmetric_triangular_pulse")
        self.assertTrue(any("not independently established" in s for s in result[0]["pl_source"]["assumptions"]))

    def test_pulse_nondefault_duty_is_preserved_not_clamped(self):
        for duty in (.01, .25, .9):
            result = convert(element("Pulse Source", **{"占空比": duty}))[0]
            self.assertEqual(result["params"]["duty"], duty)
            self.assertAlmostEqual(result["params"]["rise_s"] + result["params"]["fall_s"], duty / 10)

    def test_pulse_invalid_or_unresolved_time_scale_is_rejected(self):
        for props in ({"占空比": 0}, {"占空比": 1}, {"占空比": True},
                      {"频率": 1e30, "占空比": .1}, {"电压": math.inf}, {"偏移": 1e308, "电压": 1e308}):
            with self.subTest(props=props), self.assertRaises(ToolError):
                convert(element("Pulse Source", **props))

    def test_battery_series_resistance_is_preserved(self):
        core, resistor = convert(element())
        self.assertEqual(core["type"], "vdc")
        self.assertEqual(core["params"], {"v": 5})
        self.assertEqual(resistor["params"], {"r": 2})
        self.assertEqual(core["nodes"][0], resistor["nodes"][1])
        self.assertEqual(resistor["nodes"][0], "red")
        self.assertEqual(core["nodes"][1], "gnd")

    def test_current_resistance_is_parallel_not_series(self):
        core, shunt = convert(element("Current Source"))
        self.assertEqual(core["nodes"], ["red", "gnd"])
        self.assertEqual(shunt["nodes"], core["nodes"])
        self.assertEqual(shunt["params"], {"r": 1000})
        self.assertEqual(shunt["pl_source"]["decomposition_role"], "parallel_resistance")
        self.assertEqual(core["pl_source"]["primitive_pin_mapping"], [0, 1])

    def test_capacitor_and_inductor_series_esr_and_zero_ideal_case(self):
        for kind, native, key, value in (("Basic Capacitor", "capacitor", "c", 1e-4), ("Basic Inductor", "inductor", "l", 1)):
            core, esr = convert(element(kind))
            self.assertEqual((core["type"], core["params"]), (native, {key: value}))
            self.assertEqual(esr["params"], {"r": 100})
            self.assertEqual(core["pl_source"]["primitive_pin_mapping"], [None, 1])
            self.assertEqual(esr["pl_source"]["primitive_pin_mapping"], [0, None])
            ideal = convert(element(kind, **{"内阻": 0, "理想模式": 1}))
            self.assertEqual(len(ideal), 1)
            self.assertEqual(ideal[0]["nodes"], ["red", "gnd"])

    def test_conflicting_ideal_mode_retains_explicit_numeric_esr(self):
        for kind in ("Basic Capacitor", "Basic Inductor"):
            result = convert(element(kind, **{"理想模式": 1}))
            self.assertEqual(result[1]["params"], {"r": 100})
            self.assertTrue(any("理想模式=1" in note
                                for note in result[0]["pl_source"]["assumptions"]))

    def test_legacy_capacitor_inductor_without_ideal_flag_keep_explicit_esr(self):
        for kind in ("Basic Capacitor", "Basic Inductor"):
            el = element(kind)
            del el["properties"]["理想模式"]
            result = convert(el)
            self.assertEqual(result[1]["params"], {"r": 100})
            self.assertNotIn("理想模式", result[0]["pl_source"]["raw_properties"])
            self.assertNotIn("理想模式", el["properties"])
            self.assertTrue(any("Legacy save" in text for text in result[0]["pl_source"]["assumptions"]))
            el["properties"]["内阻"] = 0
            self.assertEqual(len(convert(el)), 1)

    @unittest.skipUnless(os.environ.get("AUREX_CHARGER_RAW_SAV"), "optional original community charger save not configured")
    def test_original_community_charger_legacy_capacitors_and_inductor(self):
        path = Path(os.environ["AUREX_CHARGER_RAW_SAV"])
        raw = json.loads(path.read_text())
        experiment = raw.get("Experiment", raw)
        self.assertEqual(experiment["Type"], 0)
        saved = experiment["StatusSave"]
        saved = json.loads(saved) if isinstance(saved, str) else saved
        selected = [e for e in saved["Elements"] if e["ModelID"] in {"Basic Capacitor", "Basic Inductor"}]
        self.assertEqual(len(selected), 3)
        expected = {"b56a33b92ca54a0cb171be154eafc2f8": ("capacitor", "c", 4e-5, 5),
                    "8ff84d9a7ef8447780d46b3b6640ea3f": ("capacitor", "c", 6e-5, 5),
                    "e50a80f5125c409092c4b45af0a390d2": ("inductor", "l", .05000000074505806, 1)}
        for original in selected:
            native, key, value, esr = expected[original["Identifier"]]
            self.assertNotIn("理想模式", original["Properties"])
            # Preserve every original physical field; the tiny test scene adds
            # only named external nodes used to invoke the importer contract.
            el = element(original["ModelID"])
            el.update(id=original["Identifier"], properties=copy.deepcopy(original["Properties"]),
                      statistics=copy.deepcopy(original["Statistics"]))
            result = convert(el)
            self.assertEqual(result[0]["type"], native)
            self.assertEqual(result[0]["params"], {key: value})
            self.assertEqual(result[1]["params"], {"r": esr})
            self.assertEqual(result[0]["pl_source"]["raw_properties"], original["Properties"])
            self.assertEqual(result[0]["pl_source"]["raw_statistics"], original["Statistics"])

    def test_sine_offset_is_separate_source_and_not_used_as_phase(self):
        core, esr, offset = convert(element("Sinewave Source"))
        self.assertEqual(core["params"], {"vp": 3, "freq_hz": 5, "phase_deg": 0})
        self.assertEqual(offset["params"], {"v": 2})
        self.assertEqual(offset["nodes"], [core["nodes"][1], "gnd"])
        self.assertEqual(core["pl_source"]["primitive_pin_mapping"], [None, None])
        self.assertEqual(offset["pl_source"]["primitive_pin_mapping"], [None, None])
        self.assertEqual(offset["pl_source"]["implicit_references"][0]["pl_pin"], 1)
        self.assertEqual(esr["params"], {"r": 100})

    def test_historical_sine_without_irrelevant_duty_imports_unchanged(self):
        original = element("Sinewave Source")
        del original["properties"]["占空比"]
        result = convert(original)
        self.assertEqual(result[0]["params"], {"vp": 3, "freq_hz": 5, "phase_deg": 0})
        self.assertNotIn("占空比", result[0]["pl_source"]["raw_properties"])
        self.assertTrue(any("omits 占空比" in note
                            for note in result[0]["pl_source"]["assumptions"]))

    def test_square_duty_and_both_offset_levels_preserved(self):
        result = convert(element("Square Source"))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["params"], {"high_v": 5, "low_v": 1, "freq_hz": 10, "duty": .25, "phase_rad": 0})

    def test_legacy_square_and_pulse_without_duty_use_public_default(self):
        for kind in ("Square Source", "Pulse Source"):
            original = element(kind)
            del original["properties"]["占空比"]
            imported = convert(original)[0]
            self.assertEqual(imported["params"]["duty"], .5)
            self.assertNotIn("占空比", imported["pl_source"]["raw_properties"])
            self.assertTrue(any(f"Legacy {kind}" in note
                                for note in imported["pl_source"]["assumptions"]))

    def test_square_saved_initial_phase_is_converted_from_degrees(self):
        result = convert(element("Square Source", 初始相位=90))
        self.assertTrue(math.isclose(result[0]["params"]["phase_rad"], math.pi / 2))
        self.assertEqual(result[0]["pl_source"]["raw_properties"]["初始相位"], 90)
        self.assertFalse(result[0]["pl_source"]["numerical_equivalence_to_original"])

    def test_saw_direction_explicit_and_irrelevant_generic_duty_disclosed(self):
        result = convert(element("Sawtooth Source"))
        self.assertEqual(result[0]["params"], {"high_v": 5, "low_v": -1, "freq_hz": 10, "phase_rad": 0})
        self.assertTrue(any("rises" in note for note in result[0]["pl_source"]["assumptions"]))
        for kind in ("Sawtooth Source", "Sinewave Source"):
            imported = convert(element(kind, **{"占空比": .2}))[0]
            self.assertEqual(imported["pl_source"]["raw_properties"]["占空比"], .2)
            self.assertTrue(any("no constitutive meaning" in note
                                for note in imported["pl_source"]["assumptions"]))

    def test_source_and_scene_unchanged_and_full_metadata_retained(self):
        el = element("Sinewave Source")
        el["properties"]["future_field"] = {"raw": [3, 2, 1]}
        scene = scene_for(el)
        before = copy.deepcopy(scene)
        result = source.import_element(el, scene=scene)
        self.assertEqual(scene, before)
        for index, component in enumerate(result):
            meta = component["pl_source"]
            self.assertEqual(meta["raw_properties"], el["properties"])
            self.assertEqual(meta["raw_statistics"], el["statistics"])
            self.assertEqual(meta["parent_identifier"], el["id"])
            self.assertEqual(meta["is_helper"], index != 0)
            self.assertEqual(len(meta["implicit_references"]), 1)
            self.assertEqual(meta["implicit_references"][0]["pl_pin"], 1)
            self.assertEqual(component["position"], el["position"])
            self.assertEqual(component["rotation"], el["rotation"])
            self.assertEqual("label" in component, index == 0)
        result[0]["pl_source"]["raw_properties"]["future_field"]["raw"].append(7)
        self.assertEqual(scene, before)
        self.assertEqual(len(result[1]["pl_source"]["raw_properties"]["future_field"]["raw"]), 3)

    def test_unwired_black_pin_remains_unwired_and_is_not_invented_ground(self):
        el = element()
        el["pins"][1]["node"] = "unwired-black"
        result = convert(el)
        self.assertEqual(result[0]["nodes"][1], "unwired-black")
        self.assertEqual(result[0]["pl_source"]["pin_mapping"], [
            {"pl_pin": 0, "node": "red", "externally_wired": True},
            {"pl_pin": 1, "node": "unwired-black", "externally_wired": False}])
        self.assertNotIn("gnd", [node for item in result for node in item["nodes"]])

    def test_single_ended_waveform_grounds_only_the_missing_terminal(self):
        positive = element("Square Source")
        positive["pins"][1]["node"] = "floating-black"
        imported = source.import_element(positive, scene=scene_for(positive))
        self.assertEqual(imported[0]["nodes"][1], "gnd")
        self.assertEqual(imported[1]["nodes"], ["red", imported[0]["nodes"][0]])
        self.assertEqual(imported[0]["pl_source"]["implicit_references"][0]["pl_pin"], 1)
        self.assertFalse(imported[0]["pl_source"]["pin_mapping"][1]["externally_wired"])

        negative = element("Square Source")
        negative["pins"][0]["node"] = "floating-red"
        negative["pins"][1]["node"] = "black"
        scene = {"components": [negative], "wires": [{"Source": negative["id"], "SourcePin": 1,
            "Target": negative["id"], "TargetPin": 1}]}
        imported = source.import_element(negative, scene=scene)
        self.assertEqual(imported[1]["nodes"][0], "gnd")
        self.assertEqual(imported[0]["nodes"][1], "black")
        self.assertEqual(imported[0]["pl_source"]["implicit_references"][0]["pl_pin"], 0)

    def test_two_terminal_waveform_does_not_invent_reference(self):
        el = element("Square Source")
        el["pins"][1]["node"] = "black"
        scene = {"components": [el], "wires": [
            {"Source": el["id"], "SourcePin": 0, "Target": el["id"], "TargetPin": 0},
            {"Source": el["id"], "SourcePin": 1, "Target": el["id"], "TargetPin": 1},
        ]}
        imported = source.import_element(el, scene=scene)
        self.assertEqual(imported[0]["nodes"][1], "black")
        self.assertEqual(imported[1]["nodes"][0], "red")
        self.assertEqual(imported[0]["pl_source"]["implicit_references"], [])

    def test_shorted_external_pins_keep_distinct_original_pin_identity(self):
        el = element("Current Source")
        el["pins"][1]["node"] = "red"
        result = convert(el)
        self.assertEqual(result[0]["pl_source"]["primitive_pin_mapping"], [0, 1])
        self.assertEqual(result[1]["pl_source"]["primitive_pin_mapping"], [0, 1])

    def test_ids_and_internal_nodes_are_stable_collision_free_and_bounded(self):
        el = element("Sinewave Source")
        el["id"] = "a" * 128
        scene = scene_for(el)
        first = source.import_element(el, scene=scene)
        collision = copy.deepcopy(element())
        collision["id"] = first[1]["id"]
        collision["pins"][0]["node"] = first[0]["nodes"][0]
        collision["pins"][1]["node"] = first[0]["nodes"][1]
        scene["components"].append(collision)
        second = source.import_element(el, scene=scene)
        self.assertEqual(second, source.import_element(el, scene=scene))
        self.assertNotEqual(first[1]["id"], second[1]["id"])
        self.assertNotEqual(first[0]["nodes"], second[0]["nodes"])
        self.assertEqual(second[0]["id"], el["id"])
        self.assertTrue(all(len(item["id"]) <= 128 for item in second))
        self.assertTrue(all(len(node) <= 128 for item in second for node in item["nodes"]))

    def test_missing_duplicate_or_malformed_pins_rejected(self):
        for pins in ([], [{"pin": 0, "node": "x"}], [{"pin": 0, "node": "x"}, {"pin": 0, "node": "y"}],
                     [{"pin": True, "node": "x"}, {"pin": 1, "node": "y"}],
                     [{"pin": 0, "node": ""}, {"pin": 1, "node": "y"}]):
            el = element()
            el["pins"] = pins
            with self.assertRaises(ToolError):
                convert(el)

    def test_invalid_source_scene_and_wires_rejected(self):
        el = element()
        for scene in ({}, {"components": [el, el], "wires": []}, {"components": [], "wires": []},
                      {"components": [el], "wires": [None]},
                      {"components": [el], "wires": [{"Source": el["id"], "SourcePin": 9}]}):
            with self.assertRaises(ToolError):
                source.import_element(el, scene=scene)

    def test_invalid_pose_broken_and_nontext_label_rejected(self):
        for change in ({"position": None}, {"rotation": [True, 0, 0]}, {"position": [math.inf, 0, 0]},
                       {"label": {}}, {"is_broken": True}):
            el = element()
            el.update(change)
            with self.assertRaises(ToolError):
                convert(el)

    def test_missing_nonfinite_negative_or_boolean_values_rejected(self):
        for value in (None, True, "2", math.inf, math.nan, -1, 10**1000):
            with self.assertRaises(ToolError):
                convert(element(**{"内阻": value}))
        for kind, key in (("Basic Capacitor", "电容"), ("Basic Inductor", "电感"), ("Current Source", "内阻")):
            with self.assertRaises(ToolError):
                convert(element(kind, **{key: 0}))

    def test_overflowed_frequency_or_levels_rejected(self):
        for changes in ({"频率": 0}, {"频率": 1e308}, {"频率": 1e-320}, {"电压": 1e308, "偏移": 1e308}):
            with self.assertRaises(ToolError):
                convert(element("Square Source", **changes))


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "native build not configured")
class NativeSourceTests(unittest.TestCase):
    def setUp(self):
        self.lib = _Lib(str(Path(os.environ["AUREX_PHY_ENGINE_BUILD"]) / "libphyengine.so"))
        self.circuits = []
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def tearDown(self):
        for circuit in self.circuits:
            circuit.close()

    def create(self, components):
        catalog = {**COMPONENTS, **source.CATALOG_ADDITIONS}
        elements, properties, nodes, wires = [0], [], {}, []
        for index, component in enumerate(components):
            meta = catalog[component["type"]]
            elements.append(meta["code"])
            parameters = {**meta["defaults"], **component.get("params", {})}
            properties.extend(parameters[key] for key in meta["props"])
            for pin, node in enumerate(component["nodes"]):
                nodes.setdefault(node, []).append((index + 1, pin))
        for node, pins in nodes.items():
            first = pins[0]
            if node == "gnd":
                wires.extend([0, 0, *first])
            elif len(pins) == 1:
                wires.extend([*first, *first])
            for other in pins[1:]:
                wires.extend([*first, *other])
        circuit = self.lib.create_circuit(elements=elements, wires=wires, properties=properties)
        self.circuits.append(circuit)
        return circuit

    def voltage(self, sample, index, pin=0):
        return sample["voltage"][sample["voltage_ord"][index] + pin]

    def resistor(self, r, a="red", b="gnd"):
        return {"type": "resistor", "nodes": [a, b], "params": {"r": r}}

    def test_battery_loaded_dc_uses_actual_series_resistance(self):
        parts = convert(element())
        circuit = self.create(parts + [self.resistor(8)])
        circuit.set_analyze_type(1)
        circuit.analyze()
        sample = circuit.sample_complex(max_pins=2)
        self.assertAlmostEqual(self.voltage(sample, len(parts)), 4, places=9)
        self.assertAlmostEqual(self.voltage(sample, 0), 5, places=9)

    def test_current_loaded_dc_obeys_parallel_norton_and_native_sign(self):
        parts = convert(element("Current Source"))
        circuit = self.create(parts + [self.resistor(1000)])
        circuit.set_analyze_type(1)
        circuit.analyze()
        self.assertAlmostEqual(self.voltage(circuit.sample_complex(max_pins=2), 0), -5, places=9)

    def test_capacitor_esr_transient_matches_independent_rc_exponential(self):
        parts = convert(element("Basic Capacitor"))
        circuit = self.create(parts + [{"type": "vdc", "nodes": ["red", "gnd"], "params": {"v": 1}}])
        result = circuit.run_transient_trace(5e-6, .02, sample_every=40,
            capture=lambda: circuit.sample_complex(max_pins=2))
        self.assertAlmostEqual(result["actual_stop_s"], .02, places=12)
        for point in result["samples"]:
            self.assertAlmostEqual(self.voltage(point["sample"], 0), 1 - math.exp(-point["time_s"] / .01), delta=.0003)

    def test_inductor_esr_transient_matches_independent_rl_exponential(self):
        parts = convert(element("Basic Inductor"))
        circuit = self.create(parts + [{"type": "vdc", "nodes": ["red", "gnd"], "params": {"v": 1}}])
        result = circuit.run_transient_trace(5e-6, .02, sample_every=40,
            capture=lambda: circuit.sample_complex(max_pins=2))
        for point in result["samples"]:
            measured_current = (1 - self.voltage(point["sample"], 0)) / 100
            expected_current = .01 * (1 - math.exp(-point["time_s"] / .01))
            self.assertAlmostEqual(measured_current, expected_current, delta=3e-6)

    def test_rc_time_step_refinement_converges_to_same_independent_oracle(self):
        errors = []
        for dt in (2e-5, 5e-6):
            circuit = self.create(convert(element("Basic Capacitor")) + [
                {"type": "vdc", "nodes": ["red", "gnd"], "params": {"v": 1}}])
            circuit.run_transient_bounded(dt, .002)
            actual = self.voltage(circuit.sample_complex(max_pins=2), 0)
            errors.append(abs(actual - (1 - math.exp(-.002 / .01))))
        self.assertGreater(errors[0], 0)
        self.assertLess(errors[1], errors[0] * .3)

    def _reactive_pair(self, parallel):
        capacitor = element("Basic Capacitor", **{"电容": .001, "内阻": 5})
        inductor = element("Basic Inductor", **{"电感": .2, "内阻": 2})
        capacitor["id"], inductor["id"] = "C-original", "L-original"
        if not parallel:
            capacitor["pins"][1]["node"] = "mid"
            inductor["pins"][0]["node"] = "mid"
        scene = {"components": [capacitor, inductor], "wires": []}
        return (source.import_element(capacitor, scene=scene),
                source.import_element(inductor, scene=scene))

    def _complex_voltage(self, sample, index, pin=0):
        at = sample["voltage_ord"][index] + pin
        return complex(sample["voltage"][at], sample["voltage_imag"][at])

    def test_series_capacitor_inductor_preserve_both_esr_complex_impedances(self):
        cap, ind = self._reactive_pair(parallel=False)
        circuit = self.create(cap + ind + [{"type": "vac", "nodes": ["red", "gnd"],
            "params": {"vp": 1, "freq_hz": 100 / math.tau, "phase_deg": 0}}])
        circuit.set_analyze_type(2)
        circuit.set_ac_omega(100)
        circuit.analyze()
        sample = circuit.sample_complex(max_pins=2)
        expected_mid = complex(2, 20) / complex(7, 10)
        # The inductor ESR helper retains its actual external terminal.
        actual = self._complex_voltage(sample, len(cap) + 1)
        self.assertAlmostEqual(actual.real, expected_mid.real, places=8)
        self.assertAlmostEqual(actual.imag, expected_mid.imag, places=8)

    def test_parallel_capacitor_inductor_preserve_both_esr_complex_impedances(self):
        cap, ind = self._reactive_pair(parallel=True)
        parts = cap + ind
        circuit = self.create(parts + [self.resistor(10, "vin", "red"),
            {"type": "vac", "nodes": ["vin", "gnd"],
             "params": {"vp": 1, "freq_hz": 100 / math.tau, "phase_deg": 0}}])
        circuit.set_analyze_type(2)
        circuit.set_ac_omega(100)
        circuit.analyze()
        sample = circuit.sample_complex(max_pins=2)
        impedance = 1 / (1 / complex(5, -10) + 1 / complex(2, 20))
        expected_red = impedance / (10 + impedance)
        actual = self._complex_voltage(sample, len(parts), 1)
        self.assertAlmostEqual(actual.real, expected_red.real, places=8)
        self.assertAlmostEqual(actual.imag, expected_red.imag, places=8)

    def test_sine_with_offset_and_loading_each_native_sample(self):
        parts = convert(element("Sinewave Source"))
        circuit = self.create(parts + [self.resistor(900)])
        result = circuit.run_transient_trace(.0001, .05, sample_every=10,
            capture=lambda: circuit.sample_complex(max_pins=2))
        for point in result["samples"]:
            expected = .9 * (2 + 3 * math.sin(math.tau * 5 * point["time_s"]))
            self.assertAlmostEqual(self.voltage(point["sample"], len(parts)), expected, places=8)

    def test_square_with_nonhalf_duty_offset_and_loading(self):
        parts = convert(element("Square Source"))
        circuit = self.create(parts + [self.resistor(900)])
        result = circuit.run_transient_trace(.0001, .09, sample_every=10,
            capture=lambda: circuit.sample_complex(max_pins=2))
        checked = 0
        for point in result["samples"]:
            phase = point["time_s"] % .1
            if abs(phase - .025) < 1e-9:
                continue  # Compare ordinary intervals, not a floating-point edge convention.
            expected = .9 * (5 if phase < .025 else 1)
            self.assertAlmostEqual(self.voltage(point["sample"], len(parts)), expected, places=8)
            checked += 1
        self.assertGreater(checked, 80)

    def test_saw_rising_convention_offset_and_loading(self):
        parts = convert(element("Sawtooth Source"))
        circuit = self.create(parts + [self.resistor(900)])
        result = circuit.run_transient_trace(.0001, .075, sample_every=25,
            capture=lambda: circuit.sample_complex(max_pins=2))
        for point in result["samples"]:
            expected = .9 * (-1 + 6 * (point["time_s"] % .1) / .1)
            self.assertAlmostEqual(self.voltage(point["sample"], len(parts)), expected, places=8)

    def student_parts(self, **props):
        el = element("Student Source", **props)
        # This authored test circuit explicitly connects both return pins to
        # ground. The importer itself never adds either connection.
        el["pins"][1]["node"] = el["pins"][3]["node"] = "gnd"
        return convert(el)

    def test_student_enabled_independent_dc_and_ac_rms_into_loads(self):
        parts = self.student_parts()
        circuit = self.create(parts + [self.resistor(1000, "dc+"), self.resistor(1000, "ac+")])
        result = circuit.run_transient_trace(.0001, .02, sample_every=1,
            capture=lambda: circuit.sample_complex(max_pins=2))
        ac_values = []
        for point in result["samples"]:
            self.assertAlmostEqual(self.voltage(point["sample"], len(parts)), 3, places=8)
            voltage = self.voltage(point["sample"], len(parts) + 1)
            ac_values.append(voltage)
            self.assertAlmostEqual(voltage, 10 * math.sqrt(2) * math.sin(math.tau * 50 * point["time_s"]), places=8)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in ac_values) / len(ac_values)), 10, delta=.03)

    def test_student_off_outputs_do_not_short_external_biased_loads(self):
        parts = self.student_parts(**{"开关": 0})
        circuit = self.create(parts + [
            {"type": "vdc", "nodes": ["bias-dc", "gnd"], "params": {"v": 9}},
            {"type": "vdc", "nodes": ["bias-ac", "gnd"], "params": {"v": 7}},
            self.resistor(1000, "bias-dc", "dc+"), self.resistor(1000, "dc+"),
            self.resistor(1000, "bias-ac", "ac+"), self.resistor(1000, "ac+")])
        circuit.set_analyze_type(1)
        circuit.analyze()
        sample = circuit.sample_complex(max_pins=2)
        self.assertAlmostEqual(self.voltage(sample, len(parts) + 3), 4.5, delta=1e-7)
        self.assertAlmostEqual(self.voltage(sample, len(parts) + 5), 3.5, delta=1e-7)

    def test_student_explicit_series_resistance_loads_both_outputs(self):
        parts = self.student_parts(**{"内阻": 100})
        circuit = self.create(parts + [self.resistor(900, "dc+"), self.resistor(900, "ac+")])
        result = circuit.run_transient_trace(.0001, .005, sample_every=1,
            capture=lambda: circuit.sample_complex(max_pins=2))
        for point in result["samples"]:
            self.assertAlmostEqual(self.voltage(point["sample"], len(parts)), 2.7, places=8)
            expected = .9 * 10 * math.sqrt(2) * math.sin(math.tau * 50 * point["time_s"])
            self.assertAlmostEqual(self.voltage(point["sample"], len(parts) + 1), expected, places=8)

    def test_pulse_real_native_timing_peak_baseline_and_series_loading(self):
        parts = convert(element("Pulse Source"))
        circuit = self.create(parts + [self.resistor(900)])
        result = circuit.run_transient_trace(.0001, .2, sample_every=10,
            capture=lambda: circuit.sample_complex(max_pins=2))
        seen = set()
        for point in result["samples"]:
            phase = point["time_s"] % .1
            if phase < .005:
                raw = -1 + 6 * phase / .005
                seen.add("rise")
            elif phase < .01:
                raw = 5 - 6 * (phase - .005) / .005
                seen.add("fall")
            else:
                raw = -1
                seen.add("baseline")
            self.assertAlmostEqual(self.voltage(point["sample"], len(parts)), .9 * raw, places=7)
        self.assertEqual(seen, {"rise", "fall", "baseline"})


if __name__ == "__main__":
    unittest.main()
