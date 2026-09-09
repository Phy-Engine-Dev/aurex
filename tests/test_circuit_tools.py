from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aurex.config import AurexConfig
from aurex.phy_engine.catalog import PL_MAX_POWER_W
from aurex.tools.circuits import (_editable_component_manifest, _render_spec, _spatial_context,
                                  _summarize_numeric_series, circuit_analyze, circuit_compare_traces, circuit_create,
                                  circuit_edit, circuit_inspect,
                                  circuit_query_many, circuit_read_trace,
                                  normalize_spec, register_circuit_tools)
from aurex.tools.phy_engine import verilog_to_sav
from aurex.tools.registry import ToolError, ToolRegistry, ToolRuntime


def resistor_spec(r=10):
    return {"components": [{"id": "V1", "type": "vdc", "nodes": ["supply", "gnd"], "params": {"v": 5}},
                           {"id": "R1", "type": "resistor", "nodes": ["supply", "gnd"], "params": {"r": r}}]}


class CircuitValidationTests(unittest.TestCase):
    def test_catalog_and_unique_registration(self):
        registry = ToolRegistry()
        register_circuit_tools(registry)
        self.assertEqual(len(registry.list()), 9)
        self.assertIsNotNone(registry.get("circuit_read_stimulus"))
        self.assertIsNotNone(registry.get("circuit_query_many"))
        self.assertIsNotNone(registry.get("circuit_compare_traces"))
        self.assertTrue(all(t.parameters["type"] == "object" for t in registry.list()))

    def test_non_electrical_and_missing_types_rejected_before_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            rt = ToolRuntime("validation", "zh", str(ROOT / "test.config.json"), AurexConfig(), folder)
            for kind in [None, True, 0.0, "0", 1, 2]:
                source = Path(folder) / "invalid.sav"
                source.write_text(json.dumps({"Experiment": {"Type": kind}}))
                with self.subTest(kind=kind), self.assertRaisesRegex(ToolError, "integer Type=0"):
                    circuit_inspect(rt, {"path": str(source)})

    def test_duplicate_ids_nan_invalid_nodes_and_unknown_params_are_rejected(self):
        for spec in [
            {"components": [resistor_spec()["components"][0]] * 2},
            resistor_spec(float("nan")),
            {"components": [{"id": "R1", "type": "resistor", "nodes": ["a"], "params": {"r": 10}}]},
            {"components": [{"id": "R1", "type": "resistor", "nodes": ["a", "b"], "params": {"r": 10, "made_up": 1}}]},
        ]:
            with self.subTest(spec=spec), self.assertRaises(ToolError):
                normalize_spec(spec)

    def test_pl_export_uses_finite_maximum_power_ratings(self):
        self.assertEqual(PL_MAX_POWER_W, float.fromhex("0x1.fffffep+127"))
        rendered, exportable = _render_spec(normalize_spec({"components": [
            {"id": "V", "type": "vdc", "nodes": ["v", "gnd"], "params": {"v": 5}},
            {"id": "QN", "type": "npn", "nodes": ["b", "c", "gnd"], "params": {"beta": 100}},
            {"id": "QP", "type": "pnp", "nodes": ["b2", "c2", "v"], "params": {"beta": 100}},
        ]}))
        self.assertTrue(exportable)
        for component in rendered["components"]:
            self.assertEqual(component["properties"]["最大功率"], PL_MAX_POWER_W)
            self.assertTrue(component["properties"]["锁定"])
        self.assertEqual(rendered["components"][0]["properties"]["内阻"], 0)

    def test_native_edit_manifest_preserves_writable_spec_not_renderer_properties(self):
        spec = normalize_spec({"components": [
            {"id": "V1", "type": "vdc", "nodes": ["supply", "gnd"],
             "params": {"v": 5}, "position": [0.1, 0.2, 0.0]},
            {"id": "R1", "type": "resistor", "nodes": ["supply", "gnd"],
             "params": {"r": 10}, "position": [0.3, 0.2, 0.0]},
        ]})
        manifest = _editable_component_manifest(spec, ["R1", "V1"], limit=1)
        self.assertEqual(manifest, [
            {"id": "R1", "type": "resistor", "nodes": ["supply", "gnd"],
             "params": {"r": 10.0}, "pin_labels": ["1", "2"]},
        ])

    def test_spatial_context_has_directions_and_never_confuses_proximity_with_connectivity(self):
        data = {"components": [
            {"id": "R", "ref": "C1", "type": "Resistor", "position": [0, 0, 0],
             "pins": [{"node": "n"}, {"node": "gnd"}]},
            {"id": "C", "ref": "C2", "type": "Basic Capacitor", "position": [-1, 0, 0],
             "pins": [{"node": "other"}, {"node": "gnd"}]},
            {"id": "V", "ref": "C3", "type": "Battery Source", "position": [0, 2, 0],
             "pins": [{"node": "v"}, {"node": "return"}]},
        ]}
        context = _spatial_context(data, ["R"])
        self.assertEqual(context["relations"][0]["nearest"][0]["direction"], "left")
        self.assertTrue(context["relations"][0]["nearest"][0]["electrically_connected"])
        self.assertEqual(context["relations"][0]["nearest"][0]["shared_nodes"], ["gnd"])
        self.assertEqual(context["relations"][0]["nearest"][1]["direction"], "above")
        self.assertFalse(context["relations"][0]["nearest"][1]["electrically_connected"])

    def test_numeric_trace_summary_distinguishes_window_stability_from_settled_remainder(self):
        series = [(0, 0.0, 0.0), (1, 1.0, 0.05), (2, 2.0, 1.0),
                  (3, 3.0, 1.02), (4, 4.0, 1.01)]
        result = _summarize_numeric_series(
            series, window_s=1.0, stability_threshold=0.1, change_threshold=0.1)
        self.assertEqual(result["first_stable_sampled_window"]["start_time_s"], 0.0)
        self.assertEqual(result["settled_for_remainder"]["start_time_s"], 2.0)
        self.assertEqual(result["changes_at_or_above_threshold"], 1)
        self.assertEqual(result["largest_consecutive_change"]["to_time_s"], 2.0)

    def test_numeric_trace_summary_is_linear_and_does_not_claim_periodicity(self):
        # Alternating values defeat every long stable window.  This is the
        # quadratic worst case for a per-start rescan, while the monotonic
        # deque implementation remains linear in the recorded sample count.
        sample_count = 30_000
        series = [(index, index * .01, float(index & 1))
                  for index in range(sample_count)]
        started = time.perf_counter()
        result = _summarize_numeric_series(
            series, window_s=150.0, stability_threshold=.1,
            change_threshold=.5)
        elapsed = time.perf_counter() - started
        self.assertIsNone(result["first_stable_sampled_window"])
        self.assertEqual(result["changes_at_or_above_threshold"], sample_count - 1)
        self.assertEqual(result["direction_reversals_at_or_above_threshold"], sample_count - 2)
        self.assertTrue(result["multiple_threshold_changes_observed"])
        self.assertFalse(result["periodic_oscillation_tested"])
        self.assertIn("does not prove periodic oscillation", result["change_evidence_note"])
        self.assertNotIn("repeated_changes_observed", result)
        # Generous enough for slow CI, but an O(n^2) implementation on this
        # adversarial series cannot satisfy it.
        self.assertLess(elapsed, 3.0)


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "set AUREX_PHY_ENGINE_BUILD to run native engine integration tests")
class CircuitNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(build),
            verilog2plsav_path=str(build / "verilog2plsav"), phyengine_lib_path=str(build / "libphyengine.so"),
            verilog2plsav_args=["-O2", "--layout", "hier"], run_timeout_sec=30))
        self.runtime = ToolRuntime(task_id="circuit-native-test", user_lang="zh", config_path=str(ROOT / "test.config.json"),
            config=cfg, cache_dir=self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_cpp_pl_templates_give_every_damage_limited_model_maximum_power(self):
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        source = Path(self.temp.name) / "ratings.json"
        output = Path(self.temp.name) / "ratings.sav"
        components = []
        for index, model_id in enumerate(("Battery Source", "Transistor", "N-MOSFET", "P-MOSFET")):
            pins = 2 if model_id == "Battery Source" else 3
            components.append({"id": f"C{index}", "model_id": model_id,
                "properties": {"电压": 5} if model_id == "Battery Source" else {},
                "nodes": [f"n{index}-{pin}" for pin in range(pins)],
                "position": [index * .2, 0, 0], "rotation": [0, 0, 180]})
        source.write_text(json.dumps({"title": "ratings", "components": components}))
        subprocess.run([str(build / "circuit_view"), "create", str(source),
            str(Path(self.temp.name) / "ratings.svg"), str(Path(self.temp.name) / "ratings.netlist.json"),
            "0", "24", str(output)], check=True, capture_output=True, text=True)
        status = json.loads(json.loads(output.read_text())["Experiment"]["StatusSave"])
        self.assertEqual([row["Properties"]["最大功率"] for row in status["Elements"]],
                         [PL_MAX_POWER_W] * 4)
        self.assertTrue(all(not row["IsBroken"] for row in status["Elements"]))

    def test_10ohm_actual_dc_measurement_and_png_and_plsav_roundtrip(self):
        result = circuit_analyze(self.runtime, {"spec": resistor_spec(), "analysis": "dc", "with_image": True})
        resistor = next(c for c in result["measurements"]["components"] if c["id"] == "R1")
        self.assertAlmostEqual(resistor["voltage"][0] - resistor["voltage"][1], 5, places=8)
        self.assertAlmostEqual(resistor["derived_current_0_to_1"]["real"], .5, places=8)
        png = Path(result["images"][0]["path"])
        self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        svg = Path(result["artifact"]["svg_path"]).read_text()
        self.assertIn("V(p0)-V(p1)=5 V", svg)
        # Native PE state has a ground node, not an invented third component.
        self.assertIn("C2.p1=gnd", svg)
        self.assertEqual(len(result["netlist"]["components"]), 2)
        self.assertTrue(Path(result["state_path"]).is_file())
        self.assertNotIn("C3.p0=gnd", svg)
        self.assertIn("C2.p0=supply; C2.p1=gnd", svg)
        self.assertIn("C2.p0:supply", svg)
        self.assertAlmostEqual(resistor["voltage_across_0_to_1"]["real"], 5, places=8)
        created = circuit_create(self.runtime, {"spec": resistor_spec()})
        self.assertEqual(created["component_manifest"], [
            {"id": "V1", "type": "vdc"}, {"id": "R1", "type": "resistor"}])
        plsav = json.loads(Path(created["sav_path"]).read_text())
        status = json.loads(plsav["Experiment"]["StatusSave"])
        source = next(element for element in status["Elements"] if element["Identifier"] == "V1")
        self.assertEqual(source["Properties"]["最大功率"], PL_MAX_POWER_W)
        self.assertEqual(source["Properties"]["内阻"], 0)
        self.assertFalse(source["IsBroken"])
        loaded = circuit_inspect(self.runtime, {"path": result["sav_path"], "with_image": True})
        self.assertIn("C3 GND (1 pin)", Path(loaded["artifact"]["svg_path"]).read_text())
        resistor = next(c for c in loaded["netlist"]["components"] if c["id"] == "R1")
        self.assertEqual(resistor["properties"]["电阻"], 10)
        self.assertEqual(len(loaded["netlist"]["nodes"]), 2)
        self.assertIn("<circle", Path(loaded["artifact"]["svg_path"]).read_text())

    def test_edit_preserves_original_and_reconnects_components(self):
        created = circuit_create(self.runtime, {"spec": resistor_spec()})
        with self.assertRaisesRegex(ToolError, "no electrical or layout change"):
            circuit_edit(self.runtime, {"path": created["circuit_path"], "operations": [
                {"action": "update", "id": "R1", "params": {"r": 10}},
            ]})
        updated = circuit_edit(self.runtime, {"path": created["circuit_path"], "operations": [
            {"action": "update", "id": "R1", "params": {"r": 20}},
            {"action": "add", "component": {"id": "R2", "type": "resistor", "nodes": ["supply", "gnd"], "params": {"r": 20}}},
        ]})
        original = json.loads(Path(created["circuit_path"]).read_text())
        self.assertEqual(original["components"][1]["params"]["r"], 10)
        measured = circuit_analyze(self.runtime, {"path": updated["circuit_path"]})
        for c in measured["measurements"]["components"]:
            if c["type"] == "resistor":
                self.assertAlmostEqual(c["derived_current_0_to_1"]["real"], .25, places=8)

        # The same edit workflow starts directly from a PLSAV import. Focused
        # inspection supplies exact writable native params; no manual rebuild.
        focused = circuit_inspect(self.runtime, {
            "path": created["sav_path"], "focus_id": "R1", "limit": 2})
        contract = focused["edit_contract"]["components"]
        self.assertEqual([(row["id"], row["params"]["r"]) for row in contract], [("R1", 10.0)])
        imported_edit = circuit_edit(self.runtime, {"path": created["sav_path"], "operations": [
            {"action": "update", "id": "R1", "params": {"r": 40}},
        ]})
        imported_result = circuit_analyze(self.runtime, {"path": imported_edit["circuit_path"]})
        resistor = next(row for row in imported_result["measurements"]["components"] if row["id"] == "R1")
        self.assertAlmostEqual(resistor["derived_current_0_to_1"]["real"], .125, places=8)

    def test_ac_preserves_imaginary_voltage(self):
        spec = {"components": [
            {"id": "V1", "type": "vac", "nodes": ["in", "gnd"], "params": {"vp": 1, "freq_hz": 1000 / (2 * 3.141592653589793)}},
            {"id": "R1", "type": "resistor", "nodes": ["in", "out"], "params": {"r": 1000}},
            {"id": "C1", "type": "capacitor", "nodes": ["out", "gnd"], "params": {"c": 1e-6}},
        ]}
        result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "ac", "ac_omega": 1000})
        c = next(c for c in result["measurements"]["components"] if c["id"] == "C1")
        self.assertAlmostEqual(c["voltage"][0], .5, places=6)
        self.assertAlmostEqual(c["voltage_imag"][0], -.5, places=6)
        self.assertNotIn("sav_path", result)  # native VAC has no certified PL mapping

    def test_transient_actual_final_state(self):
        spec = {"components": [
            {"id": "V1", "type": "vdc", "nodes": ["in", "gnd"], "params": {"v": 1}},
            {"id": "R1", "type": "resistor", "nodes": ["in", "out"], "params": {"r": 1000}},
            {"id": "C1", "type": "capacitor", "nodes": ["out", "gnd"], "params": {"c": 1e-6}},
        ]}
        result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr", "tr_step": 1e-5, "tr_stop": .001})
        c = next(c for c in result["measurements"]["components"] if c["id"] == "C1")
        self.assertGreater(c["voltage"][0], .60)
        self.assertLess(c["voltage"][0], .66)
        self.assertEqual(result["measurements"]["transient"]["actual_stop_s"], .001)
        self.assertEqual(result["measurements"]["transient"]["completed_steps"], 100)

    def test_timed_analog_switch_is_applied_inside_main_transient(self):
        spec = {"components": [
            {"id": "V", "type": "vdc", "nodes": ["src", "gnd"], "params": {"v": 5}},
            {"id": "BUTTON", "type": "switch", "nodes": ["src", "out"], "params": {"closed": 0}},
            {"id": "LOAD", "type": "resistor", "nodes": ["out", "gnd"], "params": {"r": 1000}},
        ]}
        result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr",
            "tr_step": .1, "tr_stop": .5, "tr_sample_every": 1,
            "tr_interactions": [
                {"time_s": .2, "set": {"BUTTON": 1}},
                {"time_s": .4, "set": {"BUTTON": 0}},
            ]})
        state = json.loads(Path(result["state_path"]).read_text())
        trace = state["measurements"]["transient"]
        observed = [next(c for c in point["components"] if c["id"] == "LOAD")["voltage"][0]
                    for point in trace["samples"]]
        self.assertAlmostEqual(observed[0], 0, delta=1e-6)
        self.assertAlmostEqual(observed[1], 5, delta=1e-8)
        self.assertAlmostEqual(observed[2], 5, delta=1e-8)
        self.assertAlmostEqual(observed[3], 0, delta=1e-6)
        self.assertEqual([event["native_step"] for event in trace["interaction_events"]], [2, 4])
        self.assertTrue(trace["interaction_timing"]["all_requested_events_applied"])
        controls = circuit_inspect(self.runtime, {"path": result["state_path"], "controls_only": True})
        self.assertEqual([(row["id"], row["kind"]) for row in controls["controls"]],
                         [("V", "voltage_source"), ("BUTTON", "spst")])

    def test_timed_voltage_and_digital_controls_can_share_a_mixed_tr(self):
        spec = {"components": [
            {"id": "V", "type": "vdc", "nodes": ["supply", "gnd"], "params": {"v": 1}},
            {"id": "R", "type": "resistor", "nodes": ["supply", "gnd"], "params": {"r": 1000}},
            {"id": "IN", "type": "digital_input", "nodes": ["logic"], "params": {"state": 0}},
            {"id": "OUT", "type": "digital_output", "nodes": ["logic"], "params": {}},
        ]}
        result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr",
            "tr_step": .1, "tr_stop": .3, "tr_sample_every": 1, "digital_steps_per_tr_step": 2,
            "tr_interactions": [{"time_s": .2, "set": {"V": 3.25, "IN": 1}}]})
        state = json.loads(Path(result["state_path"]).read_text())
        trace = state["measurements"]["transient"]
        at_two = trace["samples"][1]
        self.assertAlmostEqual(next(c for c in at_two["components"] if c["id"] == "R")["voltage"][0], 3.25)
        self.assertEqual(next(c for c in at_two["components"] if c["id"] == "OUT")["digital"], [1])
        self.assertEqual(at_two["interaction_states"], {"V": 3.25, "IN": 1})

    def test_random4_unconnected_reset_uses_native_default_and_clock_advances(self):
        spec = {"components": [
            {"id": "CLK", "type": "digital_input", "nodes": ["clk"],
             "params": {"state": 0, "low_v": 0, "high_v": 3}},
            {"id": "RNG", "type": "digital_random4",
             "nodes": ["q3", "q2", "q1", "q0", "clk", "floating-reset"],
             "params": {"initial": 5, "low_v": 0, "high_v": 3}},
            {"id": "Q0", "type": "digital_output", "nodes": ["q0"],
             "params": {"low_v": 0, "high_v": 3}},
        ]}
        args = {"spec": spec, "analysis": "tr", "tr_step": .1, "tr_stop": .5,
                "tr_sample_every": 1, "digital_steps_per_tr_step": 4,
                "tr_interactions": [
                    {"time_s": .2, "set": {"CLK": 1}},
                    {"time_s": .3, "set": {"CLK": 0}},
                    {"time_s": .4, "set": {"CLK": 1}},
                ]}
        traces = []
        for _ in range(2):
            result = circuit_analyze(self.runtime, args)
            trace = json.loads(Path(result["state_path"]).read_text())["measurements"]["transient"]
            rows = []
            for point in trace["samples"]:
                rng = next(c for c in point["components"] if c["id"] == "RNG")
                rows.append((rng["model_state"]["lfsr_state"],
                             rng["model_state"]["unknown"],
                             next(c for c in point["components"] if c["id"] == "Q0")["digital"][0]))
            traces.append(rows)
        self.assertEqual(traces[0], traces[1])
        self.assertTrue(all(unknown == 0 for _, unknown, _ in traces[0]))
        self.assertGreater(len({state for state, _, _ in traces[0]}), 1)
        self.assertTrue(all(value in (0, 1) for _, _, value in traces[0]))

    def test_random4_seed_is_visible_when_its_clock_driver_remains_unknown(self):
        # A table-updated stateful model must publish its initial outputs once
        # even if the combinational component driving its clock evaluates X ->
        # X and therefore does not create an ordinary node-change event.
        spec = {"components": [
            {"id": "CLOCK_X", "type": "digital_xor",
             "nodes": ["floating-a", "floating-b", "clk"], "params": {}},
            {"id": "RNG", "type": "digital_random4",
             "nodes": ["unused-q3", "unused-q2", "unused-q1", "q0", "clk", "floating-reset"],
             "params": {"initial": 5, "low_v": 0, "high_v": 3}},
            {"id": "Q0", "type": "digital_output", "nodes": ["q0"],
             "params": {"low_v": 0, "high_v": 3}},
        ]}
        result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "dc"})
        rng = next(row for row in result["measurements"]["components"] if row["id"] == "RNG")
        output = next(row for row in result["measurements"]["components"] if row["id"] == "Q0")
        self.assertEqual(rng["model_state"], {"lfsr_state": 5.0, "unknown": 0.0})
        self.assertEqual(rng["digital"][3], 1)
        self.assertEqual(output["digital"], [1])

    def test_plsav_multiplier_allows_open_high_input_and_unused_outputs(self):
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        source = Path(self.temp.name) / "partial-multiplier.json"
        saved = Path(self.temp.name) / "partial-multiplier.sav"
        components = [
            {"id": name, "model_id": "Logic Input",
             "properties": {"开关": 1, "低电平": 0, "高电平": 3},
             "nodes": [name.lower()], "position": [index * .15, 0, 0]}
            for index, name in enumerate(("A0", "A1", "B0"))
        ]
        # PhysicsLab Multiplier pin order is p3,p2,p1,p0,b1,b0,a1,a0.
        # b1 and p1..p3 are deliberately open.  PhysicsLab evaluates the
        # missing high input as zero and still emits the consumed low bit.
        components += [
            {"id": "MUL", "model_id": "Multiplier", "properties": {},
             "nodes": ["open-p3", "open-p2", "open-p1", "p0", "b1", "b0", "a1", "a0"],
             "position": [.3, .3, 0]},
            {"id": "OUT", "model_id": "Logic Output", "properties": {},
             "nodes": ["p0"], "position": [.6, .3, 0]},
        ]
        source.write_text(json.dumps({"title": "partial multiplier", "components": components}))
        subprocess.run([str(build / "circuit_view"), "create", str(source),
            str(Path(self.temp.name) / "partial-multiplier.svg"),
            str(Path(self.temp.name) / "partial-multiplier.netlist.json"), "0", "24", str(saved)],
            check=True, capture_output=True, text=True)
        result = circuit_analyze(self.runtime, {"path": str(saved), "analysis": "dc"})
        output = next(row for row in result["measurements"]["components"] if row["id"] == "OUT")
        self.assertEqual(output["digital"], [1])

    def test_circuit_query_many_groups_targeted_matches_in_one_call(self):
        created = circuit_create(self.runtime, {"spec": resistor_spec(), "with_image": False})
        result = circuit_query_many(self.runtime, {
            "path": created["circuit_path"], "queries": ["V1", "R1"], "limit": 1,
        })
        self.assertEqual(result["query_count"], 2)
        self.assertFalse(result["with_image"])
        self.assertEqual([row["query"] for row in result["results"]], ["V1", "R1"])
        self.assertEqual([row["component_ids"][0] for row in result["results"]], ["V1", "R1"])
        self.assertTrue(all(len(row["component_ids"]) == 1 for row in result["results"]))
        self.assertEqual(result["results"][0]["components"], [
            {"id": "V1", "ref": "C1", "type": "Battery Source", "label": ""}])
        self.assertEqual(result["results"][1]["components"], [
            {"id": "R1", "ref": "C2", "type": "Resistor", "label": ""}])
        self.assertNotIn("component_catalog", result)
        self.assertNotIn("pins", result["results"][0]["components"][0])
        self.assertEqual(result["selected_fields"], ["identity"])
        self.assertLess(len(json.dumps(result)), 5000)
        detailed = circuit_query_many(self.runtime, {
            "path": created["circuit_path"], "queries": ["V1", "R1"], "limit": 1,
            "fields": ["pins", "properties.电阻", "edit.r"],
        })
        voltage = detailed["results"][0]["components"][0]
        resistor = detailed["results"][1]["components"][0]
        self.assertIn("total_connections", voltage["pins"][0])
        self.assertEqual(voltage["missing_fields"], ["properties.电阻", "edit.r"])
        self.assertEqual(resistor["properties"], {"电阻": 10})
        self.assertEqual(resistor["edit"], {"r": 10})
        self.assertNotIn("measurements", resistor)
        node_id = voltage["pins"][0]["node"]
        node = circuit_query_many(self.runtime, {
            "path": created["circuit_path"], "queries": [node_id, "N999999", "R1"], "limit": 8,
        })
        self.assertEqual(node["successful_query_count"], 2)
        self.assertEqual(node["failed_query_count"], 1)
        self.assertEqual(node["results"][0]["nodes"][0]["id"], node_id)
        self.assertEqual(node["results"][0]["nodes"][0]["total_connections"], 2)
        self.assertNotIn("connections", node["results"][0]["nodes"][0])
        self.assertTrue(node["results"][0]["components"][0]["matched_pins"])
        self.assertFalse(node["results"][1]["ok"])
        self.assertEqual(node["results"][2]["components"][0]["type"], "Resistor")
        repeated = circuit_query_many(self.runtime, {
            "path": created["circuit_path"], "queries": ["R1", "R1"], "limit": 1,
        })
        self.assertEqual([[c["id"] for c in row["components"]]
                          for row in repeated["results"]], [["R1"], ["R1"]])

        complete = circuit_query_many(self.runtime, {
            "path": created["circuit_path"], "queries": ["R1"], "all": True,
        })
        self.assertEqual(complete["selected_fields"], ["all"])
        self.assertIn("properties", complete["results"][0]["components"][0])
        self.assertIn("pins", complete["results"][0]["components"][0])
        with self.assertRaisesRegex(ToolError, "Unsupported.*properties"):
            circuit_query_many(self.runtime, {
                "path": created["circuit_path"], "queries": ["R1"],
                "fields": ["properties"],
            })
        with self.assertRaisesRegex(ToolError, "mutually exclusive"):
            circuit_query_many(self.runtime, {
                "path": created["circuit_path"], "queries": ["R1"],
                "fields": ["pins"], "all": True,
            })

        analyzed = circuit_analyze(self.runtime, {
            "path": created["circuit_path"], "analysis": "dc",
        })
        recorded = circuit_query_many(self.runtime, {
            "path": analyzed["state_path"], "queries": ["R1"],
            "fields": ["native_type", "measurements.voltage"],
        })["results"][0]["components"][0]
        self.assertEqual(recorded["native_type"], "resistor")
        self.assertEqual(recorded["measurements"]["voltage"], [5.0, 0.0])
        self.assertNotIn("current", recorded["measurements"])

    def test_circuit_query_many_reads_high_and_low_levels_independently(self):
        skeleton = Path(self.temp.name) / "query-levels.json"
        saved = Path(self.temp.name) / "query-levels.sav"
        skeleton.write_text(json.dumps({"title": "query levels", "components": [
            {"id": "IN", "model_id": "Logic Input",
             "properties": {"开关": 1, "低电平": 0.25, "高电平": 3.25},
             "nodes": ["signal"], "position": [0, 0, 0]},
            {"id": "OUT", "model_id": "Logic Output", "properties": {},
             "nodes": ["signal"], "position": [.2, 0, 0]},
        ]}))
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        subprocess.run([str(build / "circuit_view"), "create", str(skeleton),
            str(Path(self.temp.name) / "query-levels.svg"),
            str(Path(self.temp.name) / "query-levels.netlist.json"), "0", "8", str(saved)],
            check=True, capture_output=True, text=True)

        high = circuit_query_many(self.runtime, {
            "path": str(saved), "queries": ["C1"], "fields": ["properties.高电平"],
        })["results"][0]["components"][0]
        self.assertEqual(high["properties"], {"高电平": 3.25})
        self.assertNotIn("低电平", high["properties"])
        self.assertNotIn("pins", high)

        low = circuit_query_many(self.runtime, {
            "path": str(saved), "queries": ["C1"], "fields": ["properties.低电平"],
        })["results"][0]["components"][0]
        self.assertEqual(low["properties"], {"低电平": 0.25})
        self.assertNotIn("高电平", low["properties"])

        spatial = circuit_query_many(self.runtime, {
            "path": str(saved), "queries": ["C1"], "fields": ["spatial"],
        })["results"][0]
        self.assertIn("spatial", spatial)
        self.assertNotIn("spatial_context", spatial)

    def test_original_plsav_ref_resolves_stable_identity_after_native_expansion(self):
        # Imported components can expand into native helpers.  Here C1 owns a
        # core and helper, so the renderer-local C2 names the helper while the
        # original PLSAV C2 must still resolve LOAD by source_ref.
        provenance = lambda source_ref, role=None: {
            "model_id": "Basic Capacitor" if source_ref == "C1" else "Resistor",
            "parent_identifier": "CAP" if source_ref == "C1" else "LOAD",
            "source_ref": source_ref,
            **({"is_helper": True, "decomposition_role": role} if role else {}),
        }
        spec = {"components": [
            {"id": "CAP", "type": "capacitor", "nodes": ["internal", "gnd"],
             "params": {"c": 1e-6}, "pl_source": provenance("C1")},
            {"id": "CAP:esr", "type": "resistor", "nodes": ["vcc", "internal"],
             "params": {"r": 1.0},
             "pl_source": provenance("C1", "series_resistance")},
            {"id": "LOAD", "type": "resistor", "nodes": ["vcc", "gnd"],
             "params": {"r": 1000}, "pl_source": provenance("C2")},
            {"id": "V", "type": "vdc", "nodes": ["vcc", "gnd"],
             "params": {"v": 5}, "pl_source": {
                 "model_id": "Battery Source", "parent_identifier": "V", "source_ref": "C3"}},
        ]}
        analyzed = circuit_analyze(self.runtime, {"spec": spec, "analysis": "dc"})
        inspected = circuit_inspect(self.runtime, {
            "path": analyzed["state_path"], "query": "C2", "limit": 4})
        self.assertEqual(inspected["source_ref_query"], {
            "source_ref": "C2", "resolved_native_ids": ["LOAD"],
            "shown_native_ids": ["LOAD"],
            "stable_across_import_revisions": True,
        })
        load = next(row for row in inspected["netlist"]["components"]
                    if row["id"] == "LOAD")
        self.assertEqual(load["source_ref"], "C2")
        self.assertNotEqual(load["ref"], "C2")
        batch = circuit_query_many(self.runtime, {
            "path": analyzed["state_path"], "queries": ["C2", "C2"], "limit": 4})
        self.assertEqual(batch["results"][0]["components"][0]["source_ref"], "C2")
        self.assertEqual(batch["results"][0]["component_ids"], ["LOAD"])
        self.assertEqual(batch["results"][1]["component_ids"], ["LOAD"])
        expanded = circuit_inspect(self.runtime, {
            "path": analyzed["state_path"], "query": "C1", "limit": 1})
        self.assertEqual(expanded["source_ref_query"]["resolved_native_ids"],
                         ["CAP", "CAP:esr"])
        self.assertEqual(expanded["source_ref_query"]["shown_native_ids"], ["CAP"])
        self.assertTrue(expanded["pagination"]["has_more"])
        expanded_batch = circuit_query_many(self.runtime, {
            "path": analyzed["state_path"], "queries": ["C1"], "limit": 1})
        self.assertTrue(expanded_batch["results"][0]["ok"])
        self.assertEqual(expanded_batch["results"][0]["match_count"], 2)
        self.assertEqual(expanded_batch["results"][0]["component_ids"], ["CAP"])

    def test_position_aware_schematic_routes_exact_nodes_and_marks_partial_context(self):
        created = circuit_create(self.runtime, {"spec": {"components": [
            {"id": "V", "type": "vdc", "nodes": ["vcc", "gnd"], "params": {"v": 5},
             "position": [0, 0, 0]},
            {"id": "R1", "type": "resistor", "nodes": ["vcc", "mid"], "params": {"r": 1000},
             "position": [.2, 0, 0]},
            {"id": "R2", "type": "resistor", "nodes": ["mid", "gnd"], "params": {"r": 2000},
             "position": [.4, 0, 0]},
            {"id": "C", "type": "capacitor", "nodes": ["mid", "gnd"], "params": {"c": 1e-6},
             "position": [.4, .2, 0]},
        ]}, "with_image": True, "view": "schematic"})
        self.assertTrue(created["camera"]["schematic"])
        self.assertFalse(created["camera"]["geometry_mutated"])
        self.assertGreaterEqual(created["camera"]["junction_count"], 1)
        mid_node = next(component for component in created["netlist"]["components"]
                        if component["id"] == "R1")["pins"][1]["node"]
        svg = Path(created["artifact"]["svg_path"]).read_text()
        self.assertEqual(svg.count("data-component-id="), 5)  # explicit generated ground
        self.assertIn(f'data-node="{mid_node}"', svg)
        self.assertIn(f'data-junction-node="{mid_node}"', svg)
        self.assertIn("Crossing lines without a dot are not connected", svg)

        partial = circuit_inspect(self.runtime, {"path": created["circuit_path"],
            "focus_ids": ["R1", "R2"], "limit": 2,
            "with_image": True, "view": "schematic"})
        self.assertEqual(partial["camera"]["rendered_components"], 2)
        self.assertGreaterEqual(partial["camera"]["external_connection_stub_count"], 2)
        self.assertTrue(partial["camera"]["minimap"]["enabled"])
        partial_svg = Path(partial["artifact"]["svg_path"]).read_text()
        self.assertIn('marker-end="url(#external-arrow)"', partial_svg)
        self.assertIn('id="global-locator"', partial_svg)
        focused_data = circuit_inspect(self.runtime, {
            "path": created["circuit_path"], "focus_id": "R1", "limit": 3})
        nearest = focused_data["spatial_context"]["relations"][0]["nearest"]
        by_neighbor = {row["id"]: row for row in nearest}
        self.assertEqual(by_neighbor["V"]["direction"], "left")
        self.assertTrue(by_neighbor["V"]["electrically_connected"])
        focused_image = circuit_inspect(self.runtime, {
            "path": created["circuit_path"], "focus_id": "R1", "limit": 3,
            "with_image": True})
        self.assertEqual(focused_image["pagination"]["view"], "schematic")
        self.assertTrue(focused_image["camera"]["schematic"])

    def test_trace_comparison_distinguishes_within_run_change_from_cross_run_difference(self):
        def run(state):
            return circuit_analyze(self.runtime, {"spec": {"components": [
                {"id": "IN", "type": "digital_input", "nodes": ["a"], "params": {"state": state}},
                {"id": "NOT", "type": "digital_not", "nodes": ["a", "b"], "params": {}},
                {"id": "OUT", "type": "digital_output", "nodes": ["b"], "params": {}},
            ]}, "analysis": "tr", "tr_step": .1, "tr_stop": .2, "tr_sample_every": 1})

        first, second, changed = run(0), run(0), run(1)
        same = circuit_compare_traces(self.runtime, {
            "left_path": first["state_path"], "right_path": second["state_path"],
            "component_ids": ["IN", "OUT"],
        })
        self.assertTrue(same["time_aligned"])
        self.assertTrue(same["exact_match_across_runs"])
        self.assertEqual(same["left_sha256"], same["right_sha256"])
        different = circuit_compare_traces(self.runtime, {
            "left_path": first["state_path"], "right_path": changed["state_path"],
            "component_ids": ["IN", "OUT"],
        })
        self.assertFalse(different["exact_match_across_runs"])
        self.assertEqual(different["mismatch_component_ids"], ["IN", "OUT"])
        self.assertIn("not evidence of cross-run nondeterminism", different["note"])

    def test_random_generator_plsav_import_is_repeatable_and_keeps_saved_levels(self):
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        source = Path(self.temp.name) / "random-pl.json"
        saved = Path(self.temp.name) / "random-pl.sav"
        source.write_text(json.dumps({"title": "random import", "components": [
            {"id": "CLK", "model_id": "Logic Input",
             "properties": {"开关": 0, "低电平": 0, "高电平": 3},
             "nodes": ["clk"], "position": [0, 0, 0]},
            {"id": "RNG", "model_id": "Random Generator",
             "properties": {"低电平": 0, "高电平": 3},
             "nodes": ["q3", "q2", "q1", "q0", "clk", "unwired-reset"],
             "position": [.2, 0, 0]},
            {"id": "Q0", "model_id": "Logic Output", "properties": {},
             "nodes": ["q0"], "position": [.4, 0, 0]},
        ]}))
        subprocess.run([str(build / "circuit_view"), "create", str(source),
            str(Path(self.temp.name) / "random-pl.svg"),
            str(Path(self.temp.name) / "random-pl.netlist.json"), "0", "24", str(saved)],
            check=True, capture_output=True, text=True)
        args = {"path": str(saved), "analysis": "tr", "tr_step": .1, "tr_stop": .3,
                "tr_sample_every": 1, "digital_steps_per_tr_step": 4,
                "tr_interactions": [{"time_s": .2, "set": {"CLK": 1}}]}
        rows = []
        for _ in range(2):
            result = circuit_analyze(self.runtime, args)
            rng = next(c for c in result["measurements"]["components"] if c["id"] == "RNG")
            recorded = json.loads(Path(result["state_path"]).read_text())
            source_rng = next(c for c in recorded["spec"]["components"] if c["id"] == "RNG")
            self.assertIn("reset_n", rng["unconnected_pins"])
            rows.append((source_rng["params"], rng["model_state"], source_rng["pl_source"]))
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[0][0]["low_v"], 0)
        self.assertEqual(rows[0][0]["high_v"], 3)
        self.assertIn(rows[0][0]["initial"], range(1, 16))
        self.assertEqual(rows[0][1]["unknown"], 0)
        self.assertFalse(rows[0][2]["numerical_equivalence_to_original"])
        self.assertIn("surrogate seed", " ".join(rows[0][2]["assumptions"]))

    @staticmethod
    def precision_rectifier_spec(*, protected_output=False):
        output_node = "oaout" if protected_output else "out"
        components = [
            {"id": "V1", "type": "vdc", "nodes": ["vin", "gnd"],
             "params": {"v": 1}},
            {"id": "Rin", "type": "resistor", "nodes": ["vin", "neg"],
             "params": {"r": 10}},
            {"id": "Rg", "type": "resistor", "nodes": ["gnd", "pos"],
             "params": {"r": 1}},
            {"id": "D1", "type": "diode", "nodes": ["neg", "out"],
             "params": {"is": 9.177923434724038e-6, "n": 2, "isr": 0,
                        "nr": 2, "temp_c": 27, "ibv": .01, "bv": 40,
                        "bv_set": 0, "area": 1}},
            {"id": "OA", "type": "clamped_op_amp",
             "nodes": ["pos", "neg", output_node, "gnd"],
             "params": {"gain": 100, "min_v": -15, "max_v": 15}},
            {"id": "Rf", "type": "resistor", "nodes": ["pos", "out"],
             "params": {"r": 99}},
        ]
        if protected_output:
            components.insert(-1, {
                "id": "P", "type": "rated_protection",
                "nodes": ["oaout", "out", "oaout", "gnd"],
                "params": {"max_current_a": .12},
            })
        return {"components": components}

    def test_clamped_op_amp_precision_rectifier_converges_to_exact_hard_clamp(self):
        result = circuit_analyze(self.runtime, {
            "spec": self.precision_rectifier_spec(), "analysis": "dc",
            "g_min_siemens": 1e-12,
        })
        opamp = next(c for c in result["measurements"]["components"]
                     if c["id"] == "OA")
        raw = 100 * (opamp["voltage"][0] - opamp["voltage"][1])
        expected = max(-15, min(raw, 15))
        actual = opamp["voltage"][2] - opamp["voltage"][3]
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertAlmostEqual(actual, -.48089245, places=7)

        for rail in (15, 1000):
            with self.subTest(symmetric_rail_v=rail):
                saturated = circuit_analyze(self.runtime, {
                    "spec": {"components": [
                        {"id": "VP", "type": "vdc",
                         "nodes": ["p", "gnd"], "params": {"v": 1}},
                        {"id": "OA", "type": "clamped_op_amp",
                         "nodes": ["p", "gnd", "out", "gnd"],
                         "params": {"gain": 2000, "min_v": -rail,
                                    "max_v": rail}},
                        {"id": "RL", "type": "resistor",
                         "nodes": ["out", "gnd"], "params": {"r": 1000}},
                    ]}, "analysis": "dc",
                })
                row = next(c for c in saturated["measurements"]["components"]
                           if c["id"] == "OA")
                self.assertAlmostEqual(row["voltage"][2] - row["voltage"][3],
                                       rail, places=9)

    def test_protection_does_not_latch_on_unconverged_newton_probe(self):
        result = circuit_analyze(self.runtime, {
            "spec": self.precision_rectifier_spec(protected_output=True),
            "analysis": "dc", "g_min_siemens": 1e-12,
        })
        protection = next(c for c in result["measurements"]["components"]
                          if c["id"] == "P")["model_state"]
        self.assertEqual(protection["broken"], 0)
        self.assertEqual(protection["trip_mask"], 0)
        self.assertAlmostEqual(abs(protection["current_a"]), .1048089245,
                               places=7)
        self.assertLess(abs(protection["current_a"]), .12)

    def test_native_rated_protection_trips_current_voltage_or_power_and_opens(self):
        def run(limits):
            result = circuit_analyze(self.runtime, {"spec": {"components": [
                {"id": "V", "type": "vdc", "nodes": ["src", "gnd"], "params": {"v": 10}},
                {"id": "P", "type": "rated_protection",
                 "nodes": ["src", "load", "load", "gnd"], "params": limits},
                {"id": "R", "type": "resistor", "nodes": ["load", "gnd"], "params": {"r": 10}},
            ]}, "analysis": "dc"})
            return (next(c for c in result["measurements"]["components"] if c["id"] == "P"),
                    next(c for c in result["measurements"]["components"] if c["id"] == "R"))

        for limits, mask in [({"max_current_a": .5}, 1), ({"max_voltage_v": 5}, 2),
                             ({"max_power_w": 5}, 4)]:
            with self.subTest(limits=limits):
                protection, load = run(limits)
                state = protection["model_state"]
                self.assertEqual(state["broken"], 1)
                self.assertEqual(state["trip_mask"], mask)
                self.assertAlmostEqual(abs(state["trip_current_a"]), 1, places=8)
                self.assertAlmostEqual(abs(state["trip_voltage_v"]), 10, places=8)
                self.assertAlmostEqual(abs(state["trip_power_w"]), 10, places=8)
                self.assertLess(abs(load["voltage"][0]), 1e-6)
        protection, load = run({"max_current_a": 2, "max_voltage_v": 20, "max_power_w": 20})
        self.assertEqual(protection["model_state"]["broken"], 0)
        self.assertAlmostEqual(load["voltage"][0], 10, places=8)

    def test_native_rated_protection_accumulates_and_reports_transient_heat(self):
        spec = {"components": [
            {"id": "V", "type": "vdc", "nodes": ["src", "gnd"], "params": {"v": 10}},
            {"id": "P", "type": "rated_protection",
             "nodes": ["src", "load", "load", "gnd"],
             "params": {"max_current_a": .5}},
            {"id": "R", "type": "resistor", "nodes": ["load", "gnd"], "params": {"r": 10}},
        ]}
        short = circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr",
            "tr_step": .1, "tr_stop": 1.0, "tr_sample_every": 1})
        state = next(c for c in short["measurements"]["components"] if c["id"] == "P")["model_state"]
        self.assertEqual(state["broken"], 0)
        self.assertGreater(state["temperature_c"], 25)
        self.assertLess(state["temperature_c"], 150)
        self.assertFalse(short["protection_summary"]["has_new_trips"])
        self.assertEqual(short["protection_summary"]["alert"], "无熔断")

        long = circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr",
            "tr_step": .1, "tr_stop": 3.5, "tr_sample_every": 5})
        state = next(c for c in long["measurements"]["components"] if c["id"] == "P")["model_state"]
        self.assertEqual(state["broken"], 1)
        self.assertEqual(state["trip_mask"], 1)
        self.assertGreaterEqual(state["temperature_c"], 150)
        self.assertTrue(long["protection_summary"]["has_new_trips"])
        self.assertEqual(long["protection_summary"]["newly_tripped_this_run"][0]["component_id"], "P")
        self.assertIn("本轮新增熔断", long["protection_summary"]["alert"])
        inspected = circuit_inspect(self.runtime, {
            "path": long["state_path"], "query": "P", "with_image": False,
        })
        self.assertEqual(inspected["protection_summary"], long["protection_summary"])
        self.assertIn("本轮新增熔断", inspected["protection_summary"]["alert"])

    def test_native_rated_protection_ambient_temperature_changes_trip_time(self):
        def run(ambient):
            result = circuit_analyze(self.runtime, {"spec": {"components": [
                {"id": "V", "type": "vdc", "nodes": ["src", "gnd"], "params": {"v": 10}},
                {"id": "P", "type": "rated_protection",
                 "nodes": ["src", "load", "load", "gnd"],
                 "params": {"max_current_a": .5, "ambient_temp_c": ambient}},
                {"id": "R", "type": "resistor", "nodes": ["load", "gnd"], "params": {"r": 10}},
            ]}, "analysis": "tr", "tr_step": .1, "tr_stop": 1.5,
                "tr_sample_every": 5})
            return next(c for c in result["measurements"]["components"]
                        if c["id"] == "P")["model_state"]
        self.assertEqual(run(25)["broken"], 0)
        self.assertEqual(run(100)["broken"], 1)

    def test_plsav_battery_rating_and_saved_broken_state_reach_native_open_circuit(self):
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        source = Path(self.temp.name) / "rated-battery.json"
        saved = Path(self.temp.name) / "rated-battery.sav"
        source.write_text(json.dumps({"title": "rated battery", "components": [
            {"id": "BAT", "model_id": "Battery Source",
             "properties": {"电压": 10, "内阻": 0, "最大功率": 5, "锁定": 1},
             "nodes": ["src", "gnd"], "position": [0, 0, 0]},
            {"id": "R", "model_id": "Resistor",
             "properties": {"电阻": 10, "最大电阻": 1e7, "最小电阻": .1, "锁定": 1},
             "nodes": ["src", "gnd"], "position": [.2, 0, 0]},
        ]}))
        subprocess.run([str(build / "circuit_view"), "create", str(source),
            str(Path(self.temp.name) / "rated.svg"), str(Path(self.temp.name) / "rated.netlist.json"),
            "0", "24", str(saved)], check=True, capture_output=True, text=True)
        original = saved.read_bytes()
        result = circuit_analyze(self.runtime, {"path": str(saved), "analysis": "dc"})
        self.assertEqual(saved.read_bytes(), original)
        protection = next(c for c in result["measurements"]["components"]
                          if c["type"] == "rated_protection")
        load = next(c for c in result["measurements"]["components"] if c["id"] == "R")
        self.assertEqual(protection["model_state"]["trip_mask"], 4)
        self.assertGreater(abs(protection["model_state"]["trip_power_w"]), 5)
        self.assertLess(abs(load["voltage"][0]), 1e-6)
        self.assertEqual(protection["pl_source"]["limits"]["max_power_w"], 5)

        broken = Path(self.temp.name) / "already-broken.sav"
        payload = json.loads(saved.read_text())
        status = json.loads(payload["Experiment"]["StatusSave"])
        next(row for row in status["Elements"] if row["Identifier"] == "BAT")["IsBroken"] = True
        payload["Experiment"]["StatusSave"] = json.dumps(status, ensure_ascii=False)
        broken.write_text(json.dumps(payload, ensure_ascii=False))
        result = circuit_analyze(self.runtime, {"path": str(broken), "analysis": "dc"})
        protection = next(c for c in result["measurements"]["components"]
                          if c["type"] == "rated_protection")
        load = next(c for c in result["measurements"]["components"] if c["id"] == "R")
        self.assertEqual(protection["model_state"]["broken"], 1)
        self.assertEqual(protection["model_state"]["trip_mask"], 8)
        self.assertLess(abs(load["voltage"][0]), 1e-6)

    def test_invalid_timed_control_never_runs_or_substitutes_an_id(self):
        spec = {"components": [
            {"id": "S", "type": "switch", "nodes": ["n", "gnd"], "params": {"closed": 0}},
        ]}
        for interactions, message in [
            ([{"time_s": .15, "set": {"S": 1}}], "tr_step grid"),
            ([{"time_s": .1, "set": {"typo": 1}}], "unknown control ID"),
            ([{"time_s": .1, "set": {"S": 2}}], "must be one of"),
        ]:
            with self.subTest(interactions=interactions), self.assertRaisesRegex(ToolError, message):
                circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr",
                    "tr_step": .1, "tr_stop": .3, "tr_interactions": interactions})

    def test_transient_can_start_from_same_circuit_dc_operating_point(self):
        spec = {"components": [
            {"id": "V1", "type": "vdc", "nodes": ["in", "gnd"], "params": {"v": 1}},
            {"id": "R1", "type": "resistor", "nodes": ["in", "out"], "params": {"r": 1000}},
            {"id": "C1", "type": "capacitor", "nodes": ["out", "gnd"], "params": {"c": 1e-6}},
        ]}
        result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr",
            "tr_step": 1e-5, "tr_stop": .001, "tr_sample_every": 10,
            "tr_initialize_dc": True})
        capacitor = next(c for c in result["measurements"]["components"] if c["id"] == "C1")
        self.assertAlmostEqual(capacitor["voltage"][0], 1, places=6)
        initial = result["measurements"]["transient"]["initial_operating_point"]
        self.assertEqual(initial, {"requested": True, "completed": True,
            "analysis": "dc", "same_live_circuit": True})
        summary = result["numerical_verification"]["trace_summary"]
        self.assertEqual(summary["sample_count"], 10)
        self.assertAlmostEqual(summary["node_ranges_v"]["out"]["first_v"], 1, places=6)
        self.assertAlmostEqual(summary["node_ranges_v"]["out"]["last_v"], 1, places=6)
        self.assertAlmostEqual(summary["node_ranges_v"]["out"]["peak_to_peak_v"], 0, places=6)

    def test_transient_dc_initialization_is_explicit_and_analog_only(self):
        for args, message in [
            ({"spec": resistor_spec(), "analysis": "dc", "tr_initialize_dc": True}, "only to analysis=tr"),
            ({"spec": resistor_spec(), "analysis": "tr", "tr_initialize_dc": 1}, "must be boolean"),
            ({"spec": {"components": [
                {"id": "I", "type": "digital_input", "nodes": ["n"], "params": {"state": 0}},
                {"id": "O", "type": "digital_output", "nodes": ["n"], "params": {}},
            ]}, "analysis": "tr", "tr_initialize_dc": True}, "analog-only"),
        ]:
            with self.subTest(args=args), self.assertRaisesRegex(ToolError, message):
                circuit_analyze(self.runtime, args)

    def test_multimodule_verilog_produces_real_wired_experiment(self):
        result = verilog_to_sav(self.runtime, {"modules": [
            "module invert(input a, output y); assign y = ~a; endmodule",
            "module top(input a, output y); wire n; invert u0(a,n); invert u1(n,y); endmodule",
        ], "top": "top"})
        self.assertTrue(Path(result["sav_path"]).is_file())
        netlist = json.loads(Path(result["artifact"]["netlist_path"]).read_text())
        self.assertGreaterEqual(len(netlist["components"]), 2)
        self.assertGreaterEqual(len(netlist["wires"]), 1)
        self.assertFalse(result["published"])

    def test_transient_trace_is_measured_and_stops_exactly_at_half_second(self):
        import math
        spec = {"components": [
            {"id": "V1", "type": "vdc", "nodes": ["in", "gnd"], "params": {"v": 1}},
            {"id": "R1", "type": "resistor", "nodes": ["in", "out"], "params": {"r": 1000}},
            {"id": "C1", "type": "capacitor", "nodes": ["out", "gnd"], "params": {"c": .001}},
        ]}
        result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr", "tr_step": .00005,
                                               "tr_stop": .5, "tr_sample_every": 100})
        trace = json.loads(Path(result["state_path"]).read_text())["measurements"]["transient"]
        self.assertEqual(trace["actual_stop_s"], .5)
        self.assertEqual(trace["completed_steps"], 10000)
        self.assertEqual(trace["sample_count"], 100)
        self.assertEqual(trace["samples"][-1]["time_s"], .5)
        last = 0
        for point in trace["samples"]:
            self.assertGreater(point["time_s"], last)
            last = point["time_s"]
            self.assertEqual([row["id"] for row in point["components"]], ["V1", "R1", "C1"])
            cap = point["components"][2]
            # Native capacitor uses trapezoidal integration with zero initial
            # history: the first step has a documented half-step startup term.
            a = .00005 / 2
            expected = 1 - 1 / (1 + a) * ((1 - a) / (1 + a)) ** (point["completed_steps"] - 1)
            self.assertAlmostEqual(cap["voltage"][0], expected, places=6)
            self.assertTrue(all(math.isfinite(v) for row in point["components"] for v in row["voltage"]))
        self.assertEqual(trace["samples"][-1]["components"], result["measurements"]["components"])
        page = circuit_read_trace(self.runtime, {"path": result["state_path"], "nodes": ["out"], "offset": 99, "limit": 1})
        self.assertEqual(page["points"][0][0], .5)
        self.assertFalse(page["interpolated"])
        self.assertFalse(page["has_more"])
        self.assertNotIn("samples", result["measurements"]["transient"])
        summary = circuit_read_trace(self.runtime, {"path": result["state_path"],
            "mode": "summary", "nodes": ["out"], "stability_window_s": .05,
            "stability_threshold_v": .1, "change_threshold_v": .01})
        self.assertEqual(summary["mode"], "analog_nodes")
        self.assertEqual(summary["sample_scope"]["selected"], 100)
        self.assertGreater(summary["nodes"]["out"]["sampled_range"], .3)
        self.assertIsNotNone(summary["nodes"]["out"]["settled_for_remainder"])

    def test_trace_summary_reads_relay_model_state_without_scanning_sample_pages(self):
        spec = {"components": [
            {"id": "V", "type": "vdc", "nodes": ["coil", "gnd"], "params": {"v": 1}},
            {"id": "CONTACT", "type": "vdc", "nodes": ["com", "gnd"], "params": {"v": 5}},
            {"id": "RELAY", "type": "relay_current_spdt",
             "nodes": ["nc", "com", "no", "coil", "gnd"],
             "params": {"i_pull": .02, "r": 20, "l": 1e-6}},
            {"id": "RNC", "type": "resistor", "nodes": ["nc", "gnd"], "params": {"r": 1000}},
            {"id": "RNO", "type": "resistor", "nodes": ["no", "gnd"], "params": {"r": 1000}},
        ]}
        result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr",
            "tr_step": .01, "tr_stop": .05, "tr_sample_every": 1})
        summary = circuit_read_trace(self.runtime, {"path": result["state_path"],
            "mode": "summary", "component_ids": ["RELAY"],
            "stability_window_s": .01})
        relay = summary["components"][0]
        self.assertEqual(relay["type"], "relay_current_spdt")
        self.assertIn("engaged", relay["model_state"])
        self.assertEqual(relay["model_state"]["engaged"]["final"], 1.0)
        self.assertGreaterEqual(len(relay["pin_voltage_v"]), 5)

    def test_generic_bjt_roundtrip_and_observed_terminal_kcl(self):
        for kind, sign in [("npn", 1), ("pnp", -1)]:
            with self.subTest(kind=kind):
                spec = {"components": [
                    {"id": "VB", "type": "vdc", "nodes": ["b", "gnd"], "params": {"v": sign * .65}},
                    {"id": "VC", "type": "vdc", "nodes": ["c", "gnd"], "params": {"v": sign * 1}},
                    {"id": "Q", "type": kind, "nodes": ["b", "c", "gnd"],
                     "params": {"beta": 120, "beta_r": 2, "nr": 1.1}},
                ]}
                result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "dc"})
                q = result["measurements"]["components"][2]
                self.assertEqual(len(q["pin_current_a"]), 3)
                self.assertAlmostEqual(sum(q["pin_current_a"]), 0, places=12)
                self.assertGreater(q["pin_current_a"][1] * sign, 0)
                repeated = circuit_analyze(self.runtime, {"path": result["sav_path"], "analysis": "dc"})
                saved_spec = json.loads(Path(repeated["circuit_path"]).read_text())
                copied = next(c for c in saved_spec["components"] if c["id"] == "Q")
                self.assertEqual(copied["type"], kind)
                self.assertEqual(copied["params"]["beta"], 120)
                self.assertEqual(copied["params"]["beta_r"], 2)
                self.assertEqual(copied["params"]["nr"], 1.1)

    def test_digital_truth_table_preserves_unknown_state(self):
        spec = {"components": [
            {"id": "A", "type": "digital_input", "nodes": ["a"], "params": {"state": 0}},
            {"id": "B", "type": "digital_input", "nodes": ["b"], "params": {"state": 0}},
            {"id": "AND", "type": "digital_and", "nodes": ["a", "b", "y"]},
            {"id": "Y", "type": "digital_output", "nodes": ["y"]},
        ]}
        result = circuit_analyze(self.runtime, {"spec": spec, "analysis": "tr", "tr_step": 1e-8,
            "stimulus": [{"set": {"A": a, "B": b}} for a, b in [(0,0), (0,1), (1,0), (1,1), (1,2)]]})
        actual = [v["digital"]["Y"][0] for v in result["measurements"]["stimulus_results"]]
        self.assertEqual(actual, [0, 0, 0, 1, 2])

    def test_sequential_verilog_sav_can_be_analyzed(self):
        result = verilog_to_sav(self.runtime, {"verilog": "module top(input d, input clk, output reg q); always @(posedge clk) q <= d; endmodule", "top": "top"})
        analyzed = circuit_analyze(self.runtime, {"path": result["sav_path"], "analysis": "tr", "tr_step": 1e-8, "tr_stop": 1e-8})
        self.assertTrue(any(c["type"] == "digital_dff" for c in analyzed["measurements"]["components"]))

    def test_lossy_tristate_export_is_rejected(self):
        with self.assertRaisesRegex(ToolError, "strict export"):
            verilog_to_sav(self.runtime, {"verilog": "module top(input a, input en, output y); assign y = en ? a : 1'bz; endmodule", "top": "top"})

    def test_spatial_view_preserves_saved_xyz_rotation_and_relative_position(self):
        spec = {"components": [
            {"id": "IN", "type": "digital_input", "nodes": ["a"], "params": {"state": 1},
             "position": [-.125, .25, .375], "rotation": [0, 0, 35]},
            {"id": "OUT", "type": "digital_output", "nodes": ["a"],
             "position": [.125, -.25, .125], "rotation": [15, 20, 30]},
        ]}
        created = circuit_create(self.runtime, {"spec": spec})
        rendered = circuit_inspect(self.runtime, {"path": created["sav_path"], "projection": "top", "with_image": True})
        self.assertEqual(rendered["pagination"]["view"], "spatial")
        by_id = {c["id"]: c for c in rendered["netlist"]["components"]}
        for expected in spec["components"]:
            actual = by_id[expected["id"]]
            self.assertEqual(actual["position"], expected["position"])
            self.assertEqual(actual["rotation"], expected["rotation"])
            self.assertEqual(actual["position_source"], "provided")
        svg = ET.parse(rendered["artifact"]["svg_path"])
        locations = {g.attrib["data-component-id"]: g.attrib for g in svg.iter() if "data-component-id" in g.attrib}
        for g in svg.iter():
            if "data-component-id" in g.attrib:
                self.assertEqual(len(g.findall("{http://www.w3.org/2000/svg}polygon")), 6)
        self.assertLess(float(locations["IN"]["data-center-x"]), float(locations["OUT"]["data-center-x"]))
        self.assertLess(float(locations["IN"]["data-center-y"]), float(locations["OUT"]["data-center-y"]))
        topology = circuit_inspect(self.runtime, {"path": created["sav_path"], "view": "topology"})
        self.assertEqual(topology["pagination"]["view"], "topology")
        self.assertEqual(topology["netlist"]["components"], rendered["netlist"]["components"])
        focused = circuit_inspect(self.runtime, {"path": created["sav_path"], "focus_id": "IN", "limit": 4})
        self.assertEqual({c["id"] for c in focused["netlist"]["components"]}, {"IN", "OUT"})
        self.assertTrue(focused["pagination"]["focused"])
        queried = circuit_inspect(self.runtime, {"path": created["sav_path"], "query": "Logic Input", "limit": 4})
        self.assertEqual(queried["pagination"]["match_count"], 1)
        self.assertEqual(queried["netlist"]["components"], focused["netlist"]["components"])
        # Community API wrappers carry Type only in the original Experiment.
        original = json.loads(Path(created["sav_path"]).read_text())
        original.pop("Type")
        wrapped = Path(self.runtime.cache_dir) / "community.sav"
        wrapped.write_text(json.dumps(original))
        community = circuit_inspect(self.runtime, {"path": str(wrapped)})
        self.assertEqual(community["netlist"]["components"], rendered["netlist"]["components"])


if __name__ == "__main__":
    unittest.main()
