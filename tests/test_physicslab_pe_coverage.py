from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aurex.config import AurexConfig
from aurex.phy_engine.catalog import PL_MAX_POWER_W
from aurex.tools.circuits import _load_spec, circuit_analyze, circuit_inspect
from aurex.tools.plar_damage import _CONTRACTS
from aurex.tools.registry import ToolRuntime


FIXTURE = ROOT / "tests" / "data" / "physicslab-all-circuit-elements-fa95b96.sav"
FIXTURE_SHA256 = "d086f2a43c9cd0fcf1197d80dbed0c6e7f41c89c6de605c4c041e506cf58389d"


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"),
                     "set AUREX_PHY_ENGINE_BUILD to run native engine integration tests")
class PhysicsLabToPECoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(
            cfg.phy_engine,
            cmake_build_dir=str(self.build),
            verilog2plsav_path=str(self.build / "verilog2plsav"),
            phyengine_lib_path=str(self.build / "libphyengine.so"),
            run_timeout_sec=60,
        ))
        self.runtime = ToolRuntime(
            task_id="physicslab-pe-coverage", user_lang="zh",
            config_path=str(ROOT / "test.config.json"), config=cfg,
            cache_dir=self.temp.name,
        )

    def tearDown(self):
        self.temp.cleanup()

    def make_sav(self, name: str, components: list[dict]) -> Path:
        source = Path(self.temp.name) / f"{name}.json"
        saved = Path(self.temp.name) / f"{name}.sav"
        source.write_text(json.dumps({"title": name, "components": components},
                                     ensure_ascii=False))
        subprocess.run([
            str(self.build / "circuit_view"), "create", str(source),
            str(Path(self.temp.name) / f"{name}.svg"),
            str(Path(self.temp.name) / f"{name}.netlist.json"),
            "0", "24", str(saved),
        ], check=True, capture_output=True, text=True)
        return saved

    @staticmethod
    def full_measurements(result: dict) -> list[dict]:
        state = json.loads(Path(result["state_path"]).read_text())
        return state["measurements"]["components"]

    def test_official_87_model_fixture_imports_and_solves_without_mutation(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE_SHA256)
        source = Path(self.temp.name) / "all-elements.sav"
        shutil.copy2(FIXTURE, source)
        original = source.read_bytes()
        status = json.loads(json.loads(original)["Experiment"]["StatusSave"])
        original_rows = status["Elements"]
        original_types = {row["ModelID"] for row in original_rows}
        self.assertEqual(len(original_rows), 91)
        self.assertEqual(len(original_types), 87)

        spec = _load_spec(self.runtime, str(source))
        self.assertEqual(spec["g_min_siemens"], 1e-12)
        self.assertEqual(len(spec["components"]), 204)
        implicit_low_inputs = [
            row for row in spec["components"]
            if row.get("pl_source", {}).get("decomposition_role")
            == "implicit_unconnected_input_low"
        ]
        self.assertEqual(len(implicit_low_inputs), 18)
        self.assertTrue(all(row["type"] == "digital_input"
                            and row["params"]["state"] == 0
                            and row.get("interaction") == {}
                            for row in implicit_low_inputs))
        imported_parent_ids = {
            row.get("pl_source", {}).get("parent_identifier", row["id"]).split(":", 1)[0]
            for row in spec["components"]
        }
        expected_parent_ids = {
            row["Identifier"] for row in original_rows
            if row["ModelID"] != "Ground Component"
        }
        self.assertEqual(imported_parent_ids, expected_parent_ids)
        parents_with_raw_properties = {
            row.get("pl_source", {}).get("parent_identifier", row["id"]).split(":", 1)[0]
            for row in spec["components"]
            if row.get("pl_source", {}).get("raw_properties") is not None
        }
        self.assertEqual(parents_with_raw_properties, expected_parent_ids)
        original_by_id = {row["Identifier"]: row for row in original_rows}
        for component in spec["components"]:
            if not component["type"].startswith("digital_"):
                continue
            parent = component.get("pl_source", {}).get(
                "parent_identifier", component["id"]
            ).split(":", 1)[0]
            properties = original_by_id[parent]["Properties"]
            if "低电平" in properties:
                self.assertEqual(component["params"]["low_v"], properties["低电平"])
            if "高电平" in properties:
                self.assertEqual(component["params"]["high_v"], properties["高电平"])
        guards_by_parent = {
            row["pl_source"]["parent_identifier"]: row
            for row in spec["components"] if row["type"] == "rated_protection"
        }
        for original_row in original_rows:
            contract = _CONTRACTS.get(original_row["ModelID"], {})
            expected_limits = {
                native_key: float(original_row["Properties"][contract[kind]])
                for kind, native_key in (("current", "max_current_a"),
                                         ("voltage", "max_voltage_v"),
                                         ("power", "max_power_w"))
                if kind in contract and contract[kind] in original_row["Properties"]
            }
            if any(expected_limits.values()) and not contract.get("mixed_only"):
                self.assertIn(original_row["Identifier"], guards_by_parent)
                for key, value in expected_limits.items():
                    self.assertEqual(guards_by_parent[original_row["Identifier"]]
                                     ["params"][key], value)

        result = circuit_analyze(self.runtime, {"path": str(source), "analysis": "dc"})
        self.assertEqual(source.read_bytes(), original)
        measured = self.full_measurements(result)
        self.assertEqual(len(measured), len(spec["components"]))
        self.assertEqual({row["id"] for row in measured},
                         {row["id"] for row in spec["components"]})
        self.assertTrue(all(math.isfinite(value)
                            for row in measured
                            for value in row.get("voltage", []) + row.get("current", [])))
        self.assertTrue({"spark_gap", "dc_motor", "ne555_timer", "rated_protection"}
                        <= {row["type"] for row in measured})

        transient_result = circuit_analyze(self.runtime, {
            "path": str(source), "analysis": "tr", "tr_step": .001,
            "tr_stop": .003, "tr_sample_every": 1,
            "digital_steps_per_tr_step": 2,
        })
        transient = json.loads(Path(transient_result["state_path"]).read_text())
        self.assertEqual(len(transient["measurements"]["components"]), len(spec["components"]))
        trace = transient["measurements"]["transient"]
        self.assertEqual(trace["completed_steps"], 3)
        self.assertEqual(trace["digital_propagation"]["completed_propagation_steps"], 6)
        self.assertEqual(source.read_bytes(), original)

    def test_official_87_model_fixture_renders_as_valid_paginated_schematic(self):
        source = Path(self.temp.name) / "all-elements-render.sav"
        shutil.copy2(FIXTURE, source)
        rendered_count = 0
        for offset in range(0, 91, 24):
            result = circuit_inspect(self.runtime, {
                "path": str(source), "with_image": True, "view": "schematic",
                "offset": offset, "limit": min(24, 91 - offset),
            })
            camera = result["camera"]
            self.assertTrue(camera["schematic"])
            self.assertFalse(camera["geometry_mutated"])
            self.assertEqual(camera["rendered_components"], min(24, 91 - offset))
            rendered_count += camera["rendered_components"]
            svg = Path(result["artifact"]["svg_path"])
            ET.parse(svg)  # valid UTF-8 XML, including Chinese labels/properties
            self.assertEqual(Path(result["images"][0]["path"]).read_bytes()[:8],
                             b"\x89PNG\r\n\x1a\n")
        self.assertEqual(rendered_count, 91)

    def test_saved_broken_state_isolates_every_official_device_terminal_set(self):
        payload = json.loads(FIXTURE.read_text())
        status = json.loads(payload["Experiment"]["StatusSave"])
        broken_ids = set()
        for row in status["Elements"]:
            if row["ModelID"] != "Ground Component":
                row["IsBroken"] = True
                broken_ids.add(row["Identifier"])
        payload["Experiment"]["StatusSave"] = json.dumps(status, ensure_ascii=False)
        source = Path(self.temp.name) / "all-elements-broken.sav"
        source.write_text(json.dumps(payload, ensure_ascii=False))

        result = circuit_analyze(self.runtime, {"path": str(source), "analysis": "dc"})
        measured = self.full_measurements(result)
        guards = [row for row in measured
                  if row["type"] == "rated_protection"
                  and row.get("model_state", {}).get("trip_mask") == 8]
        self.assertEqual({row["pl_source"]["parent_identifier"] for row in guards},
                         broken_ids)
        self.assertTrue(all(row["model_state"]["broken"] == 1 for row in guards))
        self.assertGreater(len(guards), len(broken_ids))  # multi-terminal isolation

    def test_spark_gap_breakdown_is_driven_by_saved_plsav_parameters(self):
        def solve(voltage: float) -> dict:
            saved = self.make_sav(f"spark-{voltage:g}", [
                {"id": "V", "model_id": "Battery Source",
                 "properties": {"电压": voltage, "内阻": 0, "最大功率": 1e30},
                 "nodes": ["src", "gnd"], "position": [0, 0, 0]},
                {"id": "G", "model_id": "Spark Gap",
                 "properties": {"击穿电压": 1000, "击穿电阻": 10, "维持电流": .001},
                 "nodes": ["src", "load"], "position": [.2, 0, 0]},
                {"id": "R", "model_id": "Resistor", "properties": {"电阻": 1000},
                 "nodes": ["load", "gnd"], "position": [.4, 0, 0]},
            ])
            result = circuit_analyze(self.runtime, {"path": str(saved), "analysis": "dc"})
            return next(row for row in self.full_measurements(result) if row["id"] == "G")

        below = solve(500)
        above = solve(1500)
        self.assertEqual(below["model_state"]["conducting"], 0)
        self.assertLess(abs(below["model_state"]["current_a"]), 1e-8)
        self.assertEqual(above["model_state"]["conducting"], 1)
        self.assertAlmostEqual(above["model_state"]["current_a"], 1500 / 1010,
                               places=6)
        self.assertAlmostEqual(above["model_state"]["voltage_v"],
                               above["model_state"]["current_a"] * 10, places=6)

    def test_fan_saved_electromechanical_parameters_drive_transient_state(self):
        saved = self.make_sav("fan", [
            {"id": "V", "model_id": "Battery Source",
             "properties": {"电压": 12, "内阻": 0, "最大功率": 1e30},
             "nodes": ["src", "gnd"], "position": [0, 0, 0]},
            {"id": "F", "model_id": "Electric Fan", "properties": {
                "额定电阻": 2, "电感": .01, "马达常数": .1,
                "转动惯量": .01, "负荷扭矩": .01,
                "反电动势系数": .1, "粘性摩擦系数": .01,
                "角速度": 0,
            }, "nodes": ["src", "gnd"], "position": [.2, 0, 0]},
        ])
        result = circuit_analyze(self.runtime, {
            "path": str(saved), "analysis": "tr", "tr_step": .01,
            "tr_stop": .2, "tr_sample_every": 5,
        })
        fan = next(row for row in self.full_measurements(result) if row["id"] == "F")
        state = fan["model_state"]
        self.assertGreater(state["angular_velocity_rad_s"], 0)
        self.assertGreater(state["torque_nm"], 0)
        self.assertGreater(state["back_emf_v"], 0)
        self.assertGreater(state["mechanical_power_w"], 0)
        self.assertAlmostEqual(state["voltage_v"], 12, places=8)

    def test_legacy_fan_with_only_rated_point_imports_without_inventing_dynamics(self):
        saved = self.make_sav("legacy-fan", [{
            "id": "F", "model_id": "Electric Fan",
            "properties": {"额定电压": 3.0, "额定功率": .85},
            "nodes": ["src", "gnd"], "position": [0, 0, 0],
        }])
        spec = _load_spec(self.runtime, str(saved))
        fan = next(row for row in spec["components"] if row["id"] == "F")
        self.assertEqual(fan["type"], "resistor")
        self.assertAlmostEqual(fan["params"]["r"], 3.0 * 3.0 / .85)
        self.assertEqual(fan["pl_source"]["support_level"],
                         "legacy_rated_point_electrical_load")
        self.assertIn("speed", fan["pl_source"]["unmodeled_behavior"])

    def test_legacy_simple_ammeter_uses_saved_range_as_missing_burden_resistance(self):
        saved = self.make_sav("legacy-ammeter", [{
            "id": "A", "model_id": "Simple Ammeter",
            "properties": {"量程": .0075, "名义量程": 3.0},
            "nodes": ["left", "common", "right"], "position": [0, 0, 0],
        }])
        spec = _load_spec(self.runtime, str(saved))
        branches = [row for row in spec["components"]
                    if row.get("pl_source", {}).get("parent_identifier") == "A"]
        self.assertEqual(len(branches), 2)
        self.assertTrue(all(row["params"]["r"] == .0075 for row in branches))
        self.assertTrue(all(row["pl_source"]["engineering_defaults"]
                            ["input_resistance_source"] == "量程" for row in branches))

    def test_555_saved_pin_topology_sets_and_resets_output(self):
        def solve(trigger: float, threshold: float) -> dict:
            saved = self.make_sav(f"timer-{trigger:g}-{threshold:g}", [
                {"id": "SUPPLY", "model_id": "Battery Source",
                 "properties": {"电压": 5, "内阻": 0, "最大功率": 1e30},
                 "nodes": ["vcc", "gnd"], "position": [0, 0, 0]},
                {"id": "TRIG", "model_id": "Battery Source",
                 "properties": {"电压": trigger, "内阻": 0, "最大功率": 1e30},
                 "nodes": ["trig", "gnd"], "position": [0, .2, 0]},
                {"id": "THR", "model_id": "Battery Source",
                 "properties": {"电压": threshold, "内阻": 0, "最大功率": 1e30},
                 "nodes": ["thr", "gnd"], "position": [0, .4, 0]},
                {"id": "T", "model_id": "555 Timer",
                 "properties": {"低电平": 0, "高电平": 3},
                 "nodes": ["vcc", "dis", "thr", "ctrl", "trig", "out", "reset", "gnd"],
                 "position": [.3, 0, 0]},
                {"id": "LOAD", "model_id": "Resistor", "properties": {"电阻": 1000},
                 "nodes": ["out", "gnd"], "position": [.6, 0, 0]},
            ])
            result = circuit_analyze(self.runtime, {"path": str(saved), "analysis": "dc"})
            return next(row for row in self.full_measurements(result) if row["id"] == "T")

        high = solve(0, 1)
        low = solve(3, 4)
        self.assertEqual(high["model_state"]["latched_high"], 1)
        self.assertAlmostEqual(high["voltage"][5], 3, places=5)
        self.assertEqual(low["model_state"]["latched_high"], 0)
        self.assertAlmostEqual(low["voltage"][5], 0, places=5)

    def test_current_public_schmitt_schema_runs_end_to_end(self):
        saved = self.make_sav("current-schmitt", [
            {"id": "VIN", "model_id": "Battery Source",
             "properties": {"电压": 2.5, "内阻": 0, "最大功率": 1e30},
             "nodes": ["in", "gnd"], "position": [0, 0, 0]},
            {"id": "S", "model_id": "Schmitt Trigger",
             "properties": {"低电平": 0, "高电平": 3, "反相": 0},
             "nodes": ["in", "out"], "position": [.2, 0, 0]},
            {"id": "R", "model_id": "Resistor", "properties": {"电阻": 1000},
             "nodes": ["out", "gnd"], "position": [.4, 0, 0]},
        ])
        result = circuit_analyze(self.runtime, {"path": str(saved), "analysis": "dc"})
        schmitt = next(row for row in self.full_measurements(result) if row["id"] == "S")
        self.assertEqual(schmitt["model_state"]["hysteresis_state"], 1)
        self.assertAlmostEqual(schmitt["model_state"]["output_v"], 3, places=6)
        self.assertEqual(schmitt["pl_source"]["engineering_defaults"]["schema"],
                         "current public PhysicsLab low/high/inverted")

    def test_nonideal_transformer_and_saved_winding_losses_reach_ac_solver(self):
        saved = self.make_sav("nonideal-transformer", [
            {"id": "SRC", "model_id": "Sinewave Source", "properties": {
                "电压": 10, "偏移": 0, "频率": 50, "占空比": .2, "内阻": 0,
            }, "nodes": ["pri", "gnd"], "position": [0, 0, 0]},
            {"id": "T", "model_id": "Transformer", "properties": {
                "输入电压": 10, "输出电压": 5, "额定功率": 100,
                "耦合系数": .8, "初级电阻": 1, "次级电阻": 2,
            }, "nodes": ["pri", "gnd", "sec", "gnd"], "position": [.2, 0, 0]},
            {"id": "R", "model_id": "Resistor", "properties": {"电阻": 10},
             "nodes": ["sec", "gnd"], "position": [.4, 0, 0]},
        ])
        original = saved.read_bytes()
        result = circuit_analyze(self.runtime, {
            "path": str(saved), "analysis": "ac", "ac_omega": math.tau * 50,
        })
        self.assertEqual(saved.read_bytes(), original)
        rows = self.full_measurements(result)
        transformer = next(row for row in rows if row["id"] == "T")
        helpers = [row for row in rows
                   if row.get("pl_source", {}).get("parent_identifier") == "T"
                   and row["type"] == "resistor"]
        self.assertEqual(sorted(row["effective_params"]["r"] for row in helpers), [1, 2])
        self.assertEqual(transformer["type"], "coupled_inductors")
        recorded = json.loads(Path(result["state_path"]).read_text())
        native_transformer = next(row for row in recorded["spec"]["components"]
                                  if row["id"] == "T")
        self.assertEqual(native_transformer["params"]["k"], .8)
        self.assertTrue(all(math.isfinite(value) for value in transformer["current"]
                            + transformer["current_imag"]))

    def test_plsav_ratings_trip_real_load_and_fuse_to_open_circuit(self):
        lamp_sav = self.make_sav("lamp-overload", [
            {"id": "V", "model_id": "Battery Source",
             "properties": {"电压": 6, "内阻": 0, "最大功率": 1e30},
             "nodes": ["src", "gnd"], "position": [0, 0, 0]},
            {"id": "L", "model_id": "Incandescent Lamp",
             "properties": {"额定电压": 3, "额定功率": .3},
             "nodes": ["src", "gnd"], "position": [.2, 0, 0]},
        ])
        lamp_result = circuit_analyze(self.runtime, {"path": str(lamp_sav), "analysis": "dc"})
        lamp_rows = self.full_measurements(lamp_result)
        lamp_guard = next(row for row in lamp_rows if row["type"] == "rated_protection"
                          and row["pl_source"]["parent_identifier"] == "L")
        lamp = next(row for row in lamp_rows if row["id"] == "L")
        self.assertEqual(lamp_guard["model_state"]["broken"], 1)
        self.assertEqual(lamp_guard["model_state"]["trip_mask"], 6)
        self.assertLess(abs(lamp["voltage_across_0_to_1"]["real"]), 1e-6)

        fuse_sav = self.make_sav("fuse-overload", [
            {"id": "V", "model_id": "Battery Source",
             "properties": {"电压": 10, "内阻": 0, "最大功率": 1e30},
             "nodes": ["src", "gnd"], "position": [0, 0, 0]},
            {"id": "F", "model_id": "Fuse Component",
             "properties": {"开关": 1, "额定电流": 1, "熔断电流": 2},
             "nodes": ["src", "load"], "position": [.2, 0, 0]},
            {"id": "R", "model_id": "Resistor", "properties": {"电阻": 1},
             "nodes": ["load", "gnd"], "position": [.4, 0, 0]},
        ])
        fuse_result = circuit_analyze(self.runtime, {"path": str(fuse_sav), "analysis": "dc"})
        fuse_rows = self.full_measurements(fuse_result)
        fuse_guard = next(row for row in fuse_rows if row["type"] == "rated_protection"
                          and row["pl_source"]["parent_identifier"] == "F")
        load = next(row for row in fuse_rows if row["id"] == "R")
        self.assertEqual(fuse_guard["model_state"]["broken"], 1)
        self.assertEqual(fuse_guard["model_state"]["trip_mask"], 1)
        self.assertGreater(abs(fuse_guard["model_state"]["trip_current_a"]), 2)
        self.assertLess(abs(load["voltage"][0]), 1e-6)

    def test_plsav_gate_maximum_current_trips_only_for_real_analog_load(self):
        saved = self.make_sav("gate-overload", [
            {"id": "IN", "model_id": "Logic Input",
             "properties": {"低电平": 0, "高电平": 3, "开关": 1},
             "nodes": ["in"], "position": [0, 0, 0]},
            {"id": "BUF", "model_id": "Yes Gate",
             "properties": {"低电平": 0, "高电平": 3, "最大电流": .1},
             "nodes": ["in", "out"], "position": [.2, 0, 0]},
            {"id": "LOAD", "model_id": "Resistor", "properties": {"电阻": 1},
             "nodes": ["out", "gnd"], "position": [.4, 0, 0]},
        ])
        original = saved.read_bytes()
        result = circuit_analyze(self.runtime, {"path": str(saved), "analysis": "dc"})
        self.assertEqual(saved.read_bytes(), original)
        rows = self.full_measurements(result)
        guard = next(row for row in rows if row["type"] == "rated_protection"
                     and row["pl_source"]["parent_identifier"] == "BUF")
        load = next(row for row in rows if row["id"] == "LOAD")
        self.assertEqual(guard["model_state"]["broken"], 1)
        self.assertEqual(guard["model_state"]["trip_mask"], 1)
        self.assertGreater(abs(guard["model_state"]["trip_current_a"]), .1)
        self.assertLess(abs(load["voltage"][0]), 1e-6)

        pure = self.make_sav("gate-pure-digital", [
            {"id": "IN", "model_id": "Logic Input",
             "properties": {"低电平": 0, "高电平": 3, "开关": 1},
             "nodes": ["in"], "position": [0, 0, 0]},
            {"id": "BUF", "model_id": "Yes Gate",
             "properties": {"低电平": 0, "高电平": 3, "最大电流": .1},
             "nodes": ["in", "out"], "position": [.2, 0, 0]},
            {"id": "OUT", "model_id": "Logic Output",
             "properties": {"低电平": 0, "高电平": 3, "状态": 0},
             "nodes": ["out"], "position": [.4, 0, 0]},
        ])
        pure_spec = _load_spec(self.runtime, str(pure))
        self.assertFalse(any(row["type"] == "rated_protection"
                             and row.get("pl_source", {}).get("parent_identifier") == "BUF"
                             for row in pure_spec["components"]))

    def test_unlimited_float32_power_sentinel_does_not_create_a_trip_guard(self):
        saved = self.make_sav("unlimited-source", [{
            "id": "V", "model_id": "Battery Source",
            "properties": {"电压": 5, "内阻": 0, "最大功率": PL_MAX_POWER_W},
            "nodes": ["n", "gnd"], "position": [0, 0, 0],
        }])
        original = saved.read_bytes()
        spec = _load_spec(self.runtime, str(saved))
        self.assertFalse(any(row["type"] == "rated_protection"
                             and row.get("pl_source", {}).get("parent_identifier") == "V"
                             for row in spec["components"]))
        self.assertEqual(saved.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
