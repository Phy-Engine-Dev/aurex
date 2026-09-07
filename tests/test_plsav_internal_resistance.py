"""PLSAV internal losses are preserved as explicit PE topology."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aurex.config import AurexConfig
from aurex.tools.circuits import _spec_from_sav, circuit_analyze, circuit_create
from aurex.tools.registry import ToolError, ToolRuntime


def element(model: str, properties: dict, *, cid: str = "source-id") -> dict:
    return {
        "id": cid,
        "type": model,
        "label": "original label",
        "properties": properties,
        "pins": [{"pin": 0, "node": "red"}, {"pin": 1, "node": "black"}],
        "pin_count_known": True,
        "native": None,
        "position": [1.0, 2.0, 3.0],
        "rotation": [0.0, 0.0, 180.0],
    }


class PlsavInternalResistanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = ToolRuntime(
            "internal-resistance", "zh", str(self.root / "config.json"),
            AurexConfig(), self.temp.name,
        )
        self.source = self.root / "original.sav"
        self.source.write_text("{}")

    def convert(self, item: dict, *extra: dict) -> dict:
        scene = self.root / "scene.json"
        full = {"components": [item, *extra], "wires": [], "camera": {}}
        scene.write_text(json.dumps(full, ensure_ascii=False))
        with patch("aurex.tools.circuits._view", return_value={"artifact": {"netlist_path": str(scene)}}):
            return _spec_from_sav(self.runtime, self.source)

    def test_capacitor_esr_is_a_deterministic_series_resistor(self):
        original = element("Basic Capacitor", {"电容": 5.0, "内阻": .001, "理想模式": 0})
        before = copy.deepcopy(original)
        first = self.convert(original)
        second = self.convert(original)
        self.assertEqual(first, second)
        self.assertEqual(original, before)
        core, esr = first["components"]
        self.assertEqual((core["id"], core["type"], core["params"]),
                         ("source-id", "capacitor", {"c": 5.0}))
        self.assertEqual(core["nodes"][1], "black")
        self.assertNotEqual(core["nodes"][0], "red")
        self.assertEqual(esr["nodes"], ["red", core["nodes"][0]])
        self.assertEqual(esr["params"], {"r": .001})
        self.assertEqual(esr["plsav_import"]["role"], "series_internal_resistance")
        self.assertTrue(esr["plsav_import"]["original_plsav_unchanged"])
        self.assertEqual(core["label"], "original label")
        self.assertNotIn("label", esr)

    def test_inductor_and_battery_internal_resistance_are_series(self):
        for model, props, native_type, params in (
            ("Basic Inductor", {"电感": .2, "内阻": 2, "理想模式": 0}, "inductor", {"l": .2}),
            ("Battery Source", {"电压": 5, "内阻": 2}, "vdc", {"v": 5}),
        ):
            with self.subTest(model=model):
                core, resistor = self.convert(element(model, props))["components"]
                self.assertEqual((core["type"], core["params"]), (native_type, params))
                self.assertEqual(resistor["nodes"], ["red", core["nodes"][0]])
                self.assertEqual(resistor["params"], {"r": 2.0})

    def test_current_source_resistance_is_a_parallel_norton_shunt(self):
        core, shunt = self.convert(element(
            "Current Source", {"电流": .01, "内阻": 1e9}))["components"]
        self.assertEqual(core["type"], "idc")
        self.assertEqual(core["nodes"], ["red", "black"])
        self.assertEqual(shunt["nodes"], core["nodes"])
        self.assertEqual(shunt["params"], {"r": 1e9})
        self.assertEqual(shunt["plsav_import"]["role"], "parallel_internal_resistance")

    def test_zero_resistance_does_not_change_topology(self):
        result = self.convert(element(
            "Basic Capacitor", {"电容": 1e-3, "内阻": 0, "理想模式": 1}))
        self.assertEqual(len(result["components"]), 1)
        self.assertEqual(result["components"][0]["nodes"], ["red", "black"])
        self.assertNotIn("plsav_import", result["components"][0])

    def test_invalid_or_ambiguous_resistance_fails_closed(self):
        cases = [
            element("Basic Capacitor", {"电容": 1, "内阻": -1, "理想模式": 0}),
            element("Basic Capacitor", {"电容": 1, "内阻": "0.1", "理想模式": 0}),
            element("Transistor", {"放大系数": 100, "PNP": 0, "内阻": .1}),
        ]
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ToolError):
                self.convert(item)

    def test_conflicting_legacy_ideal_flag_retains_explicit_esr(self):
        result = self.convert(element(
            "Basic Inductor", {"电感": 1e-6, "内阻": .001, "理想模式": 1}
        ))
        self.assertEqual(result["components"][1]["params"], {"r": .001})
        self.assertTrue(any("理想模式=1" in note
                            for note in result["components"][0]["pl_source"]["assumptions"]))

    def test_generated_names_cannot_collide_with_original_ids_or_nodes(self):
        original = element("Basic Capacitor", {"电容": 1, "内阻": .1, "理想模式": 0})
        first = self.convert(original)
        helper = first["components"][1]
        occupied = element("Resistor", {"电阻": 10}, cid=helper["id"])
        occupied["pins"][0]["node"] = first["components"][0]["nodes"][0]
        result = self.convert(original, occupied)
        ids = [row["id"] for row in result["components"]]
        self.assertEqual(len(ids), len(set(ids)))
        generated = next(row for row in result["components"] if row.get("plsav_import", {}).get("is_helper"))
        self.assertNotEqual(generated["id"], occupied["id"])
        self.assertNotEqual(result["components"][0]["nodes"][0], occupied["pins"][0]["node"])


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "native PE build required")
class PlsavInternalResistanceNativeTests(unittest.TestCase):
    def test_battery_internal_resistance_changes_the_solved_load_voltage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
            cfg = AurexConfig()
            cfg = replace(cfg, phy_engine=replace(
                cfg.phy_engine,
                cmake_build_dir=str(build),
                phyengine_lib_path=str(build / "libphyengine.so"),
                verilog2plsav_path=str(build / "verilog2plsav"),
                run_timeout_sec=30,
            ))
            runtime = ToolRuntime(
                "internal-resistance-native", "zh", str(root / "config.json"), cfg, folder,
            )
            created = circuit_create(runtime, {"spec": {"components": [
                {"id": "V1", "type": "vdc", "nodes": ["supply", "gnd"], "params": {"v": 5}},
                {"id": "LOAD", "type": "resistor", "nodes": ["supply", "gnd"], "params": {"r": 10}},
            ]}})
            source = Path(created["sav_path"])
            value = json.loads(source.read_text())
            status = json.loads(value["Experiment"]["StatusSave"])
            next(row for row in status["Elements"] if row["Identifier"] == "V1")["Properties"]["内阻"] = 10
            value["Experiment"]["StatusSave"] = json.dumps(status, ensure_ascii=False)
            modified = source.with_name("battery-with-internal-resistance.sav")
            modified.write_text(json.dumps(value, ensure_ascii=False))

            result = circuit_analyze(runtime, {"path": str(modified), "analysis": "dc"})
            load = next(row for row in result["measurements"]["components"] if row["id"] == "LOAD")
            helper = next(row for row in result["measurements"]["components"]
                          if row["id"].startswith("__plsav_internal_resistance_"))
            # Imported catalogs use an explicit 1 Tohm GMIN to condition
            # disconnected meter terminals. Its sub-1e-10 V loading is part
            # of the recorded solve, so do not assert impossible 12-digit
            # equivalence to the ideal divider.
            self.assertAlmostEqual(load["voltage"][0], 2.5, places=9)
            self.assertAlmostEqual(abs(helper["derived_current_0_to_1"]["real"]), .25, places=9)


if __name__ == "__main__":
    unittest.main()
