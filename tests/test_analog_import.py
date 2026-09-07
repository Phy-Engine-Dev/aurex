import copy
import json
import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from aurex.tools.plar_analog_import import import_element
from aurex.tools.registry import ToolError


def element(kind, properties, statistics, pins):
    el = {"id": "original-id", "label": "original-label", "type": kind, "properties": properties,
          "statistics": statistics, "pins": [{"pin": i, "node": name} for i, name in pins.items()],
          "position": [1, 2, 3], "rotation": [4, 5, 6]}
    scene = {"components": [el], "wires": [{"Source": el["id"], "SourcePin": p, "Target": "other", "TargetPin": p} for p in pins]}
    return el, scene


class AnalogImportTests(unittest.TestCase):
    def test_unknown_is_not_claimed(self):
        self.assertIsNone(import_element({"type": "Unmodeled thing"}, scene={}))

    def test_opamp_pin_order_reference_and_identity(self):
        el, scene = element("Operational Amplifier", {"增益系数": 1e6, "最小电压": -15, "最大电压": 15}, {"电压+": 0}, {0: "negative", 2: "output"})
        before = copy.deepcopy(scene)
        out = import_element(el, scene=scene)[0]
        self.assertEqual(out["nodes"], ["gnd", "negative", "output", "gnd"])
        self.assertEqual((out["id"], out["label"]), ("original-id", "original-label"))
        self.assertEqual(out["pl_source"]["pin_mapping"], {"0": 1, "1": 0, "2": 2, "3": None})
        self.assertEqual(scene, before)
        self.assertFalse(out["pl_source"]["numerical_equivalence_to_original"])

    def test_wired_floating_input_is_never_grounded(self):
        el, scene = element("Operational Amplifier", {"增益系数": 1e6, "最小电压": -15, "最大电压": 15}, {"电压+": 0}, {0: "negative", 1: "floating-but-wired", 2: "output"})
        self.assertEqual(import_element(el, scene=scene)[0]["nodes"][0], "floating-but-wired")

    def test_unwired_input_nonzero_stat_refused(self):
        el, scene = element("Operational Amplifier", {"增益系数": 1e6, "最小电压": -15, "最大电压": 15}, {"电压+": .1}, {0: "negative", 2: "output"})
        with self.assertRaisesRegex(ToolError, "zero reference"):
            import_element(el, scene=scene)

    def test_triangle_preserves_inner_resistance(self):
        el, scene = element("Triangle Source", {"电压": 10, "偏移": 10, "频率": 20000, "占空比": .5, "内阻": .5}, {}, {0: "drive"})
        out = import_element(el, scene=scene)[0]
        self.assertEqual(out["params"], {"high_v": 20, "low_v": 0, "freq_hz": 20000, "phase_rad": 0, "duty": .5, "r_series": .5})
        self.assertEqual(out["nodes"], ["drive", "gnd"])
        self.assertIn("absolute voltage is not recorded", out["pl_source"]["implicit_references"][0]["policy"])

    def test_legacy_triangle_without_duty_uses_symmetric_public_default(self):
        props = {"电压": 3, "偏移": 0, "频率": 50, "内阻": .5}
        el, scene = element("Triangle Source", props, {}, {0: "drive"})
        out = import_element(el, scene=scene)[0]
        self.assertEqual(out["params"]["duty"], .5)
        self.assertNotIn("占空比", out["pl_source"]["raw_properties"])
        self.assertTrue(any("Legacy Triangle Source" in note
                            for note in out["pl_source"]["assumptions"]))

    def test_schmitt_preserves_unequal_levels_without_guessing_slew_units(self):
        props = {"低电准位": -.01, "高电准位": -.01, "负向阈值": 1, "正向阈值": 3, "工作模式": 1, "切变速率": .5}
        el, scene = element("Schmitt Trigger", props, {"输入电压": 0}, {1: "out"})
        out = import_element(el, scene=scene)[0]
        self.assertEqual(out["params"]["low_v"], -.01)
        self.assertEqual(out["pl_source"]["raw_properties"]["切变速率"], .5)
        el["properties"]["高电准位"] = 5
        out = import_element(el, scene=scene)[0]
        self.assertEqual(out["params"]["high_v"], 5)
        self.assertEqual(out["params"]["slew_v_per_s"], 0)
        self.assertEqual(out["pl_source"]["engineering_defaults"]["saved_legacy_slew"], .5)

    def test_meter_is_loading_not_short_and_bad_evidence_falls_back(self):
        stats = {"瞬间电压": 2, "瞬间电流": 2e-9, "电压": 3, "电流": 3e-9}
        el, scene = element("Multimeter", {"状态": 16}, stats, {0: "out", 1: "gnd"})
        out = import_element(el, scene=scene)[0]
        self.assertEqual(out["type"], "voltage_meter")
        self.assertTrue(math.isclose(out["params"]["r_input"], 1e9, rel_tol=1e-14))
        self.assertEqual(len(out["pl_source"]["loading_evidence"]), 2)
        el["statistics"]["电流"] = 3e-6
        self.assertIsNone(import_element(el, scene=scene))
        el["statistics"]["电流"] = 0
        self.assertIsNone(import_element(el, scene=scene))

    def test_meter_unknown_mode_routes_to_safe_broad_importer(self):
        el, scene = element("Multimeter", {"状态": 7}, {}, {0: "out", 1: "gnd"})
        self.assertIsNone(import_element(el, scene=scene))

    def test_current_public_schmitt_schema_derives_explicit_default_thresholds(self):
        el, scene = element("Schmitt Trigger", {"低电平": 0, "高电平": 3, "反相": 1}, {},
                            {0: "in", 1: "out"})
        out = import_element(el, scene=scene)[0]
        self.assertEqual(out["params"], {
            "threshold_low_v": 1, "threshold_high_v": 2, "inverted": 1,
            "low_v": 0, "high_v": 3, "slew_v_per_s": 0,
        })
        self.assertEqual(out["pl_source"]["engineering_defaults"]["schema"],
                         "current public PhysicsLab low/high/inverted")

    def test_bjt_full_properties_and_native_sidecar_retained(self):
        el, scene = element("Transistor", {"PNP": 1, "放大系数": 80, "最大功率": 250}, {}, {0:"base",1:"collector",2:"emitter"})
        el["native"]={"type":"pnp","params":{"is":2e-15,"beta_r":2.0}}
        out=import_element(el,scene=scene)[0]
        self.assertEqual(out["type"],"pnp")
        self.assertEqual(out["nodes"],["base","collector","emitter"])
        self.assertEqual(out["params"]["beta"],80)
        self.assertEqual(out["params"]["is"],2e-15)
        self.assertEqual(out["params"]["beta_r"],2.0)
        self.assertEqual(out["pl_source"]["raw_properties"]["最大功率"],250)
        self.assertFalse(out["pl_source"]["numerical_equivalence_to_original"])


if __name__ == "__main__":
    # Hermetic: even accidental future imports must not contact a real service.
    with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
        unittest.main()
