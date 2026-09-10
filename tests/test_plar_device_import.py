from __future__ import annotations

import copy
import unittest

from aurex.tools.plar_device_import import import_element


def multimeter(mode: float) -> tuple[dict, dict]:
    element = {
        "id": "meter", "type": "Multimeter",
        "properties": {"状态": mode, "锁定": 1.0},
        "statistics": {},
        "pins": [{"pin": 0, "node": "before"}, {"pin": 1, "node": "after"}],
        "position": [1, 2, 3], "rotation": [4, 5, 6],
    }
    scene = {
        "components": [element],
        "wires": [
            {"Source": "meter", "SourcePin": 0, "Target": "left", "TargetPin": 0},
            {"Source": "meter", "SourcePin": 1, "Target": "right", "TargetPin": 0},
        ],
    }
    return element, scene


class DeviceImportTests(unittest.TestCase):
    def test_multimeter_mode_7_preserves_low_resistance_current_path(self):
        element, scene = multimeter(7)
        before = copy.deepcopy(scene)
        result = import_element(element, scene=scene)
        self.assertEqual(scene, before)
        self.assertEqual(len(result), 1)
        meter = result[0]
        self.assertEqual(meter["id"], "meter")
        self.assertEqual(meter["type"], "resistor")
        self.assertEqual(meter["nodes"], ["before", "after"])
        self.assertEqual(meter["params"], {"r": 1e-9})
        self.assertEqual(meter["pl_source"]["decomposition_role"],
                         "low_resistance_current_input")
        self.assertEqual(meter["pl_source"]["measurement_contract"], {
            "saved_dial_state": 7.0,
            "quantity": "current",
            "native_observable": "derived_current_0_to_1",
            "burden_resistance_ohm": 1e-9,
        })

    def test_other_multimeter_modes_remain_high_impedance_fallbacks(self):
        element, scene = multimeter(3)
        meter = import_element(element, scene=scene)[0]
        self.assertEqual(meter["type"], "voltage_meter")
        self.assertEqual(meter["params"], {"r_input": 1e9})
        self.assertFalse(meter["pl_source"]["engineering_defaults"]["calibrated"])


if __name__ == "__main__":
    unittest.main()
