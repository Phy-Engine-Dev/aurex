from __future__ import annotations

import math
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aurex.tools.plar_semiconductor_import import import_element


class PhysicsLabSemiconductorImportTests(unittest.TestCase):
    @staticmethod
    def rated_diode(kind: str, rating: float) -> dict:
        properties = {"前向压降": 0.6, "额定电流": rating, "击穿电压": 0.0}
        pin_count = 2
        if kind == "Rectifier":
            properties = {"前向压降": 0.8, "额定电流": rating}
            pin_count = 4
        nodes = [f"n{pin}" for pin in range(pin_count)]
        if pin_count == 2:
            nodes[1] = "gnd"
        return {
            "id": "D1",
            "type": kind,
            "properties": properties,
            "statistics": {},
            "pins": [{"pin": pin, "node": nodes[pin]}
                     for pin in range(pin_count)],
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
        }

    @classmethod
    def import_rated(cls, kind: str, rating: float) -> list[dict]:
        element = cls.rated_diode(kind, rating)
        result = import_element(
            element, scene={"components": [element], "wires": []})
        assert result is not None
        return result

    def test_ratings_do_not_rescale_basic_photo_or_rectifier_iv_curves(self):
        for kind in ("Basic Diode", "Photodiode", "Rectifier"):
            with self.subTest(kind=kind):
                one = self.import_rated(kind, 1.0)
                ten = self.import_rated(kind, 10.0)
                self.assertEqual(len(one), len(ten))
                for at_one, at_ten in zip(one, ten):
                    self.assertEqual(at_one["params"], at_ten["params"])
                    for diode, rating in ((at_one, 1.0), (at_ten, 10.0)):
                        derivation = diode["pl_source"]["parameter_derivation"]
                        self.assertEqual(derivation["saved_current_rating_a"], rating)
                        self.assertEqual(derivation["native_reference_current_a"], 1.0)
                        self.assertEqual(derivation["reference_current_source"],
                                         "explicit_engineering_default")
                        self.assertTrue(derivation["rating_used_only_by_damage_protection"])
                        self.assertNotIn("saved_working_current_a", derivation)

    def test_led_working_current_still_defines_its_curve(self):
        def imported(current: float) -> dict:
            element = {
                "id": "LED", "type": "Light-Emitting Diode",
                "properties": {"前向压降": 2.1, "工作电流": current,
                               "反向耐压": 6.0},
                "statistics": {},
                "pins": [{"pin": 0, "node": "a"}, {"pin": 1, "node": "k"}],
                "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
            }
            result = import_element(
                element, scene={"components": [element], "wires": []})
            assert result is not None
            return result[0]

        low, high = imported(0.01), imported(0.02)
        self.assertAlmostEqual(high["params"]["is"] / low["params"]["is"], 2.0)
        self.assertEqual(low["pl_source"]["parameter_derivation"]
                         ["saved_working_current_a"], 0.01)

    def test_real_saved_ten_amp_basic_diode_point_is_not_tenfold(self):
        # Community save `高精度运放计算电路` contains 1 A and 10 A rated
        # Basic Diodes that all record about 0.1 A at 0.4808915 V.  This is a
        # deterministic regression for that observed point without depending
        # on a mutable cache file.
        params = self.import_rated("Basic Diode", 10.0)[0]["params"]
        thermal_v = 8.617333262145e-5 * (params["temp_c"] + 273.15)
        current = params["is"] * math.expm1(
            0.480891546 / (params["n"] * thermal_v))
        self.assertAlmostEqual(current, 0.1, delta=1e-6)


@unittest.skipUnless(os.environ.get("AUREX_TEST_PHYENGINE_LIB"),
                     "optional native fixture library not configured")
class NativeSemiconductorImportTests(unittest.TestCase):
    def solve_basic_diode(self, rating: float) -> float:
        diode = PhysicsLabSemiconductorImportTests.import_rated(
            "Basic Diode", rating)[0]
        payload = {
            "lib_path": os.environ["AUREX_TEST_PHYENGINE_LIB"],
            "spec": {"analysis": "dc", "components": [
                {"id": "V", "type": "vdc", "nodes": ["src", "gnd"],
                 "params": {"v": 1.0}},
                {"id": "R", "type": "resistor", "nodes": ["src", "n0"],
                 "params": {"r": 10.0}},
                diode,
            ]},
        }
        process = subprocess.run(
            [sys.executable, "-m", "aurex.phy_engine.worker"],
            input=json.dumps(payload), text=True, capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        row = next(item for item in result["components"] if item["id"] == "D1")
        return row["voltage"][0]

    def test_native_dc_solution_is_independent_of_current_rating(self):
        at_one_amp = self.solve_basic_diode(1.0)
        at_ten_amp = self.solve_basic_diode(10.0)
        self.assertAlmostEqual(at_one_amp, at_ten_amp, places=12)
        self.assertAlmostEqual(at_one_amp, 0.4499, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
