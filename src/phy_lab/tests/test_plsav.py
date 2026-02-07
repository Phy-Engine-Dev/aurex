import json
import os
import sys
import tempfile
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from plsav import load_plsav_counts


class TestPlSav(unittest.TestCase):
    def test_counts_basic(self):
        data = {
            "Experiment": {
                "StatusSave": {
                    "Elements": [{}, {}, {}],
                    "Wires": [{}, {}],
                }
            }
        }
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.sav")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f)
            c = load_plsav_counts(p)
            self.assertEqual(c.elements, 3)
            self.assertEqual(c.wires, 2)


if __name__ == "__main__":
    unittest.main()

