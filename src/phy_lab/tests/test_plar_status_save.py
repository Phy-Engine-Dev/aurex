import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from plar import get_status_save


class TestStatusSave(unittest.TestCase):
    def test_get_status_save_falls_back_when_physicslab_helper_fails(self):
        fake_physicslab = types.ModuleType("physicsLab")

        class Category:
            class _C:
                def __init__(self, value):
                    self.value = value

            Experiment = _C("Experiment")
            Discussion = _C("Discussion")

        class OpenMode:
            load_by_plar_app = 2

        class Experiment:
            def __init__(self, *_a, **_kw):
                raise RuntimeError("boom")

        fake_physicslab.Category = Category
        fake_physicslab.OpenMode = OpenMode
        fake_physicslab.Experiment = Experiment

        status = {"Elements": [{"ModelID": "Resistor"}], "Wires": []}

        class _User:
            def get_experiment(self, *_a, **_kw):
                return {"Status": 200, "Message": "", "Data": {"StatusSave": json.dumps(status)}}

        import plar as plar_mod

        orig_ensure = plar_mod.ensure_physicslab_importable
        try:
            plar_mod.ensure_physicslab_importable = lambda **_kw: None  # type: ignore[assignment]
            with tempfile.TemporaryDirectory() as td:
                with mock.patch.dict(sys.modules, {"physicsLab": fake_physicslab}):
                    got = get_status_save(
                        _User(),
                        summary_id="sid",
                        category_value="Experiment",
                        cache_dir=td,
                        ttl_sec=0,
                    )
            self.assertEqual(got["Elements"][0]["ModelID"], "Resistor")
        finally:
            plar_mod.ensure_physicslab_importable = orig_ensure  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()

