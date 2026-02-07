import os
import sys
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from plar import PLARError, upload_sav_as_experiment


class TestUploadPublish(unittest.TestCase):
    def test_publish_raises_on_non_200_submit(self):
        fake_physicslab = types.ModuleType("physicsLab")

        class Category:
            class _C:
                def __init__(self, value):
                    self.value = value

            Experiment = _C("Experiment")
            Discussion = _C("Discussion")

        class OpenMode:
            load_by_filepath = 1

        class Experiment:
            def __init__(self, *_a, **_kw):
                pass

            def edit_publish_info(self, **_kw):
                return self

            def _Experiment__upload(self, *_a, **_kw):
                return ({"Status": 403, "Message": "invalid", "Data": None}, {"Summary": {"Image": 0}})

        fake_physicslab.Category = Category
        fake_physicslab.OpenMode = OpenMode
        fake_physicslab.Experiment = Experiment

        # patch ensure_physicslab_importable to avoid touching env/sys.path
        import plar as plar_mod

        orig_ensure = plar_mod.ensure_physicslab_importable
        try:
            plar_mod.ensure_physicslab_importable = lambda **_kw: None  # type: ignore[assignment]
            with mock.patch.dict(sys.modules, {"physicsLab": fake_physicslab}):
                with self.assertRaises(PLARError) as ctx:
                    upload_sav_as_experiment(
                        user=object(),
                        sav_path="x.sav",
                        title="t",
                        introduction="i",
                        cache_dir="cache",
                        category_value="Experiment",
                        tags=None,
                    )
            self.assertIn("status=403", str(ctx.exception))
        finally:
            plar_mod.ensure_physicslab_importable = orig_ensure  # type: ignore[assignment]

    def test_publish_unwraps_locked_user_proxy(self):
        fake_physicslab = types.ModuleType("physicsLab")

        class Category:
            class _C:
                def __init__(self, value):
                    self.value = value

            Experiment = _C("Experiment")
            Discussion = _C("Discussion")

        class OpenMode:
            load_by_filepath = 1

        captured = {"upload_user": None, "confirm_args": None}

        class _RealUser:
            def confirm_experiment(self, summary_id, category, image_counter):
                captured["confirm_args"] = (summary_id, category.value, image_counter)

        class _LockedUser:
            def __init__(self, inner):
                self._user = inner

        class Experiment:
            def __init__(self, *_a, **_kw):
                pass

            def edit_publish_info(self, **_kw):
                return self

            def _Experiment__upload(self, user, *_a, **_kw):
                captured["upload_user"] = user
                return (
                    {"Status": 200, "Message": "", "Data": {"Summary": {"ID": "sid"}}},
                    {"Summary": {"Image": 0}},
                )

        fake_physicslab.Category = Category
        fake_physicslab.OpenMode = OpenMode
        fake_physicslab.Experiment = Experiment

        import plar as plar_mod

        orig_ensure = plar_mod.ensure_physicslab_importable
        try:
            plar_mod.ensure_physicslab_importable = lambda **_kw: None  # type: ignore[assignment]
            inner = _RealUser()
            locked = _LockedUser(inner)
            with mock.patch.dict(sys.modules, {"physicsLab": fake_physicslab}):
                info = upload_sav_as_experiment(
                    user=locked,
                    sav_path="x.sav",
                    title="t",
                    introduction="i",
                    cache_dir="cache",
                    category_value="Experiment",
                    tags=None,
                )
            self.assertIs(captured["upload_user"], inner)
            self.assertEqual(captured["confirm_args"], ("sid", "Experiment", 0))
            self.assertEqual(info["summary_id"], "sid")
        finally:
            plar_mod.ensure_physicslab_importable = orig_ensure  # type: ignore[assignment]
