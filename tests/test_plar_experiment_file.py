import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from plar.api import get_experiment, get_experiment_file
from plar.errors import PLARError


SID = "6a9b5c31ec746130fc33154f"
CONTENT = "a7e6f61abe700a8782eb01a2"


class OriginalExperimentFileTests(unittest.TestCase):
    def fixture(self, kind=0):
        status = json.dumps({"Elements": [{"Identifier": "R1", "ModelID": "Resistor",
                             "Position": "0.0123,0.4567,0.8910", "Rotation": "0,90,0",
                             "Properties": {"电阻": 10}}], "Wires": [], "ExtraField": [1, 2]}, ensure_ascii=False)
        experiment = {"$type": "Original.Experiment", "Type": kind, "StatusSave": status,
                      "CameraSave": "{\"VisionCenter\":\"0,0,1\"}", "Paused": True, "Version": 2501}
        summary = {"ID": SID, "ContentID": CONTENT, "Subject": "original", "Tags": ["小作品"]}
        return summary, experiment

    def test_full_original_fields_and_positions_are_saved_with_correct_integrity_metadata(self):
        summary, experiment = self.fixture()
        original = {"Experiment": experiment, "Summary": summary, "CustomTopLevel": {"keep": 1}}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("plar.api.get_summary", return_value={"Status": 200, "Data": summary, "Token": "SECRET_TRANSPORT_TOKEN"}), \
             mock.patch("plar.api.get_experiment", return_value={"Status": 200, "Data": original, "AuthCode": "SECRET_AUTH"}), \
             mock.patch("plar.api.ensure_physicslab_importable", side_effect=AssertionError("SDK must not load/eval untrusted experiment")):
            out = get_experiment_file(object(), summary_id=SID, category_value="Experiment", cache_dir=directory)
            saved = Path(out["sav_path"]).read_bytes()
            decoded = json.loads(saved)
            self.assertEqual(decoded, original)
            self.assertEqual(decoded["Experiment"]["StatusSave"], experiment["StatusSave"])
            self.assertEqual(json.loads(decoded["Experiment"]["StatusSave"])["Elements"][0]["Position"], "0.0123,0.4567,0.8910")
            self.assertEqual(out["sha256"], hashlib.sha256(saved).hexdigest())
            self.assertEqual(out["bytes"], len(saved))
            self.assertEqual(out["elements_with_original_position"], 1)
            self.assertIs(out['external_write_performed'], False)
            self.assertNotIn('published', out)  # A download makes no assertion about source publication status.
            self.assertNotIn(b"SECRET", saved)
            again = get_experiment_file(object(), summary_id=SID, category_value="Experiment", cache_dir=directory)
            self.assertEqual(out["sav_path"], again["sav_path"])
            self.assertFalse(any(p.name.startswith(".download-") for p in Path(directory, "plar_experiments").iterdir()))

    def test_raw_experiment_api_block_is_wrapped_without_invented_layout(self):
        summary, experiment = self.fixture()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("plar.api.get_summary", return_value={"Data": summary}), \
             mock.patch("plar.api.get_experiment", return_value={"Data": experiment}):
            out = get_experiment_file(object(), summary_id=SID, category_value="Discussion", cache_dir=directory)
            saved = json.loads(Path(out["sav_path"]).read_text())
        self.assertEqual(saved, {"Experiment": experiment, "Summary": summary})
        self.assertEqual(out["category"], "Discussion")

    def test_non_electrical_or_missing_type_cannot_be_sent_to_engine(self):
        for kind in (None, 1, 3, "0", False):
            summary, experiment = self.fixture(kind)
            if kind is None:
                del experiment["Type"]
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, \
                 mock.patch("plar.api.get_summary", return_value={"Data": summary}), \
                 mock.patch("plar.api.get_experiment", return_value={"Data": experiment}), \
                 self.assertRaisesRegex(PLARError, "explicit Type=0"):
                get_experiment_file(object(), summary_id=SID, category_value="Experiment", cache_dir=directory)

    def test_get_experiment_resolves_summary_to_actual_content_id(self):
        summary, experiment = self.fixture()
        user = SimpleNamespace(token="testtoken", auth_code="testauth")
        with mock.patch("plar.api.get_summary", return_value={"Status": 200, "Data": summary}), \
             mock.patch("plar.api.post_json_no_env_proxy", return_value=(200, {"Data": experiment})) as post:
            get_experiment(user, summary_id=SID, category_value="Experiment")
        self.assertEqual(post.call_args.kwargs["payload"], {"ContentID": CONTENT})


if __name__ == "__main__":
    unittest.main()
