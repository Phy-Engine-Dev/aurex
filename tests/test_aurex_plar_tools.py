import os
import sys
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aurex.config import AurexConfig  # noqa: E402
from aurex.tools.plar_tools import plar_oldest_by_user, plar_query_experiments  # noqa: E402
from aurex.tools.registry import ToolRuntime  # noqa: E402


class TestPlarOldestByUserTool(unittest.TestCase):
    def test_scans_pages_and_picks_oldest(self):
        calls = []
        user_id = "0123456789abcdef01234567"

        def fake_qe(_user, **kwargs):
            calls.append(dict(kwargs))
            skip = int(kwargs.get("skip") or 0)
            if skip == 0:
                return [
                    {"ID": "id1", "CreationDate": 300, "Subject": "A"},
                    {"ID": "id2", "CreationDate": 200, "Subject": "B"},
                ]
            if skip == 2:
                return [{"ID": "id3", "CreationDate": 100, "Subject": "C"}]
            return []

        rt = ToolRuntime(
            task_id="T",
            user_lang="zh",
            config_path=os.path.join(ROOT, "dummy.json"),
            config=AurexConfig(),
            cache_dir=os.path.join(ROOT, ".tmp"),
            user=object(),
            planner_client=None,
        )

        with mock.patch("aurex.tools.plar_tools.plar.query_experiments", side_effect=fake_qe):
            out = plar_oldest_by_user(rt, {"user_id": user_id, "category": "Experiment", "take": 2, "max_pages": 10})

        self.assertEqual(out["category"], "Experiment")
        self.assertEqual(out["user_id"], user_id)
        self.assertEqual(out["pages_scanned"], 2)
        self.assertFalse(bool(out.get("incomplete")))
        self.assertEqual(out["item"]["id"], "id3")
        self.assertEqual(out["item"]["subject"], "C")

        self.assertEqual(calls[0]["category"], "Experiment")
        self.assertEqual(calls[0]["user_id"], user_id)
        self.assertEqual(calls[0]["sort"], "Default")
        self.assertIsNone(calls[0]["days"])
        self.assertEqual(calls[0]["skip"], 0)
        self.assertIsNone(calls[0]["from_skip"])

        self.assertEqual(calls[1]["skip"], 2)
        self.assertEqual(calls[1]["from_skip"], "id2")

    def test_category_both_picks_earlier_creation_date(self):
        user_id = "0123456789abcdef01234567"

        def fake_qe(_user, **kwargs):
            cat = kwargs.get("category")
            if cat == "Experiment":
                return [{"ID": "e1", "CreationDate": 200, "Subject": "E"}]
            if cat == "Discussion":
                return [{"ID": "d1", "CreationDate": 100, "Subject": "D"}]
            return []

        rt = ToolRuntime(
            task_id="T",
            user_lang="zh",
            config_path=os.path.join(ROOT, "dummy.json"),
            config=AurexConfig(),
            cache_dir=os.path.join(ROOT, ".tmp"),
            user=object(),
            planner_client=None,
        )

        with mock.patch("aurex.tools.plar_tools.plar.query_experiments", side_effect=fake_qe):
            out = plar_oldest_by_user(rt, {"user_id": user_id, "category": "both", "take": 24, "max_pages": 2})

        self.assertEqual(out["category"], "both")
        self.assertEqual(out["pick"]["id"], "d1")

    def test_compact_item_extracts_nested_user_id(self):
        user_id = "0123456789abcdef01234567"
        exp_id = "aaaaaaaaaaaaaaaaaaaaaaaa"

        def fake_qe(_user, **kwargs):
            return [
                {
                    "ID": exp_id,
                    "CreationDate": 100,
                    "Subject": "S",
                    "User": {"ID": user_id, "Nickname": "Nick"},
                }
            ]

        rt = ToolRuntime(
            task_id="T",
            user_lang="zh",
            config_path=os.path.join(ROOT, "dummy.json"),
            config=AurexConfig(),
            cache_dir=os.path.join(ROOT, ".tmp"),
            user=object(),
            planner_client=None,
        )

        with mock.patch("aurex.tools.plar_tools.plar.query_experiments", side_effect=fake_qe):
            out = plar_oldest_by_user(rt, {"user_id": user_id, "category": "Experiment", "take": 24, "max_pages": 1})

        self.assertEqual(out["item"]["id"], exp_id)
        self.assertEqual(out["item"]["user_id"], user_id)


class TestPlarQueryExperimentsTool(unittest.TestCase):
    def _rt(self) -> ToolRuntime:
        return ToolRuntime(
            task_id="T",
            user_lang="zh",
            config_path=os.path.join(ROOT, "dummy.json"),
            config=AurexConfig(),
            cache_dir=os.path.join(ROOT, ".tmp"),
            user=object(),
            planner_client=None,
        )

    def test_normalizes_featured_tag_name_to_value(self):
        calls = []

        def fake_qe(_user, **kwargs):
            calls.append(dict(kwargs))
            return [{"ID": "id1", "CreationDate": 1, "Subject": "S", "Category": "Experiment", "User": {"ID": "u", "Nickname": "n"}}]

        rt = self._rt()
        with mock.patch("aurex.tools.plar_tools.plar.query_experiments", side_effect=fake_qe):
            out = plar_query_experiments(rt, {"category": "Experiment", "take": 1, "tags": ["Featured"]})

        self.assertTrue(isinstance(out, list) and out)
        self.assertEqual(calls[0]["tags"], ["精选"])

    def test_normalizes_tag_prefix_form(self):
        calls = []

        def fake_qe(_user, **kwargs):
            calls.append(dict(kwargs))
            return [{"ID": "id1", "CreationDate": 1, "Subject": "S", "Category": "Experiment"}]

        rt = self._rt()
        with mock.patch("aurex.tools.plar_tools.plar.query_experiments", side_effect=fake_qe):
            _out = plar_query_experiments(rt, {"category": "Experiment", "take": 1, "tags": ["Tag.Featured"]})

        self.assertEqual(calls[0]["tags"], ["精选"])

    def test_preserves_custom_tags(self):
        calls = []

        def fake_qe(_user, **kwargs):
            calls.append(dict(kwargs))
            return [{"ID": "id1", "CreationDate": 1, "Subject": "S", "Category": "Experiment"}]

        rt = self._rt()
        with mock.patch("aurex.tools.plar_tools.plar.query_experiments", side_effect=fake_qe):
            _out = plar_query_experiments(rt, {"category": "Experiment", "take": 1, "tags": ["多体系统"]})

        self.assertEqual(calls[0]["tags"], ["多体系统"])


if __name__ == "__main__":
    unittest.main()
