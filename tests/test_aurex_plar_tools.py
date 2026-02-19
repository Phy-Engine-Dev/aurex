import os
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aurex.config import AurexConfig  # noqa: E402
from aurex.tools.plar_tools import (
    plar_check_following,
    plar_get_oldest_comment,
    plar_oldest_by_user,
    plar_query_experiments,
    plar_upload_sav,
)  # noqa: E402
from aurex.tools.registry import ToolError, ToolRuntime  # noqa: E402


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


class TestPlarCheckFollowingTool(unittest.TestCase):
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

    def test_query_then_scan_finds_match(self):
        rt = self._rt()

        def fake_get_user_by_name(_user, *, name: str):
            if name == "goodenough":
                return {"User": {"ID": "0" * 24, "Nickname": "goodenough"}, "Statistic": {}}
            if name == "MapMaths":
                return {"User": {"ID": "1" * 24, "Nickname": "MapMaths"}, "Statistic": {}}
            raise RuntimeError("unknown user")

        # First: query path returns empty; then scan returns a page containing the followee.
        calls = []

        def fake_get_relations(_user, *, user_id: str, display_type, skip: int, take: int, query: str):
            calls.append({"user_id": user_id, "display_type": display_type, "skip": skip, "take": take, "query": query})
            if query:
                return []
            if skip == 0:
                return [{"ID": "1" * 24, "Nickname": "MapMaths"}]
            return []

        with mock.patch("aurex.tools.plar_tools.plar.get_user_by_name", side_effect=fake_get_user_by_name):
            with mock.patch("aurex.tools.plar_tools.plar.get_relations", side_effect=fake_get_relations):
                out = plar_check_following(rt, {"follower_name": "goodenough", "followee_name": "MapMaths", "max_pages": 3})

        self.assertTrue(out["is_following"])
        self.assertEqual(out["follower"]["id"], "0" * 24)
        self.assertEqual(out["followee"]["id"], "1" * 24)
        self.assertTrue(any(c["query"] == "MapMaths" for c in calls))
        self.assertTrue(any(c["query"] == "" for c in calls))


class TestPlarUploadSavTool(unittest.TestCase):
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

    def test_forces_discussion_and_returns_discussion_tag(self):
        with tempfile.TemporaryDirectory() as td:
            staged_dir = os.path.join(td, "staged_sav")
            os.makedirs(staged_dir, exist_ok=True)
            staged = os.path.join(staged_dir, "T.sav")
            with open(staged, "wb") as f:
                f.write(b"x")

            rt = ToolRuntime(
                task_id="T",
                user_lang="zh",
                config_path=os.path.join(ROOT, "dummy.json"),
                config=AurexConfig(),
                cache_dir=td,
                user=object(),
                planner_client=None,
            )

            calls: list[dict] = []

            def fake_upload(*, user, sav_path: str, title: str, introduction: str, cache_dir: str, category_value: str, tags=None):
                calls.append(
                    {
                        "sav_path": sav_path,
                        "title": title,
                        "introduction": introduction,
                        "cache_dir": cache_dir,
                        "category_value": category_value,
                        "tags": tags,
                    }
                )
                return {"summary_id": "a" * 24, "category": category_value}

            with mock.patch("aurex.tools.plar_tools.plar.upload_sav_as_experiment", side_effect=fake_upload):
                out = plar_upload_sav(
                    rt,
                    {
                        "sav_path": "/definitely/not/used.sav",
                        "title": "RRR：为什么我们看到的天空是蓝色的",
                        "introduction": "Intro",
                        "category": "Experiment",
                        "tags": ["物理"],
                    },
                )

            self.assertEqual(calls[0]["category_value"], "Discussion")
            self.assertEqual(calls[0]["sav_path"], staged)
            self.assertTrue(out.get("published"))
            self.assertEqual(out.get("category"), "Discussion")
            self.assertEqual(out.get("discussion_id"), "a" * 24)
            self.assertEqual(
                out.get("discussion_tag"),
                f"<discussion={'a' * 24}>RRR：为什么我们看到的天空是蓝色的</discussion>",
            )
            self.assertTrue("<discussion=" in str(out.get("reply_suggestion_zh") or ""))
            self.assertTrue(bool(out.get("sav_archived")))
            self.assertTrue(isinstance(out.get("sav_archive_name"), str) and out.get("sav_archive_name"))
            self.assertTrue(os.path.isfile(os.path.join(td, "log", "published_sav", str(out.get("sav_archive_name") or ""))))

    def test_requires_staged_cache_sav(self):
        with tempfile.TemporaryDirectory() as td:
            rt = ToolRuntime(
                task_id="T",
                user_lang="zh",
                config_path=os.path.join(ROOT, "dummy.json"),
                config=AurexConfig(),
                cache_dir=td,
                user=object(),
                planner_client=None,
            )
            with self.assertRaises(ToolError):
                plar_upload_sav(
                    rt,
                    {
                        "sav_path": "/definitely/not/found.sav",
                        "title": "T",
                        "introduction": "I",
                        "category": "Experiment",
                    },
                )


class TestPlarGetOldestCommentTool(unittest.TestCase):
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

    def test_pages_until_oldest(self):
        rt = self._rt()
        calls: list[int] = []

        def fake_get_comments(_user, *, target_id: str, target_type: str, take: int, skip: int):
            calls.append(int(skip))
            if skip == 0:
                return [
                    {"ID": "c3", "Timestamp": 300, "UserID": "u3", "Nickname": "n3", "Content": "t3"},
                    {"ID": "c2", "Timestamp": 200, "UserID": "u2", "Nickname": "n2", "Content": "t2"},
                ]
            if skip == 199:
                return [{"ID": "c1", "Timestamp": 100, "UserID": "u1", "Nickname": "n1", "Content": "t1"}]
            return []

        with mock.patch("aurex.tools.plar_tools.plar.get_comments", side_effect=fake_get_comments):
            out = plar_get_oldest_comment(
                rt,
                {"target_type": "User", "target_id": "0" * 24, "take": 2, "max_pages": 10},
            )

        self.assertEqual(calls, [0, 199])
        self.assertFalse(bool(out.get("incomplete")))
        oldest = out.get("oldest_comment")
        self.assertTrue(isinstance(oldest, dict))
        self.assertEqual(oldest.get("id"), "c1")
        self.assertEqual(oldest.get("author_nickname"), "n1")
        self.assertEqual(oldest.get("text"), "t1")


if __name__ == "__main__":
    unittest.main()
