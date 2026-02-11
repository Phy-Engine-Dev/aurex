import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aurex.contextdb import ContextDB  # noqa: E402
from aurex.tools.local_context import local_get_target_context  # noqa: E402
from aurex.tools.registry import ToolRuntime  # noqa: E402
from aurex.config import AurexConfig  # noqa: E402


class TestContextDB(unittest.TestCase):
    def test_upsert_and_get(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "context_db.json")
            db = ContextDB(path=p)
            db.upsert_target_comments(
                target_key="User:u1",
                target={"type": "User", "id": "u1"},
                comments=[
                    {"id": "c1", "ts_ms": 1, "author_id": "a", "author_nickname": "A", "text": "hi"},
                    {"id": "c2", "ts_ms": 2, "author_id": "b", "author_nickname": "B", "text": "hello"},
                ],
                keep_last=200,
            )
            got = db.get_target_context(target_key="User:u1", take=20)
            self.assertTrue(got.get("found"))
            self.assertEqual(got.get("target_key"), "User:u1")
            self.assertEqual(got.get("target"), {"type": "User", "id": "u1"})
            self.assertEqual([c.get("id") for c in got.get("comments")], ["c1", "c2"])

    def test_keep_last_trims(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "context_db.json")
            db = ContextDB(path=p)
            db.upsert_target_comments(
                target_key="User:u1",
                target={"type": "User", "id": "u1"},
                comments=[{"id": f"c{i}", "ts_ms": i, "text": str(i)} for i in range(10)],
                keep_last=3,
            )
            got = db.get_target_context(target_key="User:u1", take=20)
            self.assertEqual([c.get("id") for c in got.get("comments")], ["c7", "c8", "c9"])


class TestLocalContextTool(unittest.TestCase):
    def test_local_get_target_context_reads_db(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "context_db.json")
            db = ContextDB(path=p)
            db.upsert_target_comments(
                target_key="User:u1",
                target={"type": "User", "id": "u1"},
                comments=[{"id": "c1", "ts_ms": 1, "author_nickname": "A", "text": "hi"}],
                keep_last=200,
            )

            rt = ToolRuntime(
                task_id="T",
                user_lang="zh",
                config_path=os.path.join(td, "cfg.json"),
                config=AurexConfig(),
                cache_dir=td,
                user=None,
                planner_client=None,
            )
            out = local_get_target_context(rt, {"target_key": "User:u1", "take": 20})
            self.assertTrue(out.get("found"))
            self.assertEqual(out.get("target_key"), "User:u1")
            self.assertEqual(out.get("comments_count"), 1)


if __name__ == "__main__":
    unittest.main()

