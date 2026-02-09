import os
import sys
import tempfile
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from state import AgentState, TargetState, load_state, save_state


class TestState(unittest.TestCase):
    def test_state_roundtrip(self):
        state = AgentState(
            schema_version=1,
            targets={
                "User:abc": TargetState(
                    last_seen_timestamp_ms=123,
                    processed_comment_keys=["c1", "c2"],
                )
            },
            closed_conversations={"User:abc|u1": 999},
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            save_state(path, state)
            loaded = load_state(path)
            self.assertEqual(loaded.schema_version, 1)
            self.assertIn("User:abc", loaded.targets)
            self.assertEqual(loaded.targets["User:abc"].last_seen_timestamp_ms, 123)
            self.assertEqual(loaded.targets["User:abc"].processed_comment_keys, ["c1", "c2"])
            self.assertIn("User:abc|u1", loaded.closed_conversations)
            self.assertEqual(loaded.closed_conversations["User:abc|u1"], 999)


if __name__ == "__main__":
    unittest.main()
