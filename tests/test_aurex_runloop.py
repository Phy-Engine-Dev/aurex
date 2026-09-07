import os
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aurex.runloop import (
    RunState,
    default_state_path,
    load_state,
    normalize_targets,
    parse_target,
    save_state,
)
from aurex.replyfmt import prefix_user_mention
from aurex.config import AgentConfig, AurexConfig
import aurex.runloop as runloop


class TestRunLoopHelpers(unittest.TestCase):
    def test_parse_target(self):
        t = parse_target("Experiment:abc")
        self.assertEqual(t.type, "Experiment")
        self.assertEqual(t.id, "abc")

    def test_normalize_targets_dedup(self):
        ts = normalize_targets(
            [
                {"type": "Experiment", "id": "x"},
                {"type": "Experiment", "id": "x"},
                "Discussion:y",
            ]
        )
        self.assertEqual([t.key for t in ts], ["Experiment:x", "Discussion:y"])

    def test_default_state_path(self):
        p = default_state_path("/a/b/cfg.json")
        self.assertTrue(p.endswith("cfg.state.json"))

    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.json")
            st = RunState(started_at_sec=1.0, seen_comment_ids={"c1": 2.0}, replied_conversations={"k": 3.0})
            save_state(st, p)
            got = load_state(p)
            self.assertEqual(got.started_at_sec, 1.0)
            self.assertEqual(got.seen_comment_ids["c1"], 2.0)
            self.assertEqual(got.replied_conversations["k"], 3.0)

    def test_prefix_user_mention(self):
        uid = "0123456789abcdef01234567"
        self.assertEqual(prefix_user_mention("hello", user_id=uid, nickname="abc"), f"<user={uid}>@abc</user> hello")
        self.assertEqual(prefix_user_mention(f"<user={uid}>@abc</user> hi", user_id=uid, nickname="abc"), f"<user={uid}>@abc</user> hi")
        self.assertEqual(prefix_user_mention("@abc hi", user_id=uid, nickname="abc"), f"<user={uid}>@abc</user> hi")
        self.assertEqual(prefix_user_mention("＠abc  hi", user_id=uid, nickname="abc"), f"<user={uid}>@abc</user> hi")
        self.assertEqual(prefix_user_mention("hi", user_id=uid, nickname=""), "hi")
        self.assertEqual(prefix_user_mention("hi", user_id="", nickname="abc"), "hi")

    def test_extract_trigger_text_normalizes_user_tags(self):
        s = "回复<user=698726c4fc064466378176b8>@aurex</user>: 你好"
        self.assertEqual(runloop._extract_trigger_text(s), "回复 @aurex : 你好")
        s2 = "回复@aurex: 你好"
        self.assertEqual(runloop._extract_trigger_text(s2), "回复@aurex: 你好")

    def test_has_explicit_mention_matches_reply_prefix(self):
        s = "回复<user=698726c4fc064466378176b8>@aurex</user>: 你好"
        self.assertTrue(runloop._has_explicit_mention(text=s, mention_tag="@aurex"))

    def test_has_explicit_mention_detects_body_mention(self):
        s = "回复<user=698726c4fc064466378176b8>@aurex</user>: @aurex 你好"
        self.assertTrue(runloop._has_explicit_mention(text=s, mention_tag="@aurex"))

    def test_has_explicit_mention_detects_fullwidth_and_casefold(self):
        self.assertTrue(runloop._has_explicit_mention(text="你好 ＠aurex", mention_tag="@aurex"))
        self.assertTrue(runloop._has_explicit_mention(text="Hello @AUREX", mention_tag="@aurex"))

    def test_normalize_post_text_preserves_newlines(self):
        s = "@abc  hello  \n\n-  a  \n-  b  \n\n\n"
        out = runloop._normalize_post_text(s)
        self.assertEqual(out, "@abc hello\n\n- a\n- b")


class TestNotificationsScan(unittest.TestCase):
    def _cfg(self) -> AurexConfig:
        return AurexConfig(
            agent=AgentConfig(
                notifications_enabled=True,
                notification_category_ids=[3],
                notification_take=20,
                bootstrap_lookback_sec=0,
                targets=[],
            )
        )

    def test_notification_scan_advances_watermark_when_seen(self):
        cfg = self._cfg()
        state = RunState(messages_last_seen_ms={"3": 1770000000000}, messages_processed_keys={"3": ["m2"]})
        fake_user = object()

        msg = {
            "ID": "m2",
            "Timestamp": 1770000002000,
            "Content": "Experiment 0123456789abcdef01234567",
        }

        with mock.patch("aurex.runloop.plar.get_messages", return_value=([msg], [])):
            out = runloop._discover_targets_from_notifications(user=fake_user, cfg=cfg, state=state)

        self.assertEqual(out, [])
        self.assertEqual(state.messages_last_seen_ms.get("3"), 1770000002000)

    def test_notification_scan_processes_unseen_at_watermark(self):
        cfg = self._cfg()
        state = RunState(messages_last_seen_ms={"3": 1770000002000}, messages_processed_keys={"3": ["m2"]})
        fake_user = object()

        msg_new_same_ts = {
            "ID": "m3",
            "Timestamp": 1770000002000,
            "Content": "Discussion 89abcdef0123456701234567",
        }

        with mock.patch("aurex.runloop.plar.get_messages", return_value=([msg_new_same_ts], [])):
            out = runloop._discover_targets_from_notifications(user=fake_user, cfg=cfg, state=state)

        self.assertEqual([t.key for t in out], ["Discussion:89abcdef0123456701234567"])
        self.assertEqual(state.messages_last_seen_ms.get("3"), 1770000002000)
        self.assertIn("m3", state.messages_processed_keys.get("3") or [])

    def test_notification_bootstraps_comment_watermark(self):
        cfg = self._cfg()
        state = RunState(messages_last_seen_ms={"3": 1770000000000}, messages_processed_keys={"3": []})
        fake_user = object()

        msg = {
            "ID": "m9",
            "Timestamp": 1770000002000,
            "Content": "User 0123456789abcdef01234567",
        }

        with mock.patch("aurex.runloop.plar.get_messages", return_value=([msg], [])):
            out = runloop._discover_targets_from_notifications(user=fake_user, cfg=cfg, state=state)

        self.assertEqual([t.key for t in out], ["User:0123456789abcdef01234567"])
        # Should set a non-zero watermark not later than the message timestamp.
        self.assertGreater(int(state.comments_last_seen_ms.get("User:0123456789abcdef01234567") or 0), 0)
        self.assertLessEqual(int(state.comments_last_seen_ms.get("User:0123456789abcdef01234567") or 0), 1770000002000)

if __name__ == "__main__":
    unittest.main()
