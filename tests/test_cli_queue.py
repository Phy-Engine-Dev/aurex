from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from aurex.terminal import AurexWebClient, DashboardState, WebAPIError


class FakeAPI:
    def __init__(self):
        self.submissions = []
        self.cancellations = []
        self.exports = []
        self._tasks = [
            {"id": "r1", "session_id": "s1", "title": "first", "status": "running"},
            {"id": "r2", "session_id": "s2", "title": "second", "status": "queued"},
        ]
        self._sessions = [
            {"id": "s1", "title": "one", "status": "running"},
            {"id": "s2", "title": "two", "status": "queued"},
        ]

    def sessions(self):
        return list(self._sessions)

    def tasks(self, limit=50):
        return list(self._tasks)

    def task(self, task_id):
        for task in self._tasks:
            if task["id"] == task_id:
                return task
        raise WebAPIError("missing")

    def events(self, session_id, task_id):
        return [{"kind": "answer", "data": {"text": f"{session_id}/{task_id}"}}]

    def submit(self, text, *, session_id=None, publish=False):
        self.submissions.append((text, session_id, publish))
        task = {"id": "r3", "session_id": session_id or "s3", "title": text,
                "status": "queued"}
        self._tasks.insert(0, task)
        if not session_id:
            self._sessions.insert(0, {"id": "s3", "title": text, "status": "queued"})
        return {"task_id": "r3", "session_id": task["session_id"]}

    def cancel(self, task_id):
        self.cancellations.append(task_id)
        return {"status": "cancelling"}

    def export_session(self, session_id, destination=None):
        self.exports.append((session_id, destination))
        return Path(destination or "/tmp/aurex-session-s1.sqlite3.zip")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def export_bytes(*, stored: bool = False, bad_header: bool = False) -> bytes:
    result = io.BytesIO()
    compression = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(result, "w", compression=compression) as archive:
        header = b"not a sqlite db!" if bad_header else b"SQLite format 3\x00"
        archive.writestr("aurex-session-s1.sqlite3", header + b"payload" * 64)
    return result.getvalue()


class DashboardTests(unittest.TestCase):
    def test_refresh_selects_running_task_and_builds_sections(self):
        state = DashboardState(FakeAPI())
        state.refresh()
        self.assertEqual(state.task_id, "r1")
        self.assertEqual(state.session_id, "s1")
        rendered = state.plain_text()
        self.assertIn("服务:", rendered)
        self.assertIn("队列", rendered)
        self.assertIn("会话", rendered)
        self.assertIn("当前任务", rendered)

    def test_normal_message_continues_selected_session(self):
        api = FakeAPI()
        state = DashboardState(api)
        state.refresh()
        self.assertFalse(state.command("继续验证"))
        self.assertEqual(api.submissions[-1], ("继续验证", "s1", False))

    def test_new_and_explicit_publish(self):
        api = FakeAPI()
        state = DashboardState(api)
        state.refresh()
        state.command("/new")
        state.command("/publish 发布已验证实验")
        self.assertEqual(api.submissions[-1], ("发布已验证实验", None, True))

    def test_cancel_targets_only_selected_task(self):
        api = FakeAPI()
        state = DashboardState(api)
        state.refresh()
        state.command("/cancel")
        self.assertEqual(api.cancellations, ["r1"])

    def test_export_and_legacy_exportsection_target_selected_session(self):
        api = FakeAPI()
        state = DashboardState(api)
        state.refresh()
        state.command("/export /tmp/backup.zip")
        state.command("/exportsection")
        self.assertEqual(api.exports, [("s1", "/tmp/backup.zip"), ("s1", None)])
        self.assertIn("会话已导出", state.notice)

    def test_export_requires_selected_session(self):
        state = DashboardState(FakeAPI())
        with self.assertRaisesRegex(WebAPIError, "没有选中的会话"):
            state.command("/export")


class ExportClientTests(unittest.TestCase):
    @staticmethod
    def _ticket() -> bytes:
        return json.dumps({
            "compressed": True,
            "filename": "aurex-session-s1.sqlite3.zip",
            "download_url": "/api/session-exports/abcdefghijklmnopqrstuvwxyz012345",
        }).encode()

    def test_streams_valid_deflated_sqlite_without_overwrite(self):
        requests = []
        responses = [FakeResponse(self._ticket()), FakeResponse(export_bytes())]

        def fake_urlopen(request, *, timeout):
            requests.append((request, timeout))
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as root, mock.patch(
            "aurex.terminal.urlopen", side_effect=fake_urlopen
        ):
            client = AurexWebClient("http://127.0.0.1:4097", "secret", timeout=1)
            output = client.export_session("s1", root)
            self.assertEqual(output.name, "aurex-session-s1.sqlite3.zip")
            self.assertTrue(output.is_file())
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertFalse(list(Path(root).glob("*.part")))
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.infolist()[0].compress_type, zipfile.ZIP_DEFLATED)
            self.assertEqual(requests[0][0].get_method(), "POST")
            self.assertTrue(requests[0][0].full_url.endswith("/api/sessions/s1/export"))
            self.assertEqual(requests[1][0].get_method(), "GET")
            self.assertEqual(requests[1][0].get_header("Authorization"), "Bearer secret")

            with self.assertRaisesRegex(WebAPIError, "不会覆盖"):
                client.export_session("s1", output)

    def test_rejects_uncompressed_or_invalid_sqlite_and_cleans_partial(self):
        for payload, message in (
            (export_bytes(stored=True), "DEFLATE"),
            (export_bytes(bad_header=True), "SQLite"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as root:
                responses = [FakeResponse(self._ticket()), FakeResponse(payload)]
                with mock.patch("aurex.terminal.urlopen", side_effect=responses):
                    client = AurexWebClient("http://127.0.0.1:4097")
                    output = Path(root) / "backup.zip"
                    with self.assertRaisesRegex(WebAPIError, message):
                        client.export_session("s1", output)
                    self.assertFalse(output.exists())
                    self.assertFalse(list(Path(root).glob("*.part")))

    def test_rejects_cross_origin_ticket_before_download(self):
        ticket = json.dumps({
            "compressed": True,
            "filename": "aurex-session-s1.sqlite3.zip",
            "download_url": "https://example.com/api/session-exports/abcdefghijklmnopqrstuvwxyz",
        }).encode()
        with tempfile.TemporaryDirectory() as root, mock.patch(
            "aurex.terminal.urlopen", return_value=FakeResponse(ticket)
        ) as opener:
            with self.assertRaisesRegex(WebAPIError, "不安全"):
                AurexWebClient("http://127.0.0.1:4097").export_session("s1", root)
            self.assertEqual(opener.call_count, 1)


if __name__ == "__main__":
    unittest.main()
