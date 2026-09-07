from __future__ import annotations

import unittest

from aurex.terminal import DashboardState, WebAPIError


class FakeAPI:
    def __init__(self):
        self.submissions = []
        self.cancellations = []
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


if __name__ == "__main__":
    unittest.main()
