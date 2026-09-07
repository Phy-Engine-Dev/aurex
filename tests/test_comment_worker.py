"""Offline contract tests: no account, credentials, or network calls."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request


_PATH = Path(__file__).resolve().parents[1] / "src/plar/comment_worker.py"
_SPEC = importlib.util.spec_from_file_location("isolated_comment_worker", _PATH)
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)

TARGET = "1" * 24
RECIPIENT = "2" * 24
COMMENT = "3" * 24


def payload():
    return {"token": "dummy-token-never-real", "auth_code": "dummy-auth-never-real",
            "target_id": TARGET, "target_type": "Experiment",
            "requester_user_id": RECIPIENT, "content": "回复@测试用户: 原正文\n第二行。"}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_one_official_call_with_explicit_user_id_and_unmodified_content(self):
        account = Mock()
        account.post_comment.return_value = {"Status": 200, "Data": {"ID": COMMENT, "Hidden": False}}
        original = payload()
        with patch.object(worker, "_configure_transport"), patch.object(worker, "_sdk_user", return_value=account):
            result = worker.submit_once(original)
        account.post_comment.assert_called_once_with(target_id=TARGET, target_type="Experiment",
            content=original["content"], reply_id=RECIPIENT, special=None)
        account.get_user.assert_not_called()
        self.assertEqual(original, payload())
        self.assertEqual(result, {"Status": 200, "Data": {"ID": COMMENT, "Hidden": False}})

    def test_sdk_failure_is_not_retried_or_switched_to_another_transport(self):
        account = Mock()
        account.post_comment.side_effect = TimeoutError("secret must not be shown")
        with patch.object(worker, "_configure_transport"), patch.object(worker, "_sdk_user", return_value=account):
            with self.assertRaises(TimeoutError):
                worker.submit_once(payload())
        self.assertEqual(account.post_comment.call_count, 1)

    def test_process_local_timeout_proxy_and_redirect_policy(self):
        with patch.object(socket, "setdefaulttimeout") as timeout, patch.object(urllib.request, "build_opener") as build, \
             patch.object(urllib.request, "install_opener") as install:
            worker._configure_transport()
        timeout.assert_called_once_with(50.0)
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertIsInstance(build.call_args.args[1], worker._NoRedirect)
        install.assert_called_once_with(build.return_value)

    def test_all_redirect_statuses_refused(self):
        handler = worker._NoRedirect()
        request = urllib.request.Request("https://example.invalid/Comment", data=b"{}", method="POST")
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code), self.assertRaises(urllib.error.HTTPError):
                handler.redirect_request(request, None, code, "redirect", {}, "https://other.invalid")

    def test_invalid_inputs_never_reach_transport(self):
        cases = [None, [], {**payload(), "extra": "ignored?"}, {**payload(), "requester_user_id": TARGET + "x"},
                 {**payload(), "target_type": "Message"}, {**payload(), "target_id": True},
                 {**payload(), "token": ""}, {**payload(), "auth_code": "bad\nheader"},
                 {**payload(), "token": "x" * 8193}]
        for value in cases:
            with self.subTest(value_type=type(value).__name__), patch.object(worker, "_configure_transport") as configure:
                with self.assertRaises(ValueError):
                    worker.submit_once(value)
                configure.assert_not_called()

    def test_ambiguous_or_unreviewed_mention_rejected(self):
        cases = ["<user=" + RECIPIENT + ">@测试</user> 原文", "回复@测试:", "回复@测试: \n ",
                 "回复@姓名 带空格: 原文", "回复@测试：用户: 原文", "回复@测试\n用户: 原文",
                 "回复@@测试: 原文"]
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ValueError):
                worker._validate({**payload(), "content": content})

    def test_reviewed_body_is_preserved_without_scanning_for_other_recipients(self):
        content = '回复@测试: 原评论引用：<user=' + TARGET + '>@另一人</user>\n  后续分析'
        result = worker._validate({**payload(), 'content': content})
        self.assertEqual(result['content'], content)
        self.assertEqual(result['requester_user_id'], RECIPIENT)

    def test_content_length_limit(self):
        value = payload()
        value["content"] = "回复@测试: " + "a" * worker.MAX_CONTENT_CHARACTERS
        with self.assertRaises(ValueError):
            worker._validate(value)

    def test_response_whitelist_flat_and_legacy(self):
        for data in ({"ID": COMMENT, "Hidden": False}, {"Comment": {"ID": COMMENT, "Hidden": False, "Token": "secret"}}):
            data["Content"] = "private original content"
            result = worker._normalize_response({"Status": 200, "Data": data, "Token": "secret", "AuthCode": "secret"})
            self.assertEqual(result, {"Status": 200, "Data": {"ID": COMMENT, "Hidden": False}})

    def test_invalid_or_missing_receipt_id_not_invented(self):
        result = worker._normalize_response({"Status": 200, "Data": {"ID": "unsafe-id", "Hidden": "false"}})
        self.assertEqual(result, {"Status": 200, "Data": {}})

    def test_failure_or_malformed_responses_rejected(self):
        for value in (None, {}, {"Status": 500, "Data": {}}, {"Status": "200", "Data": {}},
                      {"Status": 200}, {"Status": 200, "Data": []}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker._normalize_response(value)

    def _main(self, raw, account=None):
        stdin = Mock(buffer=io.BytesIO(raw))
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(worker.sys, "stdin", stdin), patch.object(worker.sys, "stdout", stdout), \
             patch.object(worker.sys, "stderr", stderr), patch.object(worker, "_configure_transport"), \
             patch.object(worker, "_sdk_user", return_value=account):
            code = worker.main()
        return code, stdout.getvalue(), stderr.getvalue()

    def test_main_stdout_is_only_normalized_json(self):
        account = Mock()
        account.post_comment.return_value = {"Status": 200, "Data": {"ID": COMMENT}, "Token": "hidden"}
        code, stdout, stderr = self._main(json.dumps(payload()).encode(), account)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), {"Status": 200, "Data": {"ID": COMMENT}})
        self.assertEqual(stderr, "")

    def test_main_failure_cannot_leak_credentials_or_server_exception(self):
        account = Mock()
        account.post_comment.side_effect = RuntimeError(json.dumps(payload()))
        code, stdout, stderr = self._main(json.dumps(payload()).encode(), account)
        self.assertEqual((code, stdout, stderr), (1, "", worker._ERROR))
        account.post_comment.assert_called_once()

    def test_main_oversized_invalid_utf8_and_duplicate_json_fail_closed(self):
        raw_inputs = [b"x" * (worker.MAX_INPUT_BYTES + 1), b"\xff", b"[]", b'{"token":"first","token":"second"}']
        for raw in raw_inputs:
            account = Mock()
            code, stdout, stderr = self._main(raw, account)
            self.assertEqual((code, stdout, stderr), (1, "", worker._ERROR))
            account.post_comment.assert_not_called()

    def test_installed_sdk_exact_body_no_lookup_with_mocked_transport(self):
        # Exercise the actual deployed SDK method, not a locally reimplemented body.
        from physicsLab.web import _request
        value = payload()
        response = {"Status": 200, "Data": {"ID": COMMENT, "Hidden": False}}
        with patch.object(worker, "_configure_transport"), patch.object(_request, "post_https", return_value=response) as post:
            result = worker.submit_once(value)
        self.assertEqual(post.call_count, 1)
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["path"], "Messages/PostComment")
        self.assertEqual(kwargs["domain"], "physics-api-cn.turtlesim.com")
        self.assertEqual(kwargs["body"], {"TargetID": TARGET, "TargetType": "Experiment", "Language": "Chinese",
            "ReplyID": RECIPIENT, "Content": value["content"], "Special": None})
        self.assertEqual(set(kwargs["header"]), {"Content-Type", "x-API-Token", "x-API-AuthCode"})
        self.assertEqual(result, response)


if __name__ == "__main__":
    unittest.main()
