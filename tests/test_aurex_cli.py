from __future__ import annotations

import os
import sys
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aurex import cli  # noqa: E402


class TestCliEntrypoints(unittest.TestCase):
    def test_only_v3_entrypoints_are_public(self):
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["cli"]).cmd, "cli")
        self.assertEqual(parser.parse_args(["web", "--config", "x.json"]).cmd, "web")
        for removed in ("chat", "console", "run"):
            with self.assertRaises(SystemExit):
                parser.parse_args([removed])

    def test_web_has_no_manual_poll_switch(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["web", "--config", "x.json", "--poll"])

    def test_web_always_logs_in_and_uses_default_polling(self):
        cfg = mock.Mock()
        cfg.llm.enabled = True
        cfg.storage.cache_dir = ".cache"
        cfg.agent.log_level = "INFO"
        cfg.resolve_path.return_value = ".cache"
        args = cli.build_parser().parse_args(["web", "--config", "x.json"])
        with mock.patch.object(cli, "load_config", return_value=cfg), \
             mock.patch.object(cli, "_access_token", return_value=""), \
             mock.patch.object(cli.AurexWebClient, "healthy", return_value=False), \
             mock.patch.object(cli, "setup_logger", return_value=mock.Mock()), \
             mock.patch.object(cli, "AurexAgent", return_value=mock.Mock()), \
             mock.patch.object(cli, "create_registry", return_value=mock.Mock()), \
             mock.patch.object(cli, "_login", return_value=object()) as login, \
             mock.patch("aurex.web.serve") as serve:
            self.assertEqual(args.func(args), 0)
        login.assert_called_once()
        self.assertNotIn("poll", serve.call_args.kwargs)

    def test_web_started_second_reuses_cli_started_server(self):
        cfg = mock.Mock()
        cfg.llm.enabled = True
        cfg.tracking.port = 4097
        cfg.tracking.token_env = "AUREX_WEB_TOKEN"
        args = cli.build_parser().parse_args(["web", "--config", "x.json"])
        with mock.patch.object(cli, "load_config", return_value=cfg), \
             mock.patch.object(cli, "_access_token", return_value="token"), \
             mock.patch.object(cli.AurexWebClient, "healthy", return_value=True), \
             mock.patch.object(cli, "_login") as login, \
             mock.patch("aurex.web.serve") as serve:
            self.assertEqual(args.func(args), 0)
        login.assert_not_called()
        serve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
