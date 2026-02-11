import os
import sys
import tempfile
import unittest
import io
import contextlib
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


from aurex.cli import main  # noqa: E402
from aurex.config import AurexConfig, save_config  # noqa: E402


class _FakeAgent:
    def __init__(self, **_kwargs):
        pass

    def handle(self, *, user_text: str, user=None, task_id=None):
        return {"answer": f"echo:{user_text}"}


class TestCliConsole(unittest.TestCase):
    def test_console_prompts_input(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "cfg.json")
            save_config(AurexConfig(), cfg_path)

            buf = io.StringIO()
            with (
                contextlib.redirect_stdout(buf),
                mock.patch("aurex.cli.AurexAgent", _FakeAgent),
                mock.patch.object(sys.stdin, "isatty", return_value=True),
                mock.patch("builtins.input", return_value="hi"),
            ):
                rc = main(["console", "--config", cfg_path])
            self.assertEqual(rc, 0)

    def test_console_accepts_text_arg(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "cfg.json")
            save_config(AurexConfig(), cfg_path)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), mock.patch("aurex.cli.AurexAgent", _FakeAgent):
                rc = main(["console", "--config", cfg_path, "--text", "hello"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
