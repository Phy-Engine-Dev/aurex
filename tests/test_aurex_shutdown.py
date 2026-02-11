import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


from aurex.shutdown import GracefulShutdown  # noqa: E402


class _Logger:
    def __init__(self):
        self.messages = []

    def warning(self, msg, *args, **kwargs):
        self.messages.append(("warning", msg))

    def error(self, msg, *args, **kwargs):
        self.messages.append(("error", msg))


class TestGracefulShutdown(unittest.TestCase):
    def test_two_stage(self):
        lg = _Logger()
        gs = GracefulShutdown(logger=lg)
        gs._on_sigint(2, None)
        self.assertTrue(gs.stop_requested)
        with self.assertRaises(SystemExit):
            gs._on_sigint(2, None)


if __name__ == "__main__":
    unittest.main()

