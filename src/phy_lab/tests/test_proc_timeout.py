import os
import sys
import time
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from proc_timeout import ProcTimeoutError, run_with_timeout


def _sleep_then_return(*, sleep_s: float, value: int) -> int:
    time.sleep(float(sleep_s))
    return int(value)


class TestProcTimeout(unittest.TestCase):
    def test_returns_result_when_under_timeout(self):
        out = run_with_timeout(fn=_sleep_then_return, kwargs={"sleep_s": 0.05, "value": 7}, timeout_sec=1.0, label="sleep")
        self.assertEqual(out, 7)

    def test_times_out_and_terminates(self):
        with self.assertRaises(ProcTimeoutError):
            run_with_timeout(fn=_sleep_then_return, kwargs={"sleep_s": 0.3, "value": 1}, timeout_sec=0.05, label="sleep")


if __name__ == "__main__":
    unittest.main()

