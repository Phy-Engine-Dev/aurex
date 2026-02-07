import os
import sys
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from phy_engine import PhyEngineError, Verilog2PlSavOptions, verilog_to_plsav


class TestVerilog2PlSav(unittest.TestCase):
    def test_wraps_typeerror_from_subprocess(self):
        with mock.patch("subprocess.run", side_effect=TypeError()):
            with self.assertRaises(PhyEngineError) as ctx:
                verilog_to_plsav(
                    verilog2plsav_bin="verilog2plsav",
                    out_sav_path="out.sav",
                    in_verilog_path="in.v",
                    options=Verilog2PlSavOptions(top="top", extra_args=["-O4"]),
                    timeout_sec=1,
                )
        self.assertIn("invocation failed", str(ctx.exception))

    def test_rejects_none_cmd_token(self):
        with self.assertRaises(PhyEngineError):
            verilog_to_plsav(
                verilog2plsav_bin=None,  # type: ignore[arg-type]
                out_sav_path="out.sav",
                in_verilog_path="in.v",
            )


if __name__ == "__main__":
    unittest.main()

