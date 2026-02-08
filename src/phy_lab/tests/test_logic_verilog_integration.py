import os
import json
import subprocess
import tempfile
import unittest


class TestLogicVerilogIntegration(unittest.TestCase):
    def test_verilog2plsav_and_logic_state_propagates(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        v2p = os.path.join(repo_root, ".phy_lab", "cache", "phy-engine-build", "verilog2plsav")
        lib = os.path.join(repo_root, ".phy_lab", "cache", "phy-engine-build", "libphyengine.so")
        if not (os.path.exists(v2p) and os.path.exists(lib)):
            self.skipTest("verilog2plsav/libphyengine.so not built; skip integration test")

        verilog = "module top(input a, input b, output y);\n  assign y = a & b;\nendmodule\n"
        with tempfile.TemporaryDirectory(prefix="phy_lab_it_") as d:
            v_path = os.path.join(d, "design.v")
            s_path = os.path.join(d, "design.sav")
            with open(v_path, "w", encoding="utf-8") as f:
                f.write(verilog)
            subprocess.run([v2p, s_path, v_path, "--top", "top", "-O0", "--layout", "fast"], check=True)

            with open(s_path, "r", encoding="utf-8") as f:
                root = json.load(f)
            ss = json.loads(root["Experiment"]["StatusSave"])
            # a=1, b=1
            for el in ss.get("Elements", []):
                if el.get("ModelID") == "Logic Input" and el.get("Label") in ("a", "b"):
                    el.setdefault("Properties", {})["开关"] = 1.0

            import sys

            phy_lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if phy_lab_dir not in sys.path:
                sys.path.insert(0, phy_lab_dir)
            from pe_sim import PhyEngineLib

            pe = PhyEngineLib(lib)
            out_ss = pe.simulate_status_save(
                status_save=ss,
                analyze_type=1,
                digital_clk_ticks=1,
            )
            y = None
            for el in out_ss.get("Elements", []):
                if el.get("ModelID") == "Logic Output" and el.get("Label") == "y":
                    y = el.get("Properties", {}).get("状态")
                    break
            self.assertIsNotNone(y)
            self.assertEqual(int(y), 1)


if __name__ == "__main__":
    unittest.main()
