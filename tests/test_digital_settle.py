"""End-to-end numerical validity gates, including the native starvation repro."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from aurex.config import AurexConfig
from aurex.phy_engine.catalog import COMPONENTS
from aurex.phy_engine.ffi import _Lib, PhyEngineError
from aurex.tools.phy_engine import pe_simulate
from aurex.tools.registry import ToolError, ToolRuntime


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "native build not configured")
class DigitalSettleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"])
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(build),
            verilog2plsav_path=str(build / "verilog2plsav"), phyengine_lib_path=str(build / "libphyengine.so")))
        self.rt = ToolRuntime("settle-regression", "zh", str(Path(self.temp.name) / "config.json"), cfg, self.temp.name)

    @staticmethod
    def spec():
        return {"analysis": "tr", "tr_step": .001, "tr_stop": .003, "tr_sample_every": 1, "components": [
            {"id": "IN", "type": "digital_input", "nodes": ["in"], "params": {"state": 1}},
            {"id": "NOT", "type": "digital_not", "nodes": ["in", "out"]},
            {"id": "OUT", "type": "digital_output", "nodes": ["out"]},
        ]}

    def test_completed_frames_include_settle_evidence(self):
        result = pe_simulate(self.rt, {"spec": self.spec()})
        self.assertEqual(result["execution_status"], "completed")
        self.assertTrue(result["digital_settle"]["settled"])
        self.assertEqual(result["digital_settle"]["attempted_ticks"], 3)
        self.assertEqual(result["digital_settle"]["settled_ticks"], 3)
        self.assertTrue(all(p["digital_settled"] for p in result["transient"]["samples"]))
        self.assertTrue(result["transient"]["digital_propagation"]["settle_verified"])

    def test_analog_only_success_does_not_report_digital_not_run_as_failure(self):
        result = pe_simulate(self.rt, {"spec": {"analysis": "dc", "components": [
            {"id": "V", "type": "vdc", "nodes": ["supply", "gnd"],
             "params": {"v": 1.0}},
            {"id": "R", "type": "resistor", "nodes": ["supply", "gnd"],
             "params": {"r": 1000.0}},
        ]}})
        self.assertEqual(result["execution_status"], "completed")
        self.assertNotIn("digital_settle", result)
        resistor = next(row for row in result["components"] if row["id"] == "R")
        self.assertAlmostEqual(resistor["voltage"][0], 1.0)

    def test_multidriver_is_failure_not_X_waveform(self):
        spec = self.spec()
        spec["components"].append({"id": "OTHER", "type": "digital_input", "nodes": ["out"], "params": {"state": 1}})
        with self.assertRaises(ToolError) as caught:
            pe_simulate(self.rt, {"spec": spec})
        report = json.loads(str(caught.exception))
        self.assertFalse(report["waveform_valid"])
        status = report["digital_settle"]
        self.assertEqual(status["reason"], "DIGITAL_MULTIPLE_DRIVERS")
        self.assertEqual(status["settled_ticks"], 0)
        self.assertEqual(status["multiple_driver_count"], 1)
        drivers = [p["component_index"] for p in status["conflicts"][0]["pins"] if p["role"] == 1]
        self.assertEqual(set(drivers), {1, 3})
        self.assertLess(len(str(caught.exception)), 5000)

    def test_unloaded_named_output_is_observable(self):
        spec = self.spec()
        spec["components"].pop()  # No output probe/load on the inverter.
        result = pe_simulate(self.rt, {"spec": spec})
        gate = next(c for c in result["components"] if c["id"] == "NOT")
        self.assertEqual(gate["digital"], [1, 0])
        self.assertTrue(result["digital_settle"]["settled"])
        self.assertIn("out", gate["unconnected_pins"])

    def test_singleton_input_keeps_open_default_and_ground_is_explicit(self):
        spec = {"analysis": "dc", "components": [
            {"id": "NOT", "type": "digital_not", "nodes": ["open-input", "named-output"]},
        ]}
        floating = pe_simulate(self.rt, {"spec": spec})
        self.assertEqual(floating["components"][0]["digital"], [2, 2])
        for ground in ("gnd", "ground", "0"):
            spec["components"][0]["nodes"][0] = ground
            result = pe_simulate(self.rt, {"spec": spec})
            self.assertEqual(result["components"][0]["digital"], [0, 1])
        # Every catalog digital primitive has an explicit direction contract.
        self.assertTrue(all("digital_output_pins" in row for name, row in COMPONENTS.items() if name.startswith("digital_")))

    def test_hybrid_and_ground_shared_digital_drivers_are_rejected(self):
        for target in ("bus", "gnd"):
            for second in (0, 1):
                spec = {"analysis": "tr", "tr_step": .001, "tr_stop": .002,
                    "components": [
                        {"id": "D1", "type": "digital_input", "nodes": [target], "params": {"state": 0}},
                        {"id": "D2", "type": "digital_input", "nodes": [target], "params": {"state": second}},
                        {"id": "R", "type": "resistor", "nodes": ["bus", "gnd"], "params": {"r": 10}},
                    ]}
                with self.subTest(target=target, second=second), self.assertRaises(ToolError) as caught:
                    pe_simulate(self.rt, {"spec": spec})
                failure = json.loads(str(caught.exception))
                self.assertEqual(failure["digital_settle"]["reason"], "DIGITAL_MULTIPLE_DRIVERS")
                self.assertEqual(failure["digital_settle"]["multiple_driver_count"], 1)
                self.assertEqual(failure["digital_settle"]["conflicts"][0]["node"], target)
                self.assertFalse(failure["waveform_valid"])
        spec["components"] = [
            {"id": "D1", "type": "digital_input", "nodes": ["bus"], "params": {"state": 1}},
            {"id": "R", "type": "resistor", "nodes": ["bus", "gnd"], "params": {"r": 10}},
        ]
        result = pe_simulate(self.rt, {"spec": spec})
        self.assertTrue(result["digital_settle"]["settled"])
        self.assertEqual(result["digital_settle"]["multiple_driver_count"], 0)
        self.assertEqual(result["digital_settle"]["undriven_count"], 0)
        resistor = next(row for row in result["components"] if row["id"] == "R")
        self.assertAlmostEqual(resistor["voltage"][0], 5)

    def test_direct_samplers_preserve_structured_settle_failure(self):
        lib = _Lib(str(Path(os.environ["AUREX_PHY_ENGINE_BUILD"]) / "libphyengine.so"))
        circuit = lib.create_circuit(elements=[0, 200, 200], wires=[1, 0, 2, 0], properties=[0, 1])
        self.addCleanup(circuit.close)
        circuit.analyze()
        with self.assertRaises(PhyEngineError):
            circuit.digital_clk()
        for sampler in (circuit.sample_u8, circuit.sample_complex):
            with self.subTest(sampler=sampler.__name__), self.assertRaises(PhyEngineError) as caught:
                sampler(max_pins=1)
            report = json.loads(str(caught.exception))
            self.assertEqual(report["execution_status"], "failed")
            self.assertFalse(report["waveform_valid"])
            self.assertEqual(report["digital_settle"]["reason"], "DIGITAL_MULTIPLE_DRIVERS")
            self.assertEqual(report["digital_settle"]["multiple_driver_count"], 1)

    def test_native_fairness_and_temporal_contracts(self):
        compiler = shutil.which("clang++-21") or shutil.which("clang++-22")
        if not compiler:
            self.skipTest("Clang 21/22 required")
        repo = Path(__file__).resolve().parents[1]
        target = Path(self.temp.name) / "digital-settle"
        result = subprocess.run([compiler, "-std=c++20", "-O0", "-Wno-braced-scalar-init",
            "-I" + str(repo / "third-parties/Phy-Engine/include"), str(repo / "tests/native/test_digital_settle.cpp"),
            "-ldl", "-o", str(target)], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run([str(target), str(Path(os.environ["AUREX_PHY_ENGINE_BUILD"]) / "libphyengine.so")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = [line for line in result.stdout.splitlines() if line.startswith("settle-json:")]
        self.assertEqual(len(expected), 2)
        # Fresh processes randomize address layout. Diagnostics must retain
        # the same selected nodes and pin ordering rather than pointer order.
        for _ in range(2):
            rerun = subprocess.run([str(target), str(Path(os.environ["AUREX_PHY_ENGINE_BUILD"]) / "libphyengine.so")], capture_output=True, text=True, timeout=30)
            self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
            self.assertEqual([line for line in rerun.stdout.splitlines() if line.startswith("settle-json:")], expected)
