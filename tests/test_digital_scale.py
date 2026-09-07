"""Offline resource-policy and native digital scale regressions, no agent calls."""
from __future__ import annotations
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from aurex.config import AurexConfig, ConfigError, load_config
from aurex.phy_engine.limits import validate_spec_size, native_digital_type
from aurex.tools.circuits import normalize_spec, _spec_from_sav, circuit_analyze, circuit_read_stimulus, circuit_inspect
from aurex.tools.phy_engine import _simulate_spec, pe_simulate
from aurex.tools.registry import ToolError, ToolRuntime


def inputs(count):
    return {"components": [{"id": f"I{i}", "type": "digital_input", "nodes": [f"n{i}"], "params": {"state": 0}} for i in range(count)]}


class DigitalScalePolicyTests(unittest.TestCase):
    def test_digital_limit_configurable_but_analog_mixed_remain_512(self):
        self.assertEqual(len(normalize_spec(inputs(513))["components"]), 513)
        self.assertEqual(validate_spec_size(inputs(4096)), 4096)
        with self.assertRaisesRegex(ToolError, "pure native digital limit 512"):
            normalize_spec(inputs(513), digital_component_limit=512)
        with self.assertRaisesRegex(ValueError, "4096"):
            validate_spec_size(inputs(4097))
        self.assertEqual(validate_spec_size(inputs(4097), 8192), 8192)
        mixed = inputs(513)
        mixed["components"][0] = {"id": "R", "type": "resistor", "nodes": ["a", "b"], "params": {"r": 1}, "device_type": "digital"}
        mixed["pure_digital"] = True
        mixed["digital_component_limit"] = 16384
        for check in (lambda: normalize_spec(mixed), lambda: _simulate_spec(mixed, "/must-not-load")):
            with self.assertRaisesRegex(ToolError, "analog/mixed limit 512"):
                check()

    def test_unknown_prefix_and_malformed_types_do_not_grant_budget(self):
        for kind in ("digital_resistor", "digital_unknown", 200, None):
            with self.assertRaises(ValueError):
                native_digital_type(kind)
        self.assertFalse(native_digital_type("resistor"))
        self.assertTrue(native_digital_type("digital_dff"))
        for invalid in (0, -1, 16385, True, 1.5, "4096"):
            with self.assertRaises(ValueError):
                validate_spec_size(inputs(1), invalid)
        with self.assertRaisesRegex(ToolError, "pure native digital limit 512"):
            _simulate_spec(inputs(513), "/must-not-load", digital_component_limit=512)

    def test_config_default_and_strict_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"
            path.write_text("{}")
            self.assertEqual(load_config(str(path)).phy_engine.digital_component_limit, 4096)
            path.write_text(json.dumps({"phy_engine": {"digital_component_limit": 2048}}))
            self.assertEqual(load_config(str(path)).phy_engine.digital_component_limit, 2048)
            path.write_text(json.dumps({"phy_engine": {"digital_component_limit": True}}))
            with self.assertRaises(ConfigError):
                load_config(str(path))

    def test_runtime_policy_not_spec_self_declared_limit(self):
        cfg = replace(AurexConfig(), phy_engine=replace(AurexConfig().phy_engine, digital_component_limit=16))
        rt = ToolRuntime(task_id="policy", user_lang="zh", config_path="/tmp/config.json", cache_dir="/tmp", config=cfg)
        spec = inputs(17)
        spec["digital_component_limit"] = 1000000
        with patch("aurex.tools.phy_engine._ensure_artifacts", side_effect=AssertionError("must fail before artifact build")):
            with self.assertRaisesRegex(ToolError, "limit 16"):
                pe_simulate(rt, {"spec": spec})


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "native build not configured")
class DigitalScaleNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(self.build),
            verilog2plsav_path=str(self.build/"verilog2plsav"), phyengine_lib_path=str(self.build/"libphyengine.so"), run_timeout_sec=30))
        self.runtime = ToolRuntime(task_id="digital-scale-offline", user_lang="zh", config_path=str(Path(self.temp.name)/"config.json"),
            config=cfg, cache_dir=self.temp.name)
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("No network in scale regression"))
        self.network.start()

    def tearDown(self):
        self.network.stop()
        self.temp.cleanup()

    def test_large_independent_gate_network_matches_small_truth_table(self):
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            base = {"components": [
                {"id":"A", "type":"digital_input", "nodes":["a"], "params":{"state":a}},
                {"id":"B", "type":"digital_input", "nodes":["b"], "params":{"state":b}},
                {"id":"G", "type":"digital_and", "nodes":["a","b","out"]},
                {"id":"Y", "type":"digital_output", "nodes":["out"]}], "digital_clock_ticks":3}
            large = {**base, "components": base["components"] + [
                {"id":f"spare{i}", "type":"digital_yes", "nodes":["a",f"spare_node{i}"]} for i in range(600)]}
            values = []
            for spec in (base, large):
                out = pe_simulate(self.runtime, {"spec":spec})
                values.append(next(c for c in out["components"] if c["id"]=="Y")["digital"])
            self.assertEqual(values[0], values[1])
            self.assertEqual(values[0], [a & b])

    @unittest.skipUnless(os.environ.get("AUREX_MATRIX_FIXTURE"), "original cached large digital fixture not configured")
    def test_original_large_import_and_three_native_propagation_steps(self):
        fixture = Path(os.environ["AUREX_MATRIX_FIXTURE"]).resolve()
        original = fixture.read_bytes()
        source = Path(self.temp.name)/"matrix.sav"
        shutil.copyfile(fixture, source)
        begin = time.monotonic()
        result = circuit_analyze(self.runtime, {"path": str(source), "digital_clock_ticks":3})
        state = json.loads(Path(result["state_path"]).read_text())
        out = state["measurements"]
        self.assertEqual(out["engine"], "Phy-Engine native")
        self.assertEqual(len(out["components"]), 981)
        self.assertEqual(len(state["spec"]["components"]), 981)
        self.assertTrue(all(v in (0,1,2,3) for c in out["components"] for v in c["digital"]))
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(fixture.read_bytes(), original)
        self.assertEqual(result["measurements"]["component_scope"]["total"], 981)
        self.assertLessEqual(len(result["measurements"]["components"]), 8)
        self.assertLess(len(json.dumps(result)), 25000)
        print("LARGE_NATIVE_EXECUTION_ONLY", json.dumps({"components":981,"propagation_steps":3,
              "elapsed_s":round(time.monotonic()-begin,3),"functional_matrix_verification":False}))

    def test_compact_measurements_and_stimulus_reader_preserve_full_raw_state(self):
        spec = inputs(20)
        spec["stimulus"] = [{"set": {"I0": value % 2, "I1": (value+1) % 2}} for value in range(12)]
        result = circuit_analyze(self.runtime, {"spec": spec})
        self.assertEqual(result["measurements"]["component_scope"]["total"], 20)
        self.assertLessEqual(len(result["measurements"]["components"]), 8)
        self.assertEqual(len(result["measurements"]["stimulus_results"]), 8)
        path = Path(result["state_path"])
        original = path.read_bytes()
        raw = json.loads(original)
        self.assertEqual(len(raw["measurements"]["components"]), 20)
        self.assertEqual(len(raw["measurements"]["stimulus_results"]), 12)
        self.assertEqual(len(raw["measurements"]["stimulus_results"][0]["digital"]), 20)
        with patch("aurex.tools.circuits.pe_simulate", side_effect=AssertionError("Reader must not rerun")):
            page = circuit_read_stimulus(self.runtime, {"path": str(path), "component_ids": ["I0", "I1"], "offset": 7, "limit": 3})
            self.assertEqual([r["step"] for r in page["steps"]], [7, 8, 9])
            self.assertEqual([r["digital"]["I0"] for r in page["steps"]], [[1], [0], [1]])
            self.assertTrue(page["has_more"])
            detail = circuit_inspect(self.runtime, {"path": str(path), "focus_id": "I19", "limit": 1})
            self.assertEqual(detail["netlist"]["components"][0]["native"]["measurements"]["id"], "I19")
        self.assertEqual(path.read_bytes(), original)
        for changes in ({"component_ids": ["missing"]}, {"component_ids": ["I0", "I0"]}, {"limit": True}, {"limit": 17}, {"offset": -1}):
            with self.assertRaises(ToolError):
                circuit_read_stimulus(self.runtime, {"path": str(path), **changes})


if __name__ == "__main__":
    unittest.main()
