from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aurex.config import AurexConfig
from aurex.tools.circuits import _camera, circuit_analyze, circuit_create, circuit_inspect, publication_cover
from aurex.tools.registry import ToolError, ToolRuntime


def fixture():
    return {"title": "相机测试", "components": [
        {"id": "V1", "type": "vdc", "nodes": ["n", "gnd"], "params": {"v": 5}, "position": [-.2, 0, 0]},
        {"id": "R1", "type": "resistor", "nodes": ["n", "gnd"], "params": {"r": 10}, "position": [.2, .1, .12], "rotation": [0, 25, 40]},
    ]}


class CameraValidationTests(unittest.TestCase):
    def test_agent_cannot_select_publication_camera_policy(self):
        for value in [{"overview": True}, {"fov_y_deg": float("nan")}, {"distance": -1},
                      {"position": [0, 1]}, {"position": [0, 1, 2], "yaw_deg": 40}]:
            with self.subTest(value=value), self.assertRaises(ToolError):
                _camera(value)
        self.assertEqual(_camera({"mode": "auto", "fit": True}), {"mode": "auto", "fit": True})


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "set AUREX_PHY_ENGINE_BUILD for native camera tests")
class CameraNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(build),
            verilog2plsav_path=str(build / "verilog2plsav"), phyengine_lib_path=str(build / "libphyengine.so")))
        self.runtime = ToolRuntime(task_id="camera-test", user_lang="zh", config_path=str(Path(self.temp.name) / "config.json"), config=cfg, cache_dir=self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_saved_camera_raw_fields_and_file_are_preserved(self):
        saved = {"Mode": 0, "Distance": 2.5, "VisionCenter": "0.1,0.3,0.2", "TargetRotation": "30,45,0"}
        spec = fixture()
        spec["camera_save"] = saved
        created = circuit_create(self.runtime, {"spec": spec})
        source = Path(created["sav_path"])
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        first = circuit_inspect(self.runtime, {"path": str(source), "with_image": True})
        changed = circuit_inspect(self.runtime, {"path": str(source), "with_image": True, "camera": {"mode": "custom", "position": [1, -2, 1], "target": [0, 0, 0], "fit": True}})
        self.assertEqual(first["camera"]["saved_raw"], saved)
        self.assertIn("saved", first["camera"]["source"])
        self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(first["netlist"], changed["netlist"])
        self.assertNotEqual(Path(first["artifact"]["svg_path"]).read_bytes(), Path(changed["artifact"]["svg_path"]).read_bytes())

    def test_state_rerender_does_not_solve_or_roundtrip_plsav(self):
        analyzed = circuit_analyze(self.runtime, {"spec": fixture()})
        source = Path(analyzed["state_path"])
        data = json.loads(source.read_text())
        self.assertEqual(data["schema"], "aurex.pe-state.v1")
        self.assertFalse(data["origin"]["live_handle_retained"])
        before = source.read_bytes()
        with patch("aurex.tools.circuits.pe_simulate", side_effect=AssertionError("camera must not re-solve")):
            first = circuit_inspect(self.runtime, {"path": str(source), "with_image": True, "camera": {"mode": "auto", "fit": True, "yaw_deg": 30, "pitch_deg": 45}})
            second = circuit_inspect(self.runtime, {"path": str(source), "with_image": True, "camera": {"mode": "auto", "fit": True, "yaw_deg": -60, "pitch_deg": 70}})
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(first["netlist"], second["netlist"])
        self.assertEqual(len(first["netlist"]["components"]), 2)
        self.assertEqual(first["state_source"], "recorded native PE snapshot; not a new simulation")
        measured = next(c for c in first["netlist"]["components"] if c["id"] == "R1")["native"]["measurements"]
        self.assertAlmostEqual(measured["derived_current_0_to_1"]["real"], .5)

    def test_publication_cover_is_fixed_and_every_component_is_in_frame(self):
        created = circuit_create(self.runtime, {"spec": fixture(), "camera": {"mode": "custom", "position": [100, 100, -100], "target": [0, 0, 0]}})
        first = publication_cover(self.runtime, created["sav_path"])
        second = publication_cover(self.runtime, created["sav_path"])
        self.assertEqual(Path(first["cover_path"]).read_bytes(), Path(second["cover_path"]).read_bytes())
        manifest = first["cover_manifest"]
        self.assertEqual(manifest["total_elements"], 3)
        self.assertEqual(manifest["visible_elements"], 3)
        self.assertEqual(manifest["clipped_ids"], [])
        self.assertEqual(manifest["view"], {"yaw": 45, "pitch": 60, "projection": "orthographic", "fit": "all"})
