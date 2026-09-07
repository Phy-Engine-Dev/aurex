"""Rendering/context regressions; offline, with all network calls forbidden."""
from __future__ import annotations
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from aurex.config import AurexConfig
from aurex.tools.circuits import _compact_view, circuit_inspect, publication_cover
from aurex.tools.registry import ToolRuntime


class CompactContextTests(unittest.TestCase):
    def test_compaction_keeps_truth_in_artifact_not_repeated_geometry(self):
        components = [{"id": f"a{i}", "ref": f"C{i}", "type": "Input" if i < 600 else "Gate", "pins": [{"pin": 0, "node": "n"}], "properties": {}} for i in range(631)]
        full = {"components": components, "nodes": [{"id": "n", "connections": [{"component": c["id"], "pin": 0} for c in components]}], "wires": [], "statistics_source": "saved"}
        camera = {"overview": True, "projected_centers": ["big"] * 10000, "pin_geometries": ["big"] * 10000}
        summary = {"components": 631, "nodes": 1, "offset": 0, "limit": 631, "view": "overview", "projection": "isometric", "focused": False, "match_count": 0, "visible_ids": [c["id"] for c in components], "camera": camera}
        result = _compact_view(full, summary)
        self.assertEqual(result["statistics"]["components"], 631)
        self.assertEqual({c["type"] for c in result["netlist"]["components"]}, {"Input", "Gate"})
        self.assertNotIn("camera", result["pagination"])
        self.assertNotIn("projected_centers", result["camera"])
        self.assertIsNone(result["pagination"]["next_offset"])
        self.assertLess(len(json.dumps(result)), 3000)
        self.assertEqual(len(full["components"]), 631)
        self.assertEqual(len(summary["camera"]["projected_centers"]), 10000)
        summary.update(camera={}, visible_ids=["a0", "a1"], limit=2, view="spatial")
        result = _compact_view(full, summary)
        self.assertEqual(result["netlist"]["nodes"][0]["external_connections"], 629)
        self.assertEqual(len(result["netlist"]["nodes"][0]["connections"]), 2)


@unittest.skipUnless(os.environ.get("AUREX_RENDER_FIXTURE"), "offline large circuit fixture not configured")
class RealLargeSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = Path(os.environ["AUREX_RENDER_FIXTURE"]).resolve()
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(build), verilog2plsav_path=str(build / "verilog2plsav"), phyengine_lib_path=str(build / "libphyengine.so")))
        cls.runtime = ToolRuntime(task_id="renderer-offline-validation", user_lang="zh", config_path=str(cls.fixture.parent / "config.json"), config=cfg, cache_dir=str(cls.fixture.parent))
        cls.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden in render tests"))
        cls.network.start()
        cls.addClassCleanup(cls.network.stop)
        cls.original = cls.fixture.read_bytes()
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        assert cls.fixture.read_bytes() == cls.original
        report = cls.fixture.parent / "render-validation.json"
        report.write_text(json.dumps(cls.results, ensure_ascii=False, indent=2))
        print("OFFLINE_RENDER_REPORT", report)

    def test_default_renders_all_positions_in_fixed_canvas(self):
        result = circuit_inspect(self.runtime, {"path": str(self.fixture), "with_image": True})
        from PIL import Image
        with Image.open(result["artifact"]["png_path"]) as image:
            self.assertEqual(image.size, (1280, 960))
        full = json.loads(Path(result["artifact"]["netlist_path"]).read_text())
        svg = Path(result["artifact"]["svg_path"]).read_text()
        self.assertEqual(len(full["components"]), 631)
        self.assertEqual(svg.count("data-component-id="), 631)
        self.assertEqual(result["camera"]["clipped_component_ids_count"], 0)
        self.assertEqual(result["pagination"]["view"], "overview")
        self.assertIsNone(result["pagination"]["next_offset"])
        self.assertLessEqual(len(result["netlist"]["components"]), 8)
        self.assertLess(len(json.dumps(result, ensure_ascii=False)), 16000)
        self.assertNotIn("projected_centers", result["camera"])
        self.assertEqual(len(result["images"]), 2)
        self.assertEqual(result["primary_viewport"]["camera"]["rendered_components"], 629)
        self.assertTrue(result["primary_viewport"]["camera"]["viewport_is_subset"])
        self.assertEqual({c["ref"] for c in result["camera"]["spatial_outliers"]}, {"C435", "C456"})
        self.assertFalse(result["artifact"]["external_write_performed"])
        self.assertNotIn("published", result["artifact"])
        raw = json.loads(self.original)
        experiment = raw.get("Experiment", raw)
        elements = json.loads(experiment["StatusSave"])["Elements"]
        positions = {c["Identifier"]: [float(n) for n in c["Position"].split(",")] for c in elements}
        for component in full["components"]:
            raw_position = positions[component["id"]]
            for actual, saved in zip(component["position"], [raw_position[0], raw_position[2], raw_position[1]]):
                self.assertAlmostEqual(actual, saved, places=8)
        self.results["overview"] = result
        self.results["overview_model_json_characters"] = len(json.dumps(result, ensure_ascii=False))

    def test_focus_and_explicit_legacy_page_have_bounded_canvas(self):
        from PIL import Image
        for name, args in (("focus", {"query": "D Flipflop", "limit": 4}), ("legacy_page", {"view": "spatial", "limit": 24})):
            result = circuit_inspect(self.runtime, {"path": str(self.fixture), "with_image": True, **args})
            with Image.open(result["artifact"]["png_path"]) as image:
                self.assertEqual(image.size, (1080, 820))
            self.assertLessEqual(result["camera"]["legend_components"], 4)
            self.assertLessEqual(len(result["netlist"]["components"]), 8)
            self.assertLess(len(json.dumps(result, ensure_ascii=False)), 18000)
            self.results[name] = result

    def test_publication_cover_remains_fixed_and_all_in_frame(self):
        result = publication_cover(self.runtime, str(self.fixture))
        self.assertEqual(result["camera"]["source"], "fixed-publication-overview")
        self.assertEqual(result["cover_manifest"]["total_elements"], 631)
        self.assertEqual(result["cover_manifest"]["clipped_ids"], [])
        self.assertTrue(result["cover_manifest"]["rendered_all"])
        self.results["publication_cover"] = result


if __name__ == "__main__":
    unittest.main()
