"""The local-view locator always includes the complete immutable scene."""
from __future__ import annotations

from dataclasses import replace
import json
import os
import re
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from aurex.config import AurexConfig
from aurex.tools.circuits import circuit_create, circuit_inspect, publication_cover
from aurex.tools.registry import ToolRuntime


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "native renderer required")
class LocatorNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("tests forbid network")).start()
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(build),
            verilog2plsav_path=str(build / "verilog2plsav"), phyengine_lib_path=str(build / "libphyengine.so")))
        self.rt = ToolRuntime("minimap", "zh", str(Path(self.temp.name) / "config.json"), cfg, self.temp.name)
        components = [{"id": f"IN{i}", "label": f"port[{i}]", "type": "digital_input", "nodes": [f"N{i}"],
                       "params": {"state": i % 2}, "position": [(i % 8) * .18, (i // 8) * .18, 0]} for i in range(40)]
        components.append({"id": "distant-original", "type": "digital_input", "nodes": ["far"], "params": {"state": 0}, "position": [-60, 0, 0]})
        self.created = circuit_create(self.rt, {"spec": {"components": components}})
        self.source = Path(self.created["sav_path"])
        self.original = self.source.read_bytes()

    def inspect(self, **args):
        result = circuit_inspect(self.rt, {"path": str(self.source), "with_image": True, **args})
        svg = ET.parse(result["artifact"]["svg_path"]).getroot()
        camera = json.loads(Path(result["artifact"]["camera_path"]).read_text())["camera"]
        metadata = next((e for e in svg.iter() if e.attrib.get("id") == "locator-markers"), None)
        markers = json.loads(metadata.text) if metadata is not None else []
        # Keep validating the actual rendered subpaths, not metadata alone.
        for selected in (False, True):
            layer = next((e for e in svg.iter() if e.attrib.get("data-mini-layer") == ("selected" if selected else "unselected")), None)
            expected = [m for m in markers if m["selected"] == selected]
            if layer is None:
                self.assertEqual(expected, [])
                continue
            moves = re.findall(r"M ([^ ]+) ([^ ]+)", layer.attrib["d"])
            self.assertEqual(len(moves), len(expected))
            for (x,y), m in zip(moves, expected):
                self.assertAlmostEqual(float(x), m["x"] - (2.7 if selected else 1.5), places=2)
                self.assertAlmostEqual(float(y), m["y"], places=2)
        self.assertEqual(self.source.read_bytes(), self.original)
        return result, svg, camera, markers

    def test_focus_highlights_actual_visible_ids_but_locator_preserves_all(self):
        result, svg, camera, markers = self.inspect(focus_id="IN3", limit=1, camera={"mode": "auto", "fit": True})
        self.assertEqual(len(markers), 41)
        yellow = {e["id"] for e in markers if e["selected"]}
        self.assertEqual(yellow, {"IN3"})
        self.assertEqual(set(camera["minimap"]["highlighted_ids"]), yellow)
        self.assertIn("distant-original", {e["id"] for e in markers})
        self.assertEqual(svg.attrib["height"], "820")
        self.assertNotIn("highlighted_ids", result["camera"]["minimap"])
        self.assertTrue(result["camera"]["minimap"]["rough"])
        x,y,w,h = camera["minimap"]["bounds"]
        for marker in markers:
            self.assertTrue(x < marker["x"] < x+w)
            self.assertTrue(y < marker["y"] < y+h)

    def test_primary_subset_is_explicit_and_original_outlier_is_not_highlighted(self):
        _, svg, camera, markers = self.inspect(view="region")
        self.assertEqual(len(markers), 41)
        self.assertEqual(camera["minimap"]["highlighted_components"], 40)
        self.assertNotIn("distant-original", camera["minimap"]["highlighted_ids"])
        self.assertEqual(camera["clipped_component_ids"], [])
        self.assertEqual(camera["spatial_outliers"][0]["id"], "distant-original")
        self.assertEqual((svg.attrib["width"], svg.attrib["height"]), ("1280", "960"))
        # Main-view clip ends above the reserved locator band; it covers no body/pin.
        clips = [e for e in svg.iter() if e.attrib.get("id") == "overview-scene"]
        rect = list(clips[0])[0]
        self.assertLessEqual(float(rect.attrib["y"])+float(rect.attrib["height"]), camera["minimap"]["bounds"][1])

    def test_clipped_components_are_not_claimed_visible_or_yellow(self):
        _, _, camera, markers = self.inspect(view="spatial", limit=8, camera={"mode": "auto", "fit": True, "zoom": 4})
        expected = {c["id"] for c in camera["projected_centers"] if not c["outside_frame"]}
        self.assertTrue(camera["clipped_components"])
        self.assertEqual(set(camera["minimap"]["highlighted_ids"]), expected)
        self.assertEqual({e["id"] for e in markers if e["selected"]}, expected)

    def test_complete_and_publication_cover_have_no_partial_locator(self):
        _, _, camera, markers = self.inspect(view="overview")
        self.assertEqual(markers, [])
        self.assertNotIn("minimap", camera)
        cover = publication_cover(self.rt, str(self.source))
        self.assertTrue(cover["cover_manifest"]["rendered_all"])
        self.assertEqual(cover["cover_manifest"]["total_elements"], 41)
        self.assertEqual(cover["cover_manifest"]["clipped_ids"], [])
        self.assertNotIn("minimap", cover["camera"])
        self.assertEqual(self.source.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
