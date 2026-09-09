"""Data-first circuit tools: actual native fixtures, no network or model calls."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from aurex.config import AurexConfig
from aurex.tools.circuits import (
    circuit_analyze, circuit_create, circuit_edit, circuit_inspect,
    circuit_read_stimulus, circuit_read_trace, publication_cover, register_circuit_tools,
)
from aurex.tools.registry import ToolError, ToolRegistry, ToolRuntime


def digital_spec(count=11):
    parts = []
    for i in range(count):
        parts.extend([
            {"id": f"in-{i}", "label": f"input[{i}]", "type": "digital_input", "nodes": [f"n{i}"],
             "params": {"state": i % 2}, "position": [i * .2, 0, 0]},
            {"id": f"out-{i}", "label": f"output[{i}]", "type": "digital_output", "nodes": [f"n{i}"],
             "position": [i * .2, .4, 0]},
        ])
    parts.append({"id": "internal", "type": "digital_not", "nodes": ["n0", "unused"], "position": [0, .2, 0]})
    return {"components": parts}


class DataModeSchemaTests(unittest.TestCase):
    def test_public_flags_default_false_without_render_alias(self):
        registry = ToolRegistry()
        register_circuit_tools(registry)
        for name in ("circuit_inspect", "circuit_create", "circuit_edit", "circuit_analyze"):
            props = registry.get(name).parameters["properties"]
            self.assertEqual(props["with_image"]["default"], False)
            self.assertNotIn("render", props)
        props = registry.get("circuit_inspect").parameters["properties"]
        self.assertEqual(props["interface_only"]["default"], False)
        self.assertEqual(props["limit"]["maximum"], 64)


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "native engine/renderer required")
class DataModeNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine, cmake_build_dir=str(build),
            verilog2plsav_path=str(build / "verilog2plsav"), phyengine_lib_path=str(build / "libphyengine.so")))
        self.rt = ToolRuntime("data-mode", "zh", str(Path(self.temp.name) / "config.json"), cfg, self.temp.name)

    def test_all_four_handlers_default_to_no_svg_no_cairo_preserving_evidence(self):
        spec = {"components": [
            {"id": "V", "type": "vdc", "nodes": ["n", "gnd"], "params": {"v": 5}},
            {"id": "R", "type": "resistor", "nodes": ["n", "gnd"], "params": {"r": 10}},
        ]}
        with patch("cairosvg.svg2png", side_effect=AssertionError("default must not rasterize")):
            created = circuit_create(self.rt, {"spec": spec})
            inspected = circuit_inspect(self.rt, {"path": created["sav_path"]})
            edited = circuit_edit(self.rt, {"path": created["sav_path"], "operations": [
                {"action": "update", "id": "R", "params": {"r": 20}},
            ]})
            analyzed = circuit_analyze(self.rt, {"path": edited["sav_path"], "analysis": "dc"})
        for result in (created, inspected, edited, analyzed):
            self.assertFalse(result["with_image"])
            self.assertEqual(result["images"], [])
            self.assertNotIn("png_path", result["artifact"])
            self.assertNotIn("svg_path", result["artifact"])
            self.assertTrue(Path(result["artifact"]["netlist_path"]).is_file())
            self.assertEqual(result["camera"]["source"], "not-rendered")
        self.assertEqual(list(Path(self.temp.name).rglob("*.svg")), [])
        self.assertEqual(list(Path(self.temp.name).rglob("*.png")), [])
        measured = next(c for c in analyzed["measurements"]["components"] if c["id"] == "R")
        self.assertAlmostEqual(measured["derived_current_0_to_1"]["real"], .25)
        for key in ("state_path", "sav_path", "analysis_table_path", "report_path"):
            self.assertTrue(Path(analyzed[key]).is_file(), key)
        original = json.loads(Path(created["circuit_path"]).read_text())
        self.assertEqual(original["components"][1]["params"]["r"], 10)

    def test_explicit_image_all_handlers_and_fixed_cover(self):
        created = circuit_create(self.rt, {"spec": digital_spec(2), "with_image": True})
        inspected = circuit_inspect(self.rt, {"path": created["sav_path"], "with_image": True, "focus_id": "in-0", "limit": 1})
        edited = circuit_edit(self.rt, {"path": created["circuit_path"], "with_image": True, "operations": [
            {"action": "update", "id": "in-0", "params": {"state": 1}},
        ]})
        analyzed = circuit_analyze(self.rt, {"path": edited["circuit_path"], "with_image": True, "digital_clock_ticks": 1})
        for result in (created, inspected, edited, analyzed):
            self.assertTrue(result["with_image"])
            self.assertEqual(Path(result["images"][0]["path"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue(inspected["camera"]["minimap"]["enabled"])
        cover = publication_cover(self.rt, created["sav_path"])
        self.assertTrue(cover["cover_manifest"]["rendered_all"])
        self.assertEqual(cover["cover_manifest"]["clipped_ids"], [])
        self.assertEqual(cover["cover_manifest"]["visible_elements"], 5)

    def test_query_pages_all_three_sources_without_overlap_or_solve(self):
        spec = digital_spec(11)
        created = circuit_create(self.rt, {"spec": spec})
        analyzed = circuit_analyze(self.rt, {"spec": spec, "digital_clock_ticks": 1})
        expected = [f"out-{i}" for i in range(11)]
        for source in (created["sav_path"], created["circuit_path"], analyzed["state_path"]):
            before = Path(source).read_bytes()
            ids = []
            with patch("aurex.tools.circuits.pe_simulate", side_effect=AssertionError("inspect cannot solve")):
                for offset in (0, 4, 8):
                    result = circuit_inspect(self.rt, {"path": source, "query": "Logic Output", "offset": offset, "limit": 4})
                    page = result["pagination"]
                    self.assertEqual(page["total_matches"], 11)
                    self.assertEqual(page["primary_ids"], expected[offset:offset + 4])
                    ids.extend(page["primary_ids"])
                    self.assertEqual(page["next_offset"], offset + 4 if offset < 8 else None)
                    self.assertIn("Optional neighbors", page["scope"])
                    self.assertTrue(all(c["selection_role"] == ("primary" if c["id"] in page["primary_ids"] else "neighbor")
                                        for c in result["netlist"]["components"]))
                with self.assertRaisesRegex(ToolError, "offset exceeds total_matches=11"):
                    circuit_inspect(self.rt, {"path": source, "query": "Logic Output", "offset": 11})
            self.assertEqual(ids, expected)
            self.assertEqual(len(set(ids)), 11)
            self.assertEqual(before, Path(source).read_bytes())

    def test_mixed_focus_query_deduplicates_before_paging_and_scopes_neighbors(self):
        created = circuit_create(self.rt, {"spec": digital_spec(5)})
        expected = ["out-4", "in-3", "out-0", "out-1", "out-2", "out-3"]
        found = []
        for offset in (0, 2, 4):
            result = circuit_inspect(self.rt, {"path": created["sav_path"], "focus_ids": ["out-4", "in-3"],
                "query": "Logic Output", "offset": offset, "limit": 2})
            found += result["pagination"]["primary_ids"]
            self.assertEqual(result["pagination"]["neighbor_ids"], [])
        self.assertEqual(found, expected)
        result = circuit_inspect(self.rt, {"path": created["sav_path"], "focus_id": "out-0", "limit": 2})
        self.assertEqual(result["pagination"]["primary_ids"], ["out-0"])
        self.assertEqual(result["pagination"]["neighbor_ids"], ["in-0"])
        self.assertEqual(result["pagination"]["total_matches"], 1)
        self.assertFalse(result["pagination"]["has_more"])

    def test_interface_all_three_sources_paginates_only_actual_io(self):
        spec = digital_spec(35)
        created = circuit_create(self.rt, {"spec": spec})
        analyzed = circuit_analyze(self.rt, {"spec": spec, "digital_clock_ticks": 1})
        expected = [c for c in spec["components"] if c["type"] != "digital_not"]
        for source in (created["sav_path"], created["circuit_path"], analyzed["state_path"]):
            before = Path(source).read_bytes()
            with patch("cairosvg.svg2png", side_effect=AssertionError("I/O is data-only")), \
                 patch("aurex.tools.circuits.pe_simulate", side_effect=AssertionError("I/O must not re-solve")):
                first = circuit_inspect(self.rt, {"path": source, "interface_only": True})
                second = circuit_inspect(self.rt, {"path": source, "interface_only": True, "offset": first["next_offset"]})
            self.assertEqual(first["total_components"], 71)
            self.assertEqual(first["total_ports"], 70)
            self.assertEqual(first["total_inputs"], 35)
            self.assertEqual(first["total_outputs"], 35)
            self.assertEqual(len(first["ports"]), 64)
            self.assertEqual(len(second["ports"]), 6)
            self.assertFalse(second["has_more"])
            ports = first["ports"] + second["ports"]
            self.assertEqual([p["id"] for p in ports], [p["id"] for p in expected])
            for actual, original in zip(ports, expected):
                self.assertEqual(actual["label"], original["label"])
                # interface_only is intentionally a compact electrical-I/O view;
                # spatial data belongs to an explicit focused/spatial inspection.
                self.assertNotIn("position", actual)
                self.assertIn("logic_source", actual)
                self.assertNotIn("pins", actual)
                if source == analyzed["state_path"]:
                    self.assertIn("recorded native", actual["logic_source"])
                    self.assertEqual(actual["logic"], int(actual["id"].split("-")[-1]) % 2)
            self.assertEqual(first["images"], [])
            self.assertNotIn("svg_path", first["artifact"])
            self.assertEqual(before, Path(source).read_bytes())

    def test_interface_missing_statistic_is_unknown_not_zero_and_original_label_not_id(self):
        created = circuit_create(self.rt, {"spec": digital_spec(1)})
        document = json.loads(Path(created["sav_path"]).read_text())
        experiment = document.get("Experiment", document)
        state = json.loads(experiment["StatusSave"])
        for component in state["Elements"]:
            if component["Identifier"] == "out-0":
                component["Statistics"] = {}
                component.pop("Label", None)
        experiment["StatusSave"] = json.dumps(state)
        source = Path(self.temp.name) / "missing.sav"
        source.write_text(json.dumps(document))
        result = circuit_inspect(self.rt, {"path": str(source), "interface_only": True})
        output = next(p for p in result["ports"] if p["id"] == "out-0")
        self.assertIsNone(output["logic"])
        self.assertEqual(output["label"], "")

    def test_interface_rejects_conflicting_options_and_invalid_flags(self):
        created = circuit_create(self.rt, {"spec": digital_spec(1)})
        for extra in ({"with_image": True}, {"query": "Logic Input"}, {"focus_id": "in-0"}, {"limit": 65}):
            with self.subTest(extra=extra), self.assertRaises(ToolError):
                circuit_inspect(self.rt, {"path": created["sav_path"], "interface_only": True, **extra})
        with self.assertRaisesRegex(ToolError, "must be boolean"):
            circuit_analyze(self.rt, {"spec": digital_spec(1), "with_image": "false"})

    def test_digital_transient_reader_does_not_invent_stimulus_or_analog_samples(self):
        spec = digital_spec(1)
        for extra in ({}, {"stimulus": [{"set": {"in-0": 1}}, {"set": {"in-0": 0}}]}):
            result = circuit_analyze(self.rt, {"spec": spec, "analysis": "tr", "tr_step": 1e-8,
                "tr_stop": 3e-8, "tr_sample_every": 1, **extra})
            trace = result["measurements"]["transient"]
            raw = json.loads(Path(result["state_path"]).read_text())["measurements"]
            self.assertEqual(len(raw["transient"]["samples"]), 3)
            self.assertEqual(trace["trace_reader"], "circuit_read_trace")
            self.assertEqual(trace["trace_access"]["kind"], "recorded_digital_solver_samples")
            if extra:
                self.assertEqual(trace["trace_access"]["separate_stimulus_reader"], "circuit_read_stimulus")
                recorded = circuit_read_stimulus(self.rt, {"path": result["state_path"], "component_ids": ["out-0"]})
                self.assertEqual(recorded["total_steps"], 2)
            else:
                self.assertFalse(trace["trace_access"]["stimulus_recorded"])
                self.assertNotIn("stimulus_results", raw)
            original = Path(result["state_path"]).read_bytes()
            page = circuit_read_trace(self.rt, {"path": result["state_path"], "component_ids": ["out-0"]})
            self.assertEqual(page["total_samples"], 3)
            self.assertEqual([p["time_s"] for p in page["points"]], [p["time_s"] for p in raw["transient"]["samples"]])
            self.assertTrue(page["recorded_not_resimulated"])
            self.assertEqual(Path(result["state_path"]).read_bytes(), original)
            with self.assertRaisesRegex(ToolError, "pure digital state"):
                circuit_read_trace(self.rt, {"path": result["state_path"], "nodes": ["n0"]})


if __name__ == "__main__":
    unittest.main()
