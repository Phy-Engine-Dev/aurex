"""Labels survive local import/render/state/export without becoming identities."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from aurex.config import AurexConfig
from aurex.tools.circuits import (
    _render_spec, _spec_from_sav, _view, circuit_analyze, circuit_create,
    circuit_edit, circuit_inspect, circuit_read_trace, normalize_spec,
)
from aurex.tools.registry import ToolError, ToolRuntime


def source_spec():
    return {"components": [
        {"id": "V1", "label": "a11[0]", "type": "vdc", "nodes": ["supply", "gnd"], "params": {"v": 5}},
        {"id": "R1", "label": "vout", "type": "resistor", "nodes": ["supply", "gnd"], "params": {"r": 10}},
        {"id": "R2", "label": "vout", "type": "resistor", "nodes": ["supply", "gnd"], "params": {"r": 20}},
        {"id": "R3", "type": "resistor", "nodes": ["supply", "gnd"], "params": {"r": 40}},
    ]}


class LabelValidationTests(unittest.TestCase):
    def test_digital_trace_error_explains_reader_instead_of_keyerror(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rt = ToolRuntime("reader-validation", "zh", str(root / "config.json"), AurexConfig(), folder)
            source = root / "digital.pe-state.json"
            source.write_text(json.dumps({"schema": "aurex.pe-state.v1", "scene": {"components": []}, "spec": {"components": [
                {"id": "in", "type": "digital_input", "nodes": ["N866"]}]}, "measurements": {
                "transient": {"actual_stop_s": .1, "samples": [{"time_s": .1, "components": [{"id": "in", "nodes": ["N866"], "voltage": [], "digital": [1]}]}]}}}))
            with self.assertRaisesRegex(ToolError, "pure digital.*component_ids.*not analog"):
                circuit_read_trace(rt, {"path": str(source), "nodes": ["N866"]})
            page = circuit_read_trace(rt, {"path": str(source), "component_ids": ["in"]})
            self.assertEqual(page["points"][0]["digital"], {"in": [1]})
            self.assertEqual(page["points"][0]["time_s"], .1)
            self.assertIsNotNone(page["warning"])

    def test_label_is_optional_display_metadata_not_unique_identity(self):
        source = source_spec()
        original = copy.deepcopy(source)
        spec = normalize_spec(source)
        scene, _ = _render_spec(spec)
        self.assertEqual(source, original)
        self.assertEqual([c["id"] for c in scene["components"]], ["V1", "R1", "R2", "R3"])
        self.assertEqual([c.get("label") for c in scene["components"]], ["a11[0]", "vout", "vout", None])
        self.assertNotIn("label", spec["components"][3])
        for bad in (1, {}, [], True, "x" * 4097):
            source["components"][0]["label"] = bad
            with self.subTest(bad_type=type(bad)), self.assertRaises(ToolError):
                normalize_spec(source)
        source["components"][0]["label"] = None
        self.assertIsNone(normalize_spec(source)["components"][0]["label"])


@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "native renderer required")
class LabelNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("tests forbid network")).start()
        build = Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()
        cfg = AurexConfig()
        cfg = replace(cfg, phy_engine=replace(cfg.phy_engine,
            cmake_build_dir=str(build), verilog2plsav_path=str(build / "verilog2plsav"),
            phyengine_lib_path=str(build / "libphyengine.so"), run_timeout_sec=30))
        self.runtime = ToolRuntime("label-fidelity", "zh", str(Path(self.temp.name) / "config.json"), cfg, self.temp.name)

    def test_plsav_import_roundtrip_keeps_labels_and_original_unchanged(self):
        created = circuit_create(self.runtime, {"spec": source_spec()})
        sav = Path(created["sav_path"])
        original = sav.read_bytes()
        spec = _spec_from_sav(self.runtime, sav)
        self.assertEqual([c["id"] for c in spec["components"]], ["V1", "R1", "R2", "R3"])
        self.assertEqual([c.get("label") for c in spec["components"]], ["a11[0]", "vout", "vout", None])
        self.assertNotIn("label", spec["components"][3])
        exported = circuit_create(self.runtime, {"spec": spec})
        view = circuit_inspect(self.runtime, {"path": exported["sav_path"], "query": "vout", "limit": 8})
        netlist = json.loads(Path(view["artifact"]["netlist_path"]).read_text())
        labels = {c["id"]: c["label"] for c in netlist["components"]}
        self.assertEqual({key: labels[key] for key in ("V1", "R1", "R2", "R3")},
                         {"V1": "a11[0]", "R1": "vout", "R2": "vout", "R3": ""})
        self.assertEqual(sav.read_bytes(), original)

    def test_measured_state_uses_native_spec_labels_and_query_can_find_them(self):
        result = circuit_analyze(self.runtime, {"spec": source_spec(), "analysis": "dc"})
        state_path = Path(result["state_path"])
        original = state_path.read_bytes()
        state = json.loads(original)
        for key in ("spec", "scene"):
            self.assertEqual(state[key]["components"][0]["label"], "a11[0]")
            self.assertNotIn("label", state[key]["components"][3])
        viewed = circuit_inspect(self.runtime, {"path": str(state_path), "query": "a11[0]", "limit": 1, "with_image": True})
        self.assertEqual(viewed["netlist"]["components"][0]["id"], "V1")
        self.assertEqual(viewed["netlist"]["components"][0]["label"], "a11[0]")
        self.assertIn("a11[0]", Path(viewed["artifact"]["svg_path"]).read_text())
        self.assertEqual(state_path.read_bytes(), original)
        state["scene"]["components"][0]["label"] = "conflicting label"
        bad = Path(self.temp.name) / "mismatch.pe-state.json"
        bad.write_text(json.dumps(state))
        with self.assertRaisesRegex(ToolError, "label must match"):
            circuit_inspect(self.runtime, {"path": str(bad)})

    def test_dff_notq_helper_does_not_inherit_public_label(self):
        scene = {"components": [{"id": "original-dff", "label": "q[0]", "model_id": "D Flipflop",
            "properties": {}, "nodes": ["Q", "notQ", "D", "clk"],
            "position": [1, 2, 3], "rotation": [0, 0, 180]}]}
        source = Path(self.temp.name) / "dff.json"
        source.write_text(json.dumps(scene))
        created = _view(self.runtime, source, create=True, exportable=True)
        sav = Path(created["sav_path"])
        before = hashlib.sha256(sav.read_bytes()).hexdigest()
        spec = _spec_from_sav(self.runtime, sav)
        self.assertEqual([c["id"] for c in spec["components"]], ["original-dff", "original-dff:notQ"])
        self.assertEqual(spec["components"][0]["label"], "q[0]")
        self.assertNotIn("label", spec["components"][1])
        result = circuit_analyze(self.runtime, {"spec": spec, "digital_clock_ticks": 1})
        rows = json.loads(Path(result["artifact"]["netlist_path"]).read_text())["components"]
        self.assertEqual([r["label"] for r in rows], ["q[0]", ""])
        self.assertEqual(hashlib.sha256(sav.read_bytes()).hexdigest(), before)

    def test_edit_label_does_not_reassign_ids_or_change_source(self):
        created = circuit_create(self.runtime, {"spec": source_spec()})
        path = Path(created["circuit_path"])
        before = path.read_bytes()
        edited = circuit_edit(self.runtime, {"path": str(path), "operations": [
            {"action": "update", "id": "R1", "label": "V1"},
            {"action": "update", "id": "R2", "label": None},
        ]})
        spec = json.loads(Path(edited["circuit_path"]).read_text())
        self.assertEqual([c["id"] for c in spec["components"]], ["V1", "R1", "R2", "R3"])
        self.assertEqual(spec["components"][1]["label"], "V1")
        self.assertIsNone(spec["components"][2]["label"])
        self.assertEqual(path.read_bytes(), before)

    def test_exact_focus_identity_precedes_ref_and_label_before_limit(self):
        # First-element labels deliberately shadow a later ID/ref. Explicit
        # focus must resolve identity globally before taking a one-item page.
        spec = {"components": [
            {"id": "other", "label": "R1", "type": "digital_input", "nodes": ["a"], "params": {"state": 0}},
            {"id": "R1", "label": "port", "type": "digital_input", "nodes": ["b"], "params": {"state": 1}},
            {"id": "C1", "label": "R1", "type": "digital_input", "nodes": ["c"], "params": {"state": 0}},
            {"id": "last", "label": "C2", "type": "digital_input", "nodes": ["d"], "params": {"state": 0}},
        ]}
        created = circuit_create(self.runtime, {"spec": spec})
        measured = circuit_analyze(self.runtime, {"spec": spec, "digital_clock_ticks": 1})
        for path in (created["circuit_path"], created["sav_path"], measured["state_path"]):
            before = Path(path).read_bytes()
            for token, expected in (("R1", "R1"), ("C1", "C1"), ("C2", "R1"), ("port", "R1")):
                result = circuit_inspect(self.runtime, {"path": path, "focus_id": token, "limit": 1})
                self.assertEqual([c["id"] for c in result["netlist"]["components"]], [expected])
            # Broad text matches cannot consume the explicit focus slot.
            result = circuit_inspect(self.runtime, {"path": path, "focus_id": "R1", "query": "Logic Input", "limit": 1})
            self.assertEqual(result["netlist"]["components"][0]["id"], "R1")
            self.assertEqual(Path(path).read_bytes(), before)

    def test_ambiguous_label_and_unknown_focus_cannot_silently_choose_first(self):
        created = circuit_create(self.runtime, {"spec": source_spec()})
        measured = circuit_analyze(self.runtime, {"spec": source_spec(), "analysis": "dc"})
        for path in (created["circuit_path"], created["sav_path"], measured["state_path"]):
            before = Path(path).read_bytes()
            for limit in (1, 4):
                with self.assertRaisesRegex(ToolError, 'Ambiguous focus label.*R1.*R2'):
                    circuit_inspect(self.runtime, {"path": path, "focus_id": "vout", "limit": limit})
            with self.assertRaisesRegex(ToolError, "Unknown focus token"):
                circuit_inspect(self.runtime, {"path": path, "focus_ids": ["V1", "missing"], "query": "Resistor", "limit": 2})
            # Explicit discovery query remains a permitted multi-match search.
            result = circuit_inspect(self.runtime, {"path": path, "query": "vout", "limit": 2})
            self.assertEqual({c["id"] for c in result["netlist"]["components"]}, {"R1", "R2"})
            self.assertEqual(Path(path).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
