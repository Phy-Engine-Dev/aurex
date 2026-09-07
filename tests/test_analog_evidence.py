import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from aurex.analog_evidence import AnalogEvidenceError, canonical_chinese_table, record_analysis, validate_report
from aurex.phy_engine.catalog import COMPONENTS


class AnalogEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = self.temp.name
        folder = Path(self.cache, "circuits", "run", "revision-fixture")
        folder.mkdir(parents=True)
        self.spec_path, self.state_path, self.sav_path = (folder / name for name in ("analyzed.circuit.json", "snapshot.pe-state.json", "circuit.sav"))
        self.spec = {"analysis": "dc", "components": [
            {"id": "V1", "type": "vdc", "nodes": ["vcc", "gnd"], "params": {"v": 5.0}, "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 180.0]},
            {"id": "R1", "type": "resistor", "nodes": ["vcc", "gnd"], "params": {"r": 10.0}, "position": [.18, .24, .0123456], "rotation": [0.0, 0.0, 180.0]}]}
        self.measured = {"analysis": "dc", "engine": "Phy-Engine native", "components": [
            {"id": "V1", "type": "vdc", "nodes": ["vcc", "gnd"], "pin_labels": ["+", "-"],
             "voltage": [5.0, 0.0], "voltage_imag": [0.0, 0.0], "current": [-.5], "current_imag": [0.0]},
            {"id": "R1", "type": "resistor", "nodes": ["vcc", "gnd"], "pin_labels": ["1", "2"],
             "voltage": [5.0, 0.0], "voltage_imag": [0.0, 0.0], "current": [], "current_imag": [],
             "derived_current_0_to_1": {"real": .5, "imag": 0.0, "method": "Ohm's law from simulated pin voltages"}}]}
        self.state = {"schema": "aurex.pe-state.v1", "spec": copy.deepcopy(self.spec), "measurements": self.measured,
            "origin": {"engine": "Phy-Engine native", "sample": "circuit_sample_complex", "live_handle_retained": False}}
        elements = []
        for c, model, properties in ((self.spec["components"][0], "Battery Source",
                                      {"电压": 5.0, **COMPONENTS["vdc"]["constant_pl_props"]}),
                                     (self.spec["components"][1], "Resistor", {"电阻": 10.0})):
            x, y, z = c["position"]
            elements.append({"Identifier": c["id"], "ModelID": model, "Position": f"{x:.6f},{z:.6f},{y:.6f}", "Rotation": "0,180,0",
                "Properties": properties, "Aurex": {"type": c["type"], "params": c["params"], "pin_count": 2}})
        elements.append({"Identifier": "G", "ModelID": "Ground Component", "Position": "0,0,-.1", "Rotation": "0,180,0", "Properties": {}})
        self.status = {"Elements": elements, "Wires": [
            {"Source": "V1", "SourcePin": 0, "Target": "R1", "TargetPin": 0},
            {"Source": "G", "SourcePin": 0, "Target": "V1", "TargetPin": 1},
            {"Source": "G", "SourcePin": 0, "Target": "R1", "TargetPin": 1}]}
        self.sav = {"Type": 0, "Experiment": {"ID": None, "Type": 0, "StatusSave": "", "CameraSave": "{}"},
                    "Summary": {"ID": None, "ContentID": None, "Price": 0}}
        self.write()

    def write(self):
        self.spec_path.write_text(json.dumps(self.spec, ensure_ascii=False))
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False))
        self.sav["Experiment"]["StatusSave"] = json.dumps(self.status, ensure_ascii=False)
        self.sav_path.write_text(json.dumps(self.sav, ensure_ascii=False))

    def record(self, with_sav=True):
        self.write()
        path = record_analysis(self.cache, spec_path=str(self.spec_path), state_path=str(self.state_path),
                               sav_path=str(self.sav_path) if with_sav else None)
        return json.loads(Path(path).read_text()), path

    def test_full_native_measurement_export_binding_and_chinese_table(self):
        report, path = self.record()
        self.assertTrue(report["verified"])
        self.assertTrue(report["simulation_verified"])
        self.assertTrue(report["export_verified"])
        self.assertFalse(report["functional_verification"])
        self.assertFalse(report["original_app_numerical_equivalence"])
        self.assertEqual(report["component_ids"], ["V1", "R1"])
        self.assertEqual(report["reference_ground"]["saved_marker_ids"], ["G"])
        self.assertEqual(report["sav"]["sha256"], hashlib.sha256(self.sav_path.read_bytes()).hexdigest())
        table_path = Path(report["table"]["path"])
        table = table_path.read_text()
        self.assertEqual(table, canonical_chinese_table(report))
        self.assertEqual(report["table"]["sha256"], hashlib.sha256(table_path.read_bytes()).hexdigest())
        self.assertEqual(table.count("| V1 |"), 1)
        self.assertEqual(table.count("| R1 |"), 1)
        self.assertIn("非独立电流采样", table)
        self.assertIn("由实测端电压按欧姆定律计算", table)

    def test_read_only_report_revalidation_recomputes_raw_evidence_and_tables(self):
        report, path = self.record()
        before = sorted(str(p) for p in Path(self.cache).rglob('*'))
        checked = validate_report(self.cache, path)
        self.assertEqual(checked['report'], report)
        self.assertFalse(checked['functional_verification'])
        self.assertEqual(before, sorted(str(p) for p in Path(self.cache).rglob('*')))
        raw = self.state_path.read_bytes()
        self.state_path.write_bytes(raw + b' ')
        with self.assertRaisesRegex(AnalogEvidenceError, 'hashes'):
            validate_report(self.cache, path)
        self.state_path.write_bytes(raw)
        table = Path(report['table']['path'])
        table.write_text(table.read_text().replace('0.5', '999'))
        with self.assertRaisesRegex(AnalogEvidenceError, 'every actual'):
            validate_report(self.cache, path)

    def test_forged_success_or_reduced_table_cannot_replace_raw_contract(self):
        report, path = self.record()
        report['rows'][1]['voltage_v'] = [99, 0]
        Path(path).write_text(json.dumps(report))
        with self.assertRaisesRegex(AnalogEvidenceError, 'measurements'):
            validate_report(self.cache, path)
        self.assertEqual(report["rows"][1]["branch_current_a"], [])
        again, again_path = self.record()
        self.assertNotEqual(path, again_path, "Do not overwrite earlier evidence")
        self.assertTrue(Path(path).exists())

    def test_missing_sav_retains_real_results_but_cannot_authorize_publication(self):
        report, _ = self.record(with_sav=False)
        self.assertFalse(report["verified"])
        self.assertTrue(report["simulation_verified"])
        self.assertFalse(report["export_verified"])
        self.assertEqual(len(report["rows"]), 2)

    def test_swapped_missing_nonfinite_or_inconsistent_pin_measurements_reject(self):
        original = copy.deepcopy(self.measured["components"])
        cases = [list(reversed(original)), original[:1],
            [{**original[0], "voltage": [float("nan"), 0]}, original[1]],
            [original[0], {**original[1], "voltage": [4.0, 0]}],
            [original[0], {**original[1], "pin_labels": ["2", "1"]}],
            [original[0], {**original[1], "current": [0], "current_imag": []}]]
        for rows in cases:
            self.measured["components"] = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.record()

    def test_changed_spec_cannot_reuse_previous_solver_snapshot(self):
        self.spec["components"][1]["params"]["r"] = 20.0
        with self.assertRaisesRegex(AnalogEvidenceError, "Snapshot spec differs"):
            self.record()

    def test_fabricated_derived_current_is_rejected(self):
        self.measured["components"][1]["derived_current_0_to_1"]["real"] = 50
        with self.assertRaisesRegex(AnalogEvidenceError, "Derived resistor current"):
            self.record()

    def test_export_parameter_layout_sidecar_pin_and_connectivity_changes_fail_closed(self):
        original = copy.deepcopy(self.status)
        mutations = [lambda s: s["Elements"][1]["Properties"].update({"电阻": 20}),
            lambda s: s["Elements"][1]["Aurex"]["params"].update(r=20),
            lambda s: s["Elements"][1].update(Position="10,0,0"),
            lambda s: s["Wires"][0].update(TargetPin=1),
            lambda s: s["Wires"].pop(0),
            lambda s: s["Elements"][0]["Properties"].update({"内阻": 2})]
        for mutation in mutations:
            self.status = copy.deepcopy(original)
            mutation(self.status)
            report, _ = self.record()
            with self.subTest(mutation=mutation):
                self.assertFalse(report["verified"])
                self.assertTrue(report["simulation_verified"])
                self.assertIsNotNone(report["export_error"])

    def test_primitive_only_excludes_switch_and_functional_black_boxes(self):
        self.spec["components"][1].update(type="switch", params={"closed": 1})
        self.state["spec"] = copy.deepcopy(self.spec)
        self.measured["components"][1].update(type="switch")
        del self.measured["components"][1]["derived_current_0_to_1"]
        report, _ = self.record(with_sav=False)
        self.assertFalse(report["verified"])
        self.assertFalse(report["primitive_only"])
        self.assertEqual(report["unsupported_types"], ["switch"])

    def make_transient(self):
        self.spec.update(analysis="tr", tr_step=.001, tr_stop=.5)
        self.state["spec"] = copy.deepcopy(self.spec)
        self.measured["analysis"] = "tr"
        self.measured["transient"] = {"actual_stop_s": .5, "completed_steps": 500,
            "requested_stop_s": .5, "requested_step_s": .001, "method": "native bounded transient solve; exact endpoint"}

    def test_real_transient_endpoint_is_recorded_without_inferring_functional_success(self):
        self.make_transient()
        report, _ = self.record()
        self.assertEqual(report["transient"]["actual_stop_s"], .5)
        self.assertEqual(report["transient"]["completed_steps"], 500)
        self.assertTrue(report["verified"])
        self.assertFalse(report["functional_verification"])
        self.assertIn("仅为末端状态", Path(report["table"]["path"]).read_text())

    def test_request_echo_cannot_replace_native_time_counter(self):
        self.make_transient()
        original = copy.deepcopy(self.measured["transient"])
        for override in ({"actual_stop_s": .499}, {"completed_steps": 3}, {"requested_stop_s": 1}, {"method": "copied request"}):
            self.measured["transient"] = {**original, **override}
            with self.subTest(override=override), self.assertRaises(AnalogEvidenceError):
                self.record()
        del self.measured["transient"]
        with self.assertRaisesRegex(AnalogEvidenceError, "actual native time"):
            self.record()

    def add_trace(self):
        self.make_transient()
        self.spec["tr_sample_every"] = 250
        self.state["spec"] = copy.deepcopy(self.spec)
        self.measured["transient"].update(sample_every=250, sample_count=2,
            method="one native transient solve, measurements sampled after completed steps; exact endpoint",
            samples=[{"time_s": .25, "completed_steps": 250, "components": copy.deepcopy(self.measured["components"])},
                     {"time_s": .5, "completed_steps": 500, "components": copy.deepcopy(self.measured["components"])}])

    def test_actual_trace_validates_every_point_and_complete_chinese_table_without_target_claims(self):
        self.add_trace()
        report, _ = self.record()
        self.assertTrue(report["trace"]["verified"])
        self.assertEqual(report["trace"]["times_s"], [.25, .5])
        self.assertEqual(report["table"]["component_rows"], 2)
        self.assertEqual(report["table"]["sample_count"], 1)
        self.assertEqual(report["trace_table"]["component_rows"], 4)
        self.assertFalse(report["functional_verification"])
        self.assertNotIn("samples", report["transient"], "Keep full raw trace in hashed state artifact, not a huge review prompt")
        table = Path(report["table"]["path"]).read_text()
        self.assertEqual(table.count("| V1 |"), 1)
        self.assertEqual(table.count("| R1 |"), 1)
        self.assertIn("仅为末端状态", table)
        trace = Path(report["trace_table"]["path"]).read_text()
        self.assertEqual(trace.count("| V1 |"), 2)
        self.assertIn("不补造t=0", trace)

    def test_trace_cannot_omit_reorder_fabricate_zero_or_mismatch_final_sample(self):
        self.add_trace()
        original = copy.deepcopy(self.measured["transient"])
        mutations = [lambda t: t["samples"][0].update(time_s=0),
            lambda t: t["samples"].reverse(), lambda t: t["samples"].pop(),
            lambda t: t["samples"][0]["components"].pop(),
            lambda t: t["samples"][0].update(completed_steps=2),
            lambda t: t["samples"][-1]["components"][0].update(current=[-.1])]
        for mutation in mutations:
            self.measured["transient"] = copy.deepcopy(original)
            mutation(self.measured["transient"])
            with self.subTest(mutation=mutation), self.assertRaises(AnalogEvidenceError):
                self.record()

    def test_bjt_native_terminal_current_is_preferred_and_never_replaced_with_empty_branch_zero(self):
        model = COMPONENTS["npn"]
        params = {**model["defaults"], **{k: v["default"] for k, v in model.get("extra_params", {}).items()}}
        self.spec["components"].append({"id": "Q1", "type": "npn", "nodes": ["base", "vcc", "gnd"],
            "params": params, "position": [.4, 0, 0], "rotation": [0, 0, 180]})
        self.state["spec"] = copy.deepcopy(self.spec)
        self.measured["components"].append({"id": "Q1", "type": "npn", "nodes": ["base", "vcc", "gnd"],
            "pin_labels": ["base", "collector", "emitter"], "voltage": [.7, 5, 0], "voltage_imag": [0, 0, 0],
            "current": [], "current_imag": [], "pin_current_a": [1e-5, 1e-3, -1.01e-3],
            "pin_current_convention": "Positive current enters the component, in catalog pin order; native model observation."})
        report, _ = self.record(with_sav=False)
        row = report["rows"][-1]
        self.assertEqual(row["branch_current_a"], [])
        self.assertEqual(row["pin_current_a"], [1e-5, 1e-3, -1.01e-3])
        table = Path(report["table"]["path"]).read_text()
        self.assertIn("正电流流入器件", table)
        self.assertIn("collector: 0.001", table)
        self.measured["components"][-1]["pin_current_a"] = [1, 1, 1]
        with self.assertRaisesRegex(AnalogEvidenceError, "charge conservation"):
            self.record(with_sav=False)


if __name__ == "__main__":
    unittest.main()
