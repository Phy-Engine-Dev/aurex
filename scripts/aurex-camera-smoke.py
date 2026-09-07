#!/usr/bin/env python3
"""Build an offline, analytically verifiable multi-cluster camera fixture.

Run from the aurex3 repository: .venv/bin/python scripts/aurex-camera-smoke.py prepare
This command never calls a language model, edits service settings or publishes.
The oracle is an evaluator artifact; do not include it in model read-image prompts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aurex.config import load_config
from aurex.tools.circuits import circuit_analyze, circuit_inspect, normalize_spec
from aurex.tools.registry import ToolRuntime


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def make_spec():
    components = [{
        "id": "V_MAIN", "type": "vdc", "nodes": ["V12", "gnd"],
        "params": {"v": 12}, "position": [-.70, .40, .10], "rotation": [0, 0, 15],
    }]
    clusters = [
        ("A", (-.45, -.20, .02), 100, 100e-9, (0, 20, 45)),
        ("B", (.02, .30, .22), 200, 220e-9, (60, 90, 120)),
        ("C", (.57, -.22, .42), 300, 470e-9, (-25, -50, -75)),
    ]
    oracle = {"source_voltage_v": 12.0, "native_component_count": 14, "branches": {},
              "nodes_v": {"V12": 12.0, "gnd": 0.0}, "component_currents_a": {},
              "component_power_w": {}, "positions_and_rotations": {}}
    for group, (x, y, z), resistance, capacitance, yaws in clusters:
        nodes = ["V12", f"{group}_J1", f"{group}_J2", "gnd"]
        current = 12 / (6 * resistance)
        ids = []
        for index, (dx, dy, dz) in enumerate([(-.15, -.06, .01), (.02, 0, .06), (.17, .06, -.02)]):
            cid = f"R_{group}{index + 1}"
            ids.append(cid)
            r = resistance * (index + 1)
            components.append({
                "id": cid, "type": "resistor", "nodes": nodes[index:index + 2],
                "params": {"r": r}, "position": [x + dx, y + dy, z + dz],
                "rotation": [0 if group == "A" else 10, 0 if group != "C" else -12, yaws[index]],
            })
            oracle["component_currents_a"][cid] = current
            oracle["component_power_w"][cid] = current * current * r
        cap_id = f"C_{group}"
        components.append({
            "id": cap_id, "type": "capacitor", "nodes": [nodes[2], "gnd"],
            "params": {"c": capacitance}, "position": [x + .04, y - .18, z + .15],
            "rotation": [15, 8, yaws[1] + 25],
        })
        oracle["branches"][group] = {"resistors": ids, "total_resistance_ohm": resistance * 6,
                                      "dc_current_a": current, "capacitor": cap_id,
                                      "capacitance_f": capacitance, "capacitor_dc_voltage_v": 6.0}
        oracle["nodes_v"].update({nodes[1]: 10.0, nodes[2]: 6.0})
    oracle["source_delivered_current_a"] = sum(branch["dc_current_a"] for branch in oracle["branches"].values())
    oracle["source_delivered_power_w"] = 12 * oracle["source_delivered_current_a"]
    oracle["positions_and_rotations"] = {c["id"]: {"position": c["position"], "rotation": c["rotation"]} for c in components}
    return normalize_spec({"title": "Multi-cluster RC camera validation", "analysis": "dc", "components": components}), oracle


def prepare(config: Path):
    cfg = load_config(str(config))
    cache = Path(cfg.resolve_path(cfg.storage.cache_dir, config_path=str(config)))
    folder = cache / "camera-validation"
    folder.mkdir(parents=True, exist_ok=True)
    runtime = ToolRuntime(task_id="camera-validation-native", user_lang="en", config_path=str(config), config=cfg, cache_dir=str(cache))
    spec, oracle = make_spec()
    write(folder / "complex-rc.circuit.json", spec)
    write(folder / "oracle.json", oracle)
    solved = circuit_analyze(runtime, {"spec": spec, "analysis": "dc"})
    measured = {component["id"]: component for component in solved["measurements"]["components"]}
    checks = []
    for component in spec["components"]:
        sample = measured[component["id"]]
        for index, node in enumerate(component["nodes"]):
            actual, expected = sample["voltage"][index], oracle["nodes_v"][node]
            assert math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7), (component["id"], node, actual, expected)
            checks.append({"component": component["id"], "pin": index, "expected_v": expected, "measured_v": actual})
        if component["type"] == "resistor":
            actual = sample["derived_current_0_to_1"]["real"]
            expected = oracle["component_currents_a"][component["id"]]
            assert math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9), (component["id"], actual, expected)
    write(folder / "measurements.json", solved["measurements"])
    if solved.get("state_path"):
        shutil.copy2(solved["state_path"], folder / "complex-rc.pe-state.json")
    # Preserve a native saved fixture, including a stable ground identity and
    # an explicit ground position. PhysicsLab serializes xyz as x,z,y.
    saved = json.loads(Path(solved["sav_path"]).read_text())
    state = json.loads(saved["Experiment"]["StatusSave"])
    grounds = [element for element in state["Elements"] if element["ModelID"] == "Ground Component"]
    assert len(grounds) == 1
    ground = grounds[0]
    previous_id = ground["Identifier"]
    ground.update({"Identifier": "GND0", "Label": "Ground reference", "Position": "-0.76,0.02,0.16", "Rotation": "0,0,0"})
    ground.setdefault("Aurex", {})["position_source"] = "provided"
    for wire in state["Wires"]:
        for endpoint in ("Source", "Target"):
            if wire[endpoint] == previous_id:
                wire[endpoint] = "GND0"
    saved["Experiment"]["StatusSave"] = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    native = folder / "complex-rc.sav"
    write(native, saved)
    viewed = circuit_inspect(runtime, {"path": str(native), "limit": 24})
    assert viewed["pagination"]["components"] == 14, viewed["pagination"]
    actual_ground = next(c for c in viewed["netlist"]["components"] if c["id"] == "GND0")
    assert actual_ground["position"] == [-.76, .16, .02], actual_ground["position"]
    oracle["positions_and_rotations"]["GND0"] = {"position": [-.76, .16, .02], "rotation": [0, 0, 0]}
    oracle["saved_camera_raw"] = saved["Experiment"]["CameraSave"]
    write(folder / "oracle.json", oracle)
    for key, name in (("svg_path", "overview.svg"), ("png_path", "overview.png"), ("netlist_path", "native-netlist.json")):
        shutil.copy2(viewed["artifact"][key], folder / name)
    report = {"fixture": str(native), "spec": str(folder / "complex-rc.circuit.json"),
              "native_state": str(folder / "complex-rc.pe-state.json"),
              "oracle": str(folder / "oracle.json"), "measurements": str(folder / "measurements.json"),
              "native_netlist": str(folder / "native-netlist.json"), "overview": str(folder / "overview.png"),
              "component_count": 14, "dc_voltage_checks_passed": len(checks), "dc_current_checks_passed": 9,
              "warning": "Oracle and measurements are evaluator-only; never inject them into blind image-read prompts."}
    write(folder / "fixture-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare"])
    parser.add_argument("--config", type=Path, default=REPO / ".config" / "aurex3.json")
    arguments = parser.parse_args()
    prepare(arguments.config.resolve())
