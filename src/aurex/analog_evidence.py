"""Generic primitive-circuit numerical evidence; never a target-design verifier.

Records solver outputs and export correspondence without introducing an experiment
solution. A successful final-state solve is NOT proof that a requested function works.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from . import publishing
from .phy_engine.catalog import COMPONENTS


PRIMITIVES = frozenset(("resistor", "capacitor", "inductor", "vdc", "idc", "diode", "npn", "pnp"))
SCHEMA = "aurex.analog-evidence.v1"


class AnalogEvidenceError(ValueError):
    pass


def _fail(message: str):
    raise AnalogEvidenceError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _read(cache_dir: str, path: str, suffixes=(".json",)):
    file, raw = publishing._read_artifact(cache_dir, path, suffixes=suffixes, limit=32 * 1024 * 1024)
    value = publishing._json(raw)
    if isinstance(value, dict) and value.get('schema') == 'aurex.pe-state.v1':
        from .trace_archive import hydrate_snapshot
        value = hydrate_snapshot(file, value)
    return value, {"path": str(file), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(label + " must be a finite numeric measurement")
    return float(value)


def _vector(value: Any, label: str, length: int | None = None) -> list[float]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        _fail(label + " has an incomplete or incorrectly ordered measurement array")
    return [_finite(number, label) for number in value]


def _ground(node: str) -> str:
    return "gnd" if node.casefold() in ("gnd", "ground", "0") else node


def _spec_components(spec: Any) -> list[dict[str, Any]]:
    if not isinstance(spec, dict) or not isinstance(spec.get("components"), list) or not 1 <= len(spec["components"]) <= 512:
        _fail("Evidence requires a complete local spec with 1..512 components")
    ids = set()
    out = []
    for item in spec["components"]:
        if not isinstance(item, dict) or item.get("type") not in COMPONENTS:
            _fail("Unknown native component in the analysis spec")
        cid, kind = item.get("id"), COMPONENTS[item["type"]]
        if not isinstance(cid, str) or not cid or cid in ids:
            _fail("Analysis component IDs must be unique")
        ids.add(cid)
        nodes = item.get("nodes")
        if not isinstance(nodes, list) or len(nodes) != kind["pins"] or not all(isinstance(n, str) and n for n in nodes):
            _fail("Analysis nodes must preserve the exact native pin order")
        if not isinstance(item.get("params", {}), dict):
            _fail("Analysis parameters must be a dictionary")
        extra = kind.get("extra_params", {})
        params = {**kind["defaults"], **{k: v["default"] for k, v in extra.items()}, **item.get("params", {})}
        if set(params) != set(kind["props"]) | set(extra):
            _fail("Analysis parameters must exactly match the native model contract")
        for key, value in params.items():
            _finite(value, f"{cid}.{key}")
        out.append({**item, "params": params, "nodes": [_ground(n) for n in nodes]})
    return out


def _measurements(spec: dict, state: dict, components: list[dict], *, validate_transient=True) -> tuple[list[dict], dict | None]:
    if (state.get("schema") != "aurex.pe-state.v1" or not isinstance(state.get("spec"), dict)
        or _canonical(state["spec"]) != _canonical(spec)):
        _fail("Snapshot spec differs from the analyzed source; measurements cannot be transferred to a different design")
    origin, measured = state.get("origin"), state.get("measurements")
    if (not isinstance(origin, dict) or origin.get("engine") != "Phy-Engine native"
        or origin.get("sample") != "circuit_sample_complex" or not isinstance(measured, dict)
        or measured.get("engine") != "Phy-Engine native"):
        _fail("Evidence must identify measurements copied from the native solver")
    analysis = str(spec.get("analysis", "dc"))
    if measured.get("analysis") != analysis:
        _fail("Measured analysis type differs from the requested analysis")
    actual_rows = measured.get("components")
    if not isinstance(actual_rows, list) or len(actual_rows) != len(components):
        _fail("Measurements do not cover every component exactly once")
    rows = []
    connected_voltages: dict[str, complex] = {}
    for component, row in zip(components, actual_rows):
        cid, ctype = component["id"], component["type"]
        kind = COMPONENTS[ctype]
        if not isinstance(row, dict) or row.get("id") != cid or row.get("type") != ctype:
            _fail("Measurement component IDs and ordering do not match the native spec")
        measured_nodes = row.get("nodes")
        if (row.get("pin_labels") != kind["pin_labels"] or not isinstance(measured_nodes, list)
            or not all(isinstance(n, str) for n in measured_nodes)
            or [_ground(n) for n in measured_nodes] != component["nodes"]):
            _fail("Measured component pin labels or nodes were changed")
        voltage = _vector(row.get("voltage"), cid + ".voltage", kind["pins"])
        imag = _vector(row.get("voltage_imag"), cid + ".voltage_imag", kind["pins"])
        current = _vector(row.get("current"), cid + ".current")
        current_imag = _vector(row.get("current_imag"), cid + ".current_imag", len(current))
        terminal = None
        if kind.get("pin_current_attributes") and analysis in kind.get("pin_current_analysis", []):
            terminal = _vector(row.get("pin_current_a"), cid + ".pin_current_a", kind["pins"])
            if row.get("pin_current_convention") != "Positive current enters the component, in catalog pin order; native model observation.":
                _fail("Terminal current sign/pin convention is missing or inconsistent")
            if abs(sum(terminal)) > 1e-9 * max(1, max(map(abs, terminal))):
                _fail("Recorded primitive terminal currents do not satisfy charge conservation")
        elif row.get("pin_current_a") is not None:
            _fail("This model/analysis does not provide validated terminal-current observations")
        for node, real, imaginary in zip(component["nodes"], voltage, imag):
            value = complex(real, imaginary)
            if node == "gnd" and abs(value) > 1e-8:
                _fail("Native ground pin voltage is not zero within tolerance")
            if node in connected_voltages and abs(connected_voltages[node] - value) > 1e-8 * max(1, abs(value)):
                _fail("Connected pins have inconsistent recorded node voltages")
            connected_voltages[node] = value
        derived = None
        if row.get("derived_current_0_to_1") is not None:
            derived = row["derived_current_0_to_1"]
            if ctype != "resistor" or not isinstance(derived, dict) or component["params"].get("r", 0) <= 0:
                _fail("Unsupported derived-current claim")
            effective = row.get("effective_params", {"r": component["params"]["r"]})
            if not isinstance(effective, dict) or set(effective) != {"r"}:
                _fail("Effective resistor parameters are malformed")
            effective_r = _finite(effective["r"], cid + ".effective_r")
            if effective_r <= 0:
                _fail("Effective resistance must be positive")
            expected = complex(voltage[0] - voltage[1], imag[0] - imag[1]) / effective_r
            recorded = complex(_finite(derived.get("real"), "derived current real"), _finite(derived.get("imag"), "derived current imaginary"))
            if abs(recorded - expected) > 1e-10 * max(1, abs(expected)):
                _fail("Derived resistor current does not match measured pin voltage divided by resistance")
            derived = {"real": recorded.real, "imag": recorded.imag, "unit": "A", "method": "由实测端电压及求解时有效电阻按欧姆定律计算，非独立电流采样"}
        rows.append({"id": cid, "type": ctype, "pin_labels": kind["pin_labels"], "nodes": component["nodes"],
            "params": component["params"], "voltage_v": voltage, "voltage_imag_v": imag,
            "branch_current_a": current, "branch_current_imag_a": current_imag,
            "branch_current_status": "measured_mna_branches" if current else "not_provided_by_model",
            "pin_current_a": terminal,
            "pin_current_convention": "正电流流入器件，顺序与原始引脚一致" if terminal is not None else None,
            "model_notes": kind.get("model_notes", ""), "effective_params": row.get("effective_params"),
            "derived_current_0_to_1": derived})
    transient = None
    if validate_transient and analysis in ("tr", "trop"):
        value = measured.get("transient")
        methods = {"native bounded transient solve; exact endpoint",
                   "one native transient solve, measurements sampled after completed steps; exact endpoint",
                   "one native controlled transient solve; exact endpoint",
                   "one native controlled transient solve, measurements sampled after completed steps; exact endpoint"}
        if not isinstance(value, dict) or value.get("method") not in methods:
            _fail("Transient evidence requires actual native time and step counters, not an echoed request")
        requested_stop = _finite(value.get("requested_stop_s"), "requested_stop_s")
        requested_step = _finite(value.get("requested_step_s"), "requested_step_s")
        actual_stop = _finite(value.get("actual_stop_s"), "actual_stop_s")
        count = value.get("completed_steps")
        if (requested_step <= 0 or requested_stop < requested_step or type(count) is not int or not 1 <= count <= 10000
            or not math.isclose(requested_stop, _finite(spec.get("tr_stop"), "spec.tr_stop"), rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(requested_step, _finite(spec.get("tr_step"), "spec.tr_step"), rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(actual_stop, requested_stop, rel_tol=1e-12, abs_tol=1e-15)):
            _fail("Native transient endpoint/counters do not match the requested simulation")
        ratio = requested_stop / requested_step
        if not max(1, math.ceil(ratio - 1e-9)) <= count <= math.ceil(ratio + 1e-9):
            _fail("Native transient completed-step count is inconsistent with its endpoint")
        transient = {k: value[k] for k in ("actual_stop_s", "completed_steps", "requested_stop_s", "requested_step_s", "method")}
        samples = value.get("samples")
        if samples is not None:
            every = value.get("sample_every")
            sampled_methods = {
                "one native transient solve, measurements sampled after completed steps; exact endpoint",
                "one native controlled transient solve, measurements sampled after completed steps; exact endpoint",
            }
            if (value["method"] not in sampled_methods
                or not isinstance(samples, list) or not 1 <= len(samples) <= 201
                or type(value.get("sample_count")) is not int or value["sample_count"] != len(samples)
                or type(every) is not int or every < 1 or spec.get("tr_sample_every") != every):
                _fail("Transient trace sampling metadata is inconsistent with the request")
            expected_steps = list(range(every, count + 1, every))
            if not expected_steps or expected_steps[-1] != count:
                expected_steps.append(count)
            if len(expected_steps) != len(samples):
                _fail("Transient trace omitted scheduled sample points")
            normalized = []
            previous_time = 0.0
            for point, expected_step in zip(samples, expected_steps):
                if not isinstance(point, dict) or type(point.get("completed_steps")) is not int or point["completed_steps"] != expected_step:
                    _fail("Transient trace sample-step ordering is inconsistent")
                actual_time = _finite(point.get("time_s"), "sample.time_s")
                if actual_time <= previous_time or not math.isclose(actual_time, min(expected_step * requested_step, actual_stop), rel_tol=1e-10, abs_tol=1e-13):
                    _fail("Transient trace times must be actual, strictly increasing completed-step times, not a synthetic zero sample")
                previous_time = actual_time
                sample_state = {**state, "measurements": {**measured, "components": point.get("components")}}
                sample_rows, _ = _measurements(spec, sample_state, components, validate_transient=False)
                normalized.append({"time_s": actual_time, "completed_steps": expected_step, "rows": sample_rows})
            if not math.isclose(previous_time, actual_stop, rel_tol=1e-12, abs_tol=1e-15) or _canonical(normalized[-1]["rows"]) != _canonical(rows):
                _fail("The final trace point does not match the measured endpoint state")
            transient.update({"sample_every": every, "sample_count": len(samples), "samples": normalized})
        elif spec.get("tr_sample_every") is not None:
            _fail("A requested transient trace is missing its actual sampled states")
    return rows, transient


def _saved_vector(raw: Any) -> list[float]:
    publishing._vector(raw, "saved position/rotation")
    x, z, y = map(float, raw.split(","))
    return [x, y, z]


def _export(cache_dir: str, sav_path: str, components: list[dict]) -> tuple[dict, list[str]]:
    saved, raw, info = publishing._source(cache_dir, sav_path)
    status = publishing._json(saved["Experiment"]["StatusSave"])
    by_id = {e["Identifier"]: e for e in status["Elements"]}
    ground_ids = [eid for eid, e in by_id.items() if e["ModelID"] == "Ground Component"]
    expected_ids = {c["id"] for c in components}
    if set(by_id) - set(ground_ids) != expected_ids:
        _fail("PLSAV added, removed or renamed a physical component")
    pins = {}
    native_nodes = {}
    for component in components:
        cid, ctype = component["id"], component["type"]
        definition = COMPONENTS[ctype]
        element = by_id[cid]
        if not definition["model_id"] or element["ModelID"] != definition["model_id"]:
            _fail("No supported original-app component mapping for " + ctype)
        native = element.get("Aurex")
        if (not isinstance(native, dict) or native.get("type") != ctype
            or _canonical(native.get("params")) != _canonical(component["params"])
            or native.get("pin_count") != definition["pins"]):
            _fail("Exported native parameter sidecar does not preserve the simulated primitive")
        props = element["Properties"]
        for key, prop in definition["pl_props"].items():
            if prop not in props or _finite(props[prop], cid + "." + prop) != float(component["params"][key]):
                _fail("PLSAV mapped electrical parameters differ from the analyzed spec")
        for prop, expected in definition.get("constant_pl_props", {}).items():
            if props.get(prop) != expected:
                _fail("PLSAV constant model properties differ from the native primitive contract")
        if ctype in ("npn", "pnp") and props.get("PNP") != (1 if ctype == "pnp" else 0):
            _fail("PLSAV transistor polarity differs from the analyzed device")
        if ctype in ("capacitor", "inductor", "vdc") and props.get("内阻") != 0:
            _fail("PLSAV introduced internal resistance absent from the native primitive model")
        for name, saved_name in (("position", "Position"), ("rotation", "Rotation")):
            expected = component.get(name)
            if not isinstance(expected, list) or len(expected) != 3:
                _fail("A reviewed exported primitive requires explicit original coordinates and rotation")
            actual = _saved_vector(element[saved_name])
            if any(not math.isclose(_finite(a, name), b, rel_tol=0, abs_tol=5.00001e-7) for a, b in zip(expected, actual)):
                _fail("PLSAV rearranged or rotated a component beyond its six-decimal serialization precision")
        # Only allowed primitive mappings have identical native/PL pin order.
        for index, node in enumerate(component["nodes"]):
            endpoint = (cid, index)
            pins[endpoint] = endpoint
            native_nodes[endpoint] = node
    for cid in ground_ids:
        pins[(cid, 0)] = (cid, 0)
        native_nodes[(cid, 0)] = "gnd"
    def find(endpoint):
        while pins[endpoint] != endpoint:
            pins[endpoint] = pins[pins[endpoint]]
            endpoint = pins[endpoint]
        return endpoint
    for wire in status["Wires"]:
        source, target = (wire["Source"], wire["SourcePin"]), (wire["Target"], wire["TargetPin"])
        if source not in pins or target not in pins:
            _fail("PLSAV uses an unmapped primitive pin")
        pins[find(source)] = find(target)
    native_to_saved, saved_to_native = {}, {}
    for endpoint, node in native_nodes.items():
        actual = find(endpoint)
        if node in native_to_saved and native_to_saved[node] != actual:
            _fail("PLSAV disconnected pins which share the same native node")
        if actual in saved_to_native and saved_to_native[actual] != node:
            _fail("PLSAV shorted distinct native nodes")
        native_to_saved[node] = actual
        saved_to_native[actual] = node
    if any("gnd" in c["nodes"] for c in components) and not ground_ids:
        _fail("PLSAV omitted the simulation voltage-reference ground")
    return {**info, "verified": True, "mapping_scope": "primitive IDs, pin connectivity, mapped properties, native parameter sidecar and saved layout",
            "original_app_numerical_equivalence": False}, ground_ids


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def canonical_chinese_table(report: dict[str, Any], *, samples: list[dict[str, Any]] | None = None) -> str:
    lines = ["# 模拟电路实际测量记录", "", "此表逐项记录求解器状态，不代表目标功能已经通过验证。空电流数组表示模型未提供该分支采样，不能当作零电流。", ""]
    transient = report.get("transient")
    if transient:
        description = f"以下覆盖全部{len(samples)}个真实采样时刻，不补造t=0数据，不推断采样点之间的状态。" if samples else "以下仅为末端状态，不据此推断未采样波形。"
        lines += [f"实际终止时刻：{transient['actual_stop_s']:.12g} 秒；完成步数：{transient['completed_steps']}。" + description, ""]
    lines += ["| 实际时刻（秒） | 元件标识 | 基础模型 | 原始引脚顺序 / 节点 | 对地电压（V） | 电流（A） | 说明 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    points = samples or [{"time_s": transient["actual_stop_s"] if transient else None, "rows": report["rows"]}]
    for point in points:
        stamp = f"{point['time_s']:.12g}" if point["time_s"] is not None else "静态"
        for row in point["rows"]:
            labels = "; ".join(f"{p}: {n}" for p, n in zip(row["pin_labels"], row["nodes"]))
            volts = "; ".join(f"{r:.9g}" + (f"{i:+.9g}j" if i else "") for r, i in zip(row["voltage_v"], row["voltage_imag_v"]))
            terminal, derived = row.get("pin_current_a"), row.get("derived_current_0_to_1")
            if terminal is not None:
                currents = "; ".join(f"{p}: {v:.9g}" for p, v in zip(row["pin_labels"], terminal))
                note = "原生模型在已求解状态的端电流；正电流流入器件"
            elif derived:
                currents = f"{derived['real']:.9g}" + (f"{derived['imag']:+.9g}j" if derived["imag"] else "")
                note = "I(0→1)，由实测端电压按欧姆定律计算；非独立电流采样"
            elif row["branch_current_a"]:
                currents = "; ".join(f"{r:.9g}" + (f"{i:+.9g}j" if i else "") for r, i in zip(row["branch_current_a"], row["branch_current_imag_a"]))
                note = "原生MNA分支电流，不等同于每引脚电流"
            else:
                currents = "未测（模型未提供电流）"
                note = "不补造端电流或开关状态，空值不等于0"
            lines.append("| " + " | ".join(_escape(x) for x in (stamp, row["id"], row["type"], labels, volts, currents, note)) + " |")
    lines += ["", "## 电压参考（不是物理器件）", "", "gnd 定义为 0 V；导出的接地标记单独计为电压参考，不混入器件测量数量。",
              "", "## 适用边界", "", "- 数值收敛及源文件对应不等于用户要求的功能已实现。",
              "- PLSAV核对只证明列明的元件、连线、参数及空间位置对应，不证明原应用与Phy-Engine逐点数值相等。", ""]
    return "\n".join(lines)


def _write(path: Path, value: bytes):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".evidence-", delete=False) as output:
        temp = Path(output.name)
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, path)


def _build_analysis(cache_dir: str, *, spec_path: str, state_path: str, sav_path: str | None = None):
    spec, spec_info = _read(cache_dir, spec_path)
    state, state_info = _read(cache_dir, state_path)
    if not isinstance(state, dict):
        _fail("Native snapshot must be a JSON object")
    components = _spec_components(spec)
    rows, transient = _measurements(spec, state, components)
    samples = transient.pop("samples", None) if transient else None
    unsupported = sorted({c["type"] for c in components} - PRIMITIVES)
    export_info, ground_ids, export_error = None, [], None
    if sav_path:
        try:
            export_info, ground_ids = _export(cache_dir, sav_path, components)
        except (ValueError, OSError) as error:
            export_error = str(error)
    else:
        export_error = "No original PLSAV export was produced; native simulation evidence alone cannot authorize publication"
    verified = not unsupported and export_info is not None
    report = {"schema": SCHEMA, "verified": verified, "simulation_verified": True,
        "export_verified": export_info is not None, "primitive_only": not unsupported, "unsupported_types": unsupported,
        "functional_verification": False, "scope": "Generic native numerical and source/export correspondence only; no target circuit function is asserted",
        "spec": spec_info, "state": state_info, "sav": export_info, "export_error": export_error,
        "analysis": spec.get("analysis", "dc"), "transient": transient, "rows": rows,
        "component_ids": [c["id"] for c in components], "component_count": len(components),
        "reference_ground": {"voltage_v": 0, "definition_not_measurement": True, "saved_marker_ids": ground_ids},
        "native_origin": state["origin"], "original_app_numerical_equivalence": False}
    if samples:
        ranges = []
        for index, row in enumerate(rows):
            points = [p["rows"][index] for p in samples]
            ranges.append({"id": row["id"], "pin_labels": row["pin_labels"],
                "voltage_min_v": [min(p["voltage_v"][pin] for p in points) for pin in range(len(row["pin_labels"]))],
                "voltage_max_v": [max(p["voltage_v"][pin] for p in points) for pin in range(len(row["pin_labels"]))]})
        # Collapse duplicate component pins onto named electrical nodes.  This
        # gives the agent a small, direct steady-state waveform summary without
        # copying the complete trace back into every model turn.
        node_series: dict[str, list[float]] = {}
        for point in samples:
            values: dict[str, float] = {}
            for row in point["rows"]:
                for node, voltage in zip(row["nodes"], row["voltage_v"]):
                    values.setdefault(node, voltage)
            for node, voltage in values.items():
                node_series.setdefault(node, []).append(voltage)
        node_ranges = {
            node: {
                "min_v": min(values), "max_v": max(values),
                "peak_to_peak_v": max(values) - min(values),
                "first_v": values[0], "last_v": values[-1],
                "mean_v": sum(values) / len(values),
            }
            for node, values in node_series.items()
        }
        report["trace"] = {"verified": True, "sample_count": len(samples), "all_components_each_sample": True,
            "times_s": [p["time_s"] for p in samples], "completed_steps": [p["completed_steps"] for p in samples],
            "component_ranges": ranges, "node_ranges_v": node_ranges, "full_raw_trace": state_info,
            "scope": "Actual sampled states only; min/max do not assert a target function or unsampled behavior"}
    else:
        report["trace"] = {"verified": False, "sample_count": 0, "reason": "No actual transient sequence was requested or recorded"}
    return report, samples


def record_analysis(cache_dir: str, *, spec_path: str, state_path: str, sav_path: str | None = None) -> str:
    """Record complete numerical/export evidence without asserting target function."""
    report, samples = _build_analysis(cache_dir, spec_path=spec_path, state_path=state_path, sav_path=sav_path)
    rows = report["rows"]
    # Each call gets a fresh report directory; no earlier evidence is overwritten.
    directory = Path(tempfile.mkdtemp(prefix="evidence-", dir=Path(report["spec"]["path"]).parent))
    table_path = directory / "analysis-table.zh.md"
    table = canonical_chinese_table(report).encode("utf-8")
    _write(table_path, table)
    report["table"] = {"path": str(table_path), "sha256": hashlib.sha256(table).hexdigest(), "bytes": len(table),
                       "language": "Chinese", "component_rows": len(rows),
                       "sample_count": 1, "scope": "Actual final state of every component, not the full waveform",
                       "generated_from_measurements": True}
    if samples:
        trace_path = directory / "analysis-trace.zh.md"
        trace_table = canonical_chinese_table(report, samples=samples).encode("utf-8")
        _write(trace_path, trace_table)
        report["trace_table"] = {"path": str(trace_path), "sha256": hashlib.sha256(trace_table).hexdigest(),
            "bytes": len(trace_table), "language": "Chinese", "component_rows": len(rows) * len(samples),
            "sample_count": len(samples), "scope": "Full actual sampled states; read in pages, not required public introduction",
            "generated_from_measurements": True}
    path = directory / "analysis-report.json"
    _write(path, (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return str(path)


def validate_report(cache_dir: str, report_path: str) -> dict[str, Any]:
    """Read-only revalidation of every source byte, measurement and canonical table.

    The report's success flags are not trusted. This replays validation, not the
    numerical solver, and never upgrades numerical evidence to functional proof.
    """
    report, info = _read(cache_dir, report_path)
    path = Path(info["path"])
    if (not isinstance(report, dict) or report.get("schema") != SCHEMA
        or path.name != "analysis-report.json" or not path.parent.name.startswith("evidence-")
        or not path.is_relative_to(Path(cache_dir).resolve() / "circuits")):
        _fail("Analysis evidence must be an original server circuit-analysis report")
    for key in ("spec", "state", "sav"):
        if not isinstance(report.get(key), dict) or not isinstance(report[key].get("path"), str):
            _fail("Publishable analysis evidence requires its complete original spec, state and PLSAV")
    rebuilt, samples = _build_analysis(cache_dir, spec_path=report["spec"]["path"],
                                      state_path=report["state"]["path"], sav_path=report["sav"]["path"])
    if _canonical({k: v for k, v in report.items() if k not in ("table", "trace_table")}) != _canonical(rebuilt):
        _fail("Analysis report does not match the current source hashes or actual native measurements")
    if not rebuilt["verified"]:
        _fail("Analysis evidence has not verified primitive-only simulation and original PLSAV correspondence")
    for key, name, points in (("table", "analysis-table.zh.md", None),
                               ("trace_table", "analysis-trace.zh.md", samples)):
        if key == "trace_table" and not samples:
            if key in report:
                _fail("A trace table cannot exist without native samples")
            continue
        metadata = report.get(key)
        if not isinstance(metadata, dict) or metadata.get("path") != str(path.parent / name):
            _fail("Canonical measurement table path is missing or changed")
        table_file, table = publishing._read_artifact(cache_dir, metadata["path"], suffixes=(".md",), limit=32 * 1024 * 1024)
        expected = canonical_chinese_table(rebuilt, samples=points).encode("utf-8")
        if (table != expected or metadata.get("sha256") != hashlib.sha256(table).hexdigest()
            or metadata.get("bytes") != len(table) or metadata.get("generated_from_measurements") is not True
            or metadata.get("language") != "Chinese"
            or metadata.get("sample_count") != (len(points) if points else 1)
            or metadata.get("component_rows") != len(rebuilt["rows"]) * (len(points) if points else 1)):
            _fail("Canonical table must cover every actual component row without modified or invented measurements")
    spec, _ = _read(cache_dir, report["spec"]["path"])
    return {"report": report, "report_artifact": info, "spec": spec, "samples": samples,
            "final_table": canonical_chinese_table(report), "functional_verification": False}
