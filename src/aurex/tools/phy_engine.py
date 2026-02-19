from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from ..phy_engine.build import PhyEngineBuildError, ensure_built
from ..phy_engine.ffi import PhyEngineError, load_library
from .registry import ToolError, ToolRuntime


def _resolve(runtime: ToolRuntime, path: str) -> str:
    return runtime.config.resolve_path(path, config_path=runtime.config_path)


def _ensure_artifacts(runtime: ToolRuntime, *, force_build: bool = False) -> tuple[str, str]:
    cfg = runtime.config.phy_engine
    v2p = (cfg.verilog2plsav_path or "").strip()
    lib = (cfg.phyengine_lib_path or "").strip()
    if v2p and lib:
        return _resolve(runtime, v2p), _resolve(runtime, lib)

    if not (cfg.auto_build or force_build):
        missing = []
        if not v2p:
            missing.append("verilog2plsav_path")
        if not lib:
            missing.append("phyengine_lib_path")
        raise ToolError(f"Phy-Engine not configured ({', '.join(missing)} missing) and auto_build=false")

    src_dir = _resolve(runtime, cfg.cmake_source_dir)
    build_dir = _resolve(runtime, cfg.cmake_build_dir)
    try:
        art = ensure_built(
            source_dir=src_dir,
            build_dir=build_dir,
            build_type=cfg.cmake_build_type,
            timeout_sec=int(cfg.build_timeout_sec),
        )
    except PhyEngineBuildError as e:
        raise ToolError(str(e)) from e

    return art.verilog2plsav_path, art.phyengine_lib_path


def phy_engine_build(runtime: ToolRuntime, _args: dict[str, Any]) -> dict[str, str]:
    cfg = runtime.config.phy_engine
    src_dir = _resolve(runtime, cfg.cmake_source_dir)
    build_dir = _resolve(runtime, cfg.cmake_build_dir)
    try:
        art = ensure_built(
            source_dir=src_dir,
            build_dir=build_dir,
            build_type=cfg.cmake_build_type,
            timeout_sec=int(cfg.build_timeout_sec),
        )
    except PhyEngineBuildError as e:
        raise ToolError(str(e)) from e
    return {"build_dir": art.build_dir, "verilog2plsav_path": art.verilog2plsav_path, "phyengine_lib_path": art.phyengine_lib_path}


def verilog_to_sav(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, str]:
    verilog = str(args.get("verilog") or "")
    if not verilog.strip():
        raise ToolError("verilog_to_sav: verilog is empty")

    # Security: always write .sav into runtime.cache_dir; never allow arbitrary output paths.
    os.makedirs(runtime.cache_dir, exist_ok=True)
    fd, out_sav_tmp = tempfile.mkstemp(prefix="aurex_", suffix=".sav", dir=runtime.cache_dir)
    os.close(fd)

    force_build = bool(args.get("force_build") or False)
    v2p, _lib = _ensure_artifacts(runtime, force_build=force_build)

    os.makedirs(os.path.dirname(out_sav_tmp) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime.cache_dir) as td:
        in_v = os.path.join(td, "in.v")
        with open(in_v, "w", encoding="utf-8") as f:
            f.write(verilog)
            if not verilog.endswith("\n"):
                f.write("\n")

        cmd = [v2p, out_sav_tmp, in_v] + list(runtime.config.phy_engine.verilog2plsav_args)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=int(runtime.config.phy_engine.run_timeout_sec))
        except subprocess.TimeoutExpired as e:
            raise ToolError(f"verilog2plsav timed out after {runtime.config.phy_engine.run_timeout_sec}s") from e
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or e.stdout or "").strip()
            raise ToolError(f"verilog2plsav failed: {msg}") from e

    # Stage the sav under cache_dir with a task-scoped, deterministic name.
    task_id = str(getattr(runtime, "task_id", "") or "").strip() or "task"
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:120].strip("._-") or "task"
    staged_dir = os.path.join(runtime.cache_dir, "staged_sav")
    os.makedirs(staged_dir, exist_ok=True)
    staged_sav = os.path.join(staged_dir, f"{safe_task}.sav")
    try:
        shutil.move(out_sav_tmp, staged_sav)
    except Exception:
        try:
            shutil.copy2(out_sav_tmp, staged_sav)
            os.remove(out_sav_tmp)
        except Exception as e:
            raise ToolError(f"verilog_to_sav: failed to stage .sav into cache: {type(e).__name__}: {e}") from e
    return {"sav_path": staged_sav}


_ANALYZE_TYPES = {"op": 0, "dc": 1, "ac": 2, "acop": 3, "tr": 4, "trop": 5}

_ELEMENTS: dict[str, dict[str, Any]] = {
    "resistor": {"code": 1, "pins": 2, "props": ["r"]},
    "capacitor": {"code": 2, "pins": 2, "props": ["c"]},
    "inductor": {"code": 3, "pins": 2, "props": ["l"]},
    "vdc": {"code": 4, "pins": 2, "props": ["v"]},
    "vac": {"code": 5, "pins": 2, "props": ["vp", "freq_hz", "phase_deg"]},
    "idc": {"code": 6, "pins": 2, "props": ["i"]},
    "iac": {"code": 7, "pins": 2, "props": ["ip", "freq_hz", "phase_deg"]},
    "digital_input": {"code": 200, "pins": 1, "props": ["state"]},
    "digital_output": {"code": 201, "pins": 1, "props": []},
    "digital_not": {"code": 205, "pins": 2, "props": []},
    "digital_yes": {"code": 203, "pins": 2, "props": []},
    "digital_and": {"code": 204, "pins": 3, "props": []},
    "digital_or": {"code": 202, "pins": 3, "props": []},
    "digital_xor": {"code": 206, "pins": 3, "props": []},
    "digital_nand": {"code": 208, "pins": 3, "props": []},
    "digital_nor": {"code": 209, "pins": 3, "props": []},
}


def pe_simulate(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    spec = args.get("spec")
    if not isinstance(spec, dict):
        raise ToolError("pe_simulate: spec must be an object")
    comps = spec.get("components")
    if not isinstance(comps, list) or not comps:
        raise ToolError("pe_simulate: spec.components must be a non-empty array")

    analysis = str(spec.get("analysis") or "dc").strip().lower()
    at = _ANALYZE_TYPES.get(analysis)
    if at is None:
        raise ToolError("pe_simulate: analysis must be one of op/dc/ac/acop/tr/trop")

    v2p, lib_path = _ensure_artifacts(runtime)
    _ = v2p  # verilog2plsav not used here; keep artifacts check unified

    lib = load_library(lib_path)

    # element 0 is ground placeholder (code=0)
    element_codes: list[int] = [0]
    properties: list[float] = []
    comp_meta: list[dict[str, Any]] = []

    for c in comps:
        if not isinstance(c, dict):
            raise ToolError("pe_simulate: each component must be an object")
        cid = str(c.get("id") or "").strip()
        ctype = str(c.get("type") or "").strip().lower()
        nodes = c.get("nodes")
        if not cid:
            raise ToolError("pe_simulate: component.id is required")
        if ctype not in _ELEMENTS:
            raise ToolError(f"pe_simulate: unsupported component type: {ctype}")
        if not isinstance(nodes, list):
            raise ToolError("pe_simulate: component.nodes must be an array")
        pins = int(_ELEMENTS[ctype]["pins"])
        if len(nodes) != pins:
            raise ToolError(f"pe_simulate: {cid} expects {pins} nodes, got {len(nodes)}")
        params = c.get("params") if isinstance(c.get("params"), dict) else {}
        element_codes.append(int(_ELEMENTS[ctype]["code"]))

        prop_keys: list[str] = list(_ELEMENTS[ctype]["props"])
        for k in prop_keys:
            if k not in params:
                raise ToolError(f"pe_simulate: {cid} missing param {k!r}")
            try:
                properties.append(float(params.get(k)))
            except Exception as e:
                raise ToolError(f"pe_simulate: {cid} param {k!r} must be a number") from e

        comp_meta.append({"id": cid, "type": ctype, "nodes": [str(n) for n in nodes], "pins": pins})

    # Build wires from named nodes
    node_map: dict[str, list[tuple[int, int]]] = {}
    for idx, meta in enumerate(comp_meta, start=1):
        for pin_i, node in enumerate(meta["nodes"]):
            n = (node or "").strip()
            if not n or n.casefold() in ("gnd", "ground", "0"):
                nkey = "gnd"
            else:
                nkey = n
            node_map.setdefault(nkey, []).append((idx, int(pin_i)))

    wires: list[int] = []
    for nkey, pins in node_map.items():
        if nkey == "gnd":
            for ele, pin in pins:
                wires.extend([0, 0, int(ele), int(pin)])
            continue
        if len(pins) <= 1:
            continue
        root_ele, root_pin = pins[0]
        for ele, pin in pins[1:]:
            wires.extend([int(root_ele), int(root_pin), int(ele), int(pin)])

    circuit = lib.create_circuit(elements=element_codes, wires=wires, properties=properties)
    try:
        circuit.set_analyze_type(at)
        if analysis in ("tr", "trop"):
            t_step = float(spec.get("tr_step") or 1e-9)
            t_stop = float(spec.get("tr_stop") or t_step)
            circuit.set_tr(t_step, t_stop)
        if analysis in ("ac", "acop"):
            omega = float(spec.get("ac_omega") or 1.0)
            circuit.set_ac_omega(omega)

        circuit.analyze()
        ticks = int(spec.get("digital_clock_ticks") or 0)
        if ticks < 0:
            ticks = 0
        if ticks > 1_000_000:
            ticks = 1_000_000
        for _ in range(ticks):
            circuit.digital_clk()

        max_pins = max(m["pins"] for m in comp_meta) if comp_meta else 1
        sample = circuit.sample_u8(max_pins=max_pins)
    except PhyEngineError as e:
        raise ToolError(str(e)) from e
    finally:
        circuit.close()

    # Slice per component based on ord arrays (component order == our comp_meta order)
    out: dict[str, Any] = {"analysis": analysis, "components": []}
    for i, meta in enumerate(comp_meta):
        v0, v1 = sample["voltage_ord"][i], sample["voltage_ord"][i + 1]
        c0, c1 = sample["current_ord"][i], sample["current_ord"][i + 1]
        d0, d1 = sample["digital_ord"][i], sample["digital_ord"][i + 1]
        out["components"].append(
            {
                "id": meta["id"],
                "type": meta["type"],
                "voltage": sample["voltage"][v0:v1],
                "current": sample["current"][c0:c1],
                "digital": sample["digital"][d0:d1],
            }
        )
    return out


PHY_ENGINE_BUILD_TOOL = {
    "name": "phy_engine_build",
    "description": "Build Phy-Engine artifacts (verilog2plsav + phyengine shared lib) via CMake.",
    "parameters": {"type": "object", "properties": {}},
}

VERILOG_TO_SAV_TOOL = {
    "name": "verilog_to_sav",
    "description": "Compile Verilog into a PhysicsLab .sav using Phy-Engine verilog2plsav.",
    "parameters": {
        "type": "object",
        "properties": {
            "verilog": {"type": "string", "description": "Verilog source code (Verilog-2001)."},
            "out_sav_path": {"type": ["string", "null"], "description": "Optional output path for .sav."},
            "force_build": {
                "type": "boolean",
                "default": False,
                "description": "If true, build Phy-Engine artifacts via CMake when paths are not configured.",
            },
        },
        "required": ["verilog"],
    },
}

PE_SIMULATE_TOOL = {
    "name": "pe_simulate",
    "description": "Run a small local simulation via Phy-Engine C ABI (limited component set).",
    "parameters": {
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string", "enum": ["op", "dc", "ac", "acop", "tr", "trop"], "default": "dc"},
                    "tr_step": {"type": "number"},
                    "tr_stop": {"type": "number"},
                    "ac_omega": {"type": "number"},
                    "digital_clock_ticks": {"type": "integer", "minimum": 0, "default": 0},
                    "components": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "nodes": {"type": "array", "items": {"type": "string"}},
                                "params": {"type": "object"},
                            },
                            "required": ["id", "type", "nodes"],
                        },
                    },
                },
                "required": ["components"],
            }
        },
        "required": ["spec"],
    },
}
