from __future__ import annotations

import os
import json
import copy
import difflib
import time
import hashlib
from pathlib import Path
import math
import re
import shutil
import subprocess
import tempfile
import sys
from typing import Any

from plar.official_publish_api import hdl_source_carrier_status

from ..phy_engine.build import PhyEngineBuildError, ensure_built
from ..phy_engine.ffi import PhyEngineError, load_library
from ..phy_engine.catalog import COMPONENTS
from ..phy_engine.limits import DEFAULT_DIGITAL_COMPONENT_LIMIT, validate_spec_size
from ..verilog_lowering import MemoryLoweringError, _mask_noncode, lower_unpacked_register_arrays
from .registry import ToolError, ToolRuntime

MAX_DIRECT_PLSAV_ELEMENTS = 5000

_VERILOG_HIGH_IMPEDANCE_LITERAL = re.compile(
    r"(?i)(?<![A-Za-z0-9_$])(?:[0-9][0-9_]*)?'\s*(?:s\s*)?"
    r"(?:z|[bodh]\s*[0-9a-f_xz?]*[z?][0-9a-f_xz?]*)")


def _celestial_hdl_source_template() -> dict[str, Any]:
    """Fixed official-compatible Type-3 carrier for oversized HDL publications.

    The HDL remains in the hash-bound publication evidence and is appended to
    the public introduction.  This shell deliberately contains no controllable
    circuit and cannot enter Phy-Engine's electrical import path.
    """
    now = int(time.time() * 1000)
    status = hdl_source_carrier_status()
    camera = {"Mode": 2, "Distance": 2.75, "VisionCenter": "0,1.08,0",
              "TargetRotation": "90,0,0"}
    summary = {"Type": 3, "ParentID": None, "ParentName": None,
        "ParentCategory": None, "ContentID": None, "Editor": None,
        "Coauthors": [], "Description": None, "LocalizedDescription": None,
        "Tags": ["Type-3", "高中", "教学实验"], "ModelID": None, "ModelName": None,
        "ModelTags": [], "Version": 0, "Language": "Chinese",
        "Visits": 0, "Stars": 0, "Supports": 0, "Remixes": 0,
        "Comments": 0, "Price": 0, "Popularity": 0,
        "CreationDate": now, "UpdateDate": 0, "SortingDate": 0,
        "ID": None, "Category": None, "Subject": "",
        "LocalizedSubject": None, "Image": 0, "ImageRegion": 0,
        "User": {"ID": None, "Nickname": None, "Signature": None,
                 "Avatar": 0, "AvatarRegion": 0, "Decoration": 0,
                 "Verification": None},
        "Visibility": 0, "Settings": {}, "Multilingual": False}
    return {"Type": 3,
        "Experiment": {"ID": None, "Type": 3, "Components": 3,
            "Subject": None, "StatusSave": json.dumps(status, ensure_ascii=False, separators=(",", ":")),
            "CameraSave": json.dumps(camera, ensure_ascii=False, separators=(",", ":")),
            "Version": 2503, "CreationDate": now, "Paused": False,
            "Summary": None, "Plots": None},
        "ID": None, "Summary": summary, "InternalName": "Aurex HDL 源码载体",
        "CreationDate": 0, "Speed": 1.0,
        "SpeedMinimum": 0.1, "SpeedMaximum": 10.0, "SpeedReal": 0.0,
        "Paused": False, "Version": 0, "CameraSnapshot": None,
        "Plots": [], "Widgets": [], "WidgetGroups": [], "Bookmarks": {},
        "Interfaces": {"Play-Expanded": False, "Chart-Expanded": False}}


def _resolve(runtime: ToolRuntime, path: str) -> str:
    return runtime.config.resolve_path(path, config_path=runtime.config_path)


def _ensure_artifacts(runtime: ToolRuntime, *, force_build: bool = False) -> tuple[str, str]:
    cfg = runtime.config.phy_engine
    v2p = (cfg.verilog2plsav_path or "").strip()
    lib = (cfg.phyengine_lib_path or "").strip()
    if v2p and lib:
        return _resolve(runtime, v2p), _resolve(runtime, lib)

    if not force_build:
        from ..phy_engine.build import _find_phyengine_lib, _find_verilog2plsav
        built = _resolve(runtime, cfg.cmake_build_dir)
        cached_v2p, cached_lib = _find_verilog2plsav(built), _find_phyengine_lib(built)
        renderer = Path(built) / ("circuit_view.exe" if os.name == "nt" else "circuit_view")
        if cached_v2p and cached_lib and renderer.is_file():
            return cached_v2p, cached_lib

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
    return {"build_dir": art.build_dir, "verilog2plsav_path": art.verilog2plsav_path,
            "phyengine_lib_path": art.phyengine_lib_path, "circuit_view_path": art.circuit_view_path}


def _verified_hdl(runtime: ToolRuntime, path: str) -> tuple[dict, str]:
    root = Path(runtime.cache_dir).resolve()
    report_path = Path(path).resolve()
    if not report_path.is_relative_to(root / "hdl") or not report_path.is_file() or report_path.stat().st_size > 1024**2:
        raise ToolError("hdl_report_path must be a local HDL verification artifact")
    report = json.loads(report_path.read_text())
    if report.get("verified") is not True:
        raise ToolError("HDL must pass verification before report-based export")
    report["_trusted_report_path"] = str(report_path)
    design_names = None
    if 'workspace_id' in report:
        from .hdl_workspace import workspace_snapshot
        info, snapshot = workspace_snapshot(runtime, report['workspace_id'], report.get('workspace_revision'))
        if info['source_sha256'] != report.get('source_sha256'):
            raise ToolError('Workspace source differs from the verification report')
        design_names = [name for name, item in sorted(snapshot.items()) if item['role'] == 'source']
        if design_names != report.get('design_source_files') or not report.get('design_top'):
            raise ToolError('Custom workspace export requires design_top and exact source-role files in its verification report')
    for stage in ("compile", "simulation"):
        status = report.get(stage) or {}
        if status.get("exit_code") != 0 or status.get("failure"):
            raise ToolError("HDL verification did not complete successfully")
    hashes, sources = {}, []
    for name, expected in (report.get("source_files_sha256") or {}).items():
        source = (report_path.parent / name).resolve()
        if source.parent != report_path.parent or source.suffix not in (".v", ".sv") or not source.is_file() or source.stat().st_size > 512000:
            raise ToolError("Invalid HDL source in verification report")
        raw = source.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if hashes[name] != expected:
            raise ToolError("HDL source changed since verification; rerun hdl_simulate")
        if design_names is None or name in design_names:
            sources.append(raw.decode("utf-8"))
    bundle = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not sources or bundle != report.get("source_sha256"):
        raise ToolError("HDL verification bundle hash does not match current sources")
    return report, "\n".join(sources)


def _lower_and_reverify_hdl(runtime: ToolRuntime, report: dict) -> tuple[str, dict | None]:
    design_names = set(report.get("design_source_files") or report.get("source_files_sha256") or {})
    files = []
    exported = []
    arrays = []
    for name in report.get("source_files_sha256") or {}:
        path = Path(report["_trusted_report_path"]).parent / name
        source = path.read_text(encoding="utf-8")
        if name in design_names:
            try:
                source, lowered = lower_unpacked_register_arrays(source)
            except MemoryLoweringError as error:
                raise ToolError("Cannot lower fixed unpacked register array for PLSAV export: " + str(error)) from error
            arrays.extend({"source": name, **item} for item in lowered)
            exported.append(source)
        files.append({"name": name, "content": source})
    if not arrays:
        return "\n".join(exported), None

    from .hdl import hdl_simulate
    args: dict[str, Any] = {"profile": report["profile"], "files": files}
    if report["profile"] == "custom":
        args["top"] = report["top"]
    else:
        args["top"] = report.get("design_top", report["top"])
    checked = hdl_simulate(runtime, args)
    if checked.get("verified") is not True:
        failure = (checked.get("simulation") or checked.get("compile") or {}).get("log", "")[-4000:]
        raise ToolError("Unpacked-array export lowering failed re-verification; no PLSAV was generated: " + failure)
    return "\n".join(exported), {
        "schema": "aurex.hdl-export-lowering.v1",
        "kind": "fixed_unpacked_register_array_to_explicit_registers",
        "arrays": arrays,
        "verification_id": checked["verification_id"],
        "verification_report_path": checked["report_path"],
        "source_sha256": checked["source_sha256"],
        "profile": checked["profile"],
        "verified": True,
        "scope": checked["scope"],
    }


def verilog_to_sav(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    verified = None
    verilog = str(args.get("verilog") or "")
    modules = args.get("modules") or []
    if args.get("hdl_report_path"):
        if verilog or modules:
            raise ToolError("Pass either hdl_report_path or inline HDL, not both")
        verified, verilog = _verified_hdl(runtime, args["hdl_report_path"])
    export_lowering = None
    if verified:
        verilog, export_lowering = _lower_and_reverify_hdl(runtime, verified)
    if modules:
        if not isinstance(modules, list) or len(modules) > 32 or not all(isinstance(v, str) for v in modules):
            raise ToolError("modules must be an array of up to 32 Verilog source strings")
        verilog = "\n".join([verilog, *modules])
    # The current PhysicsLab catalog has no tri-state device.  At -O2 the HDL
    # synthesizer may legally simplify a conditional Z into ordinary Boolean
    # gates before pe_to_pl sees a TRI model, which used to bypass that layer's
    # strict-export warning and silently turn high impedance into 0.  Reject
    # source Z literals up front, excluding comments and strings.
    if _VERILOG_HIGH_IMPEDANCE_LITERAL.search(_mask_noncode(verilog)):
        raise ToolError(
            "strict export refuses Verilog high-impedance/Z semantics because PhysicsLab cannot preserve tri-state behavior")
    if not verified:
        try:
            verilog, arrays = lower_unpacked_register_arrays(verilog)
        except MemoryLoweringError as error:
            raise ToolError("Cannot lower fixed unpacked register array for PLSAV export: " + str(error)) from error
        if arrays:
            export_lowering = {"schema": "aurex.hdl-export-lowering.v1", "kind": "unverified_inline_lowering", "arrays": arrays,
                               "verified": False}
    if not verilog.strip():
        raise ToolError("verilog_to_sav: verilog is empty")
    if len(verilog.encode()) > 1_000_000:
        raise ToolError("Verilog source exceeds 1 MB")
    # Includes from arbitrary server paths and readmem tasks are not agent inputs.
    # Submit all related modules via the modules parameter instead.
    if re.search(r"`include\b|\$(?:readmem[hb]|fopen|system)\b", verilog):
        raise ToolError("Filesystem Verilog directives are disabled; pass all module source strings in modules")
    top = str(args.get("top") or "").strip()
    if verified:
        export_top = verified.get('design_top', verified.get('top'))
        if top and top != export_top:
            raise ToolError("Export top must match the verified top")
        top = export_top
    if top and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", top):
        raise ToolError("Invalid Verilog top module name")

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
        cmd += ["--max-total-nodes", "20000", "--max-total-models", "20000", "--max-total-logic-gates", "20000", "--strict-export", "--assume-binary-inputs"]
        if top:
            cmd += ["--top", top]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=int(runtime.config.phy_engine.run_timeout_sec))
        except subprocess.TimeoutExpired as e:
            raise ToolError(f"verilog2plsav timed out after {runtime.config.phy_engine.run_timeout_sec}s") from e
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or e.stdout or "").strip()[-12000:]
            raise ToolError(f"verilog2plsav failed: {msg}") from e

    publication_fallback = None
    try:
        generated = json.loads(Path(out_sav_tmp).read_text(encoding="utf-8-sig"))
        state = json.loads(generated["Experiment"]["StatusSave"])
        element_count = len(state["Elements"])
        wire_count = len(state["Wires"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ToolError("verilog2plsav returned an invalid electrical PLSAV") from error
    if element_count > MAX_DIRECT_PLSAV_ELEMENTS:
        gate_sha256 = hashlib.sha256(Path(out_sav_tmp).read_bytes()).hexdigest()
        Path(out_sav_tmp).write_text(json.dumps(_celestial_hdl_source_template(), ensure_ascii=False,
                                                separators=(",", ":")), encoding="utf-8")
        publication_fallback = {
            "schema": "aurex.hdl-source-celestial-fallback.v1",
            "reason": "physical_gate_element_limit_exceeded",
            "max_direct_elements": MAX_DIRECT_PLSAV_ELEMENTS,
            "gate_elements": element_count,
            "gate_wires": wire_count,
            "discarded_gate_plsav_sha256": gate_sha256,
            "template_type": 3,
            "interactive_circuit": False,
            "allowed_followup": "comments_only",
        }

    # Stage the sav under cache_dir with a task-scoped, deterministic name.
    task_id = str(getattr(runtime, "task_id", "") or "").strip() or "task"
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:120].strip("._-") or "task"
    staged_dir = os.path.join(runtime.cache_dir, "staged_sav")
    os.makedirs(staged_dir, exist_ok=True)
    staged_sav = os.path.join(staged_dir, f"{safe_task}-{os.path.basename(out_sav_tmp)}")
    try:
        shutil.move(out_sav_tmp, staged_sav)
    except Exception:
        try:
            shutil.copy2(out_sav_tmp, staged_sav)
            os.remove(out_sav_tmp)
        except Exception as e:
            raise ToolError(f"verilog_to_sav: failed to stage .sav into cache: {type(e).__name__}: {e}") from e
    if publication_fallback:
        result = {"statistics": {"components": 3, "wires": 0, "nodes": 0,
                                 "component_types": {"Sun": 1, "Earth": 1, "Moon": 1}},
                  "circuit_path": staged_sav,
                  "state_source": "fixed Type-3 HDL source publication template",
                  "publication_fallback": publication_fallback,
                  "images": [], "with_image": False}
    else:
        from .circuits import circuit_inspect
        result = circuit_inspect(runtime, {"path": staged_sav})
    result.update({"sav_path": staged_sav, "top": top or "auto", "published": False})
    if verified:
        manifest = {"schema": "aurex.hdl-export.v1", "source_sha256": verified["source_sha256"],
                    "source_files_sha256": verified["source_files_sha256"], "sources_paths": verified["sources_paths"],
                    "verification_report_path": args["hdl_report_path"], "verification_id": verified["verification_id"],
                    "sav_sha256": hashlib.sha256(Path(staged_sav).read_bytes()).hexdigest(),
                    "sav_path": staged_sav, "top": top, "strict_export": True,
                    "binary_logic_export": True,
                    "export_verilog_sha256": hashlib.sha256(verilog.encode("utf-8")).hexdigest(),
                    "note": "The recorded RTL was exported directly unless export_lowering is present. Fixed unpacked register arrays are expanded to explicit registers only after the lowered bundle passes the same verification profile. These checks are not full ISA compliance, gate-level equivalence or silicon verification."}
        if export_lowering:
            manifest["export_lowering"] = export_lowering
        if publication_fallback:
            manifest["publication_fallback"] = publication_fallback
        if 'workspace_id' in verified:
            exported = verified['design_source_files']
            manifest.update(workspace_id=verified['workspace_id'], workspace_revision=verified['workspace_revision'],
                exported_source_files_sha256={name: verified['source_files_sha256'][name] for name in exported},
                excluded_testbench_files=[name for name in verified['source_files_sha256'] if name not in exported],
                simulation_top=verified['top'], verification_profile=verified['profile'],
                note='source_sha256 binds the entire tested source/testbench bundle. Only declared design source files were exported. Reported testbench success is not gate-level equivalence or full ISA compliance.')
            result.update(workspace_id=verified['workspace_id'], workspace_revision=verified['workspace_revision'])
        manifest_path = staged_sav + ".export.json"
        Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        result["export_manifest_path"] = manifest_path
        result["verification_report_path"] = args["hdl_report_path"]
        result["source_sha256"] = verified["source_sha256"]
    return result


_ANALYZE_TYPES = {"op": 0, "dc": 1, "ac": 2, "acop": 3, "tr": 4, "trop": 5}

_ELEMENTS = COMPONENTS


def pe_simulate(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    digital_limit = runtime.config.phy_engine.digital_component_limit
    try:
        validate_spec_size(args.get("spec"), digital_limit)
    except ValueError as error:
        raise ToolError(str(error)) from error
    _v2p, lib_path = _ensure_artifacts(runtime)
    # Native simulation runs outside the agent process. Timeouts, invalid native
    # models and excessive resource usage cannot bring down the conversation.
    command = [sys.executable, "-m", "aurex.phy_engine.worker"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(command, input=json.dumps({"spec": args.get("spec"), "lib_path": lib_path, "return_state": bool(args.get("return_state")), "digital_component_limit": digital_limit}),
            capture_output=True, text=True, env=env, timeout=min(300, int(runtime.config.phy_engine.run_timeout_sec)))
    except subprocess.TimeoutExpired as error:
        raise ToolError("Phy-Engine simulation exceeded its wall-clock limit; reduce circuit size or time steps") from error
    if proc.returncode:
        raise ToolError("Phy-Engine simulation failed: " + (proc.stderr.strip()[-3000:] or f"worker exit {proc.returncode}"))
    try:
        return json.loads(proc.stdout)
    except ValueError as error:
        raise ToolError("Phy-Engine returned invalid measurement data") from error


def _reported_digital(meta: dict, sample: dict, d0: int, d1: int) -> tuple[list[int], str | None]:
    """Return the externally meaningful digital value for one native model.

    A disconnected pin is correctly sampled as X at the net layer, but an
    input's configured/stimulated state and an output's decoded value are
    model attributes.  Reporting the raw net in those two cases loses real
    state precisely when a caller is testing an isolated interface port.
    """
    model_digital = sample.get("model_digital_states", {}).get(meta["id"], {})
    if meta["type"] == "digital_input" and "state" in model_digital:
        return [model_digital["state"]], (
            "native Logic Input model state, including explicit stimulus on an otherwise disconnected pin")
    if meta["type"] == "digital_output" and "value" in model_digital:
        return [model_digital["value"]], (
            "native Logic Output threshold decoder, not raw hybrid-node storage")
    return sample["digital"][d0:d1], None


def _measurement_rows(comp_meta: list[dict], sample: dict) -> list[dict]:
    rows = []
    mixed = (any(meta["type"].startswith("digital_") for meta in comp_meta)
             and any(not meta["type"].startswith("digital_") for meta in comp_meta))
    node_uses: dict[str, int] = {}
    for meta in comp_meta:
        for node in meta["nodes"]:
            node_uses[node] = node_uses.get(node, 0) + 1
    for i, meta in enumerate(comp_meta):
        v0, v1 = sample["voltage_ord"][i:i + 2]
        c0, c1 = sample["current_ord"][i:i + 2]
        d0, d1 = sample["digital_ord"][i:i + 2]
        analog = not meta["type"].startswith("digital_")
        reported_digital, digital_origin = _reported_digital(meta, sample, d0, d1)
        row = {"id": meta["id"], "type": meta["type"], "nodes": meta["nodes"],
               "pin_labels": _ELEMENTS[meta["type"]]["pin_labels"],
               "voltage": sample["voltage"][v0:v1] if analog or mixed else [],
               "voltage_imag": sample["voltage_imag"][v0:v1] if analog or mixed else [],
               "current": sample["current"][c0:c1], "current_imag": sample["current_imag"][c0:c1],
               "digital": reported_digital}
        if digital_origin is not None:
            row["digital_origin"] = digital_origin
        if meta["type"].startswith("digital_"):
            disconnected = [label for label, node in zip(row["pin_labels"], meta["nodes"])
                            if node_uses.get(node, 0) == 1]
            if disconnected:
                row["unconnected_pins"] = disconnected
                row["unconnected_pin_note"] = (
                    "Raw samples on disconnected pins are X; native model defaults, when defined, "
                    "are applied internally and are reflected by model_state/output behavior.")
        if meta.get("pl_source"):
            row["pl_source"] = copy.deepcopy(meta["pl_source"])
        if meta.get("plsav_import"):
            row["plsav_import"] = copy.deepcopy(meta["plsav_import"])
        if analog and meta["pins"] == 2:
            voltage = complex(sample["voltage"][v0] - sample["voltage"][v0 + 1],
                              sample["voltage_imag"][v0] - sample["voltage_imag"][v0 + 1])
            row["voltage_across_0_to_1"] = {"real": voltage.real, "imag": voltage.imag, "unit": "V"}
            if meta["type"] == "resistor":
                resistance = float(sample.get("model_states", {}).get(meta["id"], {}).get(
                    "resistance_ohm", meta["params"]["r"]))
                row["effective_params"] = {"r": resistance}
                current = voltage / resistance
                row["derived_current_0_to_1"] = {"real": current.real, "imag": current.imag,
                                               "method": "Ohm's law from simulated pin voltages and observed live resistance"}
        if meta["id"] in sample.get("pin_currents", {}):
            row["pin_current_a"] = sample["pin_currents"][meta["id"]]
            row["pin_current_convention"] = "Positive current enters the component, in catalog pin order; native model observation."
        if meta["id"] in sample.get("model_states", {}):
            row["model_state"] = sample["model_states"][meta["id"]]
        if meta["id"] in sample.get("model_digital_states", {}):
            row["model_digital_state"] = sample["model_digital_states"][meta["id"]]
        if _ELEMENTS[meta["type"]].get("model_notes"):
            row["model_notes"] = _ELEMENTS[meta["type"]]["model_notes"]
        rows.append(row)
    return rows


def _interaction_groups(comp_meta: list[dict]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for index, meta in enumerate(comp_meta):
        custom = meta.get("interaction")
        definition = copy.deepcopy(custom if isinstance(custom, dict)
                                   else _ELEMENTS[meta["type"]].get("interaction"))
        if not definition:
            continue
        control_id = definition.get("control_id", meta["id"])
        if not isinstance(control_id, str) or not control_id:
            raise ToolError(f"{meta['id']}: invalid interaction control_id")
        if "current" not in definition:
            definition["current"] = meta["params"].get(
                {"spst": "closed", "voltage_source": "v",
                 "digital_input": "state"}.get(definition.get("kind"), ""))
        row = groups.setdefault(control_id, {"id": control_id, "members": [],
                                             "definition": definition})
        if row["definition"].get("kind") != definition.get("kind"):
            raise ToolError(f"{control_id}: inconsistent decomposed interaction kind")
        row["members"].append({"index": index, "meta": meta,
                               "role": definition.get("role", "component")})
    return groups


def _control_catalog(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for control_id, group in groups.items():
        definition, members = group["definition"], group["members"]
        kind = definition["kind"]
        current = definition.get("current")
        if current is None and len(members) == 1:
            meta = members[0]["meta"]
            current = meta["params"].get({"spst": "closed", "voltage_source": "v",
                                          "digital_input": "state"}.get(kind, ""))
        row = {"id": control_id, "kind": kind,
               "value_name": definition.get("value_name", "value"),
               "current": current,
               "source_model_id": definition.get("source_model_id",
                    members[0]["meta"].get("pl_source", {}).get("model_id",
                    members[0]["meta"]["type"])),
               "primitive_component_ids": [member["meta"]["id"] for member in members]}
        for key in ("allowed", "minimum", "maximum", "momentary",
                    "rated_resistance_ohm", "minimum_segment_ohm"):
            if key in definition:
                row[key] = definition[key]
        rows.append(row)
    return rows


def _prepare_tr_interactions(raw: Any, groups: dict[str, dict[str, Any]],
                             step: float, stop: float) -> tuple[dict[int, list[dict]], list[dict]]:
    if raw is None:
        return {}, []
    if not isinstance(raw, list) or len(raw) > 128:
        raise ToolError("tr_interactions must be an array of at most 128 timed control frames")
    schedule: dict[int, list[dict]] = {}
    normalized, previous_time = [], -1.0
    for frame_index, frame in enumerate(raw):
        if not isinstance(frame, dict) or set(frame) != {"time_s", "set"} or not isinstance(frame["set"], dict) or not frame["set"]:
            raise ToolError(f"tr_interactions[{frame_index}] must be exactly {{time_s:number,set:{{control_id:value}}}} with a nonempty set")
        when = frame["time_s"]
        if isinstance(when, bool) or not isinstance(when, (int, float)) or not math.isfinite(float(when)):
            raise ToolError(f"tr_interactions[{frame_index}].time_s must be finite")
        when = float(when)
        if when < 0 or when > stop or when <= previous_time:
            raise ToolError("tr_interactions times must be strictly increasing and within 0..tr_stop")
        previous_time = when
        nearest = 0 if when == 0 else int(round(when / step))
        if when != 0 and (nearest < 1 or abs(when - nearest * step) > 1e-9 * max(1.0, abs(when), abs(step))):
            raise ToolError(f"tr_interactions[{frame_index}].time_s={when:g} must align exactly to the tr_step grid")
        native_step = max(1, nearest)
        if native_step in schedule:
            raise ToolError(f"tr_interactions[{frame_index}] maps to native step {native_step}, already used; combine simultaneous changes in one set")
        actions = []
        if len(frame["set"]) > 32:
            raise ToolError(f"tr_interactions[{frame_index}].set exceeds 32 controls")
        for control_id, value in frame["set"].items():
            if control_id not in groups:
                candidates = difflib.get_close_matches(str(control_id), list(groups), n=3, cutoff=.5)
                hint = f" Similar exact control IDs: {candidates}." if candidates else " Read circuit_inspect(controls_only=true) first."
                raise ToolError(f"tr_interactions[{frame_index}]: unknown control ID {control_id!r}." + hint)
            definition = groups[control_id]["definition"]
            kind = definition["kind"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ToolError(f"tr_interactions[{frame_index}].set[{control_id!r}] must be a finite number")
            numeric = float(value)
            allowed = definition.get("allowed")
            if allowed is not None and (not numeric.is_integer() or int(numeric) not in allowed):
                raise ToolError(f"tr_interactions[{frame_index}].set[{control_id!r}] must be one of {allowed}")
            if "minimum" in definition and numeric < float(definition["minimum"]):
                raise ToolError(f"tr_interactions[{frame_index}].set[{control_id!r}] is below {definition['minimum']}")
            if "maximum" in definition and numeric > float(definition["maximum"]):
                raise ToolError(f"tr_interactions[{frame_index}].set[{control_id!r}] exceeds {definition['maximum']}")
            actions.append({"control_id": control_id, "value": int(numeric) if allowed is not None else numeric,
                            "kind": kind})
        schedule[native_step] = actions
        normalized.append({"time_s": when, "native_step": native_step,
                           "set": copy.deepcopy(frame["set"])})
    return schedule, normalized


def _apply_control(circuit, group: dict[str, Any], value: int | float) -> list[dict[str, Any]]:
    definition, updates = group["definition"], []
    kind = definition["kind"]
    for member in group["members"]:
        index, meta, role = member["index"], member["meta"], member["role"]
        if kind in ("spst", "spdt", "dpdt"):
            closed = int(value) if kind == "spst" else int(
                (role.endswith("_left") and value == 1) or
                (role.endswith("_right") and value == 2))
            circuit.set_model_scalar(index, "Cut Through", closed)
            meta["params"]["closed"] = float(closed)
            updates.append({"component_id": meta["id"], "attribute": "Cut Through", "value": closed})
        elif kind == "slide_rheostat":
            if role == "wiper_link":
                continue
            rated = float(definition["rated_resistance_ohm"])
            floor = float(definition["minimum_segment_ohm"])
            resistance = max(floor, rated * (float(value) if role == "segment_left" else 1.0 - float(value)))
            circuit.set_model_scalar(index, "R", resistance)
            meta["params"]["r"] = resistance
            updates.append({"component_id": meta["id"], "attribute": "R", "value": resistance, "unit": "ohm"})
        elif kind == "voltage_source":
            circuit.set_model_scalar(index, "V", float(value))
            meta["params"]["v"] = float(value)
            updates.append({"component_id": meta["id"], "attribute": "V", "value": float(value), "unit": "V"})
        elif kind == "digital_input":
            circuit.set_model_digital(index, int(definition.get("digital_attribute", 0)), int(value))
            meta["params"]["state"] = float(value)
            updates.append({"component_id": meta["id"], "attribute": "digital state", "value": int(value)})
        else:
            raise ToolError(f"Unsupported interaction kind {kind!r}")
    definition["current"] = value
    return updates


def _simulate_spec(spec: Any, lib_path: str, *, return_state: bool = False,
                   digital_component_limit: int = DEFAULT_DIGITAL_COMPONENT_LIMIT) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ToolError("pe_simulate: spec must be an object")
    comps = spec.get("components")
    if not isinstance(comps, list) or not comps:
        raise ToolError("pe_simulate: spec.components must be a non-empty array")
    try:
        validate_spec_size(spec, digital_component_limit)
    except ValueError as error:
        raise ToolError(str(error)) from error

    analysis = str(spec.get("analysis") or "dc").strip().lower()
    at = _ANALYZE_TYPES.get(analysis)
    if at is None:
        raise ToolError("pe_simulate: analysis must be one of op/dc/ac/acop/tr/trop")
    mixed = (any(str(c.get("type", "")).startswith("digital_") for c in comps if isinstance(c, dict))
             and any(not str(c.get("type", "")).startswith("digital_") for c in comps if isinstance(c, dict)))
    if mixed and analysis not in ("op", "dc", "tr"):
        raise ToolError("Mixed-signal simulation supports op/dc/tr only; AC/trop coupling is not implemented")

    tr_initialize_dc = spec.get("tr_initialize_dc", False)
    if type(tr_initialize_dc) is not bool:
        raise ToolError("tr_initialize_dc must be boolean")
    if tr_initialize_dc and analysis != "tr":
        raise ToolError("tr_initialize_dc applies only to analysis=tr")
    if tr_initialize_dc and any(str(c.get("type", "")).startswith("digital_") for c in comps if isinstance(c, dict)):
        raise ToolError("tr_initialize_dc currently supports analog-only circuits; mixed/digital initial state is explicit")

    digital_steps = spec.get('digital_steps_per_tr_step', 1)
    if type(digital_steps) is not int or not 1 <= digital_steps <= 64:
        raise ToolError('digital_steps_per_tr_step must be an integer in 1..64; no digital time interval is inferred')
    if 'digital_steps_per_tr_step' in spec and analysis != 'tr':
        raise ToolError('digital_steps_per_tr_step applies only to analysis=tr, not post-analysis ticks or a separate stimulus sequence')

    stimulus = spec.get("stimulus", [])
    if not isinstance(stimulus, list) or len(stimulus) > 128:
        raise ToolError("stimulus must be an array of at most 128 representative input vectors")
    if stimulus:
        known = {c.get("id"): c for c in comps if isinstance(c, dict) and isinstance(c.get("id"), str)}
        if any(not str(c.get("type", "")).startswith("digital_") for c in known.values()):
            raise ToolError("stimulus currently requires a digital-only circuit; use a supported mixed transient source for analog-connected inputs")
        inputs = [cid for cid, component in known.items() if component.get("type") == "digital_input"]
        # Validate the complete sequence before loading or running native code.
        # A typo is actionable input feedback, not a solver failure to retry.
        for frame, vector in enumerate(stimulus):
            if not isinstance(vector, dict) or not isinstance(vector.get("set", {}), dict):
                raise ToolError(f"stimulus[{frame}] must be {{set:{{exact_input_component_id:0|1|2|3}}}}")
            for cid, state in vector.get("set", {}).items():
                if cid not in known:
                    candidates = difflib.get_close_matches(str(cid), inputs, n=3, cutoff=.5)
                    hint = f" Similar exact input IDs: {candidates}." if candidates else " Read circuit_inspect/interface for exact input IDs."
                    raise ToolError(f"stimulus[{frame}].set: unknown component ID {cid!r}." + hint + " No ID was automatically substituted and no simulation was run.")
                if known[cid].get("type") != "digital_input":
                    raise ToolError(f"stimulus[{frame}].set[{cid!r}] targets {known[cid].get('type')!r}, not digital_input; use the exact input component's ID")
                if type(state) not in (int, float) or state not in (0, 1, 2, 3):
                    raise ToolError(f"stimulus[{frame}].set[{cid!r}]: state must be 0=L,1=H,2=X,3=Z; got {state!r}")

    lib = load_library(lib_path)

    # element 0 is ground placeholder (code=0)
    element_codes: list[int] = [0]
    properties: list[float] = []
    comp_meta: list[dict[str, Any]] = []
    ids: set[str] = set()

    for c in comps:
        if not isinstance(c, dict):
            raise ToolError("pe_simulate: each component must be an object")
        cid = str(c.get("id") or "").strip()
        ctype = str(c.get("type") or "").strip().lower()
        nodes = c.get("nodes")
        if not cid:
            raise ToolError("pe_simulate: component.id is required")
        if cid in ids:
            raise ToolError(f"pe_simulate: duplicate component id {cid!r}")
        ids.add(cid)
        if ctype not in _ELEMENTS:
            raise ToolError(f"pe_simulate: unsupported component type: {ctype}")
        if not isinstance(nodes, list):
            raise ToolError("pe_simulate: component.nodes must be an array")
        pins = int(_ELEMENTS[ctype]["pins"])
        if len(nodes) != pins:
            raise ToolError(f"pe_simulate: {cid} expects {pins} nodes, got {len(nodes)}")
        extras = _ELEMENTS[ctype].get("extra_params", {})
        params = {**_ELEMENTS[ctype].get("defaults", {}), **{k: v["default"] for k, v in extras.items()},
                  **(c.get("params") if isinstance(c.get("params"), dict) else {})}
        for key in extras:
            try:
                val = float(params[key])
            except (ValueError, TypeError, OverflowError) as error:
                raise ToolError(f"{cid}: extra parameter {key} must be numeric") from error
            if not math.isfinite(val):
                raise ToolError(f"{cid}: extra parameter {key} must be finite")
            rule = extras[key]
            if "exclusive_minimum" in rule and val <= rule["exclusive_minimum"]:
                raise ToolError(f"{cid}: {key} must be greater than {rule['exclusive_minimum']}")
            if "minimum" in rule and val < rule["minimum"]:
                raise ToolError(f"{cid}: {key} must be at least {rule['minimum']}")
            if "maximum" in rule and val > rule["maximum"]:
                raise ToolError(f"{cid}: {key} must be at most {rule['maximum']}")
            params[key] = val
        element_codes.append(int(_ELEMENTS[ctype]["code"]))

        prop_keys: list[str] = list(_ELEMENTS[ctype]["props"])
        for k in prop_keys:
            if k not in params:
                raise ToolError(f"pe_simulate: {cid} missing param {k!r}")
            try:
                value = float(params.get(k))
                if not math.isfinite(value):
                    raise ValueError("nonfinite value")
                if k in ("r", "c", "l", "is", "n", "area", "beta", "l1", "l2") and value <= 0:
                    raise ValueError("value must be positive")
                properties.append(value)
            except Exception as e:
                raise ToolError(f"pe_simulate: {cid} param {k!r} must be a number") from e

        comp_meta.append({"id": cid, "type": ctype, "nodes": [str(n) for n in nodes], "pins": pins, "params": params,
                          **({"interaction": copy.deepcopy(c["interaction"])} if isinstance(c.get("interaction"), dict) else {}),
                          **({"pl_source": copy.deepcopy(c["pl_source"])} if c.get("pl_source") else {}),
                          **({"plsav_import": copy.deepcopy(c["plsav_import"])} if c.get("plsav_import") else {})})

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
            # A saved one-pin *digital* net is an unconnected pin, not a driven
            # X node.  Keep it disconnected so native models can apply their
            # documented open-input semantics (notably random4 reset_n=high).
            # Analog pins still need a self-edge because their floating node is
            # part of the MNA topology and may carry initial/device state.
            for ele, pin in pins:
                if not comp_meta[int(ele) - 1]["type"].startswith("digital_"):
                    wires.extend([int(ele), int(pin), int(ele), int(pin)])
            continue
        root_ele, root_pin = pins[0]
        for ele, pin in pins[1:]:
            wires.extend([int(root_ele), int(root_pin), int(ele), int(pin)])

    circuit = lib.create_circuit(elements=element_codes, wires=wires, properties=properties)
    sequence_samples = []
    transient = None
    initial_operating_point = None
    max_pins = max(m["pins"] for m in comp_meta)
    interaction_groups = _interaction_groups(comp_meta)
    interaction_catalog = _control_catalog(interaction_groups)
    interaction_schedule: dict[int, list[dict]] = {}
    normalized_interactions: list[dict] = []
    applied_interactions: list[dict] = []

    def capture():
        measured = circuit.sample_complex(max_pins=max_pins)
        measured["pin_currents"] = {}
        measured["model_states"] = {}
        measured["model_digital_states"] = {}
        measured["interaction_states"] = {
            row["id"]: interaction_groups[row["id"]]["definition"].get("current")
            for row in _control_catalog(interaction_groups)
        }
        for i, meta in enumerate(comp_meta):
            attrs = _ELEMENTS[meta["type"]].get("pin_current_attributes", [])
            if attrs and analysis in _ELEMENTS[meta["type"]].get("pin_current_analysis", []):
                measured["pin_currents"][meta["id"]] = [circuit.model_scalar(i, a) for a in attrs]
            states = _ELEMENTS[meta["type"]].get("state_attributes", {})
            if states and analysis in _ELEMENTS[meta["type"]].get("state_attribute_analysis", []):
                measured["model_states"][meta["id"]] = {
                    name: circuit.model_scalar(i, attribute) for name, attribute in states.items()
                }
            digital_states = _ELEMENTS[meta["type"]].get("digital_state_attributes", {})
            if digital_states and analysis in _ELEMENTS[meta["type"]].get("digital_state_attribute_analysis", []):
                measured["model_digital_states"][meta["id"]] = {
                    name: circuit.model_digital(i, attribute) for name, attribute in digital_states.items()
                }
        return measured

    try:
        gmin = spec.get("g_min_siemens", 0.0)
        if (isinstance(gmin, bool) or not isinstance(gmin, (int, float))
                or not math.isfinite(float(gmin)) or not 0 <= float(gmin) <= 1e-3):
            raise ToolError("g_min_siemens must be finite and in 0..1e-3 S")
        circuit.set_gmin(float(gmin))
        for i, meta in enumerate(comp_meta):
            for key, definition in _ELEMENTS[meta["type"]].get("extra_params", {}).items():
                circuit.set_model_scalar(i, definition["native_attribute"], meta["params"][key])
        circuit.set_analyze_type(at)
        if analysis in ("tr", "trop"):
            t_step = float(spec.get("tr_step") or 1e-9)
            t_stop = float(spec.get("tr_stop") or t_step)
            if not (math.isfinite(t_step) and math.isfinite(t_stop) and 0 < t_step <= t_stop and t_stop / t_step <= 10000):
                raise ToolError("transient analysis requires 0 < tr_step <= tr_stop and at most 10000 steps")
            circuit.set_tr(t_step, t_stop)
            interaction_schedule, normalized_interactions = _prepare_tr_interactions(
                spec.get("tr_interactions"), interaction_groups, t_step, t_stop)
        elif "tr_interactions" in spec:
            raise ToolError("tr_interactions requires analysis=tr")
        if analysis in ("ac", "acop"):
            omega = float(spec.get("ac_omega") or 1.0)
            if not math.isfinite(omega) or omega <= 0:
                raise ToolError("ac_omega must be finite and positive (rad/s)")
            circuit.set_ac_omega(omega)

        if analysis == "tr":
            if tr_initialize_dc:
                # A DC solve on the same live circuit establishes the biased
                # operating point before the physical transient.  This is
                # opt-in: natural power-on and RC-startup tests retain their
                # historical all-zero dynamic initial state by default.
                circuit.set_analyze_type(_ANALYZE_TYPES["dc"])
                circuit.analyze()
                initial_operating_point = {
                    "requested": True,
                    "completed": True,
                    "analysis": "dc",
                    "same_live_circuit": True,
                }
                circuit.set_analyze_type(at)
                circuit.set_tr(t_step, t_stop)
            def before_step(target_time: float, step_number: int) -> None:
                actions = interaction_schedule.get(step_number, [])
                if not actions:
                    return
                primitive_updates = []
                for action in actions:
                    primitive_updates.extend(_apply_control(
                        circuit, interaction_groups[action["control_id"]], action["value"]))
                source = next(row for row in normalized_interactions if row["native_step"] == step_number)
                applied_interactions.append({**copy.deepcopy(source), "applied_before_target_time_s": target_time,
                                             "primitive_updates": primitive_updates})

            if interaction_schedule:
                every = spec.get("tr_sample_every", 0)
                if type(every) is not int or every < 0 or (every and math.ceil(t_stop / t_step / every) > 201):
                    raise ToolError("tr_sample_every must be a positive integer and yield at most 201 actual samples")
                transient = circuit.run_transient_controlled(
                    t_step, t_stop, before_step=before_step, sample_every=every,
                    capture=capture if every else None,
                    digital_steps_per_tr_step=digital_steps)
            elif spec.get("tr_sample_every") is not None:
                every = spec["tr_sample_every"]
                if type(every) is not int or every < 1 or math.ceil(t_stop / t_step / every) > 201:
                    raise ToolError("tr_sample_every must be a positive integer and yield at most 201 actual samples")
                transient = circuit.run_transient_trace(t_step, t_stop, sample_every=every, capture=capture,
                    digital_steps_per_tr_step=digital_steps)
            else:
                transient = circuit.run_transient_bounded(t_step, t_stop, digital_steps_per_tr_step=digital_steps)
        else:
            if spec.get("tr_sample_every") is not None:
                raise ToolError("tr_sample_every requires analysis=tr")
            if mixed:
                circuit.analyze_mixed_dc(at)
            else:
                circuit.analyze()
        # New TR drivers already propagate N times on EVERY solver step, including
        # unsampled steps. Do not append the legacy implicit tick to a completed
        # trace; explicit user-requested post-analysis ticks remain additional.
        per_step = bool(transient and transient.get("digital_propagation", {}).get("verified_per_step"))
        default_ticks = 0 if per_step or mixed else (1 if any(m["type"].startswith("digital_") for m in comp_meta) else 0)
        ticks = int(spec.get("digital_clock_ticks", default_ticks))
        if ticks < 0:
            ticks = 0
        if ticks > 10000:
            raise ToolError("digital_clock_ticks exceeds 10000")
        for _ in range(ticks):
            circuit.digital_clk()
        if transient is not None:
            transient["post_trace_digital_ticks"] = {"count": ticks, "explicit": "digital_clock_ticks" in spec,
                "scope": "After the complete TR trace and before any separate stimulus sequence; these ticks never backfill sampled history."}

        sample = capture()
        if stimulus:
            circuit.set_analyze_type(_ANALYZE_TYPES["tr"])
            circuit.set_tr(1e-8, 1e-8)
            input_indices = {c["id"]: index for index, c in enumerate(comp_meta) if c["type"] == "digital_input"}
            for step, vector in enumerate(stimulus):
                for cid, state in vector.get("set", {}).items():
                    circuit.set_model_digital(input_indices[cid], 0, int(state))
                circuit.digital_clk()
                circuit.analyze()
                circuit.digital_clk()
                sample = capture()
                sequence_samples.append({"step": step, "inputs": vector.get("set", {}), "digital": {
                    c["id"]: _reported_digital(c, sample,
                        sample["digital_ord"][i], sample["digital_ord"][i + 1])[0]
                    for i, c in enumerate(comp_meta)}})
    except PhyEngineError as e:
        raise ToolError(str(e)) from e
    finally:
        circuit.close()

    # Slice per component based on ord arrays (component order == our comp_meta order)
    out: dict[str, Any] = {"analysis": analysis, "engine": "Phy-Engine native", "components": [],
        "units": {"voltage": "V", "current": "A", "frequency": "rad/s"},
        "digital_encoding": {"0": "L", "1": "H", "2": "X", "3": "Z"},
        "notes": ["Voltage arrays are per pin relative to ground; branch currents exist only for models with explicit MNA branches.",
                  "Transient results are the final state at tr_stop; no unmeasured waveform is inferred."]}
    out["components"] = _measurement_rows(comp_meta, sample)
    if interaction_catalog:
        out["interaction_controls"] = _control_catalog(interaction_groups)
        out["interaction_states"] = sample.get("interaction_states", {})
    if mixed:
        out["mixed_signal_scope"] = {
            "method": "Configured digital propagations are coupled to native MNA solves at every physical TR step.",
            "drive_model": "Ideal voltage drive at each gate's saved Ll/Hl. Imported PhysicsLab gate outputs with an analog load use an irreversible saved maximum-current guard.",
            "unsupported": ["mixed AC/trop", "finite digital output impedance", "original-app pointwise equivalence"],
        }
    if sequence_samples:
        out["stimulus_results"] = sequence_samples
    if transient:
        if initial_operating_point is not None:
            transient["initial_operating_point"] = initial_operating_point
        for point in transient.get("samples", []):
            raw_sample = point.pop("sample")
            point["components"] = _measurement_rows(comp_meta, raw_sample)
            if raw_sample.get("interaction_states"):
                point["interaction_states"] = raw_sample["interaction_states"]
        if normalized_interactions:
            transient["interaction_events"] = applied_interactions
            transient["interaction_timing"] = {
                "semantics": "Each frame is applied immediately before the named native TR step is solved; time 0 applies before step 1.",
                "grid_aligned": True, "all_requested_events_applied": len(applied_interactions) == len(normalized_interactions),
            }
        out["transient"] = transient
    if return_state:
        out["state"] = {
            "schema": "aurex.pe-state.v1", "spec": copy.deepcopy(spec),
            "measurements": copy.deepcopy(out), "captured_at": time.time(),
            "origin": {"sample": "circuit_sample_complex", "layout": "native design sidecar",
                       "live_handle_retained": False, "engine": "Phy-Engine native",
                       "note": "Measurements copied from the live solver before its handle was closed; camera changes do not rerun simulation."},
            **{k: copy.deepcopy(spec[k]) for k in ("camera", "camera_save") if k in spec},
        }
    return out


PHY_ENGINE_BUILD_TOOL = {
    "name": "phy_engine_build",
    "description": "Build Phy-Engine artifacts (verilog2plsav, phyengine shared lib and C++ circuit_view renderer) via CMake.",
    "parameters": {"type": "object", "properties": {}},
}

VERILOG_TO_SAV_TOOL = {
    "name": "verilog_to_sav",
    "description": "Export verified single or connected Verilog modules with Phy-Engine. Fixed unpacked register arrays are deterministically expanded and rerun through the same verification before export. At <=5000 elements this returns a local electrical .sav; above 5000 it discards the oversized gate PLSAV and returns a fixed non-interactive Type-3 publication carrier for title/body HDL source only, with no diagram or screenshot and comments-only follow-up. Strict export rejects unsupported/lossy mappings; nothing is published by this tool.",
    "parameters": {
        "type": "object",
        "properties": {
            "verilog": {"type": "string", "description": "Verilog source code (Verilog-2001)."},
            "modules": {"type": "array", "items": {"type": "string"}, "description": "Additional Verilog modules; use a top module to wire them together."},
            "top": {"type": "string", "description": "Explicit top module name."},
            "hdl_report_path": {"type": "string", "description": "Preferred: verification report_path returned by successful hdl_simulate. Exports those exact unmodified source files and writes a source-to-PLSAV hash manifest; do not resend inline code."},
            "force_build": {
                "type": "boolean",
                "default": False,
                "description": "If true, build Phy-Engine artifacts via CMake when paths are not configured.",
            },
        },
        "required": [],
    },
}

PE_SIMULATE_TOOL = {
    "name": "pe_simulate",
    "description": "Run a bounded native Phy-Engine simulation in an isolated worker. Use circuit_catalog for model definitions and circuit_analyze for images, saved designs and digital input vectors.",
    "parameters": {
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string", "enum": ["op", "dc", "ac", "acop", "tr", "trop"], "default": "dc"},
                    "tr_step": {"type": "number"},
                    "tr_stop": {"type": "number"},
                    "tr_sample_every": {"type": "integer", "minimum": 1, "description": "Sample actual state every N steps plus endpoint; at most 201 points, analysis=tr only."},
                    "tr_initialize_dc": {"type": "boolean", "default": False, "description": "analysis=tr analog-only opt-in: solve the DC operating point on the same live circuit before transient stepping. Use for biased steady-state small-signal analysis; leave false for natural power-on/startup behavior."},
                    "ac_omega": {"type": "number"},
                    "g_min_siemens": {"type": "number", "minimum": 0, "maximum": 0.001, "default": 0,
                        "description": "Explicit conductance from every analog node to ground. PhysicsLab imports may set 1e-15 S to regularize floating nodes; native designs default to zero."},
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
