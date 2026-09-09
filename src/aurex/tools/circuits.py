"""Local, versioned electrical-circuit tools. These tools never publish an experiment."""
from __future__ import annotations

import copy
from collections import Counter, deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from ..phy_engine.catalog import COMPONENTS
from ..phy_engine.limits import DEFAULT_DIGITAL_COMPONENT_LIMIT, native_digital_type, validate_spec_size
from .registry import ToolError, ToolRegistry, ToolRuntime, ToolSpec
from .phy_engine import _ensure_artifacts, _resolve, pe_simulate
from .plar_analog_import import import_element as import_analog_element
from .plar_passive_import import import_element as import_passive_element
from .plar_source_import import import_element as import_source_element
from .plar_power_import import import_element as import_power_element
from .plar_semiconductor_import import import_element as import_semiconductor_element
from .plar_device_import import import_element as import_device_element
from .plar_damage import apply_damage_protection, supports_broken_state

_VECTOR3 = {"type": "array", "items": {"type": "number", "minimum": -1e9, "maximum": 1e9}, "minItems": 3, "maxItems": 3}

# Saved Physics Lab pin numbers do not always match the compact native PE
# model order.  This is the same lossless mapping used by _spec_from_sav;
# exposing it during inspection keeps agents from guessing or searching the
# web for a fact the importer already knows.
_PLSAV_REORDERED = {
    "Half Adder": ("digital_half_adder", [3, 2, 0, 1]),
    "Full Adder": ("digital_full_adder", [4, 2, 3, 0, 1]),
    "Half Subtractor": ("digital_half_sub", [3, 2, 0, 1]),
    "Full Subtractor": ("digital_full_sub", [4, 2, 3, 0, 1]),
    "Multiplier": ("digital_mul2", [7, 6, 5, 4, 3, 2, 1, 0]),
    "D Flipflop": ("digital_dff", [2, 3, 0]),
    "T Flipflop": ("digital_tff", [2, 3, 0]),
    "Real-T Flipflop": ("digital_t_bar_ff", [2, 3, 0]),
    "JK Flipflop": ("digital_jkff", [2, 4, 3, 0]),
    "Counter": ("digital_counter4", [0, 1, 2, 3, 4, 5]),
    "Random Generator": ("digital_random4", [0, 1, 2, 3, 4, 5]),
}

# PhysicsLab Boolean primitives save a disconnected input as logic-low.  PE's
# native four-state gates deliberately keep an undriven net at X, so the PLSAV
# adapter must make the closed-source application's input default explicit.
# Only pins whose absence is unambiguous from the original Wire records are
# covered here; sequential/multi-bit blocks keep native X unless their own
# importer defines a documented reset/default contract.
_PLSAV_BOOLEAN_INPUT_PINS = {
    "Yes Gate": (0,), "No Gate": (0,),
    "Or Gate": (0, 1), "And Gate": (0, 1),
    "Xor Gate": (0, 1), "Xnor Gate": (0, 1),
    "Nand Gate": (0, 1), "Nor Gate": (0, 1),
    "Imp Gate": (0, 1), "Nimp Gate": (0, 1),
}

# PhysicsLab serializes the same field name for two different circuit
# topologies.  Voltage-like and reactive two-terminal devices use a series
# loss; a current source uses its internal resistance as a Norton shunt.  Keep
# this table explicit so a newly supported model cannot silently acquire the
# wrong topology merely because it happens to expose an ``内阻`` property.
_PLSAV_INTERNAL_RESISTANCE = {
    "Basic Capacitor": "series",
    "Basic Inductor": "series",
    "Battery Source": "series",
    "Current Source": "parallel",
    "Sinewave Source": "series",
    "Square Source": "series",
    "Triangle Source": "series",
    "Sawtooth Source": "series",
    "Pulse Source": "series",
}


def _saved_pin_labels(component: dict[str, Any]) -> dict[int, str]:
    """Return importer-backed native saved-pin labels, without inference."""
    kind = component.get("type")
    if kind in _PLSAV_REORDERED:
        model, saved_order = _PLSAV_REORDERED[kind]
        labels = {saved_pin: COMPONENTS[model]["pin_labels"][model_pin]
                  for model_pin, saved_pin in enumerate(saved_order)}
        if kind == "D Flipflop":
            # Physics Lab exposes both Q and ~Q; PE models ~Q as the faithful
            # companion inverter in _spec_from_sav.
            labels[1] = "q_bar"
        return labels
    candidates = [model["pin_labels"] for model in COMPONENTS.values()
                  if model.get("model_id") == kind]
    if candidates and all(labels == candidates[0] for labels in candidates[1:]):
        return dict(enumerate(candidates[0]))
    return {}
_CAMERA = {
    "type": "object", "additionalProperties": False,
    "description": "View-only camera in native xyz units. Defaults to the actual saved camera when available; does not move components. Use mode=auto,fit=true for framing; custom position+target gives an unambiguous look-at. yaw_deg/pitch_deg orbit the target; zoom>1 zooms in. Read returned clipping/camera warnings and adjust before drawing conclusions.",
    "properties": {
        "mode": {"enum": ["saved", "auto", "custom"]},
        "position": _VECTOR3, "target": _VECTOR3, "rotation": _VECTOR3,
        "distance": {"type": "number", "exclusiveMinimum": 0, "maximum": 1e9},
        "yaw_deg": {"type": "number", "minimum": -3600, "maximum": 3600},
        "pitch_deg": {"type": "number", "minimum": -90, "maximum": 90},
        "zoom": {"type": "number", "minimum": .001, "maximum": 1000},
        "fov_y_deg": {"type": "number", "minimum": 1, "maximum": 170},
        "projection": {"enum": ["perspective", "orthographic"]},
        "orthographic_height": {"type": "number", "exclusiveMinimum": 0, "maximum": 1e9},
        "fit": {"type": "boolean"},
    },
}


def _camera(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    import jsonschema
    try:
        jsonschema.validate(value, _CAMERA)
        json.dumps(value, allow_nan=False)
        if "position" in value and any(k in value for k in ("rotation", "yaw_deg", "pitch_deg")):
            raise ValueError("Use either position+target or rotation/orbit, not both")
    except (jsonschema.ValidationError, ValueError, TypeError) as error:
        raise ToolError("Invalid camera: " + str(error).splitlines()[0]) from error
    return copy.deepcopy(value)


def _artifact_dir(runtime: ToolRuntime) -> Path:
    task = re.sub(r"[^A-Za-z0-9_.-]", "_", runtime.task_id)[:100] or "task"
    base = Path(runtime.cache_dir).resolve() / "circuits" / task
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="revision-", dir=base))


def _input(runtime: ToolRuntime, path: str) -> tuple[Path, dict[str, Any]]:
    file = Path(_resolve(runtime, path)).resolve()
    roots = [Path(runtime.cache_dir).resolve(), Path(runtime.config_path).resolve().parent / "physicsLabSav"]
    if not any(file.is_relative_to(root) for root in roots):
        raise ToolError("Circuit input must be a local artifact in cache_dir or physicsLabSav")
    if not (str(file).endswith((".sav", ".plsav", ".circuit.json", ".pe-state.json")) and file.is_file()):
        raise ToolError("Expected an existing .sav, .plsav, .circuit.json or .pe-state.json artifact")
    if file.stat().st_size > 32 * 1024**2:
        raise ToolError("Circuit input exceeds 32 MiB")
    try:
        value = json.loads(file.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError) as error:
        raise ToolError("Circuit input is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ToolError("Circuit input must be a JSON object")
    if value.get("schema") == "aurex.pe-state.v1":
        if not isinstance(value.get("spec"), dict) or not isinstance(value.get("measurements"), dict) or not isinstance(value.get("scene"), dict):
            raise ToolError("Native state snapshot is missing spec, scene or measurements")
    elif value.get("schema") != "aurex.circuit.v1":
        experiment = value.get("Experiment", value)
        kind = experiment.get("Type") if isinstance(experiment, dict) else None
        if type(kind) is not int or kind != 0:
            raise ToolError("Only electrical experiments with integer Type=0 can be opened")
        if "Type" in value and (type(value["Type"]) is not int or value["Type"] != 0):
            raise ToolError("Outer experiment Type must also be integer 0")
    return file, value


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _protection_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    cause_names = ((1, "current"), (2, "voltage"), (4, "power"), (8, "saved_broken"))
    guards = []
    for row in rows:
        if row.get("type") != "rated_protection":
            continue
        state = row.get("model_state") if isinstance(row.get("model_state"), dict) else {}
        source = row.get("pl_source") if isinstance(row.get("pl_source"), dict) else {}
        mask_value = state.get("trip_mask", 0)
        mask = int(mask_value) if type(mask_value) in (int, float) and math.isfinite(mask_value) else 0
        item = {
            "component_id": source.get("parent_identifier", row.get("id")),
            "source_model_id": source.get("model_id"),
            "native_guard_id": row.get("id"),
            "broken": bool(state.get("broken", False)),
            "trip_causes": [name for bit, name in cause_names if mask & bit],
            "temperature_c": state.get("temperature_c"),
            "thermal_energy_j": state.get("thermal_energy_j"),
            "current_a": state.get("current_a"),
            "voltage_v": state.get("voltage_v"),
            "power_w": state.get("power_w"),
            "trip_current_a": state.get("trip_current_a"),
            "trip_voltage_v": state.get("trip_voltage_v"),
            "trip_power_w": state.get("trip_power_w"),
        }
        guards.append(item)
    if not guards:
        return None
    broken = [item for item in guards if item["broken"]]
    newly_tripped = [item for item in broken if any(
        cause in item["trip_causes"] for cause in ("current", "voltage", "power"))]
    return {
        "broken_components": broken,
        "newly_tripped_this_run": newly_tripped,
        "has_new_trips": bool(newly_tripped),
        "alert": ("本轮新增熔断: " + ", ".join(str(item["component_id"]) for item in newly_tripped)
                  if newly_tripped else
                  ("已有熔断: " + ", ".join(str(item["component_id"]) for item in broken)
                   if broken else "无熔断")),
        "scope": "Current/power trips are accumulated electrothermal TR results; voltage trips are immediate. saved_broken came from the input save. DC/OP is a steady-state thermal check.",
    }


def normalize_spec(value: Any, *, digital_component_limit: int = DEFAULT_DIGITAL_COMPONENT_LIMIT) -> dict[str, Any]:
    try:
        validate_spec_size(value, digital_component_limit)
    except ValueError as error:
        raise ToolError(str(error)) from error
    spec = copy.deepcopy(value)
    if "camera" in spec:
        spec["camera"] = _camera(spec["camera"])
    seen = set()
    for c in spec["components"]:
        if not isinstance(c, dict):
            raise ToolError("Each component must be an object")
        cid = c.get("id")
        if not isinstance(cid, str) or not cid or len(cid) > 128 or cid in seen:
            raise ToolError("Component IDs must be unique nonempty strings up to 128 characters")
        seen.add(cid)
        # A display label is independent of the stable component identity.
        # In particular, never fabricate a public port name from an ID.
        if "label" in c and c["label"] is not None and (not isinstance(c["label"], str) or len(c["label"]) > 4096):
            raise ToolError(f"{cid}: label must be a string up to 4096 characters or null")
        kind = COMPONENTS.get(c.get("type", ""))
        if not kind:
            raise ToolError(f"Unsupported component type {c.get('type')!r}; use circuit_catalog")
        nodes = c.get("nodes")
        if not isinstance(nodes, list) or len(nodes) != kind["pins"] or not all(isinstance(n, str) and 0 < len(n) <= 128 for n in nodes):
            raise ToolError(f"{cid}: nodes must contain {kind['pins']} node names in catalog pin order")
        c["nodes"] = ["gnd" if n.casefold() in ("0", "ground", "gnd") else n for n in nodes]
        raw = c.get("params", {})
        if not isinstance(raw, dict):
            raise ToolError(f"{cid}: params must be an object")
        extras = kind.get("extra_params", {})
        unknown = set(raw) - set(kind["props"]) - set(extras)
        if unknown:
            raise ToolError(f"{cid}: unknown parameters {sorted(unknown)}; use circuit_catalog")
        params = {**kind["defaults"], **{k: v["default"] for k, v in extras.items()}, **raw}
        for key in [*kind["props"], *extras]:
            try:
                val = float(params[key])
            except (KeyError, ValueError, TypeError) as error:
                raise ToolError(f"{cid}: missing or invalid parameter {key}") from error
            if not math.isfinite(val):
                raise ToolError(f"{cid}: {key} must be finite")
            if key in kind.get("integer_params", ()) and not val.is_integer():
                raise ToolError(f"{cid}: {key} must be an integer")
            if key in kind.get("parameter_ranges", {}):
                lower, upper = kind["parameter_ranges"][key]
                if not lower <= val <= upper:
                    raise ToolError(f"{cid}: {key} must be in [{lower}, {upper}]")
            if key in ("r", "c", "l", "is", "n", "area", "beta", "l1", "l2") and val <= 0:
                raise ToolError(f"{cid}: {key} must be positive")
            if key in extras:
                rule = extras[key]
                if "exclusive_minimum" in rule and val <= rule["exclusive_minimum"]:
                    raise ToolError(f"{cid}: {key} must be greater than {rule['exclusive_minimum']}")
                if "minimum" in rule and val < rule["minimum"]:
                    raise ToolError(f"{cid}: {key} must be at least {rule['minimum']}")
                if "maximum" in rule and val > rule["maximum"]:
                    raise ToolError(f"{cid}: {key} must be at most {rule['maximum']}")
            if key == "state" and val not in (0, 1, 2, 3):
                raise ToolError(f"{cid}: state must be 0=L, 1=H, 2=X or 3=Z")
            if key == "k" and not -1 <= val <= 1:
                raise ToolError(f"{cid}: coupling k must be in [-1, 1]")
            params[key] = val
        c["params"] = params
        if "position" not in c:
            index = len(seen) - 1
            c["position"] = [(index % 4) * .18, (index // 4) * .14, 0.0]
            c["position_source"] = "generated"
        else:
            c.setdefault("position_source", "provided")
        c.setdefault("rotation", [0.0, 0.0, 180.0])
        for field in ("position", "rotation"):
            if field not in c:
                continue
            xyz = c[field]
            if not isinstance(xyz, list) or len(xyz) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xyz):
                raise ToolError(f"{cid}: {field} must be three finite native xyz coordinates (rotation in degrees)")
            c[field] = [float(v) for v in xyz]
    spec["schema"] = "aurex.circuit.v1"
    return spec


def _render_spec(spec: dict[str, Any], measurements: dict[str, Any] | None = None,
                 interaction_states: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
    converted = {"title": spec.get("title", "Aurex circuit"), "components": [],
                 **{k: copy.deepcopy(spec[k]) for k in ("camera", "camera_save") if k in spec}}
    exportable = True
    for c in spec["components"]:
        model = COMPONENTS[c["type"]]
        exportable &= bool(model["model_id"])
        if c["type"] == "digital_input" and c["params"]["state"] not in (0, 1):
            exportable = False
        props = ({prop: c["params"][key] for key, prop in model["pl_props"].items()}
                 if model["model_id"] else dict(c["params"]))
        props.update(model.get("constant_pl_props", {}))
        if c["type"] in ("capacitor", "inductor"):
            props.update({"理想模式": 1, "内阻": 0})
        if c["type"] == "vdc":
            props["内阻"] = 0
        native = {"type": c["type"], "params": c["params"], "pin_labels": model["pin_labels"], "position_source": c.get("position_source", "provided")}
        interaction = (copy.deepcopy(c["interaction"]) if isinstance(c.get("interaction"), dict)
                       else copy.deepcopy(model.get("interaction")))
        if interaction:
            control_id = interaction.get("control_id", c["id"])
            interaction.setdefault("control_id", control_id)
            interaction.setdefault("source_model_id", model.get("model_id") or c["type"])
            interaction.setdefault("role", "component")
            if isinstance(interaction_states, dict) and control_id in interaction_states:
                interaction["current"] = interaction_states[control_id]
            elif "current" not in interaction:
                interaction["current"] = c["params"].get(
                    {"spst": "closed", "voltage_source": "v",
                     "digital_input": "state"}.get(interaction.get("kind"), ""))
            native["interaction"] = interaction
        if measurements and c["id"] in measurements:
            native["measurements"] = measurements[c["id"]]
        converted["components"].append({"id": c["id"], "model_id": model["model_id"] or "Phy-Engine " + c["type"],
            "properties": props, "nodes": c["nodes"], "native": native,
            **{k: c[k] for k in ("position", "rotation", "label") if k in c}})
    return converted, bool(exportable)


def _editable_component_manifest(spec: dict[str, Any], component_ids: list[str] | None = None,
                                 *, limit: int = 8) -> list[dict[str, Any]]:
    """Return a bounded native edit contract for selected components.

    Renderer netlists intentionally expose PhysicsLab-facing ``properties``.
    Those properties are useful for inspection, but they are not the argument
    names accepted by :func:`circuit_edit` (which edits the native PE spec).
    Keep a separate, compact manifest so a focused inspect/edit/analyze result
    remains self-sufficient after context compaction: the model can preserve
    pin/node order and change a real native parameter without reopening a
    renderer JSON sidecar or guessing a parameter name.  It is intentionally
    bounded; echoing every native component of a large imported circuit made
    the supposedly compact tool result larger than the original query.

    ``pl_source`` is deliberately reduced to provenance flags.  Imported SAV
    components may carry raw property/statistic dictionaries that are large and
    audit-oriented; those remain in the immutable artifact.
    """
    if type(limit) is not int or not 1 <= limit <= 24:
        raise ToolError("edit manifest limit must be 1..24")
    components = [component for component in spec.get("components", [])
                  if isinstance(component, dict)]
    by_id = {str(component.get("id")): component for component in components}
    ordered = ([by_id[cid] for cid in component_ids or [] if cid in by_id]
               if component_ids is not None else components)
    rows: list[dict[str, Any]] = []
    for component in ordered[:limit]:
        if not isinstance(component, dict):
            continue
        kind = component.get("type")
        model = COMPONENTS.get(kind, {}) if isinstance(kind, str) else {}
        row: dict[str, Any] = {
            "id": component.get("id"),
            "type": kind,
            "nodes": copy.deepcopy(component.get("nodes", [])),
            "params": copy.deepcopy(component.get("params", {})),
        }
        for field in ("label",):
            if field in component and component[field] not in (None, ""):
                row[field] = copy.deepcopy(component[field])
        pin_labels = model.get("pin_labels") if isinstance(model, dict) else None
        if isinstance(pin_labels, list):
            row["pin_labels"] = copy.deepcopy(pin_labels)
        interaction = component.get("interaction")
        if isinstance(interaction, dict) and interaction:
            # Only retain fields that are meaningful to a subsequent
            # tr_interactions call; do not copy arbitrary importer metadata.
            compact_interaction = {
                key: copy.deepcopy(interaction[key]) for key in (
                    "control_id", "kind", "value_name", "current", "allowed",
                    "minimum", "maximum", "momentary", "source_model_id",
                ) if key in interaction
            }
            if compact_interaction:
                row["interaction"] = compact_interaction
        source = component.get("pl_source")
        if isinstance(source, dict):
            provenance = {}
            for key in ("parent_identifier", "model_id", "is_helper",
                        "decomposition_role", "saved_is_broken", "source_ref"):
                if key in source:
                    provenance[key] = copy.deepcopy(source[key])
            if provenance:
                row["source"] = provenance
        rows.append(row)
    return rows


def _attach_edit_contract(output: dict[str, Any], spec: dict[str, Any],
                          component_ids: list[str] | None = None) -> None:
    """Attach only the edit facts needed for the currently visible targets."""
    total = len(spec.get("components", []))
    rows = _editable_component_manifest(spec, component_ids)
    output["edit_contract"] = {
        "components": rows,
        "shown": len(rows),
        "native_component_count": total,
        "omitted": max(0, total - len(rows)),
        "usage": ("Use circuit_edit(path=circuit_path, operations=[...]) with these exact native IDs, "
                  "nodes and parameter names, then circuit_analyze on the returned circuit_path. "
                  "For another component, request that exact id/ref with circuit_inspect; do not read a raw netlist."),
    }


def _source_ref_ids(spec: dict[str, Any], source_ref: str) -> list[str]:
    """Resolve a stable original PLSAV C-ref after native import expansion."""
    result = []
    for component in spec.get("components", []):
        source = component.get("pl_source") if isinstance(component, dict) else None
        if isinstance(source, dict) and source.get("source_ref") == source_ref:
            cid = component.get("id")
            if isinstance(cid, str) and cid not in result:
                result.append(cid)
    return result


def _annotate_source_refs(output: dict[str, Any], spec: dict[str, Any]) -> None:
    """Expose stable original refs beside revision-local renderer refs."""
    refs = {}
    for component in spec.get("components", []):
        if not isinstance(component, dict) or not isinstance(component.get("id"), str):
            continue
        source = component.get("pl_source")
        if isinstance(source, dict) and isinstance(source.get("source_ref"), str):
            refs[component["id"]] = source["source_ref"]
    for component in output.get("netlist", {}).get("components", []):
        if component.get("id") in refs:
            component["source_ref"] = refs[component["id"]]
    if refs:
        output["reference_semantics"] = {
            "source_ref": "stable C-number from the original imported PLSAV",
            "ref": "revision-local display number; do not reuse it across original/native/state paths",
            "stable_lookup": "On an imported native/state path, query the original source_ref to resolve its stable component identity.",
        }


def _direction(dx: float, dy: float) -> str:
    """Describe a saved top-view displacement without exposing bare xyz."""
    ax, ay = abs(dx), abs(dy)
    if max(ax, ay) <= 1e-12:
        return "same_position"
    if ax >= ay * 2:
        return "right" if dx > 0 else "left"
    if ay >= ax * 2:
        return "above" if dy > 0 else "below"
    return ("upper_right" if dx > 0 and dy > 0 else
            "lower_right" if dx > 0 else
            "upper_left" if dy > 0 else "lower_left")


def _spatial_context(data: dict[str, Any], primary_ids: list[str], *,
                     primary_limit: int = 4, neighbor_limit: int = 4) -> dict[str, Any] | None:
    """Return bounded, explicit relative-position facts for focused objects.

    PhysicsLab coordinates are meaningful only after projection.  A raw
    ``position=[x,y,z]`` forces a text model to mentally reconstruct the scene
    and led to repeated scans in real community tasks.  These relations use
    the saved top view, state their non-electrical scope, and include exact
    shared nodes so proximity is never mistaken for connectivity.
    """
    components = [row for row in data.get("components", [])
                  if isinstance(row, dict) and isinstance(row.get("position"), list)
                  and len(row["position"]) >= 2]
    by_id = {str(row.get("id")): row for row in components}
    primary = [by_id[cid] for cid in primary_ids if cid in by_id][:primary_limit]
    if not primary:
        return None
    rows = []
    for source in primary:
        sx, sy = float(source["position"][0]), float(source["position"][1])
        source_nodes = {pin.get("node") for pin in source.get("pins", [])
                        if isinstance(pin, dict) and isinstance(pin.get("node"), str)}
        candidates = []
        for target in components:
            if target is source:
                continue
            tx, ty = float(target["position"][0]), float(target["position"][1])
            dx, dy = tx - sx, ty - sy
            target_nodes = {pin.get("node") for pin in target.get("pins", [])
                            if isinstance(pin, dict) and isinstance(pin.get("node"), str)}
            candidates.append((math.hypot(dx, dy), str(target.get("id")), {
                "id": target.get("id"), "ref": target.get("ref"),
                "type": target.get("type"), "label": target.get("label", ""),
                "direction": _direction(dx, dy),
                "distance_native_xy": round(math.hypot(dx, dy), 9),
                "shared_nodes": sorted(source_nodes & target_nodes),
                "electrically_connected": bool(source_nodes & target_nodes),
            }))
        candidates.sort(key=lambda item: (item[0], item[1]))
        rows.append({
            "id": source.get("id"), "ref": source.get("ref"),
            "type": source.get("type"),
            "nearest": [item[2] for item in candidates[:neighbor_limit]],
        })
    return {
        "projection": "saved_top_view",
        "relations": rows,
        "semantics": ("left/right/above/below are derived from saved x/y positions in top view. "
                      "Distance is layout distance, not wire length. Only shared_nodes/electrically_connected "
                      "is connectivity evidence. Request with_image=true for a focused schematic when a visual relation remains ambiguous."),
    }


def _renderer(runtime: ToolRuntime) -> str:
    _ensure_artifacts(runtime)
    path = Path(_resolve(runtime, runtime.config.phy_engine.cmake_build_dir)) / ("circuit_view.exe" if os.name == "nt" else "circuit_view")
    if not path.is_file():
        raise ToolError("circuit_view is missing; run phy_engine_build to build the C++ renderer")
    return str(path)


def _compact_view(data: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Bound model context independently of scene size; raw JSON stays an artifact."""
    overview = bool(summary.get("camera", {}).get("overview"))
    components = data["components"]
    counts = Counter(c["type"] for c in components)
    selection = summary.get("selection") or {}
    if overview:
        # A type-diverse index is useful for discovery; the first eight elements
        # in a synthesized CPU are often eight indistinguishable Logic Inputs.
        chosen, seen = [], set()
        for c in components:
            if c["type"] not in seen:
                chosen.append(c)
                seen.add(c["type"])
            if len(chosen) == 8:
                break
    else:
        shown = set(summary["visible_ids"])
        index = {c["id"]: c for c in components}
        order = [*selection.get("primary_ids", []), *selection.get("neighbor_ids", [])] or [c["id"] for c in components if c["id"] in shown]
        chosen = [index[cid] for cid in order if cid in shown][:8]
    compact = []
    for c in chosen:
        # Preserve exact saved pose for fidelity and edit workflows, but never
        # leave a text consumer with bare coordinates alone: focused results
        # also carry explicit relative-position semantics below.
        item = {key: c[key] for key in ("id", "ref", "type", "label", "properties", "position", "rotation", "position_source") if key in c}
        if selection:
            item["selection_role"] = "primary" if c["id"] in selection["primary_ids"] else "neighbor"
        saved_labels = _saved_pin_labels(c)
        item["pins"] = []
        for p in c["pins"][:32]:
            pin = {key: p[key] for key in ("pin", "label", "node") if key in p}
            if "label" not in pin and type(p.get("pin")) is int and p["pin"] in saved_labels:
                pin["label"] = saved_labels[p["pin"]]
            item["pins"].append(pin)
        if saved_labels:
            item["pin_semantics_source"] = "Aurex faithful PLSAV-to-Phy-Engine import mapping"
        if c.get("native", {}).get("measurements") is not None:
            item["native"] = {key: c["native"][key] for key in ("type", "measurements") if key in c["native"]}
        item["pin_count"] = len(c["pins"])
        if len(c["pins"]) > 32:
            item["pins_truncated"] = True
        compact.append(item)
    shown = {c["id"] for c in chosen}
    nodes = []
    if not overview:
        for node in data["nodes"]:
            local = [p for p in node["connections"] if p["component"] in shown]
            if local:
                nodes.append({"id": node["id"], "connections": local[:32],
                              "total_connections": len(node["connections"]),
                              "external_connections": len(node["connections"]) - len(local),
                              "connections_truncated": len(local) > 32})
            if len(nodes) == 64:
                break
    raw_camera = summary.get("camera", data.get("camera", {}))
    camera = {key: raw_camera[key] for key in (
        "source", "projection", "position", "target", "rotation", "distance", "zoom", "fov_y_deg",
        "orthographic_height", "near_plane", "fit", "saved_raw", "warnings", "assumptions",
        "width", "height", "overview", "total_components", "rendered_components", "occlusion_possible",
        "label_overlap_count", "labels_omitted_for_overview", "legend_components", "omitted_legend_components",
        "primary_cluster_component_count", "viewport_is_subset", "external_connections",
        "frame_pan_pixels", "image_generated", "schematic", "rendered_nodes", "routed_nodes",
        "junction_count", "unconnected_pin_count", "external_connection_stub_count",
        "geometry_mutated", "layout_source", "routing", "requested_camera_ignored",
    ) if key in raw_camera}
    if "spatial_outliers" in raw_camera:
        camera["spatial_outliers"] = raw_camera["spatial_outliers"][:8]
        camera["spatial_outlier_count"] = len(raw_camera["spatial_outliers"])
    for key in ("clipped_component_ids", "clipped_components", "behind_camera"):
        if key in raw_camera:
            camera[key] = raw_camera[key][:8]
            camera[key + "_count"] = len(raw_camera[key])
    pagination = {key: summary[key] for key in ("components", "nodes", "offset", "limit", "view", "projection", "focused", "match_count") if key in summary}
    pagination.update({"next_offset": selection.get("next_offset") if selection else None if overview else
                       summary["offset"] + summary["limit"] if summary["offset"] + summary["limit"] < len(components) else None,
                       "text_components": len(compact), "text_components_truncated": not overview and len(summary["visible_ids"]) > len(compact)})
    if selection:
        pagination.update(selection)
    output = {"netlist": {"components": compact, "nodes": nodes,
                        "statistics_source": data["statistics_source"],
                        "scope": selection["scope"] if selection else "type-diverse examples, not the complete netlist" if overview else "selected components; connections to unselected components counted but not repeated"},
            "statistics": {"components": len(components), "wires": len(data["wires"]), "nodes": len(data["nodes"]),
                           "component_types": dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))},
            "pagination": pagination, "camera": camera,
            "detail_hint": "For digital I/O IDs, labels and recorded values use circuit_inspect(interface_only=true), not a scan of the full internal netlist. To follow any displayed N<number> node, call circuit_inspect again on the SAME original circuit/state path with query set to that exact node; offset/limit pages its connected components. Never use read_context/find on the netlist artifact for an exact node. Explicit component IDs/C<number> refs can likewise be focused directly. Request with_image=true only when visual/spatial evidence is needed. Do not enumerate every component merely for a general introduction."}
    if selection:
        spatial = _spatial_context(data, [str(value) for value in selection.get("primary_ids", [])])
        if spatial:
            output["spatial_context"] = spatial
    return output


def _view(runtime: ToolRuntime, source: Path, *, create: bool = False, exportable: bool = False,
          offset: int = 0, limit: int = 8, view: str = "spatial", projection: str = "isometric",
          focus_ids: list[str] | None = None, query: str = "", camera: dict | None = None,
          state: bool = False, _publication_cover: bool = False, with_image: bool = True) -> dict[str, Any]:
    if _publication_cover:
        with_image = True
    if type(with_image) is not bool:
        raise ToolError("with_image must be boolean")
    if offset < 0 or not 1 <= limit <= 24:
        raise ToolError("offset must be >= 0 and limit must be 1..24")
    folder = _artifact_dir(runtime)
    svg, netlist, png = folder / "circuit.svg", folder / "netlist.json", folder / "circuit.png"
    command = [_renderer(runtime), "state" if state else "create" if create else "render", str(source), str(svg), str(netlist), str(offset), str(limit)]
    if exportable:
        command.append(str(folder / "circuit.sav"))
    else:
        command.append("-")
    camera_options = {"overview": True} if _publication_cover else _camera(camera)
    # A displayed C<n> token is an exact selectable reference, not a substring
    # of UUID/type/properties. Reuse the renderer's ID > ref > label resolver.
    if re.fullmatch(r"C[1-9][0-9]*", query) and not focus_ids and not _publication_cover:
        focus_ids, query = [query], ""
    # Node names are connectivity selectors, not component text. Resolve them
    # from one lossless data-only render, then reuse the renderer's exact-ID
    # focus path. This avoids paging a million-character netlist just to find
    # the drivers and loads of N24. The original source is never modified.
    if re.fullmatch(r"N(?:0|[1-9][0-9]*)", query) and not focus_ids and not _publication_cover:
        node_query = query
        discovery = _view(runtime, source, create=create, exportable=False, offset=0, limit=8,
                          view="overview", projection=projection, focus_ids=None, query="",
                          camera=camera, state=state, with_image=False)
        full = json.loads(Path(discovery["artifact"]["netlist_path"]).read_text())
        matches = [c["id"] for c in full.get("components", []) if any(
            isinstance(pin, dict) and pin.get("node") == node_query for pin in c.get("pins", []))]
        # The focused renderer intentionally shows at most eight primary
        # components per view, even when its general query limit is larger.
        # Advance by the number it can actually return; otherwise limit=16/24
        # silently skips connected components between pages.
        requested_limit = limit
        page_limit = min(limit, 8)
        page = matches[offset:offset + page_limit]
        if not page:
            raise ToolError(f"No components connect to exact node {node_query}; no substring or inferred node was used")
        selected = _view(runtime, source, create=create, exportable=exportable, offset=0, limit=len(page),
                         view=view, projection=projection, focus_ids=page, query="", camera=camera,
                         state=state, _publication_cover=False, with_image=with_image)
        selected["node_query"] = {"node": node_query, "exact": True, "match_count": len(matches),
                                  "offset": offset, "limit": page_limit,
                                  "requested_limit": requested_limit,
                                  "next_offset": offset + len(page) if offset + len(page) < len(matches) else None,
                                  "scope": "components whose saved/native pin node exactly equals this node"}
        selected["pagination"].update({"query": node_query, "match_count": len(matches), "offset": offset,
                                         "limit": page_limit, "requested_limit": requested_limit,
                                         "next_offset": selected["node_query"]["next_offset"]})
        selected["warnings"].append("Node query is exact connectivity evidence; it does not identify driver direction or prove dynamic behavior by itself.")
        if requested_limit > page_limit:
            selected["warnings"].append(
                f"Exact node views return at most {page_limit} connected components per page; "
                "follow next_offset so no connection is skipped.")
        return selected
    command += [view, projection, json.dumps(focus_ids or []), query, json.dumps(camera_options, allow_nan=False)]
    command += [json.dumps({"with_image": with_image})]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ToolError("C++ circuit renderer failed: " + str(getattr(error, "stderr", "") or error)[-2000:]) from error
    data = json.loads(netlist.read_text())
    summary = json.loads(result.stdout)
    if not with_image and summary.get("with_image") is not False:
        raise ToolError("circuit_view must be rebuilt to support with_image=false without rendering")
    if with_image:
        try:
            import cairosvg
            # C++ SVG contains no external images, URLs, fonts, or scripts.
            cairosvg.svg2png(url=str(svg), write_to=str(png))
        except (ImportError, OSError) as error:
            raise ToolError("PNG rendering requires cairosvg and the system Cairo library: " + str(error)) from error
    output = {
        "artifact": {"netlist_path": str(netlist), "external_write_performed": False,
                     **({"svg_path": str(svg), "png_path": str(png)} if with_image else {})},
        "images": [{"path": str(png), "mime_type": "image/png"}] if with_image else [],
        "with_image": with_image,
        **_compact_view(data, summary),
        "warnings": data["warnings"][:32],
        "state_source": "recorded native PE snapshot; not a new simulation" if state else "native design scene" if create else "original PLSAV",
        "evidence": ("C++ Phy-Engine .sav loader / component state. The position-aware schematic uses original native xyz "
                     "only to form non-overlapping spatial ranks and routes exact saved/native nodes; it never changes the circuit. "
                     "Spatial view preserves native xyz and Euler rotation. Connectivity/values come from saved state, while "
                     "fresh functional claims still require circuit_analyze. Coordinates are generated only when missing."),
    }
    # Locator geometry/complete ID lists stay in the lossless camera artifact.
    # Expose only a bounded orientation summary to the model.
    minimap = summary.get("camera", {}).get("minimap")
    if minimap:
        output["camera"]["minimap"] = {key: minimap[key] for key in (
            "enabled", "total_components", "highlighted_components", "wire_total", "wire_drawn",
            "projection", "rough", "geometry_mutated", "note") if key in minimap}
        output["camera"]["minimap"]["highlighted_refs_sample"] = minimap.get("highlighted_refs", [])[:8]
    if exportable:
        output["sav_path"] = str(folder / "circuit.sav")
    view_path = folder / "camera-view.json"
    _write(view_path, {"source_path": str(source), "source_kind": output["state_source"],
                       "requested_camera": _camera(camera), "camera": summary.get("camera", data.get("camera", {})),
                       "view": view, "projection": projection, "focus_ids": focus_ids or [],
                       "query": query, "offset": offset, "limit": limit})
    output["artifact"]["camera_path"] = str(view_path)
    if (with_image and view in {"auto", "overview"} and not _publication_cover and summary.get("camera", {}).get("overview")
            and summary["camera"].get("primary_cluster_component_count", summary["components"]) < summary["components"]):
        # Preserve a complete scene image even when genuinely distant saved
        # components make it uninformative. A second explicitly partial image
        # is a view-only crop; never discard or move those original components.
        region = _view(runtime, source, create=create, state=state, view="region", camera=camera)
        output["images"] = region["images"] + output["images"]
        output["artifact"]["primary_viewport"] = region["artifact"]
        output["primary_viewport"] = {"camera": region["camera"], "scope": "First image is the main spatial cluster only; second image is the complete original scene including distant elements. Neither image rearranges components."}
    return output


def publication_cover(runtime: ToolRuntime, sav_path: str) -> dict[str, Any]:
    """Server-only fixed-angle full-scene cover; never exposed as agent options."""
    from PIL import Image
    source, value = _input(runtime, sav_path)
    if value.get("schema") in {"aurex.pe-state.v1", "aurex.circuit.v1"}:
        raise ToolError("Publication requires an actual exported PLSAV, not a native-only state")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = _view(runtime, source, _publication_cover=True)
    camera = result["camera"]
    total, visible = camera.get("total_components"), camera.get("rendered_components")
    clipped = camera.get("clipped_component_ids")
    if (camera.get("source") != "fixed-publication-overview" or type(total) is not int or
            total <= 0 or visible != total or clipped != []):
        raise ToolError("Publication cover failed its all-components-in-frame check")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ToolError("Circuit changed while rendering its publication cover")
    target = Path(result["artifact"]["png_path"]).with_name("publication-cover.jpg")
    with Image.open(result["artifact"]["png_path"]) as image:
        rgb = image.convert("RGB")
        for quality in (92, 85, 75):
            rgb.save(target, "JPEG", quality=quality, optimize=True)
            if target.stat().st_size <= 1024**2:
                break
    if target.stat().st_size > 1024**2:
        raise ToolError("Publication cover exceeds the 1 MiB upload limit")
    manifest = {"source_sha256": digest, "cover_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "rendered_all": True, "total_elements": total, "visible_elements": visible,
                "clipped_ids": clipped, "occlusion_possible": True,
                "view": {"yaw": 45, "pitch": 60, "projection": "orthographic", "fit": "all"}}
    _write(target.with_suffix(".manifest.json"), manifest)
    return {"cover_path": str(target), "cover_manifest": manifest, "camera": camera,
            "images": [{"path": str(target), "mime_type": "image/jpeg"}]}


def circuit_catalog(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    return {"components": {name: {k: v for k, v in c.items() if k != "code"} for name, c in COMPONENTS.items()},
        "pin_order": "nodes[i] connects pin i; two equal node names are electrically connected; gnd/0 is the voltage reference",
        "units": "SI (ohm, F, H, V, A), freq_hz in Hz, phase_deg in degrees, temp_c in Celsius",
        "analysis": ["op", "dc", "ac", "acop", "tr", "trop"],
        "component_limits": {"analog_or_mixed": 512, "pure_native_digital": runtime.config.phy_engine.digital_component_limit,
                             "classification": "server catalog native ABI model types; component count is after import expansion"},
        "transient": "tr_step/tr_stop are seconds; at most 10000 solver steps. tr_sample_every optionally records actual per-pin samples every N steps and at the exact endpoint (at most 201 samples). Sampling does not invent values between points; use finer step/sample spacing to investigate convergence and switching.",
        "export": "model_id empty means native Phy-Engine simulation only; not exported as a PhysicsLab .sav",
        "example": {"components": [{"id": "V1", "type": "vdc", "nodes": ["supply", "gnd"], "params": {"v": 5}},
                                     {"id": "R1", "type": "resistor", "nodes": ["supply", "gnd"], "params": {"r": 10}}]}}


def _interface_result(result: dict[str, Any], offset: int, limit: int) -> dict[str, Any]:
    data = json.loads(Path(result["artifact"]["netlist_path"]).read_text())
    node_connection_counts = {
        node.get("id"): len(node.get("connections", []))
        for node in data.get("nodes", []) if isinstance(node, dict)
    }
    ports = []
    for component in data["components"]:
        kind = component["type"]
        if kind not in ("Logic Input", "Logic Output"):
            continue
        measured = (component.get("native") or {}).get("measurements")
        if measured is not None:
            values = measured.get("digital") or []
            logic = values[0] if len(values) == 1 else None
            source = "recorded native PE snapshot, not a new solve"
        elif kind == "Logic Input":
            logic = component.get("properties", {}).get("开关")
            source = "saved input setting, not a new solve"
        else:
            logic = component.get("statistics", {}).get("状态")
            source = "saved output statistic, not a new solve"
        logic = int(logic) if isinstance(logic, (int, float)) and not isinstance(logic, bool) and logic in (0, 1, 2, 3) else None
        node = component["pins"][0]["node"] if len(component["pins"]) == 1 else None
        connection_count = node_connection_counts.get(node) if node is not None else None
        ports.append({"id": component["id"], "ref": component["ref"], "label": component.get("label", ""),
                      "direction": "input" if kind == "Logic Input" else "output",
                      "node": node, "node_connection_count": connection_count,
                      "connected_to_other_components": connection_count > 1 if connection_count is not None else None,
                      "logic": logic, "logic_text": "LHXZ"[logic] if logic is not None else None,
                      "logic_source": source if logic is not None else "no recorded logic value available"})
    return {"interface_only": True, "with_image": False, "images": [], "ports": ports[offset:offset + limit],
            "total_ports": len(ports), "total_inputs": sum(p["direction"] == "input" for p in ports),
            "total_outputs": sum(p["direction"] == "output" for p in ports), "total_components": len(data["components"]),
            "offset": offset, "limit": limit, "has_more": offset + limit < len(ports),
            "next_offset": offset + limit if offset + limit < len(ports) else None,
            "scope": "Actual Logic Input/Output devices in saved order, with original IDs/labels and exact saved-node connection counts; internal gates and bare coordinates omitted. Use a focused schematic/spatial_context only for a real layout question. A port with connected_to_other_components=false is isolated and cannot stimulate or observe the circuit. This is not an inferred module signature: unlabelled inputs may be constants. No stimulus or waveform is invented.",
            "artifact": result["artifact"],
            **{k: result[k] for k in ("circuit_path", "state_path", "state_source", "measurement_source") if k in result}}


def _controls_result(result: dict[str, Any], offset: int, limit: int) -> dict[str, Any]:
    data = json.loads(Path(result["artifact"]["netlist_path"]).read_text())
    controls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for component in data.get("components", []):
        native = component.get("native") if isinstance(component.get("native"), dict) else {}
        interaction = native.get("interaction") if isinstance(native.get("interaction"), dict) else None
        props = component.get("properties") if isinstance(component.get("properties"), dict) else {}
        if interaction:
            control_id = interaction.get("control_id", component.get("id"))
            descriptor = {"id": control_id, "kind": interaction.get("kind"),
                          "value_name": interaction.get("value_name", "value"),
                          "current": interaction.get("current"),
                          "source_model_id": interaction.get("source_model_id", component.get("type")),
                          "primitive_component_ids": []}
            for key in ("allowed", "minimum", "maximum", "momentary",
                        "rated_resistance_ohm", "minimum_segment_ohm"):
                if key in interaction:
                    descriptor[key] = interaction[key]
        else:
            kind = component.get("type")
            control_id = component.get("id")
            if kind in ("Simple Switch", "Push Switch", "Air Switch"):
                descriptor = {"id": control_id, "kind": "spst",
                    "value_name": "pressed" if kind == "Push Switch" else "closed",
                    "allowed": [0, 1], "current": props.get("开关"),
                    "momentary": kind == "Push Switch", "source_model_id": kind,
                    "primitive_component_ids": []}
            elif kind in ("SPDT Switch", "DPDT Switch"):
                descriptor = {"id": control_id, "kind": "spdt" if kind == "SPDT Switch" else "dpdt",
                    "value_name": "position", "allowed": [0, 1, 2], "current": props.get("开关"),
                    "source_model_id": kind, "primitive_component_ids": []}
            elif kind == "Slide Rheostat":
                descriptor = {"id": control_id, "kind": "slide_rheostat", "value_name": "position",
                    "minimum": 0.0, "maximum": 1.0, "current": props.get("滑块位置"),
                    "rated_resistance_ohm": props.get("额定电阻"), "source_model_id": kind,
                    "primitive_component_ids": []}
            elif kind == "Battery Source":
                descriptor = {"id": control_id, "kind": "voltage_source", "value_name": "voltage_v",
                    "current": props.get("电压"), "finite_number": True, "source_model_id": kind,
                    "primitive_component_ids": []}
            elif kind == "Logic Input":
                descriptor = {"id": control_id, "kind": "digital_input", "value_name": "state",
                    "allowed": [0, 1, 2, 3], "current": props.get("开关"), "source_model_id": kind,
                    "primitive_component_ids": []}
            else:
                continue
        if not isinstance(control_id, str) or not control_id:
            continue
        if control_id not in controls:
            controls[control_id] = descriptor
            order.append(control_id)
        controls[control_id]["primitive_component_ids"].append(component.get("id"))
    rows = [controls[control_id] for control_id in order]
    return {"controls_only": True, "with_image": False, "images": [],
            "controls": rows[offset:offset + limit], "total_controls": len(rows),
            "offset": offset, "limit": limit, "has_more": offset + limit < len(rows),
            "next_offset": offset + limit if offset + limit < len(rows) else None,
            "usage": "Use exact id in circuit_analyze.tr_interactions. These are time-varying physical/model controls; switch and rheostat controls remain analog devices, not fabricated digital ports.",
            "artifact": result["artifact"],
            **{k: result[k] for k in ("circuit_path", "state_path", "state_source", "measurement_source") if k in result}}


def circuit_inspect(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    with_image = args.get("with_image", False)
    interface_only = args.get("interface_only", False)
    controls_only = args.get("controls_only", False)
    if type(with_image) is not bool or type(interface_only) is not bool or type(controls_only) is not bool:
        raise ToolError("with_image, interface_only and controls_only must be boolean")
    if interface_only and controls_only:
        raise ToolError("Choose one data-only listing: interface_only or controls_only")
    listing_only = interface_only or controls_only
    if listing_only and with_image:
        raise ToolError("Data-only listings are mutually exclusive with with_image=true")
    if listing_only and any(args.get(k) for k in ("focus_id", "focus_ids", "query", "camera", "view", "projection")):
        raise ToolError("interface_only/controls_only use original order; omit focus/query/camera/view/projection")
    source, value = _input(runtime, str(args.get("path") or ""))
    offset, limit = args.get("offset", 0), args.get("limit", 64 if listing_only else 8)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= (64 if listing_only else 24):
        raise ToolError("Inspection offset must be nonnegative; limit must be 1..64 for interface_only/controls_only, otherwise 1..24")
    page_offset, page_limit = offset, limit
    if listing_only:
        offset, limit = 0, 8
    targeted = bool(any(args.get(k) for k in ("focus_id", "focus_ids", "query")))
    # A requested image of a focused object should be an actual wired
    # schematic.  Data-only focus keeps the cheaper spatial selector and
    # returns explicit left/right/adjacency facts.
    default_view = ("schematic" if with_image and targeted else "spatial"
                    if targeted or args.get("offset", 0) > 0 else "auto")
    view, projection = str(args.get("view", default_view)), str(args.get("projection", "isometric"))
    focus = args.get("focus_ids", [args["focus_id"]] if args.get("focus_id") else [])
    query = str(args.get("query", ""))
    camera = _camera(args.get("camera"))
    known_spec = None
    if value.get("schema") == "aurex.pe-state.v1":
        known_spec = normalize_spec(
            value["spec"], digital_component_limit=runtime.config.phy_engine.digital_component_limit)
    elif value.get("schema") == "aurex.circuit.v1":
        known_spec = normalize_spec(
            value, digital_component_limit=runtime.config.phy_engine.digital_component_limit)
    source_ref_query = None
    source_ref_ids = None
    source_ref_offset = offset
    if known_spec is not None and re.fullmatch(r"C[1-9][0-9]*", query) and not focus:
        stable_ids = _source_ref_ids(known_spec, query)
        if stable_ids:
            source_ref_query, source_ref_ids, query = query, stable_ids, ""
            focus = stable_ids[offset:offset + limit]
            if not focus:
                raise ToolError(f"Inspection offset {offset} exceeds {len(stable_ids)} native matches for {source_ref_query}")
            # The renderer requires focus_ids to fit inside its page limit.
            # Slice the stable alias here and expose the complete alias count
            # below instead of silently resolving the revision-local C-ref.
            offset = 0

    def attach_source_ref_resolution(result: dict[str, Any]) -> None:
        if not source_ref_query or source_ref_ids is None:
            return
        total = len(source_ref_ids)
        result["source_ref_query"] = {
            "source_ref": source_ref_query,
            "resolved_native_ids": copy.deepcopy(source_ref_ids),
            "shown_native_ids": copy.deepcopy(focus),
            "stable_across_import_revisions": True,
        }
        pagination = result.setdefault("pagination", {})
        pagination.update({
            "primary_ids": copy.deepcopy(focus),
            "offset": source_ref_offset, "limit": limit,
            "total_matches": total, "match_count": len(focus),
            "has_more": source_ref_offset + len(focus) < total,
            "next_offset": (source_ref_offset + len(focus)
                            if source_ref_offset + len(focus) < total else None),
        })
    if value.get("schema") == "aurex.pe-state.v1":
        output = _view(runtime, source, state=True, offset=offset, limit=limit, view=view, projection=projection, focus_ids=focus, query=query, camera=camera, with_image=with_image)
        output["circuit_path"] = str(source)
        output["state_path"] = str(source)
        # A recorded state carries the exact native spec used by the solver.
        # Return only the currently visible targets' writable fields.
        try:
            state_spec = known_spec
            _annotate_source_refs(output, state_spec)
            visible_ids = [str(row.get("id")) for row in output.get("netlist", {}).get("components", [])]
            if not args.get("_skip_edit_contract", False):
                _attach_edit_contract(output, state_spec, visible_ids)
        except (KeyError, ToolError):
            pass
        output["measurement_source"] = "recorded PE state; changing camera does not execute the solver"
        if isinstance(value.get("protection_summary"), dict):
            output["protection_summary"] = copy.deepcopy(value["protection_summary"])
        attach_source_ref_resolution(output)
        return (_interface_result(output, page_offset, page_limit) if interface_only
                else _controls_result(output, page_offset, page_limit) if controls_only else output)
    if value.get("schema") == "aurex.circuit.v1":
        spec = known_spec
        rendered_spec, exportable = _render_spec(spec)
        rendered = _artifact_dir(runtime) / "render-input.json"
        _write(rendered, rendered_spec)
        output = _view(runtime, rendered, create=True, exportable=exportable and not listing_only, offset=offset, limit=limit, view=view, projection=projection, focus_ids=focus, query=query, camera=camera, with_image=with_image)
        _annotate_source_refs(output, spec)
        visible_ids = [str(row.get("id")) for row in output.get("netlist", {}).get("components", [])]
        if not args.get("_skip_edit_contract", False):
            _attach_edit_contract(output, spec, visible_ids)
    else:
        output = _view(runtime, source, offset=offset, limit=limit, view=view, projection=projection, focus_ids=focus, query=query, camera=camera, with_image=with_image)
        # Import an editable native workspace only for a targeted request.  An
        # overview remains one fast renderer call and cannot echo the full
        # imported circuit back into context.
        if targeted and not args.get("_skip_edit_contract", False):
            editable = _load_spec(runtime, str(source))
            _annotate_source_refs(output, editable)
            visible_ids = [str(row.get("id")) for row in output.get("netlist", {}).get("components", [])]
            _attach_edit_contract(output, editable, visible_ids)
    output["circuit_path"] = str(source)
    attach_source_ref_resolution(output)
    return (_interface_result(output, page_offset, page_limit) if interface_only
            else _controls_result(output, page_offset, page_limit) if controls_only else output)


_QUERY_MANY_SIMPLE_FIELDS = {"pins", "native_type", "spatial"}
_QUERY_MANY_EXACT_FIELD = re.compile(
    r"^(?:properties|measurements|edit)\.[^.]{1,64}(?:\.[^.]{1,64})*$")


def _query_many_nested_value(value: Any, path: list[str]) -> tuple[bool, Any]:
    """Read one explicitly named nested field without widening the result."""
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, copy.deepcopy(current)


def _query_many_set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    current = target
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _query_many_pins(component: dict[str, Any],
                     node_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pins = []
    for pin in component.get("pins", []):
        projected = {key: copy.deepcopy(pin[key]) for key in ("pin", "label", "node")
                     if key in pin}
        node = node_index.get(str(pin.get("node")))
        if node is not None:
            projected["total_connections"] = node.get("total_connections")
            projected["connected"] = bool((node.get("total_connections") or 0) > 1)
        pins.append(projected)
    return pins


def circuit_query_many(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Inspect exact circuit targets while returning only requested fields.

    Target selection and field selection are deliberately independent.  A
    default call returns identity/match evidence only.  Callers must name each
    property, measurement or editable parameter they need; only ``all=true``
    requests the complete matched component records.
    """
    path = args.get("path")
    queries = args.get("queries")
    limit = args.get("limit", 1)
    fields = args.get("fields", [])
    include_all = args.get("all", False)
    if not isinstance(path, str) or not path:
        raise ToolError("circuit_query_many requires path")
    if (not isinstance(queries, list) or not 1 <= len(queries) <= 24
            or any(not isinstance(query, str) or not query.strip() or len(query) > 128
                   for query in queries)):
        raise ToolError("queries must contain 1..24 nonempty strings of at most 128 characters")
    normalized = [query.strip() for query in queries]
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ToolError("circuit_query_many limit must be 1..8 per query")
    if type(include_all) is not bool:
        raise ToolError("circuit_query_many all must be boolean")
    if (not isinstance(fields, list) or len(fields) > 24 or
            any(not isinstance(field, str) or not field.strip() or len(field) > 196
                for field in fields)):
        raise ToolError("circuit_query_many fields must contain at most 24 nonempty selectors")
    selected_fields = [field.strip() for field in fields]
    if len(set(selected_fields)) != len(selected_fields):
        raise ToolError("circuit_query_many fields must be unique")
    invalid_fields = [field for field in selected_fields
                      if field not in _QUERY_MANY_SIMPLE_FIELDS and
                      not _QUERY_MANY_EXACT_FIELD.fullmatch(field)]
    if invalid_fields:
        raise ToolError(
            "Unsupported circuit_query_many field selector(s): " + repr(invalid_fields) +
            ". Use pins, native_type, spatial, properties.<exact-name>, "
            "measurements.<exact-path>, edit.<exact-param>, or all=true.")
    if include_all and selected_fields:
        raise ToolError("circuit_query_many all=true is mutually exclusive with fields")
    if "spatial_context" in args:
        raise ToolError(
            "circuit_query_many spatial_context was replaced by fields=['spatial']; "
            "request spatial data only when it is actually needed")

    include_spatial = include_all or "spatial" in selected_fields
    include_edit = include_all or any(field.startswith("edit.") for field in selected_fields)
    identity_fields = ("id", "ref", "source_ref", "type", "label")

    rows = []
    circuit_path = None
    state_path = None
    for query in normalized:
        exact_node_query = bool(re.fullmatch(r"N\d+", query, re.IGNORECASE))
        try:
            result = circuit_inspect(runtime, {
                "path": path, "query": query, "limit": limit, "with_image": False,
                "_skip_edit_contract": not include_edit,
            })
        except ToolError as error:
            # A batch is a set of independent lookups.  One stale/mistyped
            # node must not discard every valid answer or force the model to
            # resend the remaining queries.
            rows.append({"query": query, "ok": False, "error": str(error)})
            continue
        if circuit_path is None:
            circuit_path = result.get("circuit_path")
            state_path = result.get("state_path")
        netlist = result.get("netlist") or {}
        pagination = result.get("pagination") or {}
        primary_ids = {str(value) for value in pagination.get("primary_ids", [])}
        node_index = {str(node.get("id")): node for node in netlist.get("nodes", [])
                      if isinstance(node, dict) and node.get("id") is not None}
        edit_rows = ((result.get("edit_contract") or {}).get("components", [])
                     if isinstance(result.get("edit_contract"), dict) else [])
        edit_by_id = {str(item.get("id")): item for item in edit_rows
                      if isinstance(item, dict) and item.get("id") is not None}
        component_ids = []
        selected_components = []
        for component in netlist.get("components", []):
            component_id = str(component.get("id") or "")
            if not component_id or (primary_ids and component_id not in primary_ids):
                continue
            component_ids.append(component_id)
            if include_all:
                compact = copy.deepcopy(component)
                compact["pins"] = _query_many_pins(component, node_index)
                edit_row = edit_by_id.get(component_id)
                if edit_row is not None:
                    compact["edit"] = copy.deepcopy(edit_row)
                selected_components.append(compact)
                continue

            compact = {key: copy.deepcopy(component[key]) for key in identity_fields
                       if key in component}
            if exact_node_query:
                matching_pins = [
                    {key: copy.deepcopy(pin[key]) for key in ("pin", "label", "node")
                     if key in pin}
                    for pin in component.get("pins", [])
                    if str(pin.get("node", "")).casefold() == query.casefold()
                ]
                if matching_pins:
                    compact["matched_pins"] = matching_pins
            if "pins" in selected_fields:
                compact["pins"] = _query_many_pins(component, node_index)
            native = component.get("native")
            missing = []
            if "native_type" in selected_fields:
                if isinstance(native, dict) and "type" in native:
                    compact["native_type"] = copy.deepcopy(native["type"])
                else:
                    missing.append("native_type")
            for selector in selected_fields:
                if selector in _QUERY_MANY_SIMPLE_FIELDS:
                    continue
                root, *nested = selector.split(".")
                if root == "properties":
                    source = component.get("properties")
                elif root == "measurements":
                    source = native.get("measurements") if isinstance(native, dict) else None
                else:
                    source = (edit_by_id.get(component_id) or {}).get("params")
                found, value = _query_many_nested_value(source, nested)
                if found:
                    _query_many_set_nested(compact, [root, *nested], value)
                else:
                    missing.append(selector)
            if missing:
                compact["missing_fields"] = missing
            selected_components.append(compact)

        # An exact node selector itself asks which components touch that node,
        # so retain the node identity/count.  Its complete connection records
        # are still field-controlled by pins/all.
        exact_nodes = []
        for node in netlist.get("nodes", []):
            if str(node.get("id", "")).casefold() != query.casefold():
                continue
            node_fields = ("id", "total_connections", "connections", "connections_truncated") \
                if include_all or "pins" in selected_fields else ("id", "total_connections")
            exact_nodes.append({key: copy.deepcopy(node[key]) for key in node_fields if key in node})
        row = {
            "query": query, "ok": True,
            "component_ids": component_ids,
            "components": selected_components,
            "match_count": pagination.get("total_matches", pagination.get("match_count", len(component_ids))),
            "has_more": bool(pagination.get("has_more")),
        }
        if exact_nodes:
            row["nodes"] = exact_nodes
        if pagination.get("next_offset") is not None:
            row["next_offset"] = pagination.get("next_offset")
        if include_spatial and isinstance(result.get("spatial_context"), dict):
            row["spatial"] = copy.deepcopy(result["spatial_context"])
        elif "spatial" in selected_fields:
            row["missing_fields"] = ["spatial"]
        rows.append({
            **row,
        })
    output = {
        "batch": True, "with_image": False, "query_count": len(rows),
        "successful_query_count": sum(row.get("ok") is True for row in rows),
        "failed_query_count": sum(row.get("ok") is False for row in rows),
        "limit_per_query": limit,
        "selected_fields": ["all"] if include_all else ["identity", *selected_fields],
        "results": rows,
        "scope": ("Read-only exact matches. Identity and match evidence are always returned; every other value "
                  "appears only when named in fields, or when all=true. missing_fields means the exact requested "
                  "field is absent on that component. Repeated queries are allowed."),
    }
    if circuit_path:
        output["circuit_path"] = circuit_path
    if state_path:
        output["state_path"] = state_path
    return output


def circuit_create(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    spec = normalize_spec(args.get("spec"), digital_component_limit=runtime.config.phy_engine.digital_component_limit)
    path = _artifact_dir(runtime) / "design.circuit.json"
    _write(path, spec)
    result = circuit_inspect(runtime, {"path": str(path), "view": args.get("view", "spatial"), "projection": args.get("projection", "isometric"), "camera": args.get("camera"), "with_image": args.get("with_image", False)})
    # Always return the complete lightweight identity list even when the
    # rendered/netlist rows are projected for context size.  An omitted row is
    # not an absent component and must never prompt a duplicate add operation.
    complete_manifest = [
        {"id": component["id"], "type": component["type"],
         **({"label": component["label"]} if component.get("label") else {})}
        for component in spec["components"]
    ]
    result["component_manifest"] = complete_manifest[:24]
    result["component_manifest_scope"] = {
        "total": len(complete_manifest), "shown": min(24, len(complete_manifest)),
        "omitted": max(0, len(complete_manifest) - 24),
        "lookup": "Use circuit_inspect/circuit_query_many on circuit_path; do not recreate omitted components.",
    }
    visible_ids = [str(row.get("id")) for row in result.get("netlist", {}).get("components", [])]
    _attach_edit_contract(result, spec, visible_ids)
    return result


def _spec_from_sav(runtime: ToolRuntime, path: Path) -> dict[str, Any]:
    view = _view(runtime, path, with_image=False)
    full = json.loads(Path(view["artifact"]["netlist_path"]).read_text())
    components: list[dict[str, Any]] = []
    used_random_seeds: set[int] = set()
    reserved_ids = {el["id"] for el in full["components"]}
    mapping = {c["model_id"]: (name, c) for name, c in COMPONENTS.items() if c["model_id"]}
    for el in full["components"]:
        if el["type"] == "Ground Component":
            continue
        imported_element = None
        converter_element = el
        if el.get("is_broken", False):
            # Individual audited importers still reject broken devices when
            # called alone. The complete SAV pipeline imports the intact core
            # from a copy, then apply_damage_protection wraps the original
            # topology in an initially-open native guard.
            converter_element = copy.deepcopy(el)
            converter_element["is_broken"] = False
        for converter in (import_analog_element, import_passive_element,
                          import_source_element, import_power_element,
                          import_semiconductor_element, import_device_element):
            imported_element = converter(converter_element, scene=full)
            if imported_element is not None:
                for row in imported_element:
                    row.setdefault("pl_source", {}).setdefault("source_ref", el.get("ref"))
                # Preserve the stable metadata contract used by existing
                # Aurex documents while retaining richer audited provenance.
                resistance_rows = [row for row in imported_element
                    if row.get("pl_source", {}).get("decomposition_role")
                    in {"series_resistance", "parallel_resistance"}]
                if resistance_rows:
                    resistance = resistance_rows[0]["pl_source"]
                    topology = ("series" if resistance["decomposition_role"] == "series_resistance"
                                else "parallel")
                    common = {
                        "source_component_id": el["id"],
                        "source_model_id": el["type"],
                        "saved_property": "内阻",
                        "resistance_ohm": float(el["properties"]["内阻"]),
                        "topology": topology,
                        "original_plsav_unchanged": True,
                    }
                    for row in imported_element:
                        role = row.get("pl_source", {}).get("decomposition_role")
                        if role == "core":
                            row["plsav_import"] = {**common, "is_helper": False, "role": "ideal_core"}
                        elif role == topology + "_resistance":
                            row["plsav_import"] = {**common, "is_helper": True,
                                "role": topology + "_internal_resistance"}
                components.extend(imported_element)
                break
        if imported_element is not None:
            continue
        # Native inspection exposes absent/null PL labels as an empty display
        # string. Only carry an actual nonempty name into the native spec;
        # unlabelled components remain unlabelled, including helper gates.
        label = {"label": el["label"]} if el.get("label") else {}
        if el["type"] in _PLSAV_REORDERED:
            kind, order = _PLSAV_REORDERED[el["type"]]
            pins = {p["pin"]: p["node"] for p in el["pins"]}
            if not all(p in pins for p in order):
                raise ToolError(f"Missing mapped pins on {el['id']}")
            params: dict[str, Any] = {}
            source: dict[str, Any] = {
                "model_id": el["type"], "parent_identifier": el["id"],
                "source_ref": el.get("ref"),
                "raw_properties": copy.deepcopy(el.get("properties", {})),
                "raw_statistics": copy.deepcopy(el.get("statistics", {})),
                "assumptions": [], "numerical_equivalence_to_original": False,
            }
            if kind == "digital_random4":
                # PhysicsLab does not serialize the hidden generator state.
                # Use an explicit stable non-zero surrogate seed and avoid
                # duplicate streams for the first 15 instances in one save.
                seed = int.from_bytes(hashlib.sha256(el["id"].encode()).digest()[:2], "big") % 15 + 1
                for _ in range(15):
                    if seed not in used_random_seeds:
                        break
                    seed = seed % 15 + 1
                used_random_seeds.add(seed)
                params["initial"] = seed
                source["assumptions"] = [
                        f"PhysicsLab saved no hidden Random Generator state; PE uses explicit deterministic surrogate seed {seed}.",
                        "This permits repeatable transition/reset and downstream-response tests, but does not reproduce or certify the original app's initial random value or exact sequence.",
                    ]
            components.append({"id": el["id"], "type": kind, "nodes": [pins[p] for p in order], "params": params,
                               "pl_source": source,
                               "position": el["position"], "rotation": el["rotation"], **label})
            if el["type"] in {"D Flipflop", "T Flipflop", "Real-T Flipflop", "JK Flipflop"} and 1 in pins:
                helper_id = el["id"] + ":notQ"
                if len(helper_id) > 128 or helper_id in reserved_ids:
                    base = "__pl_notQ_" + hashlib.sha256(el["id"].encode()).hexdigest()
                    helper_id, suffix = base, 0
                    while helper_id in reserved_ids:
                        suffix += 1
                        helper_id = base + "_" + str(suffix)
                reserved_ids.add(helper_id)
                components.append({"id": helper_id, "type": "digital_not", "nodes": [pins[0], pins[1]], "params": {},
                                   "position": el["position"], "rotation": el["rotation"],
                                   "pl_source": {**copy.deepcopy(source),
                                       "parent_identifier": el["id"], "is_helper": True,
                                       "decomposition_role": "complementary_output_not_gate"}})
            continue
        mapped_type = {"Eight Bit Display": "8bit Display", "Eight Bit Input": "8bit Input"}.get(el["type"], el["type"])
        if mapped_type not in mapping:
            raise ToolError(f"Cannot simulate/import {el['id']} ({el['type']}) faithfully; use native circuit_create with an explicit model. Rendering remains available.")
        candidates = [(name, c) for name, c in COMPONENTS.items() if c["model_id"] == mapped_type
                      and all(el["properties"].get(k) == v for k, v in c.get("constant_pl_props", {}).items())]
        if len(candidates) != 1:
            raise ToolError(f"Ambiguous model variant on {el['id']}; check saved variant properties")
        kind, model = candidates[0]
        if not el["pin_count_known"] or len(el["pins"]) != model["pins"]:
            raise ToolError(f"Unmapped pin count for {el['id']}")
        params = dict(model["defaults"])
        fallback_assumptions: list[str] = []
        native = el.get("native") or {}
        if native.get("type") == kind and isinstance(native.get("params"), dict):
            # Preserve PE-only model parameters from our explicit sidecar. The
            # actual PL properties below remain authoritative for shared fields.
            params.update(native["params"])
        for key, prop in model["pl_props"].items():
            if prop not in el["properties"]:
                if key in model.get("extra_params", {}):
                    continue
                if mapped_type == "Logic Input" and key == "state":
                    params[key] = 0
                    fallback_assumptions.append(
                        "Legacy Logic Input save omits 开关 and has no saved output statistic: explicit safe low state 0 is used."
                    )
                    continue
                raise ToolError(f"Missing saved property {prop} on {el['id']}")
            params[key] = el["properties"][prop]
        components.append({"id": el["id"], "type": kind,
                           "nodes": [p["node"] for p in sorted(el["pins"], key=lambda p: p["pin"])],
                           "params": params, "position": el["position"],
                           "rotation": el["rotation"],
                           "pl_source": {"model_id": el["type"],
                               "parent_identifier": el["id"],
                               "source_ref": el.get("ref"),
                               "raw_properties": copy.deepcopy(el.get("properties", {})),
                               "raw_statistics": copy.deepcopy(el.get("statistics", {})),
                               "assumptions": fallback_assumptions,
                               "numerical_equivalence_to_original": False}, **label})

    wired_pins: set[tuple[str, int]] = set()
    for wire in full["wires"]:
        for id_key, pin_key in (("Source", "SourcePin"), ("Target", "TargetPin")):
            component_id, pin = wire.get(id_key), wire.get(pin_key)
            if isinstance(component_id, str) and type(pin) is int:
                wired_pins.add((component_id, pin))
    for original in full["components"]:
        input_pins = _PLSAV_BOOLEAN_INPUT_PINS.get(original["type"])
        if input_pins is None:
            continue
        pins = {pin.get("pin"): pin.get("node") for pin in original.get("pins", [])
                if isinstance(pin, dict)}
        for pin in input_pins:
            if (original["id"], pin) in wired_pins:
                continue
            node = pins.get(pin)
            if not isinstance(node, str):
                raise ToolError(f"{original['id']}: missing disconnected Boolean input pin {pin}")
            base = "__pl_unwired_low_" + hashlib.sha256(
                f"{original['id']}:{pin}".encode()).hexdigest()
            helper_id, suffix = base, 0
            while helper_id in reserved_ids:
                suffix += 1
                helper_id = base + "_" + str(suffix)
            reserved_ids.add(helper_id)
            components.append({
                "id": helper_id,
                "type": "digital_input",
                "nodes": [node],
                "params": {"state": 0},
                "position": copy.deepcopy(original["position"]),
                "rotation": copy.deepcopy(original["rotation"]),
                # An empty explicit descriptor suppresses the native
                # digital_input control: this compatibility constant is not a
                # user-facing port and cannot be changed by tr_interactions.
                "interaction": {},
                "pl_source": {
                    "model_id": original["type"],
                    "parent_identifier": original["id"],
                    "source_ref": original.get("ref"),
                    "is_helper": True,
                    "decomposition_role": "implicit_unconnected_input_low",
                    "original_pin": pin,
                    "raw_properties": copy.deepcopy(original.get("properties", {})),
                    "raw_statistics": copy.deepcopy(original.get("statistics", {})),
                    "assumptions": [
                        "The original PLSAV has no Wire record on this Boolean input pin.",
                        "PhysicsLab saved-state evidence represents the disconnected input as 0; a non-interactive native low driver prevents PE four-state X propagation.",
                        "No original wire, public Logic Input port or editable control is created.",
                    ],
                    "numerical_equivalence_to_original": False,
                },
            })

    components = apply_damage_protection(components, full)

    # A PL gate's saved voltage levels are part of the mixed-signal contract,
    # not cosmetic metadata. Keep defaults only for old saves that omit them.
    mixed = (any(c["type"].startswith("digital_") for c in components)
             and any(not c["type"].startswith("digital_") for c in components))
    analog_nodes = {node for c in components
                    if not c["type"].startswith("digital_")
                    and not (c["type"] == "rated_protection"
                             and c.get("pl_source", {}).get("saved_is_broken"))
                    for node in c["nodes"]}
    originals = {el["id"]: el for el in full["components"]}
    for component in components:
        if not component["type"].startswith("digital_"):
            continue
        source = component.setdefault("pl_source", {})
        original = originals.get(source.get("parent_identifier", component["id"]), {})
        props = original.get("properties", {})
        source.setdefault("model_id", original.get("type", component["type"]))
        source.setdefault("raw_properties", copy.deepcopy(props))
        assumptions = source.setdefault("assumptions", [])
        if COMPONENTS[component["type"]].get("extra_params", {}).get("high_v"):
            for key, field, default in (("low_v", "低电平", 0.0), ("high_v", "高电平", 5.0)):
                value = props.get(field, default)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ToolError(f"{component['id']}: invalid original {field}")
                component["params"][key] = value
                if field not in props:
                    assumptions.append(f"Saved {field} missing: explicit native default {default:g} V.")
            assumptions.append(
                "Saved high/low voltages drive and decode mixed analog nodes; "
                "PhysicsLab does not enforce digital-device maximum-current protection, so this PLSAV import maps its effective current limit to PE's maximum finite value while preserving the saved field as provenance. "
                "This compatibility rule does not change native PE rated_protection behavior.")
            if component["type"] == "digital_output":
                # PhysicsLab Logic Output is an indicator, not a timing-check
                # primitive.  A saved analog voltage must therefore decode in
                # a zero-time operating-point/fixed-point solve.  PE's native
                # generic OUTPUT keeps nonzero setup/hold defaults for native
                # timing studies, so the import makes this source semantic
                # difference explicit instead of leaving every DC reading X.
                component["params"]["setup_time_s"] = 0.0
                component["params"]["hold_time_s"] = 0.0
                assumptions.append(
                    "PhysicsLab Logic Output is imported as an immediate voltage indicator: native setup/hold delays are explicitly zero for DC and transient observation."
                )
        elif mixed and any(node in analog_nodes for node in component["nodes"]) and any(props.get(field, default) != default
                           for field, default in (("低电平", 0), ("高电平", 5))):
            raise ToolError(
                f"{component['id']}: {component['type']} has no configurable analog levels; "
                "it is directly connected to an analog node, so refusing a silent 0/5 V substitution")

    imported = {"components": components, "g_min_siemens": 1e-12, "import_scope": {
        "original_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_component_count": len(full["components"]),
        "original_wire_count": len(full["wires"]),
        "native_component_count": len(components),
        "original_unchanged": True,
        "scope": ("Native engineering model of original connectivity. Declared approximations "
                  "are not a claim of pointwise PhysicsLab numerical equivalence."),
        "solver_regularization": {
            "g_min_siemens": 1e-12,
            "equivalent_resistance_to_ground_ohm": 1e12,
            "reason": "regularize floating original terminals and mixed milli-ohm meter ranges; no source wire was added",
        },
    }}
    saved_camera = (full.get("camera") or {}).get("saved_raw")
    if isinstance(saved_camera, dict) and saved_camera:
        imported["camera_save"] = saved_camera
    return normalize_spec(imported, digital_component_limit=runtime.config.phy_engine.digital_component_limit)


def _load_spec(runtime: ToolRuntime, path: str) -> dict[str, Any]:
    source, value = _input(runtime, path)
    if value.get("schema") == "aurex.circuit.v1":
        return normalize_spec(value, digital_component_limit=runtime.config.phy_engine.digital_component_limit)
    if value.get("schema") == "aurex.pe-state.v1":
        return normalize_spec(value["spec"], digital_component_limit=runtime.config.phy_engine.digital_component_limit)
    return _spec_from_sav(runtime, source)


def circuit_edit(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    spec = _load_spec(runtime, str(args.get("path") or ""))
    original_spec = copy.deepcopy(spec)
    changes = args.get("operations")
    if not isinstance(changes, list) or not 1 <= len(changes) <= 100:
        raise ToolError("operations must contain 1..100 add/update/remove/connect operations")
    for op in changes:
        if not isinstance(op, dict):
            raise ToolError("Operation must be an object")
        action = op.get("action")
        by_id = {c["id"]: c for c in spec["components"]}
        if action == "add":
            spec["components"].append(op.get("component"))
        elif action in ("update", "remove", "connect"):
            cid = op.get("id")
            if cid not in by_id:
                raise ToolError(f"Unknown component {cid!r}")
            c = by_id[cid]
            if action == "remove":
                spec["components"].remove(c)
            elif action == "update":
                if "params" in op:
                    c["params"].update(op["params"])
                if "nodes" in op:
                    c["nodes"] = op["nodes"]
                for key in ("position", "rotation", "label"):
                    if key in op:
                        c[key] = op[key]
            else:
                pin = op.get("pin")
                if not isinstance(pin, int) or not 0 <= pin < len(c["nodes"]):
                    raise ToolError("connect pin is out of range")
                c["nodes"][pin] = op.get("node")
        else:
            raise ToolError("action must be add/update/remove/connect")
        spec = normalize_spec(spec, digital_component_limit=runtime.config.phy_engine.digital_component_limit)
    if json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == json.dumps(
            original_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False):
        raise ToolError(
            "circuit_edit made no electrical or layout change; no new revision was created. "
            "Do not repeat default/current parameter values. Inspect the existing evidence and change only a value justified by it."
        )
    result = circuit_create(runtime, {"spec": spec, "camera": args.get("camera"), "with_image": args.get("with_image", False)})
    result["source_path"] = args["path"]
    return result


def _expand_stimulus_table(spec: dict[str, Any], table: Any) -> list[dict[str, Any]]:
    """Lossless syntax conversion only; native timing and sampling stay unchanged."""
    if not isinstance(table, dict) or set(table) != {"inputs", "vectors"}:
        raise ToolError("stimulus_table must contain exactly inputs and vectors")
    inputs, vectors = table["inputs"], table["vectors"]
    if not isinstance(inputs, list) or not inputs:
        raise ToolError("stimulus_table.inputs must list at least one exact digital_input component ID")
    if any(not c["type"].startswith("digital_") for c in spec["components"]):
        raise ToolError("stimulus_table currently requires a digital-only circuit, like legacy stimulus")
    known = {c["id"]: c for c in spec["components"]}
    seen = set()
    for column, cid in enumerate(inputs):
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ToolError(f"stimulus_table.inputs[{column}] must be a unique nonempty exact input ID")
        if cid not in known:
            raise ToolError(f"stimulus_table.inputs[{column}]: unknown component ID {cid!r}; read circuit_inspect(interface_only=true) for exact IDs. No ID was substituted.")
        if known[cid]["type"] != "digital_input":
            raise ToolError(f"stimulus_table.inputs[{column}]: {cid!r} is {known[cid]['type']!r}, not digital_input")
        seen.add(cid)
    if not isinstance(vectors, list) or not 1 <= len(vectors) <= 128:
        raise ToolError("stimulus_table.vectors must contain 1..128 frames (the existing per-call stimulus limit); use separate batches for additional requested samples")
    expanded = []
    for frame, vector in enumerate(vectors):
        if not isinstance(vector, list) or len(vector) != len(inputs):
            actual = len(vector) if isinstance(vector, list) else type(vector).__name__
            raise ToolError(f"stimulus_table.vectors[{frame}] must contain exactly {len(inputs)} values in inputs order; got {actual}; no values are padded or inferred. "
                "No simulation was executed. For a few changed inputs, omit stimulus_table and use stimulus=[{set:{exact_discovered_input_id:0_or_1}}, ...]; "
                "an empty set holds every current input. Unlisted inputs retain their saved/current state, not an implicit zero. "
                "Alternatively list only the intended input IDs in a narrower table and give matching-width rows. "
                "Do not assume that the first input is a clock/reset or infer bit significance from listing order.")
        values = {}
        for column, state in enumerate(vector):
            if type(state) is not int or state not in (0, 1, 2, 3):
                raise ToolError(f"stimulus_table.vectors[{frame}][{column}] for {inputs[column]!r} must be an integer 0=L,1=H,2=X,3=Z (not boolean)")
            values[inputs[column]] = state
        expanded.append({"set": values})
    node_connection_counts: dict[str, int] = {}
    for component in spec["components"]:
        for node in set(component["nodes"]):
            node_connection_counts[node] = node_connection_counts.get(node, 0) + 1
    isolated = [cid for cid in inputs if all(node_connection_counts.get(node, 0) <= 1 for node in known[cid]["nodes"])]
    if isolated:
        raise ToolError(
            "stimulus_table selected isolated digital_input component ID(s) with no connection beyond the input itself: "
            + ", ".join(repr(cid) for cid in isolated)
            + ". No simulation was executed because toggling these inputs cannot affect another component. "
              "Use circuit_inspect(interface_only=true) and require connected_to_other_components=true; "
              "then confirm a candidate clock by exact node query (for example circuit_inspect(query=\"N24\")) instead of position or listing order."
        )
    return expanded


def circuit_analyze(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    with_image = args.get("with_image", False)
    if type(with_image) is not bool:
        raise ToolError("with_image must be boolean")
    has_table = "stimulus_table" in args
    inline = args.get("spec")
    if has_table and ("stimulus" in args or isinstance(inline, dict) and "stimulus" in inline):
        raise ToolError("Specify only one explicit stimulus format: stimulus or stimulus_table, not both")
    spec = normalize_spec(args["spec"], digital_component_limit=runtime.config.phy_engine.digital_component_limit) if "spec" in args else _load_spec(runtime, str(args.get("path") or ""))
    if "stimulus_table" in spec:
        raise ToolError("stimulus_table belongs at the top level of circuit_analyze arguments, not inside the native spec")
    for key in ("analysis", "tr_step", "tr_stop", "tr_sample_every", "tr_initialize_dc", "digital_steps_per_tr_step", "tr_interactions", "ac_omega", "g_min_siemens", "digital_clock_ticks", "stimulus"):
        if key in args:
            spec[key] = args[key]
    if has_table:
        # An explicit new sequence replaces a path-loaded historical sequence,
        # just like the existing top-level stimulus argument. The source file
        # and any inputs absent from this table are not modified.
        spec["stimulus"] = _expand_stimulus_table(spec, args["stimulus_table"])
    ground = args.get("ground_node")
    if ground:
        if not any(ground in c["nodes"] for c in spec["components"]):
            raise ToolError("ground_node is not present in the circuit")
        for c in spec["components"]:
            c["nodes"] = ["gnd" if n == ground else n for n in c["nodes"]]
    has_analog = any(not c["type"].startswith("digital_") for c in spec["components"])
    has_digital = any(c["type"].startswith("digital_") for c in spec["components"])
    if has_analog and not has_digital and not any("gnd" in c["nodes"] for c in spec["components"]):
        raise ToolError("Analog circuit has no voltage reference; choose a node from circuit_inspect and set ground_node")
    if args.get("camera") is not None:
        spec["camera"] = _camera(args["camera"])
    samples = pe_simulate(runtime, {"spec": spec, "return_state": True})
    folder = _artifact_dir(runtime)
    path = folder / "analyzed.circuit.json"
    _write(path, spec)
    snapshot = samples.pop("state")
    protection_summary = _protection_summary(samples.get("components", []))
    if protection_summary is not None:
        snapshot["protection_summary"] = protection_summary
    measurements = {c["id"]: c for c in samples["components"]}
    rendered, exportable = _render_spec(spec, measurements, samples.get("interaction_states"))
    snapshot["scene"] = rendered
    source = folder / "snapshot.pe-state.json"
    from ..trace_archive import compact_snapshot
    _write(source, compact_snapshot(source, snapshot))
    try:
        output = _view(runtime, source, state=True,
                       view=str(args.get("view", "spatial")), projection=str(args.get("projection", "isometric")),
                       camera=args.get("camera"), focus_ids=args.get("focus_ids"), query=str(args.get("query", "")),
                       limit=int(args.get("limit", 8)), with_image=with_image)
    except ToolError as error:
        # Solving has already succeeded and the complete state is durable.
        # Presentation failure must not erase evidence or demand another solve.
        output = {'images': [], 'with_image': False, 'state_source': 'native solver snapshot',
            'simulation_completed': True, 'presentation_error': str(error),
            'recovery': 'Read state_path with circuit_read_trace/circuit_read_stimulus/circuit_inspect. Do not rerun solely because presentation failed.',
            'statistics': {'components': len(spec['components'])}}
    # Native-state rendering above never requires a PLSAV roundtrip. Export a
    # separate compatible design only when the mapping is known to be faithful.
    if exportable and 'presentation_error' not in output:
        export_source = folder / "export-input.json"
        _write(export_source, rendered)
        exported = _view(runtime, export_source, create=True, exportable=True, camera=args.get("camera"), with_image=False)
        output["sav_path"] = exported["sav_path"]
    output["circuit_path"] = str(path)
    output["state_path"] = str(source)
    _annotate_source_refs(output, spec)
    # Keep only the visible components' native edit schema. Measurements and
    # renderer properties use different field names; a focused inspection can
    # retrieve another exact writable row without echoing the whole design.
    visible_ids = [str(row.get("id")) for row in output.get("netlist", {}).get("components", [])]
    _attach_edit_contract(output, spec, visible_ids)
    if protection_summary is not None:
        output["protection_summary"] = protection_summary
    if all(not c["type"].startswith("digital_") for c in spec["components"]) and spec.get("analysis", "dc") != "trop":
        from ..analog_evidence import record_analysis
        report_path = record_analysis(runtime.cache_dir, spec_path=str(path), state_path=str(source), sav_path=output.get("sav_path"))
        output["report_path"] = report_path
        output["analysis_table_path"] = str(Path(report_path).with_name("analysis-table.zh.md"))
        report = json.loads(Path(report_path).read_text())
        output["numerical_verification"] = {k: report[k] for k in ("verified", "functional_verification") if k in report}
        trace = report.get("trace")
        if isinstance(trace, dict) and trace.get("verified"):
            node_ranges = trace.get("node_ranges_v", {})
            selected_ranges = dict(list(node_ranges.items())[:32]) if isinstance(node_ranges, dict) else {}
            output["numerical_verification"]["trace_summary"] = {
                "sample_count": trace.get("sample_count"),
                "time_start_s": trace.get("times_s", [None])[0],
                "time_end_s": trace.get("times_s", [None])[-1],
                "node_ranges_v": selected_ranges,
                "node_count": len(node_ranges) if isinstance(node_ranges, dict) else 0,
                "node_ranges_omitted": max(0, len(node_ranges) - len(selected_ranges)) if isinstance(node_ranges, dict) else 0,
                "scope": "Actual recorded sample points only; peak-to-peak is max-min over those samples and does not infer unsampled extrema.",
            }
    # Keep the full sampled state immutable on disk. A compact paged reader
    # prevents long traces from flooding every subsequent agent request.
    recorded_stimulus_count = len(samples["stimulus_results"]) if isinstance(samples.get("stimulus_results"), list) else None
    transient = samples.get("transient")
    if transient and "samples" in transient:
        sample_index_guide = _trace_sample_index_guide(
            transient["samples"], args.get("tr_interactions"))
        samples = {**samples, "transient": {k: v for k, v in transient.items() if k != "samples"}}
        if sample_index_guide:
            samples["transient"]["sample_index_guide"] = sample_index_guide
        only_digital = all(native_digital_type(c["type"]) for c in spec["components"])
        if only_digital:
            samples["transient"]["trace_reader"] = "circuit_read_trace"
            samples["transient"]["trace_access"] = {
                "kind": "recorded_digital_solver_samples", "state_path": str(source),
                "location": "measurements.transient.samples",
                "reader": "circuit_read_trace", "selector": "component_ids",
                "note": "Read actual timestamped digital TR samples with circuit_read_trace(component_ids=[exact IDs]). Digital propagation policy is recorded separately; older samples may predate per-step propagation. No changes between recorded samples are inferred.",
            }
            if samples.get("stimulus_results"):
                samples["transient"]["trace_access"]["separate_stimulus_reader"] = "circuit_read_stimulus"
                samples["transient"]["trace_access"]["stimulus_note"] = "This reader accesses the separately recorded stimulus sequence, not the transient solver samples."
            else:
                samples["transient"]["trace_access"]["stimulus_recorded"] = False
        else:
            samples["transient"]["trace_reader"] = "circuit_read_trace"
            samples["transient"]["trace_access"] = {"kind": "recorded_analog_voltage_samples", "state_path": str(source),
                "note": "Read actual sampled node voltages with circuit_read_trace(nodes=[...]); select digital component_ids separately for timestamped logic samples. Complete per-component measurements remain in the immutable state artifact."}
    if len(samples.get("components", [])) > 16:
        # The immutable state above already contains the complete solver output.
        # Context size must not grow with every gate or every sampled time step.
        selected = {c["id"] for c in output.get("netlist", {}).get("components", samples["components"])[:8]}
        count = len(samples["components"])
        samples = {**samples, "components": [c for c in samples["components"] if c["id"] in selected]}
        samples["component_scope"] = {"total": count, "shown": len(samples["components"]),
                                      "omitted": count - len(samples["components"]),
                                      "complete_state_path": str(source),
                                      "read_more": "circuit_inspect on this state_path with focus_id/query returns the selected component's recorded native measurements without rerunning"}
        if samples.get("stimulus_results"):
            sequence = samples["stimulus_results"]
            samples["stimulus_results"] = [{"step": row["step"],
                "inputs": {k: v for k, v in row.get("inputs", {}).items() if k in selected},
                "changed_input_count": len(row.get("inputs", {})),
                "digital": {k: v for k, v in row["digital"].items() if k in selected}} for row in sequence[-8:]]
            samples["stimulus_scope"] = {"total_steps": len(sequence), "shown_steps": len(samples["stimulus_results"]),
                                         "omitted_steps": max(0, len(sequence) - 8), "shown_component_ids": sorted(selected),
                                         "reader": "circuit_read_stimulus", "state_path": str(source)}
    if has_table and isinstance(samples.get("stimulus_results"), list):
        # The immutable snapshot above contains every expanded set and actual
        # sample. Do not undo the compact input format by echoing per-frame UUID
        # maps back into the agent's context; use the existing paged reader.
        samples = {key: value for key, value in samples.items() if key != "stimulus_results"}
        samples["stimulus_scope"] = {
            "total_steps": recorded_stimulus_count, "shown_steps": 0,
            "omitted_steps": recorded_stimulus_count, "reader": "circuit_read_stimulus",
            "state_path": str(source),
            "note": "All actual stimulus results are recorded in the immutable state. Read selected components and steps without rerunning; no per-frame UUID maps are echoed here.",
        }
    output["measurements"] = samples
    output["measurement_source"] = "fresh Phy-Engine solve; resistor currents explicitly derived from simulated pin voltage"
    if has_table:
        output["stimulus_input_format"] = {
            "format": "stimulus_table", "input_count": len(args["stimulus_table"]["inputs"]),
            "frame_count": len(spec["stimulus"]), "native_frame_step_s": 1e-8,
            "recorded_frame_count": recorded_stimulus_count,
            "unselected_input_initial_states_preserved": True,
            "expanded_sequence_location": "spec.stimulus in the immutable state_path",
            "recorded_results_reader": "circuit_read_stimulus",
            "note": "Columns follow the exact supplied input ID order; no bit significance, clock edges or values were inferred. Same native execution as legacy stimulus; no functional pass is implied.",
        }
    return output


def circuit_read_stimulus(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    source, snapshot = _input(runtime, str(args.get("path") or ""))
    if snapshot.get("schema") != "aurex.pe-state.v1":
        raise ToolError("circuit_read_stimulus requires a .pe-state.json artifact")
    from ..trace_archive import read_series
    rows = read_series(source, snapshot.get("measurements") or {}, "stimulus_results")
    if not isinstance(rows, list) or not rows:
        raise ToolError("This state has no recorded digital stimulus; request stimulus vectors during circuit_analyze")
    catalog = {c["id"]: c for c in snapshot["spec"]["components"]}
    selected = args.get("component_ids", list(catalog)[:8])
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 8 or
            any(not isinstance(cid, str) or cid not in catalog for cid in selected) or len(set(selected)) != len(selected)):
        raise ToolError("component_ids must select 1..8 unique IDs from the state; discover IDs with circuit_inspect")
    offset, limit = args.get("offset", 0), args.get("limit", 8)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16:
        raise ToolError("Stimulus offset must be nonnegative and limit must be 1..16")
    steps = []
    for row in rows[offset:offset + limit]:
        # Zero is a real logic value. Missing samples remain missing, not zero.
        values = {cid: row["digital"][cid] for cid in selected if cid in row["digital"]}
        changes = {cid: row.get("inputs", {})[cid] for cid in selected if cid in row.get("inputs", {})}
        steps.append({"step": row["step"], "digital": values, "input_changes": changes,
                      "omitted_input_changes": len(row.get("inputs", {})) - len(changes)})
    return {"state_path": str(source), "recorded_not_resimulated": True,
            "encoding": {"0": "L", "1": "H", "2": "X", "3": "Z"},
            "components": [{"id": cid, "type": catalog[cid]["type"], "nodes": catalog[cid]["nodes"],
                            "pin_labels": COMPONENTS[catalog[cid]["type"]]["pin_labels"]} for cid in selected],
            "steps": steps, "offset": offset, "total_steps": len(rows), "total_components": len(catalog),
            "has_more": offset + len(steps) < len(rows),
            "note": "Digital arrays are in catalog pin order. Step is the actual stored stimulus index, not an inferred timestamp. Only selected components/input changes are shown; full samples remain in the immutable state."}


def _trace_sample_index_guide(points, interactions):
    """Build a small exact index/time guide around requested interactions."""
    if not isinstance(points, list) or not points:
        return None
    indices = {0, len(points) - 1}
    boundaries = []
    for interaction in interactions if isinstance(interactions, list) else []:
        event_time = interaction.get("time_s") if isinstance(interaction, dict) else None
        if type(event_time) not in (int, float) or not math.isfinite(event_time):
            continue
        before = next((index for index in range(len(points) - 1, -1, -1)
                       if points[index].get("time_s", math.inf) < event_time), None)
        at_or_after = next((index for index, point in enumerate(points)
                            if point.get("time_s", -math.inf) >= event_time), None)
        after = next((index for index, point in enumerate(points)
                      if point.get("time_s", -math.inf) > event_time), None)
        for index in (before, at_or_after, after):
            if index is not None:
                indices.add(index)
        boundaries.append({"interaction_time_s": event_time, "before_index": before,
                           "at_or_after_index": at_or_after, "after_index": after})
    ordered = sorted(indices)
    return {
        "zero_based_index_range": [0, len(points) - 1],
        "total_samples": len(points),
        "suggested_samples": [{"sample_index": index,
                                "time_s": points[index].get("time_s"),
                                "completed_steps": points[index].get("completed_steps")}
                               for index in ordered],
        "interaction_boundaries": boundaries,
        "usage": "Pass selected sample_index values to circuit_read_trace(sample_indices=[...]); these are recorded frames, not solver-step numbers.",
    }


def _select_trace_points(points, args):
    """Return a bounded page or exact sparse samples without rereading a trace."""
    requested = args.get("sample_indices")
    if requested is not None:
        if "offset" in args or "limit" in args:
            raise ToolError("sample_indices is mutually exclusive with trace offset/limit")
        if (not isinstance(requested, list) or not 1 <= len(requested) <= 16
                or any(type(index) is not int or index < 0 for index in requested)
                or len(set(requested)) != len(requested)):
            raise ToolError("sample_indices must contain 1..16 unique nonnegative integers")
        invalid = [index for index in requested if index >= len(points)]
        if invalid:
            raise ToolError(
                f"sample_indices {invalid} exceed the recorded range 0..{len(points) - 1}")
        return [(index, points[index]) for index in requested], {
            "selection_mode": "sample_indices",
            "sample_indices": requested,
            "has_more": False,
        }
    offset, limit = args.get("offset", 0), args.get("limit", 32)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 64:
        raise ToolError("Trace offset must be nonnegative; limit must be 1..64")
    selected = list(enumerate(points[offset:offset + limit], start=offset))
    return selected, {
        "selection_mode": "page",
        "offset": offset,
        "has_more": offset + len(selected) < len(points),
    }


def _digital_trace_page(source, snapshot, trace, points, args):
    # A stored assertion is not proof: validate timestamps and native count
    # metadata before presenting a historical artifact as a propagated trace.
    previous_time, previous_step = 0.0, 0
    for point in points:
        time_s = point.get('time_s')
        if type(time_s) not in (int, float) or not math.isfinite(time_s) or time_s <= previous_time:
            raise ToolError('Invalid recorded TR timestamps: expected finite, strictly increasing positive times')
        previous_time = time_s
        completed = point.get('completed_steps')
        if completed is not None:
            if type(completed) is not int or completed <= previous_step:
                raise ToolError('Invalid recorded TR completed_steps: expected strictly increasing positive integers')
            previous_step = completed
    stop = trace.get('actual_stop_s')
    if type(stop) not in (int, float) or not math.isfinite(stop) or stop < previous_time:
        raise ToolError('Invalid recorded TR actual_stop_s')
    catalog = {c["id"]: c for c in snapshot["spec"]["components"] if native_digital_type(c.get("type"))}
    selected = args.get("component_ids", list(catalog)[:8])
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 8 or
        any(not isinstance(cid, str) or cid not in catalog for cid in selected) or len(set(selected)) != len(selected)):
        raise ToolError("component_ids must select 1..8 unique native digital component IDs from the state; use circuit_inspect for IDs")
    selected_points, selection = _select_trace_points(points, args)
    output = []
    for sample_index, point in selected_points:
        recorded = {c["id"]: c for c in point["components"]}
        values, missing = {}, []
        for cid in selected:
            row = recorded.get(cid, {})
            logic = row.get("digital")
            if logic is None:
                missing.append(cid)
                continue
            expected_pins = COMPONENTS[catalog[cid]["type"]]["pins"]
            if (not isinstance(logic, list) or len(logic) != expected_pins or
                any(type(value) is not int or value not in (0, 1, 2, 3) for value in logic)):
                raise ToolError("Invalid recorded four-state pin samples for " + cid + "; no values were inferred")
            values[cid] = logic
        output.append({"sample_index": sample_index, "time_s": point["time_s"],
                       "completed_steps": point.get("completed_steps"),
                       "digital": values, "missing_component_ids": missing})
    saved = trace.get('digital_propagation')
    policy = dict(saved) if isinstance(saved, dict) else {'version': None,
        'policy': 'unreported_in_historical_record', 'completed_propagation_steps': None}
    per_step, total = policy.get('per_tr_step'), policy.get('completed_propagation_steps')
    completed = trace.get('completed_steps')
    expected_policy = ('once_after_each_native_solve_before_sampling' if per_step == 1 else
                       'configured_after_each_native_solve_before_sampling')
    policy['verified_per_step'] = bool(
        policy.get('verified_per_step') is True and type(policy.get('version')) is int and policy['version'] == 1
        and type(per_step) is int and 1 <= per_step <= 64
        and type(total) is int and type(completed) is int and completed > 0
        and total == per_step * completed and policy.get('policy') == expected_policy
        and all(type(p.get('completed_steps')) is int for p in points)
        and previous_step == completed and previous_time == stop
        and (policy.get('count_origin') != 'native_counter' or
             (type(policy.get('configured_version')) is int and policy['configured_version'] == 1))
        and (per_step == 1 or (type(policy.get('configured_version')) is int
             and policy['configured_version'] == 1 and policy.get('count_origin') == 'native_counter')))
    return {"kind": "recorded_digital_transient_samples", "state_path": str(source),
            "recorded_not_resimulated": True, "interpolated": False, "units": {"time": "s"},
            "encoding": {"0": "L", "1": "H", "2": "X", "3": "Z"},
            "components": [{"id": cid, "type": catalog[cid]["type"], "nodes": catalog[cid]["nodes"],
                "pin_labels": COMPONENTS[catalog[cid]["type"]]["pin_labels"]} for cid in selected],
            "points": output, **selection, "total_samples": len(points),
            "actual_stop_s": trace["actual_stop_s"],
            "digital_propagation": policy,
            "warning": None if policy.get("verified_per_step") is True else
                "This historical/library record does not confirm per-step digital propagation. Values are returned unchanged, not certified as a propagated digital waveform.",
            "note": "Actual TR timestamps and recorded pin states only, not the separate stimulus sequence. Missing values remain missing. Unknown X/Z are not zero or a functional PASS."}


def _summarize_numeric_series(series: list[tuple[int, float, float]], *,
                              window_s: float, stability_threshold: float,
                              change_threshold: float, event_limit: int = 3) -> dict[str, Any]:
    """Summarize recorded samples without interpolation or inferred extrema."""
    if not series:
        raise ToolError("Cannot summarize an empty recorded series")
    values = [row[2] for row in series]
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ToolError("Recorded trace contains a non-finite numeric value")
    changes = []
    largest = {"magnitude": 0.0, "from_index": series[0][0], "to_index": series[0][0],
               "from_time_s": series[0][1], "to_time_s": series[0][1]}
    change_count = 0
    direction_reversal_count = 0
    direction_reversals = []
    previous_significant_direction = None
    for before, after in zip(series, series[1:]):
        delta = after[2] - before[2]
        magnitude = abs(delta)
        if magnitude > largest["magnitude"]:
            largest = {"magnitude": magnitude, "from_index": before[0], "to_index": after[0],
                       "from_time_s": before[1], "to_time_s": after[1],
                       "from": before[2], "to": after[2]}
        if magnitude > 0 and magnitude >= change_threshold:
            change_count += 1
            direction = 1 if delta > 0 else -1
            if (previous_significant_direction is not None
                    and direction != previous_significant_direction):
                direction_reversal_count += 1
                if len(direction_reversals) < event_limit:
                    direction_reversals.append({
                        "sample_index": after[0], "time_s": after[1],
                        "from_direction": ("rising" if previous_significant_direction > 0 else "falling"),
                        "to_direction": ("rising" if direction > 0 else "falling"),
                    })
            previous_significant_direction = direction
            if len(changes) < event_limit:
                changes.append({"from_index": before[0], "to_index": after[0],
                                "from_time_s": before[1], "to_time_s": after[1],
                                "from": before[2], "to": after[2], "magnitude": magnitude})

    # The end index needed to span ``window_s`` is monotonic as the start
    # advances.  Maintain min/max deques over that moving window instead of
    # rescanning every suffix (the earlier implementation was O(n^2)).
    first_window = None
    minimum_queue: deque[int] = deque()
    maximum_queue: deque[int] = deque()
    end = -1
    required_window = max(0.0, window_s - 1e-12)
    for start, row in enumerate(series):
        while minimum_queue and minimum_queue[0] < start:
            minimum_queue.popleft()
        while maximum_queue and maximum_queue[0] < start:
            maximum_queue.popleft()
        while (end + 1 < len(series)
               and (end < start or series[end][1] - row[1] < required_window)):
            end += 1
            while minimum_queue and values[minimum_queue[-1]] >= values[end]:
                minimum_queue.pop()
            minimum_queue.append(end)
            while maximum_queue and values[maximum_queue[-1]] <= values[end]:
                maximum_queue.pop()
            maximum_queue.append(end)
        if end < start or series[end][1] - row[1] < required_window:
            break
        minimum, maximum = values[minimum_queue[0]], values[maximum_queue[0]]
        span = maximum - minimum
        if span <= stability_threshold:
            first_window = {"start_index": row[0], "end_index": series[end][0],
                            "start_time_s": row[1], "end_time_s": series[end][1],
                            "sampled_range": span, "minimum": minimum, "maximum": maximum}
            break

    suffix_min = [0.0] * len(values)
    suffix_max = [0.0] * len(values)
    suffix_min[-1] = suffix_max[-1] = values[-1]
    for index in range(len(values) - 2, -1, -1):
        suffix_min[index] = min(values[index], suffix_min[index + 1])
        suffix_max[index] = max(values[index], suffix_max[index + 1])
    settled = None
    for index, row in enumerate(series):
        if (series[-1][1] - row[1] >= window_s - 1e-12
                and suffix_max[index] - suffix_min[index] <= stability_threshold):
            settled = {"start_index": row[0], "start_time_s": row[1],
                       "through_index": series[-1][0], "through_time_s": series[-1][1],
                       "sampled_range": suffix_max[index] - suffix_min[index],
                       "minimum": suffix_min[index], "maximum": suffix_max[index]}
            break
    return {
        "sample_count": len(series),
        "first": {"sample_index": series[0][0], "time_s": series[0][1], "value": values[0]},
        "final": {"sample_index": series[-1][0], "time_s": series[-1][1], "value": values[-1]},
        "minimum": min(values), "maximum": max(values), "sampled_range": max(values) - min(values),
        "largest_consecutive_change": largest,
        "changes_at_or_above_threshold": change_count,
        "change_examples": changes,
        "change_examples_omitted": max(0, change_count - len(changes)),
        "multiple_threshold_changes_observed": change_count >= 2,
        "direction_reversals_at_or_above_threshold": direction_reversal_count,
        "direction_reversal_examples": direction_reversals,
        "direction_reversal_examples_omitted": max(0, direction_reversal_count - len(direction_reversals)),
        "first_stable_sampled_window": first_window,
        "settled_for_remainder": settled,
        "periodic_oscillation_tested": False,
        "change_evidence_note": ("Repeated sampled changes or direction reversals are event evidence only; "
                                 "this bounded summary does not prove periodic oscillation or estimate a period."),
    }


def _project_trace_numeric_summary(summary: dict[str, Any], *,
                                   include_examples: bool) -> dict[str, Any]:
    """Project detailed internal statistics to compact model-facing evidence."""
    first_window = summary.get("first_stable_sampled_window")
    settled = summary.get("settled_for_remainder")
    largest = summary.get("largest_consecutive_change") or {}
    result: dict[str, Any] = {
        "sample_count": summary["sample_count"],
        "first_value": summary["first"]["value"],
        "final_value": summary["final"]["value"],
        "minimum": summary["minimum"], "maximum": summary["maximum"],
        "sampled_range": summary["sampled_range"],
        "largest_consecutive_change": {
            key: largest[key] for key in
            ("magnitude", "from_time_s", "to_time_s", "from", "to") if key in largest
        },
        "changes_at_or_above_threshold": summary["changes_at_or_above_threshold"],
        "multiple_threshold_changes_observed": summary["multiple_threshold_changes_observed"],
        "direction_reversals_at_or_above_threshold":
            summary["direction_reversals_at_or_above_threshold"],
        "first_stable_sampled_window": ({
            key: first_window[key] for key in
            ("start_time_s", "end_time_s", "sampled_range") if key in first_window
        } if isinstance(first_window, dict) else None),
        "settled_for_remainder": ({
            key: settled[key] for key in
            ("start_time_s", "through_time_s", "sampled_range") if key in settled
        } if isinstance(settled, dict) else None),
        "periodic_oscillation_tested": False,
    }
    if include_examples and summary.get("change_examples"):
        result["change_examples"] = [{
            key: event[key] for key in
            ("from_time_s", "to_time_s", "from", "to", "magnitude") if key in event
        } for event in summary["change_examples"]]
        omitted = summary.get("change_examples_omitted", 0)
        if omitted:
            result["change_examples_omitted"] = omitted
    if include_examples and summary.get("direction_reversal_examples"):
        result["direction_reversal_examples"] = copy.deepcopy(
            summary["direction_reversal_examples"])
        omitted = summary.get("direction_reversal_examples_omitted", 0)
        if omitted:
            result["direction_reversal_examples_omitted"] = omitted
    return result


def _trace_summary(source: Path, snapshot: dict[str, Any], trace: dict[str, Any],
                   points: list[dict[str, Any]], args: dict[str, Any]) -> dict[str, Any]:
    """Compute one bounded evidence summary over a recorded TR trace."""
    window_s = args.get("stability_window_s", 1.0)
    threshold_v = args.get("stability_threshold_v", 0.1)
    change_threshold_v = args.get("change_threshold_v", threshold_v)
    start_s, stop_s = args.get("from_time_s"), args.get("to_time_s")
    for name, value, allow_zero in (
            ("stability_window_s", window_s, False),
            ("stability_threshold_v", threshold_v, True),
            ("change_threshold_v", change_threshold_v, True)):
        if type(value) not in (int, float) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            raise ToolError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
    for name, value in (("from_time_s", start_s), ("to_time_s", stop_s)):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ToolError(f"{name} must be a finite nonnegative number")
    if start_s is not None and stop_s is not None and start_s > stop_s:
        raise ToolError("from_time_s must not exceed to_time_s")
    selected_points = [(index, point) for index, point in enumerate(points)
                       if (start_s is None or point.get("time_s", -math.inf) >= start_s)
                       and (stop_s is None or point.get("time_s", math.inf) <= stop_s)]
    if not selected_points:
        raise ToolError("No recorded samples fall inside the requested time range")
    previous_time = -math.inf
    for _, point in selected_points:
        time_s = point.get("time_s")
        if type(time_s) not in (int, float) or not math.isfinite(time_s) or time_s <= previous_time:
            raise ToolError("Recorded trace timestamps must be finite and strictly increasing")
        previous_time = time_s

    # Index each recorded sample once.  A summary can request up to eight
    # nodes/components; rebuilding these maps per target made long traces cost
    # O(samples * targets * components) and encouraged paged rereads.
    point_views = []
    for index, point in selected_points:
        component_map: dict[str, dict[str, Any]] = {}
        voltage_map: dict[str, float] = {}
        for component in point.get("components", []):
            if not isinstance(component, dict):
                continue
            cid = component.get("id")
            if isinstance(cid, str):
                component_map[cid] = component
            for node, value in zip(component.get("nodes", []), component.get("voltage", [])):
                if isinstance(node, str) and type(value) in (int, float) and math.isfinite(value):
                    voltage_map[node] = float(value)
        point_views.append((index, point, component_map, voltage_map))

    catalog = {c["id"]: c for c in snapshot["spec"]["components"]}
    nodes = args.get("nodes")
    component_ids = args.get("component_ids")
    if nodes is not None and component_ids is not None:
        raise ToolError("Trace summary accepts nodes or component_ids, not both")
    interaction_events = trace.get("interaction_events")
    if not isinstance(interaction_events, list):
        interaction_events = []
    output: dict[str, Any] = {
        "kind": "recorded_transient_summary", "state_path": str(source),
        "recorded_not_resimulated": True, "interpolated": False,
        "sample_scope": {"total": len(points), "selected": len(selected_points),
                         "first_index": selected_points[0][0], "last_index": selected_points[-1][0],
                         "from_time_s": selected_points[0][1]["time_s"],
                         "to_time_s": selected_points[-1][1]["time_s"]},
        "criteria": {"stability_window_s": float(window_s),
                     "stability_threshold_v": float(threshold_v),
                     "change_threshold_v": float(change_threshold_v),
                     "stability_definition": ("first_stable_sampled_window is the earliest recorded window of at least "
                                              "stability_window_s whose sampled max-min is within the threshold; "
                                              "settled_for_remainder additionally requires every later recorded sample to remain within it."),
                     "change_evidence_definition": ("Counts and transition examples describe recorded sample-to-sample events only. "
                                                    "Repeated changes do not by themselves prove a periodic oscillation.")},
        "interaction_events": copy.deepcopy(interaction_events[:32]),
        "interaction_events_omitted": max(0, len(interaction_events) - 32),
    }
    if nodes is not None or component_ids is None:
        available = sorted({node for _, _, _, voltages in point_views for node in voltages})
        selected_nodes = nodes if nodes is not None else available[:8]
        if (not isinstance(selected_nodes, list) or not 1 <= len(selected_nodes) <= 8
                or len(set(selected_nodes)) != len(selected_nodes)
                or any(not isinstance(node, str) or node not in available for node in selected_nodes)):
            raise ToolError("nodes must select 1..8 unique nodes with recorded analog voltages")
        node_results = {}
        for node in selected_nodes:
            series = []
            for index, point, _, voltages in point_views:
                if node not in voltages:
                    raise ToolError(f"No recorded analog voltage for {node} at sample {index}")
                series.append((index, float(point["time_s"]), voltages[node]))
            summary = _summarize_numeric_series(
                series, window_s=float(window_s), stability_threshold=float(threshold_v),
                change_threshold=float(change_threshold_v))
            node_results[node] = _project_trace_numeric_summary(
                summary, include_examples=True)
        output.update({"mode": "analog_nodes", "units": {"time": "s", "value": "V relative to ground"},
                       "available_node_count": len(available), "nodes": node_results})
        return output

    selected_components = component_ids
    if (not isinstance(selected_components, list) or not 1 <= len(selected_components) <= 8
            or len(set(selected_components)) != len(selected_components)
            or any(not isinstance(cid, str) or cid not in catalog for cid in selected_components)):
        raise ToolError("component_ids must select 1..8 unique native component IDs from the state")
    component_results = []
    for cid in selected_components:
        spec = catalog[cid]
        samples = []
        for index, point, components_by_id, _ in point_views:
            row = components_by_id.get(cid)
            if row is None:
                raise ToolError(f"No recorded component state for {cid} at sample {index}")
            samples.append((index, float(point["time_s"]), row))
        result: dict[str, Any] = {"id": cid, "type": spec["type"], "nodes": spec["nodes"],
                                  "pin_labels": COMPONENTS[spec["type"]]["pin_labels"]}
        state_keys = sorted({key for _, _, row in samples
                             for key, value in (row.get("model_state") or {}).items()
                             if type(value) in (int, float, bool)})
        state_summaries = {}
        for key in state_keys[:12]:
            sequence = [(index, time_s, (row.get("model_state") or {}).get(key))
                        for index, time_s, row in samples]
            if any(value is None or type(value) not in (int, float, bool) for _, _, value in sequence):
                continue
            changes = []
            for before, after in zip(sequence, sequence[1:]):
                if after[2] != before[2]:
                    changes.append({"sample_index": after[0], "time_s": after[1],
                                    "from": before[2], "to": after[2]})
            state_summaries[key] = {"first": sequence[0][2], "final": sequence[-1][2],
                                    "change_count": len(changes), "changes": changes[:6],
                                    "changes_omitted": max(0, len(changes) - 6)}
        if state_summaries:
            result["model_state"] = state_summaries
        pin_labels = result["pin_labels"]
        voltage_count = max((len(row.get("voltage", [])) for _, _, row in samples), default=0)
        voltage_summaries = []
        for pin in range(min(voltage_count, 8)):
            series = [(index, time_s, float(row["voltage"][pin])) for index, time_s, row in samples
                      if len(row.get("voltage", [])) > pin]
            if len(series) != len(samples):
                continue
            summary = _summarize_numeric_series(series, window_s=float(window_s),
                stability_threshold=float(threshold_v), change_threshold=float(change_threshold_v),
                event_limit=0)
            voltage_summaries.append({"pin": pin,
                "label": pin_labels[pin] if pin < len(pin_labels) else None,
                "node": spec["nodes"][pin] if pin < len(spec["nodes"]) else None,
                **_project_trace_numeric_summary(summary, include_examples=False)})
        if voltage_summaries:
            result["pin_voltage_v"] = voltage_summaries
        digital_count = max((len(row.get("digital", [])) for _, _, row in samples), default=0)
        digital_summaries = []
        for pin in range(min(digital_count, 16)):
            sequence = [(index, time_s, row["digital"][pin]) for index, time_s, row in samples
                        if len(row.get("digital", [])) > pin]
            if len(sequence) != len(samples):
                continue
            changes = [{"sample_index": after[0], "time_s": after[1], "from": before[2], "to": after[2]}
                       for before, after in zip(sequence, sequence[1:]) if after[2] != before[2]]
            digital_summaries.append({"pin": pin,
                "label": pin_labels[pin] if pin < len(pin_labels) else None,
                "first": sequence[0][2], "final": sequence[-1][2],
                "change_count": len(changes), "changes": changes[:6],
                "changes_omitted": max(0, len(changes) - 6)})
        if digital_summaries:
            result["digital"] = digital_summaries
        component_results.append(result)
    output.update({"mode": "component_state", "encoding": {"0": "L", "1": "H", "2": "X", "3": "Z"},
                   "components": component_results})
    return output


def circuit_read_trace(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    source, snapshot = _input(runtime, str(args.get("path") or ""))
    if snapshot.get("schema") != "aurex.pe-state.v1":
        raise ToolError("circuit_read_trace requires a .pe-state.json artifact")
    try:
        components = snapshot["spec"]["components"]
        only_digital = bool(components) and all(native_digital_type(c.get("type")) for c in components)
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError("State has an invalid native component catalog") from error
    if "nodes" in args and "component_ids" in args:
        raise ToolError("Select nodes for analog voltages or component_ids for digital TR samples, not both")
    if only_digital and "nodes" in args:
        raise ToolError("This is a pure digital state: use component_ids, not analog voltage nodes")
    trace = (snapshot.get("measurements") or {}).get("transient") or {}
    from ..trace_archive import read_series
    points = read_series(source, trace, "samples")
    if not isinstance(points, list) or not points:
        has_stimulus = bool((snapshot.get("measurements") or {}).get("stimulus_results"))
        guidance = ("Use circuit_read_stimulus with component_ids to read the separately recorded stimulus sequence. " if has_stimulus else
                    "No stimulus sequence was recorded, so circuit_read_stimulus is not available for this artifact. ")
        raise ToolError("This state has no measured trace; request tr_sample_every in a transient analysis. " + guidance)
    mode = args.get("mode", "samples")
    if mode not in ("samples", "summary"):
        raise ToolError("circuit_read_trace mode must be samples or summary")
    if mode == "summary":
        if any(key in args for key in ("sample_indices", "offset", "limit")):
            raise ToolError("summary mode scans the bounded recorded trace once; omit sample_indices/offset/limit")
        return _trace_summary(source, snapshot, trace, points, args)
    if only_digital or "component_ids" in args:
        return _digital_trace_page(source, snapshot, trace, points, args)
    # Mixed states may contain digital-only nodes with no MNA voltage sample.
    # Never invent a zero or leak KeyError merely because the node exists in
    # the topology. Available voltage nodes come from actual solver records.
    available = sorted({n for point in points for c in point["components"]
                        for n, _ in zip(c["nodes"], c.get("voltage", []))})
    nodes = args.get("nodes", available[:8])
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 8 or len(set(nodes)) != len(nodes) or any(n not in available for n in nodes):
        raise ToolError("nodes must select 1..8 unique nodes with recorded analog voltages; select component_ids with circuit_read_trace for digital TR samples")
    selected_points, selection = _select_trace_points(points, args)
    output = []
    selected_indices = []
    for sample_index, point in selected_points:
        values = {node: value for c in point["components"]
                  for node, value in zip(c.get("nodes", []), c.get("voltage", []))}
        missing = [node for node in nodes if node not in values]
        if missing:
            raise ToolError(f"No recorded analog voltage for {missing} at time {point['time_s']}; use component_ids for digital TR samples. Missing values are not zero.")
        selected_indices.append(sample_index)
        output.append([point["time_s"], *[values[node] for node in nodes]])
    return {"state_path": str(source), "units": {"time": "s", "voltage": "V relative to ground"},
            "columns": ["time_s", *nodes], "points": output,
            "available_node_count": len(available),
            **selection, "selected_sample_indices": selected_indices, "total_samples": len(points),
            "actual_stop_s": trace["actual_stop_s"], "interpolated": False,
            "note": "Only actual post-solve samples are returned; no interpolation, rerun, or inferred waveform between samples. Use circuit_inspect/query on the original circuit when another node must be identified."}


def circuit_compare_traces(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Compare aligned recorded digital TR samples across two immutable runs."""
    left_source, left = _input(runtime, str(args.get("left_path") or ""))
    right_source, right = _input(runtime, str(args.get("right_path") or ""))
    if left.get("schema") != "aurex.pe-state.v1" or right.get("schema") != "aurex.pe-state.v1":
        raise ToolError("circuit_compare_traces requires two .pe-state.json artifacts")
    selected = args.get("component_ids")
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 32
            or len(set(selected)) != len(selected)
            or any(not isinstance(cid, str) or not cid for cid in selected)):
        raise ToolError("component_ids must contain 1..32 unique nonempty component IDs")
    offset, limit = args.get("offset", 0), args.get("limit", 64)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 64:
        raise ToolError("Compare offset must be nonnegative and limit must be 1..64")
    from ..trace_archive import read_series

    def page(source, snapshot):
        trace = (snapshot.get("measurements") or {}).get("transient") or {}
        points = read_series(source, trace, "samples")
        if not isinstance(points, list) or not points:
            raise ToolError("Both states must contain recorded transient samples")
        merged = None
        for start in range(0, len(selected), 8):
            part = _digital_trace_page(source, snapshot, trace, points, {
                "component_ids": selected[start:start + 8], "offset": offset, "limit": limit,
            })
            if merged is None:
                merged = part
            else:
                if [(p["time_s"], p.get("completed_steps")) for p in merged["points"]] != [
                        (p["time_s"], p.get("completed_steps")) for p in part["points"]]:
                    raise ToolError("Recorded trace pages are not internally aligned")
                for target, addition in zip(merged["points"], part["points"]):
                    target["digital"].update(addition["digital"])
                    target["missing_component_ids"].extend(addition["missing_component_ids"])
                merged["components"].extend(part["components"])
        return merged

    left_page, right_page = page(left_source, left), page(right_source, right)
    left_points, right_points = left_page["points"], right_page["points"]
    aligned = (left_page["total_samples"] == right_page["total_samples"]
               and left_page["actual_stop_s"] == right_page["actual_stop_s"]
               and [(p["time_s"], p.get("completed_steps")) for p in left_points]
                   == [(p["time_s"], p.get("completed_steps")) for p in right_points])
    mismatches = []
    mismatch_components = set()
    if aligned:
        for left_point, right_point in zip(left_points, right_points):
            for cid in selected:
                before = left_point["digital"].get(cid)
                after = right_point["digital"].get(cid)
                if before != after:
                    mismatch_components.add(cid)
                    if len(mismatches) < 16:
                        mismatches.append({"time_s": left_point["time_s"], "component_id": cid,
                                           "left": before, "right": after})
    canonical_left = json.dumps(left_points, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    canonical_right = json.dumps(right_points, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    exact = aligned and canonical_left == canonical_right
    return {
        "kind": "recorded_digital_trace_comparison",
        "left_path": str(left_source), "right_path": str(right_source),
        "recorded_not_resimulated": True, "offset": offset,
        "compared_samples": len(left_points) if aligned else 0,
        "selected_component_count": len(selected), "time_aligned": aligned,
        "exact_match_across_runs": exact,
        "left_sha256": hashlib.sha256(canonical_left.encode()).hexdigest(),
        "right_sha256": hashlib.sha256(canonical_right.encode()).hexdigest(),
        "mismatch_component_ids": sorted(mismatch_components),
        "mismatch_examples": mismatches,
        "mismatch_examples_truncated": len(mismatch_components) > len({m["component_id"] for m in mismatches}),
        "digital_propagation_verified": bool(
            left_page["digital_propagation"].get("verified_per_step") is True
            and right_page["digital_propagation"].get("verified_per_step") is True),
        "note": ("This compares the same selected component pins at aligned recorded timestamps across two runs. "
                 "Changes between timestamps inside either run are not evidence of cross-run nondeterminism."),
    }


_SPEC = {"type": "object", "description": "Circuit spec: components[{id,type,nodes,params,label?:string,position?:[x,y,z],rotation?:[x,y,z]}], optional title and camera. IDs are stable unique connection identities; optional labels are display/port names, not IDs. Position is native xyz; rotation is Euler degrees. Saved labels/coordinates are preserved; only missing positions use generated layout. Use circuit_catalog for exact models/pins."}
_PATH = {"type": "string", "description": "Local .sav/.plsav/.circuit.json/.pe-state.json artifact path returned by experiment/circuit tools. A .pe-state.json reopens a recorded solver state without rerunning it."}
_WITH_IMAGE = {"type": "boolean", "default": False, "description": "Default false: return structured data and immutable artifacts without generating or attaching SVG/PNG. Set true for an explicit visual/spatial question, or once for a targeted view=schematic after data-only inspection proves a complex topology still has a concrete spatial/wiring ambiguity. Images are supplementary and do not change simulation or saved circuit state."}
_DIGITAL_STATE = {"type": "integer", "enum": [0, 1, 2, 3]}
_STIMULUS = {"type": "array", "maxItems": 128,
    "description": "Digital-only representative input frames [{set:{exact_input_component_id:0|1|2|3}}]. Omitted inputs keep their current values; an empty frame holds all inputs. Native advances 10ns/frame on the same circuit instance within this call. For repeated UUIDs prefer stimulus_table. This is a per-call limit, not a task budget; additional requested samples can use separate batches, but separate calls initialize from the source spec, not implicit continuation of hidden sequential state.",
    "items": {"type": "object", "additionalProperties": False,
              "properties": {"set": {"type": "object", "additionalProperties": _DIGITAL_STATE}}}}
_STIMULUS_TABLE = {"type": "object", "additionalProperties": False, "required": ["inputs", "vectors"],
    "description": "Compact alternative to stimulus: list only the exact discovered Logic Input IDs that this test needs to change, then one matching-width row per frame. You do not need to repeat all circuit inputs. For one/few changing pins, sparse stimulus [{set:{exact_input_id:0}}, {set:{exact_input_id:1}}, {set:{}}] is simpler. Never zero/reset unrelated program or control inputs just to fill a table. Columns follow the supplied inputs order, not guessed bit order; no clock/reset role is inferred. Values are strict integers 0=L,1=H,2=X,3=Z; omitted inputs retain their current values. Native timing remains 10ns/frame within the call. Mutually exclusive with explicit stimulus (including inline spec.stimulus). Use representative samples, not an unsolicited exhaustive sweep; the existing 128-frame per-call limit is unchanged. Separate batches initialize from the source spec, not implicit continuation of hidden sequential state.",
    "properties": {"inputs": {"type": "array", "minItems": 1, "maxItems": 15, "uniqueItems": True,
        "items": {"type": "string", "minLength": 1, "maxLength": 128},
        "description": "Exact digital_input component IDs from circuit_inspect(interface_only=true); not display labels, output IDs or internal gates. For more than 15 explicitly changed inputs use sparse stimulus frames instead of repeating a wide row on every frame."},
        "vectors": {"type": "array", "minItems": 1, "maxItems": 128,
            "items": {"type": "array", "minItems": 1, "items": _DIGITAL_STATE}}}}
_TR_INTERACTIONS = {"type": "array", "maxItems": 128,
    "description": "Time-aligned physical/model controls for analysis=tr. First read circuit_inspect(controls_only=true), then supply [{time_s, set:{exact_control_id:value}}]. Times are seconds, strictly increasing and exactly on the tr_step grid; time 0 applies before step 1. Omitted controls hold state. Buttons/switches/rheostats remain analog devices; digital inputs and source voltage may share the same mixed TR timeline.",
    "items": {"type": "object", "additionalProperties": False, "required": ["time_s", "set"],
        "properties": {"time_s": {"type": "number", "minimum": 0},
            "set": {"type": "object", "minProperties": 1, "maxProperties": 32,
                    "additionalProperties": {"type": "number"}}}}}


def register_circuit_tools(registry: ToolRegistry) -> None:
    definitions = [
        ("circuit_catalog", "List supported real Phy-Engine electrical models, exact pin order, SI parameters, defaults and .sav export support.", {}, [], circuit_catalog),
        ("circuit_read_stimulus", "Read actual recorded digital stimulus results from an immutable native state, without rerunning. Select up to 8 component IDs and at most 16 steps. Use circuit_inspect to discover IDs; arrays are real L/H/X/Z samples in native pin order.",
         {"path": _PATH, "component_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8, "uniqueItems": True},
          "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 16}}, ["path"], circuit_read_stimulus),
        ("circuit_read_trace", "Read or summarize actual timestamped samples from a saved native TR state, without rerun or interpolation. Prefer mode=summary for stability time, first/last/largest change, repeated-change/direction-reversal evidence (not proof of periodic oscillation), relay/switch model-state transitions, op-amp pin ranges, and a whole-trace answer in one call; supply up to 8 analog nodes OR component_ids and explicit thresholds. Use mode=samples only for a few exact boundary frames via sample_indices (up to 16); never page an entire trace to calculate a summary manually. Historical records retain original values. For the separate digital stimulus sequence use circuit_read_stimulus.",
         {"path": _PATH, "nodes": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8, "uniqueItems": True},
          "component_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8, "uniqueItems": True,
                            "description": "summary: any exact native component IDs, including relays/op-amps/switches; samples: native digital component IDs only."},
          "mode": {"enum": ["samples", "summary"], "default": "samples"},
          "stability_window_s": {"type": "number", "exclusiveMinimum": 0, "default": 1.0,
                                  "description": "summary only: minimum duration of a sampled stability window."},
          "stability_threshold_v": {"type": "number", "minimum": 0, "default": 0.1,
                                     "description": "summary only: maximum sampled voltage range for stability."},
          "change_threshold_v": {"type": "number", "minimum": 0,
                                  "description": "summary only: minimum consecutive voltage change reported as an event; defaults to stability_threshold_v."},
          "from_time_s": {"type": "number", "minimum": 0,
                           "description": "summary only: optional inclusive recorded-time start."},
          "to_time_s": {"type": "number", "minimum": 0,
                         "description": "summary only: optional inclusive recorded-time end."},
          "sample_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1, "maxItems": 16, "uniqueItems": True,
                             "description": "Exact zero-based recorded sample indices. Use sparse boundary/final frames instead of paging a long interaction trace. Mutually exclusive with offset/limit."},
          "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 64}}, ["path"], circuit_read_trace),
        ("circuit_compare_traces", "Compare two immutable digital TR state artifacts at aligned recorded timestamps. Use this for repeatability/determinism claims; it distinguishes changes inside one run from differences between runs and returns exact hashes plus bounded mismatch examples.",
         {"left_path": _PATH, "right_path": _PATH,
          "component_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                            "maxItems": 32, "uniqueItems": True,
                            "description": "Exact native digital component IDs to compare across both runs."},
          "offset": {"type": "integer", "minimum": 0},
          "limit": {"type": "integer", "minimum": 1, "maximum": 64, "default": 64}},
         ["left_path", "right_path", "component_ids"], circuit_compare_traces),
        ("circuit_inspect", "Inspect original PLSAV, native design or recorded state. Default is bounded data-only output; focused results include explicit saved-top-view nearest/left/right/above/below relations and exact shared nodes instead of asking the model to interpret bare xyz. with_image=true on a focus/query defaults to a PE-rendered wired schematic. interface_only=true lists digital I/O. controls_only=true lists exact agent-operable analog controls plus digital inputs. Exact focus also returns a bounded native edit contract so imported analog circuits can be edited then simulated without rebuilding them. An exact N<number> query returns connected components per bounded page. Images preserve positions, never prove fresh simulation.",
         {"path": _PATH, "with_image": _WITH_IMAGE, "interface_only": {"type": "boolean", "default": False, "description": "Data-only digital I/O listing, mutually exclusive with with_image=true and focus/query/camera/view. Original order, default/max 64 per page."},
          "controls_only": {"type": "boolean", "default": False, "description": "Data-only interactive-control listing. Use returned exact IDs and value contracts in circuit_analyze.tr_interactions. Mutually exclusive with interface_only, image and focus/query/camera/view."},
          "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 64, "description": "interface_only: 1..64 (default64). Otherwise 1..24 (default8); 4..8 recommended for readable diagrams and up to 12 for a targeted schematic."}, "view": {"enum": ["auto", "overview", "region", "spatial", "topology", "schematic"]}, "projection": {"enum": ["isometric", "top"]},
          "focus_ids": {"type": "array", "items": {"type": "string"}}, "focus_id": {"type": "string"},
          "query": {"type": "string", "description": "Exact N<number> selects all components connected to that node. On an original PLSAV, exact C<number> is its displayed ref; on imported native/state paths, an available original source_ref C<number> takes precedence over the revision-local display ref and resolves stable native IDs. Otherwise use literal component text search. Never read_context/find a full netlist artifact."}, "camera": _CAMERA}, ["path"], circuit_inspect),
        ("circuit_query_many", "Read several exact circuit targets in one read-only call with strict field selection. A call without fields returns identity and match evidence only. Request only the values needed for the next decision: pins; native_type; spatial; one exact saved property such as properties.高电平 or properties.低电平; one exact recorded measurement such as measurements.digital; or one exact native edit parameter such as edit.r. Only all=true returns complete matched records. fields and all are mutually exclusive. Each query independently reports ok/error/missing_fields, and normal repeated reads remain allowed. No images or simulation.",
         {"path": _PATH,
          "queries": {"type": "array", "minItems": 1, "maxItems": 24,
                      "items": {"type": "string", "minLength": 1, "maxLength": 128},
                      "description": "Up to 24 exact refs/Identifiers/nodes or literal component queries to inspect together."},
          "limit": {"type": "integer", "minimum": 1, "maximum": 8, "default": 1,
                    "description": "Maximum ordinary circuit_inspect matches returned for each query."},
          "fields": {"type": "array", "maxItems": 24, "uniqueItems": True,
                     "items": {"type": "string", "minLength": 1, "maxLength": 196},
                     "description": "Optional exact output selectors. Allowed: pins, native_type, spatial, properties.<exact-name>, measurements.<exact-path>, edit.<exact-param>. Omit for identity-only results. Broad properties/measurements/edit selectors are rejected."},
          "all": {"type": "boolean", "default": False,
                  "description": "Return every available field for matched records. Expensive; use only when the complete selected records are genuinely required. Mutually exclusive with fields."}},
         ["path", "queries"], circuit_query_many),
        ("circuit_create", "Create a local electrical design and native circuit artifact. Data-only by default; explicitly set with_image=true for a diagram. view=schematic produces an exact-node, position-aware, automatically wired C++ projection. Exports .sav only when model mappings are compatible; never publishes. Use Verilog tool for digital modules.",
         {"spec": _SPEC, "with_image": _WITH_IMAGE, "view": {"enum": ["spatial", "topology", "schematic"]}, "projection": {"enum": ["isometric", "top"]}, "camera": _CAMERA}, ["spec"], circuit_create),
        ("circuit_edit", "Create an editable native workspace revision from an original PLSAV, native design, or recorded state by adding/removing components, changing native params, or connecting a pin to a named node. First focus the target with circuit_inspect and use its edit_contract exact ID/params/nodes. Original stays intact; run circuit_analyze on the returned circuit_path; nothing is published. A no-op update is rejected.",
         {"path": _PATH, "with_image": _WITH_IMAGE, "camera": _CAMERA, "operations": {"type": "array", "items": {"type": "object", "properties": {
             "action": {"enum": ["add", "update", "remove", "connect"]}, "id": {"type": "string"}, "component": {"type": "object"},
             "params": {"type": "object"}, "nodes": {"type": "array", "items": {"type": "string"}}, "label": {"type": ["string", "null"], "maxLength": 4096}, "position": {"type": "array", "items": {"type": "number"}}, "rotation": {"type": "array", "items": {"type": "number"}}, "pin": {"type": "integer"}, "node": {"type": "string"}}}}}, ["path", "operations"], circuit_edit),
        ("circuit_analyze", "Run actual native Phy-Engine DC, AC, transient or digital analysis. On PLSAV import, PhysicsLab digital-device maximum-current fields are preserved as provenance but mapped to PE's maximum finite threshold because the PhysicsLab runtime does not enforce them; native PE protection models are unchanged. For buttons, analog switches, SPDT/DPDT selectors, slide rheostats, source voltage or mixed digital inputs, first discover exact IDs with circuit_inspect(controls_only=true), then use tr_interactions so each change occurs before its specified native TR solve. Legacy stimulus is a separate digital-only post-TR sequence. Returns measurements and immutable state; image only when explicitly requested.",
         {"path": _PATH, "spec": _SPEC, "with_image": _WITH_IMAGE, "analysis": {"enum": ["op", "dc", "ac", "acop", "tr", "trop"]},
          "view": {"enum": ["spatial", "topology", "schematic"]}, "projection": {"enum": ["isometric", "top"]}, "camera": _CAMERA,
          "focus_ids": {"type": "array", "items": {"type": "string"}}, "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 24},
          "ground_node": {"type": "string"}, "ac_omega": {"type": "number"},
          "g_min_siemens": {"type": "number", "minimum": 0, "maximum": 0.001, "description": "Optional explicit numerical leakage to ground; PhysicsLab imports use 1e-12 S (1 Tohm) to condition disconnected catalog terminals without materially loading real circuits."},
          "tr_step": {"type": "number", "description": "Seconds per native solver step; at most 10000 steps."}, "tr_stop": {"type": "number", "description": "Transient stop time in seconds."},
          "tr_sample_every": {"type": "integer", "minimum": 1, "description": "Sample actual state every N solver steps plus endpoint; at most 201 samples. For analysis=tr only. Omit for final state only."},
          "tr_initialize_dc": {"type": "boolean", "default": False, "description": "analysis=tr analog-only opt-in: solve the DC operating point on the same live circuit before transient stepping. Use for biased steady-state small-signal gain/phase analysis; leave false for power-on/startup tests."},
          "digital_steps_per_tr_step": {"type": "integer", "minimum": 1, "maximum": 64, "default": 1,
              "description": "analysis=tr only: execute this many complete digital propagation ticks after EACH physical TR solver step, before any sample. Default 1; not a digital time interval or accuracy/convergence knob. Larger counts advance tick-based digital devices; tr_step still controls physical time. Independent of sample_every and of the separate stimulus sequence."},
          "tr_interactions": _TR_INTERACTIONS,
          "digital_clock_ticks": {"type": "integer", "minimum": 0, "maximum": 10000},
          "stimulus": _STIMULUS, "stimulus_table": _STIMULUS_TABLE}, [], circuit_analyze),
    ]
    for name, description, props, required, handler in definitions:
        parameters = {"type": "object", "properties": props, "required": required}
        if name == "circuit_analyze":
            parameters["not"] = {"required": ["stimulus", "stimulus_table"]}
        if name == "circuit_read_trace":
            parameters["not"] = {"required": ["nodes", "component_ids"]}
        registry.register(ToolSpec(name=name, description=description, parameters=parameters, handler=handler))
