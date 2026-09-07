"""Audited PhysicsLab semiconductor-to-PE engineering mappings.

The public PhysicsLab schema gives terminal order and a small number of
editor parameters, but not the closed-source device equations.  These
mappings therefore preserve every terminal and use the saved forward working
point to parameterise PE's Shockley model.  The approximation and all defaults
remain attached to every primitive as provenance; saved Statistics are never
used as a fresh solve or as a fitted answer.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from .registry import ToolError


_SDK = ("https://github.com/SekaiArendelle/physicslab/blob/"
        "fa95b96910dd0fd4e09cf27e24cefaf9b91798ad/"
        "physicslab/circuit/elements/")
_PIN_COUNTS = {
    "Basic Diode": 2,
    "Light-Emitting Diode": 2,
    "Photodiode": 2,
    "Color Light-Emitting Diode": 4,
    "Dual Light-Emitting Diode": 2,
    "Rectifier": 4,
    "Comparator": 3,
    "N-MOSFET": 3,
    "P-MOSFET": 3,
}


def _number(props: dict, key: str, cid: str, *, optional: bool = False,
            default: float = 0.0) -> float:
    if optional and key not in props:
        return default
    value = props.get(key)
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ToolError(f"{cid}: original {key} must be an explicit finite number")
    return float(value)


def _pins(el: dict, count: int) -> dict[int, str]:
    raw = el.get("pins")
    if not isinstance(raw, list) or len(raw) != count:
        raise ToolError(f"{el.get('id')}: expected all {count} original semiconductor pins")
    pins: dict[int, str] = {}
    for item in raw:
        if (not isinstance(item, dict) or type(item.get("pin")) is not int
                or item["pin"] not in range(count) or item["pin"] in pins
                or not isinstance(item.get("node"), str)
                or not 0 < len(item["node"]) <= 128):
            raise ToolError(f"{el.get('id')}: missing, duplicate or invalid original pin mapping")
        pins[item["pin"]] = item["node"]
    return pins


def _pose(el: dict, key: str) -> list[float]:
    value = el.get(key)
    try:
        valid = isinstance(value, list) and len(value) == 3 and all(
            type(item) in (int, float) and math.isfinite(item) for item in value)
    except OverflowError:
        valid = False
    if not valid:
        raise ToolError(f"{el.get('id')}: original finite {key} is required")
    return [float(item) for item in value]


def _pn_params(forward_v: float, working_current: float, reverse_v: float,
               cid: str) -> tuple[dict[str, float], dict[str, Any]]:
    if forward_v <= 0 or working_current <= 0 or reverse_v < 0:
        raise ToolError(f"{cid}: diode forward voltage/current must be positive and reverse rating nonnegative")
    # N=2 is an explicit generic silicon/LED engineering choice.  Solve Is so
    # that the saved (Vf, If) pair lies exactly on the PE Shockley curve at 27C.
    ideality = 2.0
    temp_c = 27.0
    thermal_v = 8.617333262145e-5 * (temp_c + 273.15)
    exponent = forward_v / (ideality * thermal_v)
    if not math.isfinite(exponent) or exponent > 700:
        raise ToolError(f"{cid}: diode working point exceeds finite native exponential range")
    saturation = working_current / math.expm1(exponent)
    if not math.isfinite(saturation) or saturation <= 0:
        raise ToolError(f"{cid}: diode working point cannot parameterise a finite PN model")
    params = {"is": saturation, "n": ideality, "isr": 0.0, "nr": 2.0,
              "temp_c": temp_c, "ibv": max(working_current * .01, 1e-12),
              "bv": reverse_v if reverse_v > 0 else 40.0,
              "bv_set": int(reverse_v > 0), "area": 1.0}
    derivation = {
        "kind": "shockley_working_point",
        "saved_forward_voltage_v": forward_v,
        "saved_working_current_a": working_current,
        "native_ideality": ideality,
        "native_temperature_c": temp_c,
        "derived_saturation_current_a": saturation,
        "saved_reverse_rating_v": reverse_v,
        "reverse_breakdown_enabled": bool(reverse_v > 0),
    }
    return params, derivation


def import_element(el: dict, *, scene: dict) -> list[dict[str, Any]] | None:
    if not isinstance(el, dict) or el.get("type") not in _PIN_COUNTS:
        return None
    kind, cid = el["type"], el.get("id")
    if not isinstance(cid, str) or not 0 < len(cid) <= 128:
        raise ToolError("Semiconductor import requires the original component ID")
    if el.get("is_broken", el.get("IsBroken", False)):
        raise ToolError(f"{cid}: broken-device behavior requires the complete SAV damage adapter")
    props = el.get("properties")
    if not isinstance(props, dict):
        raise ToolError(f"{cid}: original semiconductor properties are required")
    if (not isinstance(scene, dict) or not isinstance(scene.get("components"), list)
            or not isinstance(scene.get("wires"), list)):
        raise ToolError(f"{cid}: complete original semiconductor scene is required")
    ids = [item.get("id") for item in scene["components"] if isinstance(item, dict)]
    if len(ids) != len(scene["components"]) or any(not isinstance(item, str) for item in ids):
        raise ToolError(f"{cid}: malformed original scene component IDs")
    if ids.count(cid) != 1 or len(ids) != len(set(ids)):
        raise ToolError(f"{cid}: scene must contain every original component exactly once")
    pins = _pins(el, _PIN_COUNTS[kind])
    position, rotation = _pose(el, "position"), _pose(el, "rotation")
    label = el.get("label")
    if label is not None and not isinstance(label, str):
        raise ToolError(f"{cid}: original label must be text or null")

    wired: set[int] = set()
    for wire in scene["wires"]:
        if not isinstance(wire, dict):
            raise ToolError(f"{cid}: malformed original wire record")
        for id_key, pin_key in (("Source", "SourcePin"), ("Target", "TargetPin")):
            if wire.get(id_key) == cid:
                pin = wire.get(pin_key)
                if type(pin) is not int or pin not in pins:
                    raise ToolError(f"{cid}: wire references an invalid original pin")
                wired.add(pin)

    used = set(ids)
    digest = hashlib.sha256(cid.encode()).hexdigest()

    def allocate(role: str) -> str:
        base = "__pl_semiconductor_" + digest + "_" + role
        value, suffix = base, 0
        while value in used:
            suffix += 1
            value = base + "_" + str(suffix)
        used.add(value)
        return value

    base = {
        "model_id": kind,
        "identifier": cid,
        "raw_properties": copy.deepcopy(props),
        "raw_statistics": copy.deepcopy(el.get("statistics", {})),
        "pin_mapping": [{"pl_pin": pin, "node": pins[pin],
                         "externally_wired": pin in wired} for pin in range(len(pins))],
        "implicit_references": [],
        "mapping_reference": [_SDK + ("sensor.py" if kind == "Photodiode" else
                                       "other_circuit.py" if "Light-Emitting" in kind else
                                       "artificial_circuit.py")],
        "statistics_source": "original saved values, never fresh PE measurements",
        "numerical_equivalence_to_original": False,
        "support_level": "explicit_native_engineering_mapping",
    }
    output: list[dict[str, Any]] = []

    def append(component_type: str, nodes: list[str], params: dict, role: str,
               primitive_pins: list[int], assumptions: list[str],
               extra_source: dict[str, Any] | None = None) -> None:
        primary = not output
        source = copy.deepcopy(base)
        source.update({"parent_identifier": cid, "is_helper": not primary,
                       "decomposition_role": role,
                       "primitive_pin_mapping": primitive_pins,
                       "assumptions": assumptions})
        if extra_source:
            source.update(copy.deepcopy(extra_source))
        item = {"id": cid if primary else allocate(role), "type": component_type,
                "nodes": nodes, "params": params, "position": copy.deepcopy(position),
                "rotation": copy.deepcopy(rotation), "pl_source": source}
        if primary and label:
            item["label"] = label
        output.append(item)

    common = [
        "Every original external terminal and node is preserved; no saved wire is added, removed or grounded.",
        "Saved Statistics are retained only as historical observations and do not initialise or fit this solve.",
    ]
    if kind in {"Basic Diode", "Light-Emitting Diode", "Photodiode"}:
        forward = _number(props, "前向压降", cid)
        current_key = "工作电流" if kind == "Light-Emitting Diode" else "额定电流"
        current = _number(props, current_key, cid)
        reverse = (_number(props, "反向耐压", cid) if "反向耐压" in props
                   else _number(props, "击穿电压", cid, optional=True))
        params, derivation = _pn_params(forward, current, reverse, cid)
        assumptions = common + [
            "PE uses a Shockley PN approximation anchored exactly at the saved forward-voltage/current working point; the original app equation is not public.",
            "A zero/missing reverse-breakdown field disables native avalanche instead of inventing a threshold.",
        ]
        if kind == "Photodiode":
            assumptions += [
                "The save contains sensitivity and response time but no illumination sample. This solve is the dark PN characteristic; no photocurrent is invented.",
                "Photodiode optical transients require an explicit future illumination stimulus and are not inferred from archived Statistics.",
            ]
        append("diode", [pins[0], pins[1]], params, "pn_junction", [0, 1],
               assumptions, {"parameter_derivation": derivation})
    elif kind in {"Color Light-Emitting Diode", "Dual Light-Emitting Diode"}:
        forward = _number(props, "前向压降", cid)
        current = _number(props, "工作电流", cid)
        reverse = _number(props, "反向耐压", cid)
        params, derivation = _pn_params(forward, current, reverse, cid)
        assumptions = common + [
            "Each optical die is an independent Shockley PN approximation anchored at the saved working point; brightness is proportional to positive die current and is not a calibrated photometric value.",
        ]
        if kind == "Color Light-Emitting Diode":
            # PL0/1/2 are independent dies and PL3 is their common cathode.
            pairs = [(0, 3), (1, 3), (2, 3)]
        else:
            # The two-colour, two-terminal package contains antiparallel dies.
            pairs = [(0, 1), (1, 0)]
        for index, (anode, cathode) in enumerate(pairs):
            append("diode", [pins[anode], pins[cathode]], copy.deepcopy(params),
                   f"die_{index}", [anode, cathode], assumptions,
                   {"parameter_derivation": copy.deepcopy(derivation), "optical_die": index})
    elif kind == "Rectifier":
        forward = _number(props, "前向压降", cid)
        rated = _number(props, "额定电流", cid)
        params, derivation = _pn_params(forward, rated, 0.0, cid)
        assumptions = common + [
            "Bridge convention is PL0/PL1 AC and PL2(+)/PL3(-), matching PE's documented full-bridge terminal order.",
            "Four explicit PN primitives preserve the saved forward drop rather than using PE's unrelated default bridge diodes.",
        ]
        for role, a, b in (("a_to_plus", 0, 2), ("b_to_plus", 1, 2),
                           ("minus_to_a", 3, 0), ("minus_to_b", 3, 1)):
            append("diode", [pins[a], pins[b]], copy.deepcopy(params), role, [a, b],
                   assumptions, {"parameter_derivation": copy.deepcopy(derivation)})
    elif kind == "Comparator":
        low, high = _number(props, "低电平", cid), _number(props, "高电平", cid)
        if low > high:
            raise ToolError(f"{cid}: comparator low level exceeds high level")
        append("comparator", [pins[1], pins[2], pins[0]],
               {"low_v": low, "high_v": high}, "comparator", [1, 2, 0],
               common + [
                   "PL1 is non-inverting input, PL2 is inverting input and PL0 is output, exactly matching the public SDK pin names.",
                   "Native comparator is ideal and instantaneous; no undocumented input impedance, delay, output resistance or damage is invented.",
               ])
    else:
        gain = _number(props, "放大系数", cid)
        threshold = _number(props, "阈值电压", cid)
        max_power = _number(props, "最大功率", cid)
        if gain <= 0 or threshold <= 0 or max_power <= 0:
            raise ToolError(f"{cid}: MOSFET gain, threshold and maximum power must be positive")
        if kind == "N-MOSFET":
            component_type, order, vth = "nmos", [2, 0, 1], threshold
        else:
            component_type, order, vth = "pmos", [1, 0, 2], -threshold
        append(component_type, [pins[pin] for pin in order],
               {"kp": gain, "lambda": 0.0, "vth": vth}, "mosfet", order,
               common + [
                   "Saved 放大系数 is used as PE Level-1 Kp (A/V^2); saved threshold magnitude and public G/D/S pin identities are preserved.",
                   "Channel-length modulation lambda is explicitly 0 because the save has no corresponding field; body is tied to source by the PE model.",
                   "Saved maximum power is enforced by the complete-SAV damage adapter on the drain path; no thermal time constant is present in the public format.",
               ], {"engineering_defaults": {"lambda_per_v": 0.0,
                                              "bulk_connection": "source"}})
    return output
