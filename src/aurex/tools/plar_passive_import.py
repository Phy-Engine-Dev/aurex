"""Conservative PL passive-device decomposition into existing PE primitives.

The input is circuit_view's full netlist, not an SDK object. The original scene
and component dictionaries are never mutated. This module does not decide any
external publication permission or make network calls.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from .registry import ToolError


_SDK = "https://github.com/SekaiArendelle/physicslab/blob/fa95b96910dd0fd4e09cf27e24cefaf9b91798ad/"
_RECOGNIZED = {"Simple Switch", "Push Switch", "Air Switch", "SPDT Switch",
               "DPDT Switch", "Resistance Box", "Slide Rheostat", "Resistance Law"}


def _number(props: dict, key: str, cid: str) -> int | float:
    value = props.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{cid}: original {key} must be an explicit finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ToolError(f"{cid}: original {key} must be finite")
    return value


def _state(props: dict, key: str, allowed: tuple[int, ...], cid: str) -> int:
    value = _number(props, key, cid)
    if value not in allowed:
        raise ToolError(f"{cid}: unknown original {key} state {value!r}; expected one of {allowed}")
    return int(value)


def _pins(el: dict, count: int) -> dict[int, str]:
    cid = el["id"]
    raw = el.get("pins")
    if not isinstance(raw, list) or len(raw) != count:
        raise ToolError(f"{cid}: expected all {count} original pins, including unconnected pins")
    out: dict[int, str] = {}
    for pin in raw:
        if not isinstance(pin, dict) or type(pin.get("pin")) is not int:
            raise ToolError(f"{cid}: invalid original pin index")
        index, node = pin["pin"], pin.get("node")
        if index in out or index not in range(count) or not isinstance(node, str) or not 0 < len(node) <= 128:
            raise ToolError(f"{cid}: missing, duplicate or invalid original pin/node mapping")
        out[index] = node
    return out


def _pose(el: dict, key: str) -> list:
    value = el.get(key)
    try:
        valid = isinstance(value, list) and len(value) == 3 and all(
            not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in value)
    except OverflowError:
        valid = False
    if not valid:
        raise ToolError(f"{el['id']}: expected original finite {key}, not a generated layout")
    return copy.deepcopy(value)


def import_element(el: dict, *, scene: dict) -> list[dict] | None:
    """Return a faithful static connectivity decomposition or fail explicitly.

    Snapshot switch state is retained; no button press/release timing, contact
    bounce or interaction event is invented. PE's switch primitive represents an
    open contact with its configured large r_open, not infinite resistance.
    """
    if not isinstance(el, dict) or el.get("type") not in _RECOGNIZED:
        return None
    kind, cid = el["type"], el.get("id")
    if not isinstance(cid, str) or not 0 < len(cid) <= 128:
        raise ToolError("Passive PL import requires the original unique component ID")
    if kind == "Resistance Law":
        # The broader device adapter owns the four aligned sample-wire
        # topology and its SI R=rho*L/A derivation.
        return None
    props = el.get("properties")
    if not isinstance(props, dict):
        raise ToolError(f"{cid}: original properties are required")
    if el.get("is_broken", el.get("IsBroken", False)):
        raise ToolError(f"{cid}: broken-device behavior requires the complete SAV damage adapter")
    for resistance_field in ("内阻", "接触电阻"):
        if resistance_field in props and _number(props, resistance_field, cid) != 0:
            raise ToolError(f"{cid}: nonzero {resistance_field} cannot be discarded by an ideal-contact mapping")
    if not isinstance(scene, dict) or not isinstance(scene.get("components"), list) or not isinstance(scene.get("wires"), list):
        raise ToolError(f"{cid}: full original scene components and wires are required")
    if not all(isinstance(c, dict) for c in scene["components"]):
        raise ToolError(f"{cid}: malformed original scene component")
    ids = [c.get("id") for c in scene["components"] if isinstance(c, dict)]
    if ids.count(cid) != 1 or any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ToolError(f"{cid}: scene must identify the source exactly once with unique original IDs")
    used = set(ids)
    pin_count = {"Simple Switch": 2, "Push Switch": 2, "Air Switch": 2,
                 "SPDT Switch": 3, "DPDT Switch": 6, "Resistance Box": 2,
                 "Slide Rheostat": 4}[kind]
    pins = _pins(el, pin_count)
    position, rotation = _pose(el, "position"), _pose(el, "rotation")
    label = el.get("label")
    if label is not None and not isinstance(label, str):
        raise ToolError(f"{cid}: original display label must be text or null")

    # A pin's existence in the renderer's schema is not evidence it is wired.
    wired = set()
    for wire in scene["wires"]:
        if not isinstance(wire, dict):
            raise ToolError(f"{cid}: malformed original wire record")
        for id_key, pin_key in (("Source", "SourcePin"), ("Target", "TargetPin")):
            if wire.get(id_key) == cid:
                pin = wire.get(pin_key)
                if type(pin) is not int or pin not in pins:
                    raise ToolError(f"{cid}: wire references an invalid original pin")
                wired.add(pin)

    assumptions = [
        "Original static component state only; no mechanical interaction timing or contact bounce is synthesized.",
        "External node names and all original pins are preserved; an unwired pin is not grounded or assigned an invented voltage.",
        "PE open switches use the engine's finite r_open approximation; contact loading is not claimed pointwise-identical to the original app.",
    ]
    base_source = {"model_id": kind, "identifier": cid,
        "raw_properties": copy.deepcopy(props), "raw_statistics": copy.deepcopy(el.get("statistics", {})),
        "pin_mapping": [{"pl_pin": p, "node": node, "externally_wired": p in wired} for p, node in sorted(pins.items())],
        "implicit_references": [], "assumptions": assumptions,
        "mapping_reference": [_SDK + "physicslab/circuit/elements/basic_circuit.py",
                              _SDK + "physicslab/enums.py"],
        "numerical_equivalence_to_original": False}
    output = []

    def allocate(role: str) -> str:
        # Fixed-length digest keeps even a 128-character original ID in bounds.
        prefix = "__pl_passive_" + hashlib.sha256(cid.encode("utf-8")).hexdigest() + "_" + role
        candidate, suffix = prefix, 0
        while candidate in used:
            suffix += 1
            candidate = prefix + "_" + str(suffix)
        used.add(candidate)
        return candidate

    def append(component_type: str, pair: tuple[int, int], params: dict, role: str,
               interaction: dict | None = None):
        primary = not output
        source = copy.deepcopy(base_source)
        source.update({"decomposition_role": role, "parent_identifier": cid,
                       "is_helper": not primary, "primitive_pin_mapping": list(pair)})
        item = {"id": cid if primary else allocate(role), "type": component_type,
                "nodes": [pins[p] for p in pair], "params": params,
                "position": copy.deepcopy(position), "rotation": copy.deepcopy(rotation), "pl_source": source}
        if interaction is not None:
            item["interaction"] = {"control_id": cid, "source_model_id": kind,
                                   "role": role, **copy.deepcopy(interaction)}
        if primary and label:
            item["label"] = label
        output.append(item)

    if kind == "Resistance Box":
        resistance = _number(props, "电阻", cid)
        if resistance < 0:
            raise ToolError(f"{cid}: negative resistance-box value is not supported")
        # Range sliders are preserved as source metadata, not applied as clamps.
        if resistance == 0:
            append("switch", (0, 1), {"closed": 1}, "zero_ohm_contact")
        else:
            append("resistor", (0, 1), {"r": resistance}, "resistance")
    elif kind in {"Simple Switch", "Push Switch", "Air Switch"}:
        current = _state(props, "开关", (0, 1), cid)
        if kind == "Push Switch" and "默认开关" in props:
            _state(props, "默认开关", (0, 1), cid)
        # PhysicsLab uses a 1 nOhm closed contact rather than an ideal short.
        # This is observable in public flash-ADC saves where several clamped
        # amplifier outputs are bussed through closed switches: the recorded
        # voltage/current differences are exactly consistent with 1e-9 ohm.
        # Keeping it finite also avoids turning a valid lossy bus into
        # contradictory ideal voltage-source constraints in native MNA.
        append("switch", (0, 1), {"closed": current, "r_closed": 1e-9}, "contact",
               {"kind": "spst", "value_name": "pressed" if kind == "Push Switch" else "closed",
                "allowed": [0, 1], "current": current,
                "momentary": kind == "Push Switch"})
    elif kind in {"SPDT Switch", "DPDT Switch"}:
        state = _state(props, "开关", (0, 1, 2), cid)
        # SDK pins: SPDT l/mid/r = 0/1/2. DPDT lower = 0/1/2,
        # upper = 3/4/5. Enum explicitly defines OFF=0, LEFT=1, RIGHT=2.
        for offset in ((0,) if kind == "SPDT Switch" else (0, 3)):
            pole = "pole0" if offset == 0 else "pole1"
            common = {"kind": "spdt" if kind == "SPDT Switch" else "dpdt",
                      "value_name": "position", "allowed": [0, 1, 2], "current": state}
            append("switch", (offset + 1, offset), {"closed": int(state == 1), "r_closed": 1e-9},
                   pole + "_left", common)
            append("switch", (offset + 1, offset + 2), {"closed": int(state == 2), "r_closed": 1e-9},
                   pole + "_right", common)
    else:
        rated = float(_number(props, "额定电阻", cid))
        position_value = float(_number(props, "滑块位置", cid))
        floor = 1e-5
        expected1, expected2 = max(floor, rated * position_value), max(floor, rated * (1 - position_value))
        r1 = float(_number(props, "电阻1", cid)) if "电阻1" in props else expected1
        r2 = float(_number(props, "电阻2", cid)) if "电阻2" in props else expected2
        if rated <= 0 or not 0 <= position_value <= 1 or r1 <= 0 or r2 <= 0:
            raise ToolError(f"{cid}: Slide Rheostat requires positive rated/segment resistances and position in 0..1")
        floor = min(r1, r2, floor)
        tolerance = max(2e-5, rated * 1e-5)
        expected1, expected2 = max(floor, rated * position_value), max(floor, rated * (1 - position_value))
        if abs(r1 - expected1) > tolerance or abs(r2 - expected2) > tolerance:
            raise ToolError(f"{cid}: saved Slide Rheostat segments disagree with rated resistance and slider position; refusing to guess")
        if "电阻1" not in props or "电阻2" not in props:
            assumptions.append(
                "Legacy save omits computed 电阻1/电阻2 fields: native segments are derived from exact 额定电阻 and 滑块位置, with a 10 micro-ohm numerical endpoint floor."
            )
        common = {"kind": "slide_rheostat", "value_name": "position",
                  "minimum": 0.0, "maximum": 1.0, "current": position_value,
                  "rated_resistance_ohm": rated, "minimum_segment_ohm": floor}
        # The official SDK names pins 0/1 l_low/r_low and pins 2/3 l_up/r_up.
        # The two lower binding posts are duplicate contacts to the moving
        # wiper; the two upper posts are the resistance-wire ends.  Two
        # variable segments plus an ideal 0<->1 wiper link preserve all four
        # terminals, including experiments that use either lower post.
        append("resistor", (0, 2), {"r": r1}, "segment_left", common)
        append("resistor", (1, 3), {"r": r2}, "segment_right", common)
        append("switch", (0, 1), {"closed": 1}, "wiper_link", common)
    return output
