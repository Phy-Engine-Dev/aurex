"""Explicit PhysicsLab coupled-device import; no implicit model substitution.

Only explicit PE electrical families are used. Serialized winding resistances
are expanded as series primitives; where a save lacks the absolute inductance
needed for a unique non-ideal transformer, the declared normalization is kept
in provenance. No network, publication or source-file mutation.
"""
from __future__ import annotations

import copy
import hashlib
import math

from .registry import ToolError


_SDK = "https://github.com/SekaiArendelle/physicslab/blob/fa95b96910dd0fd4e09cf27e24cefaf9b91798ad/physicslab/circuit/elements/artificial_circuit.py"
_PE = "Phy-Engine 01367a3b337c9b1bae9081c818d14d791f96b32e"
_COUNTS = {"Transformer": 4, "Mutual Inductor": 4,
           "Tapped Transformer": 5, "Relay Component": 5}


def _number(props: dict, key: str, cid: str) -> float:
    value = props.get(key)
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ToolError(f"{cid}: original {key} must be an explicit finite number")
    return value


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
    """Map complete original pins/properties, preserving source provenance.

    Transformer: only k=1, n=Vp/Vs, PL 0/1 primary and 2/3 secondary.
    Mutual Inductor: L1/L2 are raw SI henries, 0<=k<=1; same pin pairs.
    Neither route adds grounding, estimates missing values, or claims identical
    application simulation history. Original saved statistics remain evidence,
    not initial conditions or newly measured PE output.
    """
    if not isinstance(el, dict) or el.get("type") not in _COUNTS:
        return None
    kind, cid = el["type"], el.get("id")
    if not isinstance(cid, str) or not 0 < len(cid) <= 128:
        raise ToolError("Power PL import requires the original unique component ID")
    props = el.get("properties")
    if not isinstance(props, dict):
        raise ToolError(f"{cid}: original properties are required")
    if el.get("is_broken", el.get("IsBroken", False)):
        raise ToolError(f"{cid}: broken-device behavior requires the complete SAV damage adapter")
    winding_losses = {}
    for field in ("内阻", "内阻1", "内阻2", "线圈电阻", "线圈电阻1", "线圈电阻2", "初级电阻", "次级电阻"):
        if kind == "Relay Component" and field == "线圈电阻":
            continue
        if field in props:
            value = _number(props, field, cid)
            if value < 0:
                raise ToolError(f"{cid}: original {field} cannot be negative")
            if value:
                winding_losses[field] = value
    if not isinstance(scene, dict) or not isinstance(scene.get("components"), list) or not isinstance(scene.get("wires"), list):
        raise ToolError(f"{cid}: full original scene components and wires are required")
    if not all(isinstance(c, dict) for c in scene["components"]):
        raise ToolError(f"{cid}: malformed original scene component")
    ids = [c.get("id") for c in scene["components"]]
    if ids.count(cid) != 1 or any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ToolError(f"{cid}: scene must identify source exactly once with unique original IDs")
    pins = {}
    count = _COUNTS[kind]
    raw_pins = el.get("pins")
    if not isinstance(raw_pins, list) or len(raw_pins) != count:
        raise ToolError(f"{cid}: expected all {count} original pins, including unconnected pins")
    for pin in raw_pins:
        if not isinstance(pin, dict) or type(pin.get("pin")) is not int:
            raise ToolError(f"{cid}: invalid original pin index")
        index, node = pin["pin"], pin.get("node")
        if index in pins or index not in range(count) or not isinstance(node, str) or not 0 < len(node) <= 128:
            raise ToolError(f"{cid}: missing, duplicate or invalid original pin/node mapping")
        pins[index] = node
    position, rotation = _pose(el, "position"), _pose(el, "rotation")
    label = el.get("label")
    if label is not None and not isinstance(label, str):
        raise ToolError(f"{cid}: original display label must be text or null")
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
    k = None
    if kind != "Relay Component":
        k = _number(props, "耦合系数", cid)
        if not 0 <= k <= 1:
            raise ToolError(f"{cid}: original 耦合系数 must be in [0,1]")
    assumptions = [
        "Original external nodes and every pin are preserved; unwired terminals remain floating, never grounded implicitly.",
        "Saved PL statistics are retained as original observations, not imported initial conditions or new PE measurements.",
    ]
    engineering_defaults = {}
    primitive_pins = list(range(count))
    if kind in ("Transformer", "Tapped Transformer"):
        vin, vout = _number(props, "输入电压", cid), _number(props, "输出电压", cid)
        if vin <= 0 or vout <= 0:
            raise ToolError(f"{cid}: original transformer input/output rated voltages must both be positive")
        ratio = vin / vout
        if not math.isfinite(ratio) or ratio <= 0:
            raise ToolError(f"{cid}: transformer voltage ratio is not finite and positive")
        rating = _number(props, "额定功率", cid)
        if rating <= 0:
            raise ToolError(f"{cid}: original transformer 额定功率 must be positive")
        ctype, params = "transformer", {"ratio": ratio}
        if kind == "Transformer" and k != 1:
            # Voltage ratio fixes sqrt(Lp/Ls) but the save does not fix either
            # absolute inductance. A 1 H primary is therefore an explicit,
            # scale-visible normalization rather than a hidden guessed value.
            ctype, params = "coupled_inductors", {"l1": 1.0, "l2": 1.0 / (ratio * ratio), "k": k}
            engineering_defaults["nonideal_transformer_normalization"] = {
                "primary_inductance_h": 1.0,
                "secondary_inductance_h": 1.0 / (ratio * ratio),
                "reason": "save fixes voltage ratio and coupling but has no absolute winding inductance",
            }
            assumptions.append("For non-unity coupling, native coupled inductors preserve saved k and voltage ratio using an explicit 1 H primary normalization; absolute magnetizing impedance is therefore an engineering normalization, not recovered app data.")
        if kind == "Tapped Transformer":
            if not math.isfinite(2 * ratio) or 2 * ratio == 0:
                raise ToolError(f"{cid}: center-tapped half-winding voltage ratio is not representable")
            ctype = "transformer_center_tap"
            primitive_pins = [0, 1, 2, 4, 3]
            engineering_defaults["output_voltage_definition"] = "full_secondary_end_to_end"
            assumptions.append("Declared engineering convention: 输出电压 is total secondary PL2-to-PL3; PL4 is an equal center tap. The closed-source app's rating definition was not independently verified. Native pins are [PL0,PL1,PL2,PL4,PL3].")
            if k != 1:
                engineering_defaults["saved_coupling_not_identifiable"] = {
                    "saved_k": k,
                    "native_k": 1.0,
                    "reason": "five-pin save has no absolute primary/half-secondary inductances or cross-coupling matrix",
                }
                assumptions.append("Tapped-transformer saved k is retained but cannot uniquely determine a three-winding leakage matrix without absolute inductances; the native center-tap ratio remains ideal instead of inventing an unstable matrix.")
        assumptions += [
            "Ideal linear magnetic model only: no inferred core saturation, hysteresis, winding losses, thermal damage or saved dynamic state.",
            "Voltage fields determine V(primary 0-1)/V(secondary 2-3); no nominal voltage source is inserted.",
            "额定功率 is retained as a rating, not silently converted into a voltage/current clamp; overload or thermal failure is not simulated.",
            "The ideal PE transformer's DC solution is an algebraic ratio constraint, not evidence of real-transformer steady DC operation.",
        ]
    elif kind == "Mutual Inductor":
        l1, l2 = _number(props, "电感1", cid), _number(props, "电感2", cid)
        if l1 <= 0 or l2 <= 0 or not math.isfinite(l1 * l2) or l1 * l2 == 0:
            raise ToolError(f"{cid}: original winding inductances and their product must be finite and positive")
        ctype, params = "coupled_inductors", {"l1": l1, "l2": l2, "k": k}
        assumptions.append("Raw 电感1/电感2 use the existing PE PL-wrapper SI-H mapping; M=k*sqrt(L1*L2), with each pin pair retaining original polarity.")
        assumptions.append("Ideal linear coupled inductors; no inferred core saturation, winding losses or retained magnetic initial state.")
    else:
        inductance = _number(props, "线圈电感", cid)
        resistance = _number(props, "线圈电阻", cid)
        pickup = _number(props, "接通电流", cid)
        rating = _number(props, "额定电流", cid)
        state = _number(props, "开关", cid)
        if inductance <= 0 or resistance <= 0 or pickup <= 0 or rating <= 0 or state not in (0, 1):
            raise ToolError(f"{cid}: relay requires positive coil L/R/pickup/rated current and explicit 开关 0 or 1")
        dropout = .8 * pickup
        if not math.isfinite(dropout) or not 0 <= dropout < pickup:
            raise ToolError(f"{cid}: relay dropout engineering default is not representable")
        # These are explicit native-model parameters, not invented PL fields.
        contact_resistance = _number(props, "接触电阻", cid) if "接触电阻" in props else .01
        if contact_resistance <= 0:
            raise ToolError(f"{cid}: relay contact resistance must be positive")
        engineering_defaults = {"release_current_fraction": .8, "contact_closed_resistance_ohm": contact_resistance,
                                "contact_open_resistance_ohm": 1e12, "operate_delay_s": 0.0,
                                "release_delay_s": 0.0, "deenergized_contact_pl_pin": 0,
                                "energized_contact_pl_pin": 2}
        ctype, params = "relay_current_spdt", {"l": inductance, "r": resistance, "i_pull": pickup,
            "i_drop": dropout, "r_on": contact_resistance, "r_off": 1e12, "t_on": 0.0, "t_off": 0.0, "initial": int(state)}
        assumptions += [
            "Generic five-pin engineering relay, not a claim of pointwise PhysicsLab simulator equivalence; original coil L/R and pickup current are physically stamped/used.",
            "PL pins 3/4 are coil terminals and 1 is changeover common, corroborated by original password-lock wires; pin 0=NC and pin 2=NO is a declared engineering orientation convention, not a proved app animation contract.",
            "Unspecified dropout is 0.8*pickup, contact Ron=0.01 ohm/Roff=1e12 ohm, operate/release delays=0 s; all are explicit editable native parameters. No contact bounce, arcing or mechanical inertia is modeled.",
            "Non-polarized activation uses absolute solved coil current; transient coil uses backward Euler and threshold/delay timing is quantized at actual solve steps. AC holds operating-point contacts, not an AC phasor actuator simulation.",
            "开关 initializes the contact state only, not a permanent forced switch; coil current starts from the native initial condition. 额定电流 remains a rating, not a simulated overload trip or thermal failure threshold.",
        ]
    source = {
        "model_id": kind, "identifier": cid,
        "raw_properties": copy.deepcopy(props), "raw_statistics": copy.deepcopy(el.get("statistics", {})),
        "pin_mapping": [{"pl_pin": p, "node": pins[p], "externally_wired": p in wired} for p in range(count)],
        "implicit_references": [], "assumptions": assumptions,
        "mapping_reference": [_SDK, _PE + ":include/phy_engine/phy_lab_wrapper/pe_sim.h coupled-device adapter"],
        "numerical_equivalence_to_original": False,
        "decomposition_role": "engineering_relay" if kind == "Relay Component" else "ideal_coupled_device", "parent_identifier": cid,
        "is_helper": False, "primitive_pin_mapping": primitive_pins,
    }
    if engineering_defaults:
        source["engineering_defaults"] = engineering_defaults
        if kind == "Relay Component":
            source["mapping_reference"].append("Original public experiment 6363de186e31f337bfb88e29, raw SHA256 ec2f6055767c3e1ba188ce3fab836f8e5d7b687416930bb26acecc76eda99df6: six relays share coil pins 3/4, with pin4 grounded; pins0/1/2 carry the switched signal network")
    item = {"id": cid, "type": ctype, "nodes": [pins[p] for p in primitive_pins], "params": params,
            "position": position, "rotation": rotation, "pl_source": source}
    if label:
        item["label"] = label
    output = [item]
    if winding_losses and kind != "Relay Component":
        digest = hashlib.sha256(cid.encode("utf-8")).hexdigest()
        primary_loss = sum(value for field, value in winding_losses.items()
                           if field in {"内阻", "内阻1", "线圈电阻", "线圈电阻1", "初级电阻"})
        secondary_loss = sum(value for field, value in winding_losses.items()
                             if field in {"内阻2", "线圈电阻2", "次级电阻"})
        loss_parts = [(0, primary_loss, "primary_winding_resistance")]
        if kind == "Tapped Transformer":
            loss_parts += [(2, secondary_loss / 2, "secondary_left_half_resistance"),
                           (3, secondary_loss / 2, "secondary_right_half_resistance")]
        else:
            loss_parts.append((2, secondary_loss, "secondary_winding_resistance"))
        for pl_pin, resistance_value, role in loss_parts:
            if resistance_value == 0:
                continue
            native_pin = primitive_pins.index(pl_pin)
            internal_node = f"__pl_winding_node_{digest}_{pl_pin}"
            helper_id = f"__pl_winding_resistance_{digest}_{pl_pin}"
            item["nodes"][native_pin] = internal_node
            helper_source = copy.deepcopy(source)
            helper_source.update({"parent_identifier": cid, "is_helper": True,
                                  "decomposition_role": role,
                                  "primitive_pin_mapping": [pl_pin, None],
                                  "combined_saved_fields": copy.deepcopy(winding_losses)})
            output.append({"id": helper_id, "type": "resistor",
                           "nodes": [pins[pl_pin], internal_node],
                           "params": {"r": resistance_value},
                           "position": copy.deepcopy(position),
                           "rotation": copy.deepcopy(rotation),
                           "pl_source": helper_source})
        item["pl_source"]["winding_resistance_mapping"] = {
            "saved_fields": copy.deepcopy(winding_losses),
            "primary_series_ohm": primary_loss,
            "secondary_total_series_ohm": secondary_loss,
        }
    return output
