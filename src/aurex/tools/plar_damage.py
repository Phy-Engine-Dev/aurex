"""PhysicsLab rating and broken-state adaptation for native PE simulations.

This module only adds explicit native protection around an already audited
component mapping.  It never mutates the source scene and never interprets
saved Statistics as a fresh measurement. PhysicsLab's public file schema does
not expose thermal constants, so the adapter records conservative configurable
engineering defaults instead of treating one nonlinear iteration as elapsed
heating time.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from .registry import ToolError


# PhysicsLab writes this largest finite float32 value for parts whose maximum
# power is intentionally unlimited (notably generated ideal voltage sources).
# It is a serialization sentinel, not a physical trip threshold.  Treating it
# as a live protection rating adds a native-only helper and makes an otherwise
# lossless .sav edit non-exportable.
_PL_UNLIMITED_POWER_W = float.fromhex("0x1.fffffep+127")


# line_pin is opened on failure. sense pins measure the original device's
# terminal voltage. A missing rating key means this contract only preserves an
# already-broken source state.
_CONTRACTS: dict[str, dict[str, Any]] = {
    "Resistor": {"line_pin": 0, "sense_pins": (0, 1)},
    "Resistance Box": {"line_pin": 0, "sense_pins": (0, 1)},
    "Simple Switch": {"line_pin": 0, "sense_pins": (0, 1)},
    "Push Switch": {"line_pin": 0, "sense_pins": (0, 1)},
    "Air Switch": {"line_pin": 0, "sense_pins": (0, 1), "current": "额定电流"},
    "Fuse Component": {"line_pin": 0, "sense_pins": (0, 1), "current": "熔断电流"},
    "Incandescent Lamp": {"line_pin": 0, "sense_pins": (0, 1), "voltage": "额定电压", "power": "额定功率"},
    "Buzzer": {"line_pin": 0, "sense_pins": (0, 1), "voltage": "额定电压", "power": "额定功率"},
    "Electric Bell": {"line_pin": 0, "sense_pins": (0, 1), "voltage": "额定电压", "power": "额定功率"},
    "Musical Box": {"line_pin": 0, "sense_pins": (0, 1), "voltage": "额定电压", "power": "额定功率"},
    "Simple Instrument": {"line_pin": 0, "sense_pins": (0, 1), "voltage": "额定电压", "power": "额定功率"},
    "Basic Capacitor": {"line_pin": 0, "sense_pins": (0, 1), "voltage": "耐压"},
    "Basic Inductor": {"line_pin": 0, "sense_pins": (0, 1), "current": "额定电流"},
    "Basic Diode": {"line_pin": 0, "sense_pins": (0, 1), "current": "额定电流"},
    "Photodiode": {"line_pin": 0, "sense_pins": (0, 1), "current": "额定电流"},
    "Light-Emitting Diode": {"line_pin": 0, "sense_pins": (0, 1), "current": "工作电流", "voltage": "反向耐压"},
    "Color Light-Emitting Diode": {"line_pin": 3, "sense_pins": (3, 0), "current": "工作电流", "voltage": "反向耐压"},
    "Dual Light-Emitting Diode": {"line_pin": 0, "sense_pins": (0, 1), "current": "工作电流", "voltage": "反向耐压"},
    "Rectifier": {"line_pin": 0, "sense_pins": (0, 1), "current": "额定电流"},
    "Battery Source": {"line_pin": 0, "sense_pins": (0, 1), "power": "最大功率"},
    "Current Source": {"line_pin": 0, "sense_pins": (0, 1)},
    "Sinewave Source": {"line_pin": 0, "sense_pins": (0, 1)},
    "Square Source": {"line_pin": 0, "sense_pins": (0, 1)},
    "Triangle Source": {"line_pin": 0, "sense_pins": (0, 1)},
    "Sawtooth Source": {"line_pin": 0, "sense_pins": (0, 1)},
    "Pulse Source": {"line_pin": 0, "sense_pins": (0, 1)},
    "Multimeter": {"line_pin": 0, "sense_pins": (0, 1)},
    "Electricity Meter": {"line_pin": 0, "sense_pins": (0, 1), "current": "额定电流"},
    "Photoresistor": {"line_pin": 0, "sense_pins": (0, 1), "voltage": "最大电压"},
    "Schmitt Trigger": {"line_pin": 1, "sense_pins": (1, None)},
    "Operational Amplifier": {"line_pin": 2, "sense_pins": (2, None)},
    "Transistor": {"line_pin": 1, "sense_pins": (1, 2), "power": "最大功率"},
    "N-MOSFET": {"line_pin": 2, "sense_pins": (2, 1), "power": "最大功率"},
    "P-MOSFET": {"line_pin": 1, "sense_pins": (1, 2), "power": "最大功率"},
    "Transformer": {"line_pin": 0, "sense_pins": (0, 1), "power": "额定功率"},
    "Tapped Transformer": {"line_pin": 0, "sense_pins": (0, 1), "power": "额定功率"},
    "Relay Component": {"line_pin": 3, "sense_pins": (3, 4), "current": "额定电流"},
    # Boolean gate 最大电流 is a thermally accumulated live rating.  It is
    # meaningful only when the output is coupled to an analog MNA load; pure
    # digital nets have no physical branch current.
    "No Gate": {"line_pin": 1, "sense_pins": (1, None), "current": "最大电流", "mixed_only": True},
    "Yes Gate": {"line_pin": 1, "sense_pins": (1, None), "current": "最大电流", "mixed_only": True},
    "And Gate": {"line_pin": 2, "sense_pins": (2, None), "current": "最大电流", "mixed_only": True},
    "Or Gate": {"line_pin": 2, "sense_pins": (2, None), "current": "最大电流", "mixed_only": True},
    "Nor Gate": {"line_pin": 2, "sense_pins": (2, None), "current": "最大电流", "mixed_only": True},
    "Nand Gate": {"line_pin": 2, "sense_pins": (2, None), "current": "最大电流", "mixed_only": True},
    "Xor Gate": {"line_pin": 2, "sense_pins": (2, None), "current": "最大电流", "mixed_only": True},
    "Xnor Gate": {"line_pin": 2, "sense_pins": (2, None), "current": "最大电流", "mixed_only": True},
    "Imp Gate": {"line_pin": 2, "sense_pins": (2, None), "current": "最大电流", "mixed_only": True},
    "Nimp Gate": {"line_pin": 2, "sense_pins": (2, None), "current": "最大电流", "mixed_only": True},
}


def supports_broken_state(model_id: str) -> bool:
    # Any imported electrical device can be represented as disconnected when
    # the source save already marks it broken. _CONTRACTS additionally define
    # live overload criteria and the terminal that opens when a new trip occurs.
    return isinstance(model_id, str) and bool(model_id)


def _number(properties: dict, key: str, cid: str) -> float:
    value = properties.get(key)
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ToolError(f"{cid}: protection rating {key} must be an explicit finite nonnegative number")
    return float(value)


def _rating(properties: dict, key: str, cid: str) -> float:
    # Older real saves and intentionally minimal fixtures may predate a rating
    # field. Absence means that criterion is unavailable, never zero-rated.
    if key not in properties:
        return 0.0
    value = _number(properties, key, cid)
    if key == "最大功率" and value >= _PL_UNLIMITED_POWER_W:
        return 0.0
    return value


def _source_owner(component: dict, cid: str) -> bool:
    if component.get("id") == cid:
        return True
    source = component.get("pl_source") or {}
    imported = component.get("plsav_import") or {}
    return (source.get("parent_identifier") == cid or source.get("identifier") == cid or
            imported.get("source_component_id") == cid)


def apply_damage_protection(components: list[dict], scene: dict) -> list[dict]:
    """Add collision-free protection models around recognized source devices."""
    if not isinstance(scene, dict) or not isinstance(scene.get("components"), list):
        raise ToolError("Damage adaptation requires the complete original component scene")
    output = copy.deepcopy(components)
    used_ids = {component["id"] for component in output}
    used_nodes = {node for component in output for node in component["nodes"]}

    for original in scene["components"]:
        model_id = original.get("type")
        contract = _CONTRACTS.get(model_id)
        initial_broken = bool(original.get("is_broken", False))
        if contract is None and not initial_broken:
            continue
        cid = original.get("id")
        if not isinstance(cid, str) or not cid:
            raise ToolError("Damage adaptation requires each original component ID")
        props = original.get("properties")
        pins = original.get("pins")
        if not isinstance(props, dict) or not isinstance(pins, list):
            raise ToolError(f"{cid}: original properties and pins are required for protection")
        pin_nodes = {pin.get("pin"): pin.get("node") for pin in pins if isinstance(pin, dict)}
        live_limits = {"max_current_a": 0.0, "max_voltage_v": 0.0, "max_power_w": 0.0}
        if contract is not None:
            line_pin = contract["line_pin"]
            sense_positive, sense_negative = contract["sense_pins"]
            needed = [line_pin, sense_positive] + ([] if sense_negative is None else [sense_negative])
            if any(type(pin) is not int or not isinstance(pin_nodes.get(pin), str) for pin in needed):
                raise ToolError(f"{cid}: protection contract references a missing original pin")
            live_limits = {
                "max_current_a": _rating(props, contract["current"], cid) if "current" in contract else 0.0,
                "max_voltage_v": _rating(props, contract["voltage"], cid) if "voltage" in contract else 0.0,
                "max_power_w": _rating(props, contract["power"], cid) if "power" in contract else 0.0,
            }
            if contract.get("mixed_only"):
                external_line = pin_nodes[line_pin]
                analog_load = any(
                    not component["type"].startswith("digital_")
                    and external_line in component["nodes"]
                    and not _source_owner(component, cid)
                    for component in output
                )
                if not analog_load:
                    live_limits = {key: 0.0 for key in live_limits}
        if not initial_broken and not any(live_limits.values()):
            continue

        # A newly tripped live device opens its audited line terminal. A save
        # that is already broken is isolated at every distinct external node,
        # including multi-terminal analog/digital devices. Pins already tied
        # together share one guard so original shorted topology is preserved.
        if initial_broken:
            targets = []
            seen_nodes = set()
            for pin, node in sorted(pin_nodes.items()):
                if type(pin) is not int or not isinstance(node, str) or node in seen_nodes:
                    continue
                seen_nodes.add(node)
                is_live_line = contract is not None and pin == contract["line_pin"]
                targets.append((pin, node, live_limits if is_live_line else {
                    "max_current_a": 0.0, "max_voltage_v": 0.0, "max_power_w": 0.0
                }, pin, None))
        else:
            line_pin = contract["line_pin"]
            sense_positive, sense_negative = contract["sense_pins"]
            targets = [(line_pin, pin_nodes[line_pin], live_limits,
                        sense_positive, sense_negative)]

        digest = hashlib.sha256(cid.encode("utf-8")).hexdigest()
        owned_components = [component for component in output if _source_owner(component, cid)]
        for target_index, (line_pin, external_line, limits,
                           sense_positive, sense_negative) in enumerate(targets):
            guard_id = "__pl_damage_" + digest + (f"_{target_index}" if len(targets) > 1 else "")
            suffix = 0
            base_guard_id = guard_id
            while guard_id in used_ids:
                suffix += 1
                guard_id = base_guard_id + "_" + str(suffix)
            used_ids.add(guard_id)
            internal_node = "__pl_damage_node_" + digest + (f"_{target_index}" if len(targets) > 1 else "")
            suffix = 0
            base_internal_node = internal_node
            while internal_node in used_nodes:
                suffix += 1
                internal_node = base_internal_node + "_" + str(suffix)
            used_nodes.add(internal_node)

            changed = 0
            for component in owned_components:
                for index, node in enumerate(component["nodes"]):
                    if node == external_line:
                        component["nodes"][index] = internal_node
                        changed += 1
            if changed == 0:
                if initial_broken:
                    # An importer may intentionally leave an originally
                    # unwired/reference-only terminal out of the native
                    # primitive. It is already disconnected and needs no
                    # redundant open branch.
                    continue
                raise ToolError(f"{cid}: could not isolate original broken pin {line_pin}")

            sense_positive_node = pin_nodes.get(sense_positive, external_line)
            sense_negative_node = "gnd" if sense_negative is None else pin_nodes[sense_negative]
            source = {
                "model_id": model_id,
                "parent_identifier": cid,
                "decomposition_role": "saved_broken_terminal_isolation" if initial_broken else "irreversible_damage_protection",
                "is_helper": True,
                "raw_properties": copy.deepcopy(props),
                "raw_statistics": copy.deepcopy(original.get("statistics", {})),
                "saved_is_broken": initial_broken,
                "limits": copy.deepcopy(limits),
                "line_pin": line_pin,
                "sense_pins": [sense_positive, sense_negative],
                "assumptions": [
                    "PhysicsLab exposes no saved thermal constants; the native compatibility model explicitly uses ambient/rating-reference 25 C, trip 150 C, Rth 1 K/W and Cth 10 J/K.",
                    "Current and power use a first-order electrothermal RC in transient analysis; voltage remains an immediate breakdown limit. DC/OP represents steady-state thermal overload.",
                    "At a saved current or power rating the normalized thermal equilibrium equals trip temperature; overload duration and cooling, not one solver iteration, determine a transient trip.",
                    "An already-broken saved device is isolated at every distinct external node; tied original pins remain tied behind one open guard.",
                    "The source .sav is unchanged; Ron=1e-12 ohm and Roff=1e12 ohm are explicit native engineering approximations.",
                    *( ["A saved digital gate current rating is thermally active only because its output is electrically coupled to an analog MNA load; a pure digital net has no analog current to heat the model."]
                       if contract is not None and contract.get("mixed_only") else []),
                ],
                "numerical_equivalence_to_original": False,
            }
            output.append({
                "id": guard_id,
                "type": "rated_protection",
                "nodes": [external_line, internal_node, sense_positive_node, sense_negative_node],
                "params": {"r_on": 1e-12, "r_open": 1e12,
                           **limits, "initial_broken": int(initial_broken),
                           "ambient_temp_c": 25.0, "rating_reference_temp_c": 25.0,
                           "trip_temp_c": 150.0,
                           "thermal_resistance_k_per_w": 1.0,
                           "thermal_capacitance_j_per_k": 10.0},
                "position": copy.deepcopy(original.get("position", [0, 0, 0])),
                "rotation": copy.deepcopy(original.get("rotation", [0, 0, 0])),
                "pl_source": source,
            })
    return output
