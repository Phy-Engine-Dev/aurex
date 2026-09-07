"""Broader PhysicsLab device decompositions using explicit PE primitives.

This module closes import holes without pretending the closed-source simulator
is known.  Exact editor fields drive every available equation; missing
environmental inputs (light, acceleration, proximity) receive documented zero
stimuli instead of archived Statistics or fabricated measurements.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from .registry import ToolError


_SDK_COMMIT = "fa95b96910dd0fd4e09cf27e24cefaf9b91798ad"
_PIN_COUNTS = {
    "555 Timer": 8,
    "Incandescent Lamp": 2, "Buzzer": 2, "Electric Bell": 2,
    "Musical Box": 2, "Simple Instrument": 2, "Fuse Component": 2,
    "Multimeter": 2, "Galvanometer": 3, "Microammeter": 3,
    "Simple Ammeter": 3, "Simple Voltmeter": 3, "Electricity Meter": 4,
    "Resistance Law": 8, "Spark Gap": 2, "Tesla Coil": 2,
    "Solenoid": 4, "Electric Fan": 2,
    "Accelerometer": 3, "Attitude Sensor": 3, "Gravity Sensor": 3,
    "Gyroscope": 3, "Linear Accelerometer": 3, "Magnetic Field Sensor": 3,
    "Analog Joystick": 6, "Photoresistor": 2, "Proximity Sensor": 1,
}
_LOADS = {"Incandescent Lamp", "Buzzer", "Electric Bell", "Musical Box",
          "Simple Instrument"}
_MEMS = {"Accelerometer", "Attitude Sensor", "Gravity Sensor", "Gyroscope",
         "Linear Accelerometer", "Magnetic Field Sensor"}


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
        raise ToolError(f"{el.get('id')}: expected all {count} original device pins")
    pins: dict[int, str] = {}
    for item in raw:
        if (not isinstance(item, dict) or type(item.get("pin")) is not int
                or item["pin"] not in range(count) or item["pin"] in pins
                or not isinstance(item.get("node"), str)
                or not 0 < len(item["node"]) <= 128):
            raise ToolError(f"{el.get('id')}: missing, duplicate or invalid original pin mapping")
        pins[item["pin"]] = item["node"]
    return pins


def _pose(el: dict, field: str) -> list[float]:
    value = el.get(field)
    try:
        valid = isinstance(value, list) and len(value) == 3 and all(
            type(item) in (int, float) and math.isfinite(item) for item in value)
    except OverflowError:
        valid = False
    if not valid:
        raise ToolError(f"{el.get('id')}: original finite {field} is required")
    return [float(item) for item in value]


def import_element(el: dict, *, scene: dict) -> list[dict[str, Any]] | None:
    if not isinstance(el, dict) or el.get("type") not in _PIN_COUNTS:
        return None
    kind, cid = el["type"], el.get("id")
    if not isinstance(cid, str) or not 0 < len(cid) <= 128:
        raise ToolError("Device import requires the original component ID")
    if el.get("is_broken", el.get("IsBroken", False)):
        raise ToolError(f"{cid}: broken-device behavior requires the complete SAV damage adapter")
    props = el.get("properties")
    if not isinstance(props, dict):
        raise ToolError(f"{cid}: original device properties are required")
    if (not isinstance(scene, dict) or not isinstance(scene.get("components"), list)
            or not isinstance(scene.get("wires"), list)):
        raise ToolError(f"{cid}: complete original device scene is required")
    ids = [row.get("id") for row in scene["components"] if isinstance(row, dict)]
    if (len(ids) != len(scene["components"]) or any(not isinstance(value, str) for value in ids)
            or ids.count(cid) != 1 or len(ids) != len(set(ids))):
        raise ToolError(f"{cid}: malformed or duplicate original scene component IDs")
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

    used_ids = set(ids)
    used_nodes = {pin.get("node") for row in scene["components"]
                  if isinstance(row, dict) for pin in row.get("pins", [])
                  if isinstance(pin, dict) and isinstance(pin.get("node"), str)}
    digest = hashlib.sha256(cid.encode()).hexdigest()

    def allocate(role: str, *, node: bool = False) -> str:
        used = used_nodes if node else used_ids
        base = "__pl_device_" + ("node_" if node else "") + digest + "_" + role
        value, suffix = base, 0
        while value in used:
            suffix += 1
            value = base + "_" + str(suffix)
        used.add(value)
        return value

    base = {
        "model_id": kind, "identifier": cid,
        "raw_properties": copy.deepcopy(props),
        "raw_statistics": copy.deepcopy(el.get("statistics", {})),
        "pin_mapping": [{"pl_pin": index, "node": pins[index],
                         "externally_wired": index in wired} for index in range(len(pins))],
        "implicit_references": [],
        "mapping_reference": [
            "https://github.com/SekaiArendelle/physicslab/blob/" + _SDK_COMMIT +
            "/physicslab/circuit/elements/"
        ],
        "statistics_source": "original saved values, never fresh PE measurements",
        "numerical_equivalence_to_original": False,
        "support_level": "explicit_native_engineering_mapping",
    }
    output: list[dict[str, Any]] = []

    def append(component_type: str, nodes: list[str], params: dict, role: str,
               primitive_pins: list[int | None], assumptions: list[str],
               *, extra_source: dict[str, Any] | None = None,
               interaction: dict[str, Any] | None = None) -> None:
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
        if interaction:
            item["interaction"] = {"control_id": cid, "source_model_id": kind,
                                   "role": role, **copy.deepcopy(interaction)}
        output.append(item)

    common = [
        "Original external terminal nodes and pose are preserved; saved Statistics do not initialise or fit this solve.",
        "The closed-source PhysicsLab constitutive equation is not public; engineering defaults are explicit and editable in provenance.",
    ]

    if kind == "555 Timer":
        low, high = _number(props, "低电平", cid), _number(props, "高电平", cid)
        if low > high:
            raise ToolError(f"{cid}: 555 low output exceeds high output")
        append("ne555_timer", [pins[index] for index in range(8)],
               {"low_v": low, "high_v": high, "output_r": 1e-3,
                "discharge_on_r": 1e-3, "discharge_off_r": 1e12,
                "internal_control": int(3 not in wired),
                "internal_reset_pullup": int(6 not in wired)},
               "behavioral_555", list(range(8)), common + [
                   "Trigger and threshold comparators update an SR latch; OUT is a finite-resistance driver and DIS is an open-collector discharge path.",
                   "Unwired CTRL uses 2/3 VCC and unwired RESET uses the internal pull-up; actual externally wired pins remain authoritative.",
                   "No undocumented delay, output saturation curve, thermal behavior or internal bipolar transistor network is invented.",
               ], extra_source={"engineering_defaults": {"output_resistance_ohm": 1e-3,
                   "discharge_on_resistance_ohm": 1e-3,
                   "discharge_off_resistance_ohm": 1e12}})
    elif kind in _LOADS:
        rated_v = _number(props, "额定电压", cid)
        rated_p = _number(props, "额定功率", cid)
        if rated_v <= 0 or rated_p <= 0:
            raise ToolError(f"{cid}: rated voltage and power must be positive")
        resistance = rated_v * rated_v / rated_p
        append("resistor", [pins[0], pins[1]], {"r": resistance}, "rated_load",
               [0, 1], common + [
                   "Electrical loading is the exact rated-point equivalent R=V_rated^2/P_rated; acoustic, light and animation output are not electrical solver states.",
                   "The complete-SAV damage adapter enforces saved voltage and power limits instantaneously and opens the path when exceeded.",
               ], extra_source={"parameter_derivation": {"rated_voltage_v": rated_v,
                   "rated_power_w": rated_p, "equivalent_resistance_ohm": resistance}})
    elif kind == "Fuse Component":
        closed = _number(props, "开关", cid)
        rated = _number(props, "额定电流", cid)
        melt = _number(props, "熔断电流", cid)
        if closed not in (0, 1) or rated <= 0 or melt <= 0 or melt < rated:
            raise ToolError(f"{cid}: fuse requires switch 0/1 and 0 < rated <= melt current")
        append("switch", [pins[0], pins[1]], {"closed": int(closed)}, "fuse_contact",
               [0, 1], common + [
                   "Saved contact state is preserved. The complete-SAV damage adapter irreversibly opens at the saved melt current; rated current remains metadata below that threshold.",
               ])
    elif kind in {"Galvanometer", "Microammeter", "Simple Ammeter"}:
        if kind == "Simple Ammeter":
            # Saves made by the old (2018/2019-era) PhysicsLab meter only
            # contain 量程 and 名义量程.  The current public SDK added
            # 内阻 with the same default/value semantics as 量程.  Preserve
            # an explicit modern 内阻, but use the saved legacy 量程 as the
            # burden resistance when that newer field is genuinely absent.
            resistance_field = "内阻" if "内阻" in props else "量程"
            resistance = _number(props, resistance_field, cid)
            nominal = _number(props, "名义量程", cid)
        else:
            resistance = 0.1
            resistance_field = "explicit PE low-burden default"
            nominal = _number(props, "量程", cid)
        if resistance <= 0 or nominal <= 0:
            raise ToolError(f"{cid}: meter resistance/range must be positive")
        for role, a, b in (("left_range", 0, 1), ("right_range", 2, 1)):
            append("resistor", [pins[a], pins[b]], {"r": resistance}, role, [a, b],
                   common + [
                       "PL1 is the shared meter terminal; each outer terminal is an independent current range. PE reports the actual branch current.",
                       "Simple Ammeter uses saved 内阻. Galvanometer/Microammeter expose no input resistance, so 0.1 ohm is an explicit low-burden, numerically conditioned sensing default, not an app calibration.",
                   ], extra_source={"engineering_defaults": {
                       "range": nominal, "input_resistance_ohm": resistance,
                       "input_resistance_source": resistance_field}})
    elif kind == "Simple Voltmeter":
        movement = _number(props, "量程", cid)
        nominal = _number(props, "名义量程", cid)
        if movement <= 0 or nominal <= 0:
            raise ToolError(f"{cid}: voltmeter movement/range must be positive")
        resistance = nominal / movement
        for role, a, b in (("left_range", 0, 1), ("right_range", 2, 1)):
            append("voltage_meter", [pins[a], pins[b]], {"r_input": resistance}, role,
                   [a, b], common + [
                       "PL1 is the shared terminal. Input resistance is derived as nominal voltage divided by saved full-scale movement current.",
                   ], extra_source={"parameter_derivation": {
                       "full_scale_current_a": movement, "nominal_voltage_v": nominal,
                       "input_resistance_ohm": resistance}})
    elif kind == "Multimeter":
        mode = _number(props, "状态", cid)
        append("voltage_meter", [pins[0], pins[1]], {"r_input": 1e9}, "safe_voltage_input",
               [0, 1], common + [
                   "The public SDK does not define dial-state semantics or input impedance. Unknown/non-calibrated modes use an explicit 1 Gohm voltage-observation load instead of a 0 V source or archived V/I fitting.",
               ], extra_source={"engineering_defaults": {"saved_dial_state": mode,
                   "input_resistance_ohm": 1e9, "calibrated": False}})
    elif kind == "Electricity Meter":
        rated = _number(props, "额定电流", cid)
        if rated <= 0:
            raise ToolError(f"{cid}: electricity-meter rated current must be positive")
        append("resistor", [pins[0], pins[1]], {"r": 1e-6}, "current_coil", [0, 1],
               common + ["PL0/1 form a near-ideal current coil; actual PE current and power are observable."],
               extra_source={"engineering_defaults": {"current_coil_resistance_ohm": 1e-6,
                                                        "rated_current_a": rated}})
        append("voltage_meter", [pins[2], pins[3]], {"r_input": 1e9}, "voltage_coil",
               [2, 3], common + [
                   "PL2/3 form a 1 Gohm voltage coil. Saved 示数 is retained as historical editor state; PE does not replay old accumulated energy.",
               ], extra_source={"engineering_defaults": {"voltage_coil_resistance_ohm": 1e9}})
    elif kind == "Resistance Law":
        length, radius = _number(props, "长度", cid), _number(props, "半径", cid)
        if length <= 0 or radius <= 0:
            raise ToolError(f"{cid}: wire length and radius must be positive")
        area = math.pi * radius * radius
        rhos = [_number(props, "电阻率", cid), _number(props, "电阻率2", cid),
                _number(props, "电阻率3", cid), _number(props, "电阻率", cid)]
        explicit = [props.get("电阻"), props.get("电阻2"), props.get("电阻3"), None]
        for index, (a, b) in enumerate(zip(range(4), range(4, 8))):
            derived = rhos[index] * length / area
            stored = explicit[index]
            resistance = float(stored) if type(stored) in (int, float) and math.isfinite(stored) and stored > 0 else derived
            if resistance <= 0 or not math.isfinite(resistance):
                raise ToolError(f"{cid}: resistance-law branch {index} is not finite and positive")
            append("resistor", [pins[a], pins[b]], {"r": resistance}, f"wire_{index}",
                   [a, b], common + [
                       "Aligned left/right pins form four independent sample wires. Explicit saved resistance wins; otherwise R=rho*length/(pi*radius^2) is derived in SI.",
                   ], extra_source={"parameter_derivation": {"resistivity_ohm_m": rhos[index],
                       "length_m": length, "radius_m": radius, "derived_resistance_ohm": derived,
                       "used_explicit_saved_resistance": stored is not None}})
    elif kind == "Analog Joystick":
        rated = _number(props, "额定电阻", cid)
        if rated <= 0:
            raise ToolError(f"{cid}: joystick rated resistance must be positive")
        for axis, left, wiper, right in (("x", 0, 1, 2), ("y", 3, 4, 5)):
            for side, a, b in (("left", left, wiper), ("right", wiper, right)):
                append("resistor", [pins[a], pins[b]], {"r": rated / 2}, f"{axis}_{side}",
                       [a, b], common + [
                           "The save has no joystick position; each axis starts at the explicit electrical midpoint, preserving all six terminals without inventing archived motion.",
                       ], extra_source={"engineering_defaults": {"axis": axis,
                           "position": .5, "total_resistance_ohm": rated}})
    elif kind in _MEMS:
        output_r = _number(props, "输出阻抗", cid)
        offset = _number(props, "偏移", cid)
        ranges = _number(props, "量程", cid)
        sensitivity = _number(props, "响应系数", cid)
        if output_r <= 0 or ranges <= 0:
            raise ToolError(f"{cid}: MEMS range and output impedance must be positive")
        for axis in range(3):
            internal = allocate(f"axis_{axis}", node=True)
            append("vdc", [internal, "gnd"], {"v": offset}, f"axis_{axis}_zero_stimulus",
                   [axis, None], common + [
                       "No physical acceleration/rotation/field sample is serialized. Zero environmental stimulus yields the exact saved output offset; no archived measurement is invented.",
                   ], extra_source={"engineering_defaults": {"axis": axis,
                       "environmental_input": 0.0, "range": ranges,
                       "response_factor": sensitivity, "output_offset_v": offset}})
            append("resistor", [internal, pins[axis]], {"r": output_r}, f"axis_{axis}_output_impedance",
                   [None, axis], common + ["Saved output impedance is explicitly in series with the sensor output."],
                   extra_source={"engineering_defaults": {"axis": axis,
                                                           "output_impedance_ohm": output_r}})
    elif kind == "Photoresistor":
        dark = _number(props, "暗电阻", cid)
        bright = _number(props, "亮电阻", cid)
        max_v = _number(props, "最大电压", cid)
        if dark <= 0 or bright <= 0 or max_v <= 0:
            raise ToolError(f"{cid}: photoresistor light/dark resistance and maximum voltage must be positive")
        append("resistor", [pins[0], pins[1]], {"r": dark}, "dark_resistance", [0, 1],
               common + [
                   "No illumination is serialized. The native solve uses the exact saved dark resistance; bright resistance, sensitivity and response time remain provenance for a future explicit light stimulus.",
               ], extra_source={"engineering_defaults": {"illumination": 0.0,
                   "dark_resistance_ohm": dark, "bright_resistance_ohm": bright,
                   "maximum_voltage_v": max_v}})
    elif kind == "Proximity Sensor":
        low, high = _number(props, "低电平", cid), _number(props, "高电平", cid)
        output_r = _number(props, "输出阻抗", cid)
        if low > high or output_r <= 0:
            raise ToolError(f"{cid}: invalid proximity output levels/impedance")
        internal = allocate("proximity", node=True)
        append("vdc", [internal, "gnd"], {"v": low}, "zero_proximity_source", [0, None],
               common + ["No proximity state is serialized; explicit not-detected state drives saved low level."],
               extra_source={"engineering_defaults": {"detected": False, "low_v": low, "high_v": high}})
        append("resistor", [internal, pins[0]], {"r": output_r}, "output_impedance", [None, 0],
               common + ["Saved output impedance is explicitly in series with the sensor output."])
    elif kind == "Spark Gap":
        breakdown = _number(props, "击穿电压", cid)
        arc_r = _number(props, "击穿电阻", cid)
        hold = _number(props, "维持电流", cid)
        if breakdown <= 0 or arc_r <= 0 or hold < 0:
            raise ToolError(f"{cid}: spark-gap breakdown/resistance/holding current is invalid")
        append("spark_gap", [pins[0], pins[1]], {
                   "breakdown_v": breakdown, "arc_r": arc_r,
                   "holding_current": hold, "off_r": 1e12,
                   "initial_conducting": 0,
               }, "dynamic_arc", [0, 1],
               common + [
                   "Saved breakdown voltage closes the native PE arc during nonlinear iteration; saved holding current extinguishes it at a transient-step boundary. The open state uses an explicit 1 Tohm resistance.",
               ], extra_source={"support_level": "electrical_dynamic_arc",
                   "unmodeled_behavior": "stochastic ignition, plasma temperature, electrode erosion and RF radiation"})
    elif kind == "Tesla Coil":
        l1, l2 = _number(props, "电感1", cid), _number(props, "电感2", cid)
        cap = _number(props, "次级电容", cid)
        resistance = _number(props, "次级电阻", cid)
        breakdown = _number(props, "击穿电压", cid)
        if min(l1, l2, cap, resistance, breakdown) <= 0:
            raise ToolError(f"{cid}: Tesla-coil electrical parameters must be positive")
        s1, s2 = allocate("secondary_a", node=True), allocate("secondary_b", node=True)
        append("coupled_inductors", [pins[0], pins[1], s1, s2],
               {"l1": l1, "l2": l2, "k": .98}, "coupled_windings", [0, 1, None, None],
               common + [
                   "Saved primary/secondary inductances are coupled with explicit engineering k=0.98; the save has no coupling coefficient.",
                   "Secondary is an internal closed R-C load. Arc radiation and dynamic breakdown are not inferred from saved Statistics.",
               ], extra_source={"engineering_defaults": {"coupling": .98,
                                                           "breakdown_voltage_v": breakdown}})
        internal = allocate("secondary_rc", node=True)
        append("resistor", [s1, internal], {"r": resistance}, "secondary_resistance", [None, None], common)
        append("capacitor", [internal, s2], {"c": cap}, "secondary_capacitance", [None, None], common)
    elif kind == "Solenoid":
        turns = _number(props, "线圈匝数", cid)
        radius = _number(props, "内线圈半径", cid)
        inserted = _number(props, "插入铁芯", cid)
        if turns <= 0 or radius <= 0 or inserted not in (0, 1):
            raise ToolError(f"{cid}: invalid solenoid turns/radius/core state")
        # Air-core solenoid with declared length=2r; inserted core uses a
        # conservative explicit relative permeability, not a hidden app fit.
        l_air = 4e-7 * math.pi * turns * turns * math.pi * radius / 2
        relative_mu = 100.0 if inserted else 1.0
        inductance = l_air * relative_mu
        append("coupled_inductors", [pins[0], pins[1], pins[2], pins[3]],
               {"l1": inductance, "l2": inductance, "k": .95 if inserted else .5},
               "coupled_coils", [0, 1, 2, 3], common + [
                   "Both original coil pairs are preserved. Inductance uses the air-solenoid formula with explicit length=2*radius and relative permeability 100 when the saved core is inserted.",
               ], extra_source={"engineering_defaults": {"assumed_length_m": 2 * radius,
                   "relative_permeability": relative_mu, "inductance_h": inductance,
                   "coupling": .95 if inserted else .5}})
    else:  # Electric Fan
        modern_fields = {
            "额定电阻", "电感", "马达常数", "转动惯量", "负荷扭矩",
            "反电动势系数", "粘性摩擦系数",
        }
        if not modern_fields.issubset(props) and {"额定电压", "额定功率"}.issubset(props):
            # The legacy fan predates PhysicsLab's electromechanical state and
            # serializes only a rated operating point.  There is no hidden
            # inertia/back-EMF state to reconstruct faithfully.  Its exact
            # rated-point electrical load is nevertheless well-defined.
            rated_v = _number(props, "额定电压", cid)
            rated_p = _number(props, "额定功率", cid)
            if rated_v <= 0 or rated_p <= 0:
                raise ToolError(f"{cid}: legacy fan rated voltage and power must be positive")
            resistance = rated_v * rated_v / rated_p
            append("resistor", [pins[0], pins[1]], {"r": resistance},
                   "legacy_rated_load", [0, 1], common + [
                       "This legacy PhysicsLab fan save contains only rated voltage and rated power, not the later armature/mechanical parameters.",
                       "PE therefore preserves the exact rated operating point as R=V_rated^2/P_rated; speed, torque and transient motor behavior are explicitly unavailable for this old schema.",
                   ], extra_source={"support_level": "legacy_rated_point_electrical_load",
                       "parameter_derivation": {"rated_voltage_v": rated_v,
                           "rated_power_w": rated_p,
                           "equivalent_resistance_ohm": resistance},
                       "unmodeled_behavior": "motor speed, inertia, back EMF and torque were not serialized by this legacy schema"})
            return output
        resistance = _number(props, "额定电阻", cid)
        inductance = _number(props, "电感", cid)
        torque_constant = _number(props, "马达常数", cid)
        inertia = _number(props, "转动惯量", cid)
        load_torque = _number(props, "负荷扭矩", cid)
        back_emf = _number(props, "反电动势系数", cid)
        friction = _number(props, "粘性摩擦系数", cid)
        initial_omega_raw = props.get("角速度", 0.0)
        if isinstance(initial_omega_raw, bool) or not isinstance(initial_omega_raw, (int, float)) or not math.isfinite(float(initial_omega_raw)):
            raise ToolError(f"{cid}: saved fan angular velocity must be finite when present")
        initial_omega = float(initial_omega_raw)
        if min(resistance, inductance, torque_constant, inertia, back_emf, friction) <= 0:
            raise ToolError(f"{cid}: fan R/L/Kt/J/Ke/friction must be positive")
        append("dc_motor", [pins[0], pins[1]], {
                   "resistance": resistance, "inductance": inductance,
                   "torque_constant": torque_constant, "inertia": inertia,
                   "load_torque": load_torque,
                   "back_emf_constant": back_emf,
                   "viscous_friction": friction,
                   "initial_omega": initial_omega,
               }, "electromechanical_motor", [0, 1], common + [
                   "All serialized electrical and mechanical fan parameters feed the native PE DC-motor equations: armature R/L, Kt, inertia, load torque, Ke, viscous friction and initial angular velocity.",
                   "Transient speed is integrated from the previous solved current and feeds back as back EMF; PE returns electrical current/voltage plus speed, torque and mechanical power.",
               ], extra_source={"support_level": "lumped_electromechanical",
                   "initial_omega_origin": "saved Properties.角速度" if "角速度" in props else "explicit zero for legacy save without dynamic state",
                   "unmodeled_behavior": "blade aerodynamics, nonlinear bearings, commutator ripple and thermal winding failure"})
    return output
