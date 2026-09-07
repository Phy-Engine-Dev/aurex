"""PL sources and reactive passives with explicit, provenance-bearing parasitics.

SDK fields establish serialized pins and values, not the closed-source solver's
complete constitutive equations. Native waveform conventions and unsupported
rating/initial-state behavior are therefore recorded, never claimed identical.
"""
from __future__ import annotations

import copy
import hashlib
import math

from .registry import ToolError


_SDK = "https://github.com/SekaiArendelle/physicslab/blob/fa95b96910dd0fd4e09cf27e24cefaf9b91798ad/"
_RECOGNIZED = {"Battery Source", "Current Source", "Basic Capacitor", "Basic Inductor",
               "Sinewave Source", "Square Source", "Sawtooth Source", "Pulse Source", "Student Source"}

# Integration metadata only. This module never mutates the shared catalog.
CATALOG_ADDITIONS = {
    "square": {"code": 21, "pins": 2, "pin_labels": ["+", "-"],
        "props": ["high_v", "low_v", "freq_hz", "duty", "phase_rad"],
        "defaults": {"phase_rad": 0}, "model_id": "", "pl_props": {}, "constant_pl_props": {}},
    "sawtooth": {"code": 20, "pins": 2, "pin_labels": ["+", "-"],
        "props": ["high_v", "low_v", "freq_hz", "phase_rad"],
        "defaults": {"phase_rad": 0}, "model_id": "", "pl_props": {}, "constant_pl_props": {}},
    "pulse": {"code": 22, "pins": 2, "pin_labels": ["+", "-"],
        "props": ["high_v", "low_v", "freq_hz", "duty", "phase_rad", "rise_s", "fall_s"],
        "defaults": {"phase_rad": 0, "rise_s": 0, "fall_s": 0},
        "model_id": "", "pl_props": {}, "constant_pl_props": {}},
}


def _number(props: dict, key: str, cid: str) -> float:
    value = props.get(key)
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ToolError(f"{cid}: original {key} must be an explicit finite number")
    return float(value)


def _optional_number(props: dict, key: str, cid: str, default: float = 0.0) -> float:
    """Read a legacy optional numeric property without inventing it in metadata."""
    return _number(props, key, cid) if key in props else default


def _pose(el: dict, field: str) -> list:
    raw = el.get(field)
    try:
        valid = isinstance(raw, list) and len(raw) == 3 and all(
            type(v) in (int, float) and math.isfinite(v) for v in raw)
    except OverflowError:
        valid = False
    if not valid:
        raise ToolError(f"{el['id']}: original finite {field} is required")
    return copy.deepcopy(raw)


def _pins(el: dict, count: int = 2) -> dict[int, str]:
    raw = el.get("pins")
    if not isinstance(raw, list) or len(raw) != count:
        raise ToolError(f"{el['id']}: all {count} original source/passive pins are required")
    pins = {}
    for pin in raw:
        if (not isinstance(pin, dict) or type(pin.get("pin")) is not int
            or pin["pin"] not in range(count) or pin["pin"] in pins
            or not isinstance(pin.get("node"), str) or not 0 < len(pin["node"]) <= 128):
            raise ToolError(f"{el['id']}: missing, duplicate or invalid original pin mapping")
        pins[pin["pin"]] = pin["node"]
    return pins


def import_element(el: dict, *, scene: dict) -> list[dict] | None:
    """Preserve external connectivity; expand explicit losses into native R.

    Return None for other importer groups. Recognized but ambiguous devices fail
    explicitly instead of being replaced by a differently functioning circuit.
    """
    if not isinstance(el, dict) or el.get("type") not in _RECOGNIZED:
        return None
    kind, cid = el["type"], el.get("id")
    if not isinstance(cid, str) or not 0 < len(cid) <= 128:
        raise ToolError("Source import requires the original component ID")
    props = el.get("properties")
    if not isinstance(props, dict):
        raise ToolError(f"{cid}: original properties are required")
    if el.get("is_broken", el.get("IsBroken", False)):
        raise ToolError(f"{cid}: broken-device behavior requires the complete SAV damage adapter")
    if not isinstance(scene, dict) or not isinstance(scene.get("components"), list) or not isinstance(scene.get("wires"), list):
        raise ToolError(f"{cid}: full original scene components and wires are required")
    if not all(isinstance(c, dict) for c in scene["components"]):
        raise ToolError(f"{cid}: invalid original scene component")
    ids = [c.get("id") for c in scene["components"]]
    if any(not isinstance(x, str) for x in ids) or ids.count(cid) != 1 or len(ids) != len(set(ids)):
        raise ToolError(f"{cid}: scene must contain each original ID exactly once")
    pins = _pins(el, 4 if kind == "Student Source" else 2)
    position, rotation = _pose(el, "position"), _pose(el, "rotation")
    label = el.get("label")
    if label is not None and not isinstance(label, str):
        raise ToolError(f"{cid}: original label must be text or null")
    wired = set()
    for wire in scene["wires"]:
        if not isinstance(wire, dict):
            raise ToolError(f"{cid}: invalid original wire record")
        for id_key, pin_key in (("Source", "SourcePin"), ("Target", "TargetPin")):
            if wire.get(id_key) == cid:
                pin = wire.get(pin_key)
                if type(pin) is not int or pin not in pins:
                    raise ToolError(f"{cid}: wire references an invalid original pin")
                wired.add(pin)

    resistance = 0.0 if kind == "Student Source" and "内阻" not in props else _number(props, "内阻", cid)
    if resistance < 0:
        raise ToolError(f"{cid}: negative internal resistance is not supported")
    assumptions = [
        "Original pin numbers, external nodes, pose and source properties are preserved. Any compatibility reference added for a single-ended source is enumerated in implicit_references; other unwired terminals remain floating.",
        "This is an explicit native engineering model, not a claim of numerical equivalence to the closed-source PhysicsLab simulator.",
        "Saved Statistics are retained only as historical observations; they never initialize or override a new native simulation.",
        "Rated voltage/current/power and failure behavior are metadata, not native overload, breakdown, heating or saturation models.",
    ]
    native_type = ""
    params = {}
    offset = 0.0
    engineering_defaults = {}
    student = None
    if kind == "Student Source":
        vrms = _number(props, "交流电压", cid)
        vdc = _number(props, "直流电压", cid)
        frequency = _number(props, "频率", cid)
        state = _number(props, "开关", cid)
        peak = math.sqrt(2) * vrms
        if vrms < 0 or frequency <= 0 or state not in (0, 1) or not all(
            math.isfinite(v) for v in (peak, frequency * math.tau, 1 / frequency)):
            raise ToolError(f"{cid}: Student Source requires finite nonnegative AC RMS voltage, positive frequency and explicit 开关 0 or 1")
        student = {"dc_v": vdc, "ac_peak_v": peak, "frequency_hz": frequency, "closed": int(state)}
        engineering_defaults = {"dc_terminals": {"positive": 0, "negative": 1},
            "ac_terminals": {"positive": 2, "negative": 3}, "ac_voltage_definition": "rms",
            "switch_definition": "0=both_outputs_disconnected,1=both_outputs_enabled",
            "phase_deg": 0, "output_series_resistance_ohm_per_channel": resistance}
        assumptions += [
            "Student Source has separate DC PL0/1 and sinusoidal AC PL2/3 channels; no connection between their returns is added unless present in original wires.",
            "AC PL2/3 and RMS interpretation are corroborated by public charger 5f6f3c8b7cb6f90001ef3e46: its 220 V setting has saved instantaneous voltage -239.727 V, exceeding 220 V in magnitude. RMS converts to native peak by sqrt(2).",
            "DC polarity PL0 positive/PL1 negative, AC positive PL2/negative PL3, switch 1 enabling both channels and 0 disconnecting them are explicit engineering conventions; the SDK does not independently specify all these behaviors.",
            "A disabled output is isolated through an open native contact, not replaced by a 0 V source short; finite engine r_open permits tiny leakage. No unwired return is grounded or linked to another channel.",
            "The SDK has no Student Source internal-resistance field; absent 内阻 means an explicitly ideal source. If present, the saved value is retained as series resistance on each independent channel by declared convention.",
            "AC phase starts at zero in a new simulation; original saved voltage/current Statistics are not used to infer the old simulation time or phase.",
        ]
    elif kind in {"Basic Capacitor", "Basic Inductor"}:
        native_type, field, key = ("capacitor", "电容", "c") if kind == "Basic Capacitor" else ("inductor", "电感", "l")
        number = _number(props, field, cid)
        if number <= 0:
            raise ToolError(f"{cid}: {field} must be positive")
        # Historical PL saves predate this SDK property. Their explicit ESR is
        # still meaningful; absence is not a license to discard it or add a
        # synthetic field to the archived original properties.
        legacy_mode_absent = "理想模式" not in props
        ideal = 0 if legacy_mode_absent else _number(props, "理想模式", cid)
        if ideal not in (0, 1):
            raise ToolError(f"{cid}: unknown ideal-mode value")
        params = {key: number}
        assumptions += [
            "An explicit nonzero saved 内阻 is modeled as series ESR. A zero saved resistance remains ideal.",
            "Reactive state starts under native initialization, not by replaying saved capacitor voltage or inductor current Statistics.",
        ]
        if ideal == 1 and resistance != 0:
            assumptions.append(
                "The archived save simultaneously marks 理想模式=1 and stores a nonzero 内阻. "
                "The explicit numeric resistance is retained as series ESR instead of making the whole experiment unusable; "
                "this declared compatibility choice does not claim pointwise parity with the closed-source app."
            )
        if legacy_mode_absent:
            assumptions.append("Legacy save has no 理想模式 field: the explicitly saved 内阻 is modeled as nonideal series ESR. No 理想模式 field is inserted into raw_properties and no saved resistance is ignored.")
    elif kind == "Battery Source":
        native_type, params = "vdc", {"v": _number(props, "电压", cid)}
        assumptions.append("Battery emf is V(native pin0)-V(native pin1); explicit 内阻 is a series resistance, not a changed emf.")
    elif kind == "Current Source":
        if resistance <= 0:
            raise ToolError(f"{cid}: current-source shunt resistance must be positive; zero is not silently reinterpreted as an ideal infinite resistance")
        native_type, params = "idc", {"i": _number(props, "电流", cid)}
        assumptions += [
            "Current-source 内阻 is a parallel Norton shunt, never a series resistor.",
            "Native positive current flows from mapped PL pin0 (red) to pin1 (black); original-app current arrow parity is not independently established.",
        ]
    else:
        amplitude = _number(props, "电压", cid)
        frequency = _number(props, "频率", cid)
        offset = _number(props, "偏移", cid)
        # Duty is constitutive only for square/pulse waveforms.  Historical
        # sine and sawtooth saves often omit it, and rejecting such a save for
        # a property the native equation never reads is not faithful import.
        # Modern saves serialize the public waveform-source default.  Very old
        # square/pulse saves omit it entirely; for those schemas the only
        # reproducible contract is the public default 0.5, not a rejection or
        # a value fitted from archived Statistics.
        duty = _optional_number(props, "占空比", cid, .5)
        if amplitude < 0 or frequency <= 0 or not 0 < duty < 1:
            raise ToolError(f"{cid}: waveform amplitude must be nonnegative, frequency positive and duty strictly between zero and one")
        if not math.isfinite(frequency * math.tau) or not math.isfinite(1 / frequency):
            raise ToolError(f"{cid}: frequency exceeds finite native angular-frequency or period range")
        if kind not in {"Square Source", "Pulse Source"} and duty != .5:
            assumptions.append(
                f"Saved generic duty={duty:g} is retained as provenance but has no constitutive meaning for {kind}; waveform output is not duty-gated."
            )
        phase_deg = _optional_number(props, "初始相位", cid)
        phase_rad = math.radians(phase_deg)
        if not math.isfinite(phase_rad):
            raise ToolError(f"{cid}: initial phase is outside the native finite range")
        assumptions += [
            "Waveform engineering convention: 电压 is peak amplitude, 偏移 is a DC voltage offset, 频率 is Hz and 初始相位 is degrees; the SDK does not prove the original app's pointwise formula.",
            "Explicit waveform source resistance is series loading, preserved independently of the signal and its DC offset.",
        ]
        if kind not in {"Square Source", "Pulse Source"} and "占空比" not in props:
            assumptions.append(
                f"Archived {kind} omits 占空比; no field is invented in raw_properties and the native waveform does not use duty."
            )
        elif kind in {"Square Source", "Pulse Source"} and "占空比" not in props:
            assumptions.append(
                f"Legacy {kind} save omits 占空比; the public PhysicsLab waveform-source default 0.5 is used explicitly, while raw_properties remains unchanged."
            )
        if kind == "Sinewave Source":
            native_type, params = "vac", {"vp": amplitude, "freq_hz": frequency, "phase_deg": phase_deg}
            assumptions.append("Native transient waveform is offset + amplitude*sin(2*pi*f*t); a separate ideal DC source supplies the exact saved offset. AC phasor conventions follow native VAC, not the saved app.")
        elif kind == "Square Source":
            native_type, params = "square", {"high_v": offset + amplitude, "low_v": offset - amplitude,
                "freq_hz": frequency, "duty": duty, "phase_rad": phase_rad}
            offset = 0.0  # Offset already occurs in both explicit levels.
            assumptions.append("Native square starts at high level and stays high for duty/f seconds per cycle; edge timing/parity with the app is not asserted.")
        elif kind == "Pulse Source":
            ramp = (duty / frequency) / 2
            if not math.isfinite(ramp) or ramp < 1e-30:
                raise ToolError(f"{cid}: pulse rise/fall duration is below the native finite-ramp resolution")
            native_type, params = "pulse", {"high_v": offset + amplitude, "low_v": offset - amplitude,
                "freq_hz": frequency, "duty": duty, "phase_rad": phase_rad, "rise_s": ramp, "fall_s": ramp}
            offset = 0.0
            engineering_defaults = {"pulse_shape": "symmetric_triangular_pulse", "active_width_s": duty / frequency,
                "rise_s": ramp, "fall_s": ramp, "phase_rad": phase_rad,
                "level_definition": "low=offset-amplitude,peak=offset+amplitude"}
            assumptions.append("Declared finite-width spike engineering waveform: each period starts at offset-amplitude, rises linearly to offset+amplitude over duty/(2*f), falls over the same duration, then stays low for the rest of the period. Native pulse ABI22 supplies the actual waveform; original-app formula is not independently established.")
            assumptions.append("Rise/fall durations and phase are explicit editable native parameters. This is not an ideal Dirac impulse; time steps must resolve its active width. Native small-signal AC excitation is zero; use transient analysis to observe the pulse.")
        else:
            native_type, params = "sawtooth", {"high_v": offset + amplitude, "low_v": offset - amplitude,
                "freq_hz": frequency, "phase_rad": phase_rad}
            offset = 0.0
            assumptions.append("Native sawtooth rises from offset-amplitude to offset+amplitude, then resets; app direction/phase parity is not established. Default duty .5 is retained as metadata because native sawtooth has no duty control.")
        if not all(math.isfinite(value) for value in params.values()):
            raise ToolError(f"{cid}: derived waveform levels exceed finite native range")

    # PhysicsLab waveform generators are commonly used as one-terminal signal
    # sources.  Archived community saves prove that such a source can carry a
    # nonzero current even though its other serialized pin has no Wire record,
    # so the closed-source app supplies an implicit reference for that pin.
    # Native PE sources are strictly differential; leaving the second pin
    # floating makes GMIN split the voltage around zero and changes threshold
    # crossings (for example a 0..1 V clock becomes +1/3..-2/3 V under load).
    # Apply this compatibility rule only to waveform sources with exactly one
    # externally wired terminal. Batteries/current/student sources retain their
    # literal two-terminal topology.
    waveform_kinds = {"Sinewave Source", "Square Source", "Sawtooth Source", "Pulse Source"}
    terminal_positive, terminal_negative = pins[0], pins[1]
    implicit_references = []
    if kind in waveform_kinds and len(wired) == 1:
        missing_pin = 1 if 0 in wired else 0
        if missing_pin == 0:
            terminal_positive = "gnd"
        else:
            terminal_negative = "gnd"
        implicit_references.append({
            "pl_pin": missing_pin,
            "original_node": pins[missing_pin],
            "native_reference_node": "gnd",
            "reason": (
                "PhysicsLab one-terminal waveform-source compatibility: archived community saves record "
                "nonzero source current with this terminal absent from every Wire record"
            ),
            "scope": "native simulation only; no original wire or PLSAV field is added",
        })
        assumptions.append(
            f"Only PL pin {next(iter(wired))} is externally wired; unwired PL pin {missing_pin} is referenced to native ground for PhysicsLab single-ended waveform-source compatibility."
        )

    source = {"model_id": kind, "identifier": cid, "raw_properties": copy.deepcopy(props),
        "raw_statistics": copy.deepcopy(el.get("statistics", {})),
        "pin_mapping": [{"pl_pin": pin, "node": node, "externally_wired": pin in wired} for pin, node in sorted(pins.items())],
        "implicit_references": implicit_references, "assumptions": assumptions,
        "mapping_reference": [_SDK + "physicslab/circuit/elements/" + ("basic_circuit.py" if kind in {"Battery Source", "Student Source"} else "artificial_circuit.py")],
        "numerical_equivalence_to_original": False, "support_level": "explicit_native_engineering_mapping"}
    if engineering_defaults:
        source["engineering_defaults"] = engineering_defaults
    if student is not None:
        source["inactive_unconnected_subchannel"] = [
            {"channel": channel, "pl_pins": [a, b], "pin_ids": [f"{cid}:{a}", f"{cid}:{b}"],
             "nodes": [pins[a], pins[b]], "reason": "neither original terminal has a wire",
             "retained_in_native_spec": True, "scope_effect": "informational only; no component removal or numerical grounding was performed"}
            for channel, a, b in (("dc", 0, 1), ("ac", 2, 3)) if a not in wired and b not in wired]
        source["assumptions"].append("A fully disconnected channel is retained and identified by inactive_unconnected_subchannel. Its floating voltage-source island may require an explicitly disclosed numerical reference or connected-scope analysis; this importer does not invent either.")
    used_ids = set(ids)
    used_nodes = set()
    for component in scene["components"]:
        if not isinstance(component.get("pins"), list):
            raise ToolError(f"{cid}: complete scene pins are required to reserve collision-free internal nodes")
        for pin in component["pins"]:
            if not isinstance(pin, dict) or not isinstance(pin.get("node"), str):
                raise ToolError(f"{cid}: malformed scene pin/node record")
            used_nodes.add(pin["node"])
    digest = hashlib.sha256(cid.encode("utf-8")).hexdigest()

    def allocate(role: str, *, node: bool = False) -> str:
        used = used_nodes if node else used_ids
        # Retain Aurex's established helper prefixes for internal resistance so
        # existing documents and diagnostics recognize the same decomposition.
        if role == "series" and node:
            base = "__plsav_internal_node_" + digest
        elif role in {"series_resistance", "parallel_resistance"} and not node:
            base = "__plsav_internal_resistance_" + digest
        else:
            base = "__pl_source_" + ("node_" if node else "") + digest + "_" + role
        value, suffix = base, 0
        while value in used:
            suffix += 1
            value = base + "_" + str(suffix)
        used.add(value)
        return value

    output = []

    def append(kind: str, nodes: list[str], values: dict, role: str, pin_mapping: list[int | None]):
        primary = not output
        provenance = copy.deepcopy(source)
        provenance.update({"parent_identifier": cid, "is_helper": not primary, "decomposition_role": role,
            "primitive_pin_mapping": pin_mapping})
        item = {"id": cid if primary else allocate(role), "type": kind, "nodes": nodes, "params": values,
                "position": copy.deepcopy(position), "rotation": copy.deepcopy(rotation), "pl_source": provenance}
        if primary and label:
            item["label"] = label
        output.append(item)

    if student is not None:
        for channel, a, b, device_type, values in (
            ("dc", 0, 1, "vdc", {"v": student["dc_v"]}),
            ("ac", 2, 3, "vac", {"vp": student["ac_peak_v"], "freq_hz": student["frequency_hz"], "phase_deg": 0}),
        ):
            core_positive = allocate(channel + "_emf", node=True)
            append(device_type, [core_positive, pins[b]], values, channel + "_core", [None, b])
            contact_inner = core_positive
            if resistance > 0:
                contact_inner = allocate(channel + "_series", node=True)
                append("resistor", [core_positive, contact_inner], {"r": resistance}, channel + "_series_resistance", [None, None])
            # PhysicsLab contacts are numerically finite.  A 1 nOhm series
            # contact preserves the enabled voltage to engineering precision
            # while keeping two equal Student Sources in parallel solvable;
            # ideal duplicate voltage-source constraints are singular in MNA.
            append("switch", [contact_inner, pins[a]],
                   {"closed": student["closed"], "r_closed": 1e-9},
                   channel + "_enable_contact", [None, a])
        return output

    positive = terminal_positive
    negative = terminal_negative
    series = resistance > 0 and kind != "Current Source"
    if series:
        positive = allocate("series", node=True)
    if offset != 0:
        negative = allocate("offset", node=True)
    append(native_type, [positive, negative], params, "core", [None if series else 0, None if offset != 0 else 1])
    if series:
        append("resistor", [terminal_positive, positive], {"r": resistance}, "series_resistance", [0 if 0 in wired else None, None])
    if offset != 0:
        append("vdc", [negative, terminal_negative], {"v": offset}, "dc_offset", [None, 1 if 1 in wired else None])
    if kind == "Current Source":
        append("resistor", [pins[0], pins[1]], {"r": resistance}, "parallel_resistance", [0, 1])
    return output
