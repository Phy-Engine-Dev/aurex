"""Audited analog PL-to-PE adapters, with explicit provenance and limitations.

The SDK describes file/pin structure, not the closed-source solver. Saved
Statistics are never used as fresh simulation output or a fitted circuit answer.
An instrument's archived V/I may establish an explicitly *inferred* input load.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from .registry import ToolError

SOURCE = "https://github.com/SekaiArendelle/physicslab/blob/v2.0.6/physicsLab/circuit/elements/"


def _number(obj: dict, key: str, cid: str) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolError(f"{cid}: missing/nonfinite saved numeric property {key}")
    return float(value)


def _used_pins(scene: dict, cid: str) -> set[int]:
    used = set()
    for wire in scene.get("wires", []):
        for side in ("Source", "Target"):
            if wire.get(side) == cid:
                used.add(wire[side + "Pin"])
    return used


def import_element(el: dict, *, scene: dict) -> list[dict[str, Any]] | None:
    """Import one recognized original component without changing any source wire.

    Native reference pins absent from the PL element are an explicit sidecar,
    never exported as fabricated original wiring. Unknown native semantics fail
    closed rather than replacing the device with an unrelated ideal source.
    """
    kind = el.get("type")
    if kind not in {"Operational Amplifier", "Triangle Source", "Schmitt Trigger", "Multimeter", "Transistor"}:
        return None
    cid = el["id"]
    props = el.get("properties", {})
    stats = el.get("statistics", {})
    pins = {p["pin"]: p["node"] for p in el.get("pins", [])}
    used = _used_pins(scene, cid)
    assumptions: list[str] = []
    references: list[dict[str, Any]] = []
    mapping: dict[str, int | None] = {}
    source = {"model_id": kind, "raw_properties": copy.deepcopy(props), "raw_statistics": copy.deepcopy(stats),
              "pin_mapping": mapping, "implicit_references": references, "assumptions": assumptions,
              "statistics_source": "original saved values, not fresh simulation measurements",
              "numerical_equivalence_to_original": False, "mapping_source": SOURCE + "artificialCircuit.py"}

    def actual(pin: int) -> str:
        if pin not in pins:
            raise ToolError(f"{cid}: missing connected saved pin {pin}")
        return pins[pin]

    def reference(pin: int, role: str, saved_zero: str | None = None) -> str:
        if pin in used:
            return actual(pin)  # an actually wired floating net must NEVER be grounded
        if pin in pins and saved_zero is not None and saved_zero not in stats:
            # The renderer preserves every known terminal even when the SDK's
            # old all-elements fixture has no Statistics entry. Keep that
            # actual isolated terminal; absence of evidence is not evidence of
            # a zero-volt reference.
            references.append({"pl_pin": pin, "role": role, "node": actual(pin),
                               "external_wire_present": False,
                               "policy": "unwired original terminal retained as its isolated native node; saved reference statistic absent"})
            assumptions.append(f"Unwired {role} (PL pin {pin}) remains floating because no saved zero-reference statistic exists.")
            return actual(pin)
        if saved_zero is not None and _number(stats, saved_zero, cid) != 0:
            raise ToolError(f"{cid}: unconnected {role} does not have a saved zero reference")
        evidence = ({"saved_statistic": saved_zero, "saved_value": 0.0} if saved_zero else
                    {"policy": "unconnected source return terminal uses 0 V reference; absolute voltage is not recorded"})
        references.append({"pl_pin": pin, "role": role, "node": "gnd", "external_wire_present": False, **evidence})
        assumptions.append(f"Unwired {role} (PL pin {pin}) uses a 0 V simulation reference; no original wire was changed.")
        return "gnd"

    def implicit(native_pin: int) -> str:
        mapping[str(native_pin)] = None
        references.append({"native_pin": native_pin, "pl_pin": None, "node": "gnd", "role": "implicit output reference"})
        return "gnd"

    if kind == "Transistor":
        variant, beta = (_number(props, p, cid) for p in ("PNP", "放大系数"))
        if variant not in (0, 1) or beta <= 0:
            raise ToolError(f"{cid}: invalid saved transistor polarity or forward beta")
        native_type = "pnp" if variant else "npn"
        from ..phy_engine.catalog import COMPONENTS
        model = COMPONENTS[native_type]
        params = copy.deepcopy(model["defaults"])
        native = el.get("native") or {}
        if native.get("type") == native_type and isinstance(native.get("params"), dict):
            params.update(copy.deepcopy(native["params"]))
        params["beta"] = beta
        mapping.update({"0": 0, "1": 1, "2": 2})
        nodes = [actual(p) for p in range(3)]
        assumptions.append("Original B/C/E pins and PNP/forward beta are preserved. Undocumented junction parameters use explicitly recorded PE defaults unless an original PE sidecar supplies them; no fitting to a desired circuit answer. Quasi-static Ebers-Moll lacks original-App calibration, charge storage and thermal damage; any saved maximum-power rating is retained metadata, not a silently invented thermal model.")
    elif kind == "Operational Amplifier":
        gain, low, high = (_number(props, p, cid) for p in ("增益系数", "最小电压", "最大电压"))
        if gain < 0 or low > high:
            raise ToolError(f"{cid}: invalid finite gain or voltage rails")
        mapping.update({"0": 1, "1": 0, "2": 2})
        nodes = [reference(1, "positive amplifier input", "电压+"), reference(0, "negative amplifier input", "电压-"), actual(2), implicit(3)]
        native_type, params = "clamped_op_amp", {"gain": gain, "min_v": low, "max_v": high}
        assumptions.append("Finite-gain hard-clamped behavioral amplifier: zero input current, ideal output; no undocumented bandwidth, slew or current limit was invented.")
    elif kind == "Triangle Source":
        amplitude, offset, frequency, resistance = (
            _number(props, p, cid) for p in ("电压", "偏移", "频率", "内阻"))
        duty = _number(props, "占空比", cid) if "占空比" in props else .5
        if amplitude < 0 or frequency <= 0 or not 0 < duty < 1 or resistance < 0:
            raise ToolError(f"{cid}: invalid triangle amplitude/frequency/duty/internal resistance")
        mapping.update({"0": 0, "1": 1})
        nodes = [actual(0), reference(1, "waveform source return")]
        native_type, params = "triangle_loaded", {"high_v": offset + amplitude, "low_v": offset - amplitude,
            "freq_hz": frequency, "phase_rad": 0, "duty": duty, "r_series": resistance}
        assumptions.append("Saved voltage is interpreted as peak excursion about offset; phase is not saved, so native t=0 starts at offset-amplitude. Saved Statistics have unknown phase/time and are not a transient initial condition.")
        if "占空比" not in props:
            assumptions.append(
                "Legacy Triangle Source save omits 占空比; the public PhysicsLab waveform-source default 0.5 is used explicitly for the symmetric triangle, while raw_properties remains unchanged."
            )
    elif kind == "Schmitt Trigger":
        # Preserve the historical adjustable-threshold schema. Current SDK
        # saves expose only levels/inversion; for that documented schema PE
        # uses explicit one-third/two-thirds thresholds, matching the legacy
        # defaults without pretending that hidden app state was serialized.
        legacy = "高电准位" in props and "低电准位" in props
        if legacy:
            low, high, t_low, t_high, inverted, slew = (_number(props, p, cid) for p in
                ("低电准位", "高电准位", "负向阈值", "正向阈值", "工作模式", "切变速率"))
        elif all(key in props for key in ("低电平", "高电平", "反相")):
            low, high, inverted = (_number(props, p, cid) for p in
                                   ("低电平", "高电平", "反相"))
            span = high - low
            t_low, t_high, slew = low + span / 3.0, low + 2.0 * span / 3.0, 0.0
        else:
            raise ToolError(f"{cid}: Schmitt save matches neither the legacy nor current public schema")
        if t_low > t_high or inverted not in (0, 1) or slew < 0:
            raise ToolError(f"{cid}: invalid Schmitt mode/thresholds/slew")
        mapping.update({"0": 0, "1": 1})
        nodes = [reference(0, "Schmitt input", "输入电压"), actual(1), implicit(2)]
        native_type, params = "analog_schmitt", {"threshold_low_v": t_low, "threshold_high_v": t_high,
            "inverted": inverted, "low_v": low, "high_v": high, "slew_v_per_s": 0}
        if legacy:
            assumptions.append(
                "Saved thresholds, inversion and unequal output levels are preserved. The legacy 切变速率 value is retained only in raw_properties because its unit is undocumented; native slew is explicitly zero (instantaneous transition), never a guessed V/s conversion."
            )
            source["engineering_defaults"] = {
                "native_slew_v_per_s": 0.0,
                "reason": "legacy PhysicsLab 切变速率 unit is undocumented",
                "saved_legacy_slew": slew,
            }
        else:
            assumptions.append(
                "Current public saves serialize only low/high levels and inversion. Native thresholds are explicitly derived at one-third/two-thirds of that saved range, the same values used by the legacy default schema; native slew is instantaneous."
            )
            source["engineering_defaults"] = {
                "native_slew_v_per_s": 0.0,
                "threshold_derivation": "low + (high-low)/3 and low + 2*(high-low)/3",
                "schema": "current public PhysicsLab low/high/inverted",
            }
        source["mapping_source"] = SOURCE + "logicCircuit.py"
    else:
        mode = _number(props, "状态", cid)
        if mode != 16:
            # The broad device adapter provides a safe high-impedance voltage
            # observation load for undocumented dial modes.  Mode 16 retains
            # this stricter archived-V/I loading calibration path.
            return None
        evidence = []
        for v_key, i_key in (("瞬间电压", "瞬间电流"), ("电压", "电流")):
            v, i = _number(stats, v_key, cid), _number(stats, i_key, cid)
            if v == 0 or i == 0 or not math.isfinite(v / i) or v / i <= 0:
                return None
            evidence.append({"voltage_key": v_key, "current_key": i_key, "voltage": v, "current": i, "inferred_ohm": v / i})
        r = evidence[0]["inferred_ohm"]
        if not math.isclose(r, evidence[1]["inferred_ohm"], rel_tol=1e-6, abs_tol=0):
            return None
        for v_key, i_key, p_key in (("瞬间电压", "瞬间电流", "瞬间功率"), ("电压", "电流", "功率")):
            if p_key in stats and not math.isclose(_number(stats, p_key, cid), _number(stats, v_key, cid) * _number(stats, i_key, cid), rel_tol=1e-5, abs_tol=1e-18):
                return None
        mapping.update({"0": 0, "1": 1})
        nodes = [actual(0), actual(1)]
        native_type, params = "voltage_meter", {"r_input": r}
        source["loading_evidence"] = evidence
        source["mapping_source"] = SOURCE + "basicCircuit.py"
        assumptions.append("Meter is a finite passive input conductance inferred from two consistent archived V/I pairs, not an official dial-impedance specification or an independently calibrated original-App model. New voltages/currents come only from the solver.")
    component = {"id": cid, "type": native_type, "nodes": nodes, "params": params, "pl_source": source,
                 **{key: copy.deepcopy(el[key]) for key in ("position", "rotation", "position_source") if key in el}}
    if el.get("label"):
        component["label"] = el["label"]
    return [component]
