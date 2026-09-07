"""PE C ABI component and pin definitions, with explicit PL export mappings.

Native-only models are valid Phy-Engine models but are deliberately not presented
as PhysicsLab-compatible .sav exports. All values are SI unless the key says otherwise.
"""
from __future__ import annotations

from typing import Any


# PhysicsLab stores circuit properties as single-precision values (community
# saves retain their characteristic float32 rounding). Use the largest finite
# float32 value: JSON cannot represent Infinity, while omitting the property is
# interpreted as zero and immediately marks an ideal source as broken.
PL_MAX_POWER_W = float.fromhex("0x1.fffffep+127")


def component(code: int, labels: list[str], props: list[str], *, defaults=None,
              model_id: str = "", pl_props=None, constant_pl_props=None) -> dict[str, Any]:
    return {"code": code, "pins": len(labels), "pin_labels": labels, "props": props,
            "defaults": defaults or {}, "model_id": model_id, "pl_props": pl_props or {},
            "constant_pl_props": constant_pl_props or {}}


COMPONENTS = {
    "resistor": component(1, ["1", "2"], ["r"], model_id="Resistor", pl_props={"r": "电阻"}),
    "capacitor": component(2, ["+", "-"], ["c"], model_id="Basic Capacitor", pl_props={"c": "电容"}),
    "inductor": component(3, ["1", "2"], ["l"], model_id="Basic Inductor", pl_props={"l": "电感"}),
    "vdc": component(4, ["+", "-"], ["v"], model_id="Battery Source", pl_props={"v": "电压"},
        constant_pl_props={"内阻": 0, "最大功率": PL_MAX_POWER_W, "锁定": 1}),
    "vac": component(5, ["+", "-"], ["vp", "freq_hz", "phase_deg"], defaults={"phase_deg": 0}),
    "idc": component(6, ["+", "-"], ["i"], model_id="Current Source", pl_props={"i": "电流"}),
    "iac": component(7, ["+", "-"], ["ip", "freq_hz", "phase_deg"], defaults={"phase_deg": 0}),
    "vccs": component(8, ["control+", "control-", "out+", "out-"], ["g"]),
    "vcvs": component(9, ["control+", "control-", "out+", "out-"], ["gain"]),
    "cccs": component(10, ["sense+", "sense-", "out+", "out-"], ["gain"]),
    "ccvs": component(11, ["sense+", "sense-", "out+", "out-"], ["r"]),
    # C ABI 33 extends legacy code 12 with explicit R_on without breaking old
    # native callers that still provide the original one-property payload.
    "switch": component(33, ["1", "2"], ["closed", "r_closed"], defaults={"r_closed": 0},
        model_id="Simple Switch", pl_props={"closed": "开关"}),
    "diode": component(13, ["anode", "cathode"], ["is", "n", "isr", "nr", "temp_c", "ibv", "bv", "bv_set", "area"],
        defaults={"is": 1e-14, "n": 1, "isr": 0, "nr": 2, "temp_c": 27, "ibv": 1e-3, "bv": 40, "bv_set": 1, "area": 1}),
    "transformer": component(14, ["primary+", "primary-", "secondary+", "secondary-"], ["ratio"]),
    "coupled_inductors": component(15, ["L1+", "L1-", "L2+", "L2-"], ["l1", "l2", "k"]),
    "transformer_center_tap": component(16, ["P+", "P-", "S1", "CT", "S2"], ["ratio"]),
    "op_amp": component(17, ["+", "-", "out+", "out-"], ["gain"], defaults={"gain": 1e6}),
    "comparator": component(19, ["+", "-", "out"], ["low_v", "high_v"], defaults={"low_v": 0, "high_v": 5}),
    "sawtooth": component(20, ["+", "-"], ["high_v", "low_v", "freq_hz", "phase_rad"], defaults={"phase_rad": 0}),
    "square": component(21, ["+", "-"], ["high_v", "low_v", "freq_hz", "duty", "phase_rad"],
        defaults={"phase_rad": 0}),
    "pulse": component(22, ["+", "-"], ["high_v", "low_v", "freq_hz", "duty", "phase_rad", "rise_s", "fall_s"],
        defaults={"phase_rad": 0, "rise_s": 0, "fall_s": 0}),
    "clamped_op_amp": component(24, ["+", "-", "out", "ref"], ["gain", "min_v", "max_v"],
        defaults={"gain": 1e6, "min_v": -15, "max_v": 15}),
    "analog_schmitt": component(26, ["in", "out", "ref"],
        ["threshold_low_v", "threshold_high_v", "inverted", "low_v", "high_v", "slew_v_per_s"],
        defaults={"threshold_low_v": 1.6666666666666667, "threshold_high_v": 3.3333333333333335,
                  "inverted": 0, "low_v": 0, "high_v": 5, "slew_v_per_s": 0}),
    "triangle_loaded": component(27, ["+", "-"], ["high_v", "low_v", "freq_hz", "phase_rad", "duty", "r_series"],
        defaults={"high_v": 5, "low_v": 0, "freq_hz": 1000, "phase_rad": 0, "duty": .5, "r_series": 0}),
    "relay_current_spdt": component(28, ["NC", "COM", "NO", "COIL+", "COIL-"],
        ["l", "r", "i_pull", "i_drop", "r_on", "r_off", "t_on", "t_off", "initial"],
        defaults={"l": .2, "r": 20, "i_pull": .02, "i_drop": .016, "r_on": .01,
                  "r_off": 1e12, "t_on": 0, "t_off": 0, "initial": 0}),
    "rated_protection": component(29, ["line_in", "line_out", "sense+", "sense-"],
        ["r_on", "r_open", "max_current_a", "max_voltage_v", "max_power_w", "initial_broken",
         "ambient_temp_c", "rating_reference_temp_c", "trip_temp_c",
         "thermal_resistance_k_per_w", "thermal_capacitance_j_per_k"],
        defaults={"r_on": 1e-9, "r_open": 1e12, "max_current_a": 0,
                  "max_voltage_v": 0, "max_power_w": 0, "initial_broken": 0,
                  "ambient_temp_c": 25, "rating_reference_temp_c": 25, "trip_temp_c": 150,
                  "thermal_resistance_k_per_w": 1, "thermal_capacitance_j_per_k": 10}),
    "ne555_timer": component(30, ["vcc", "dis", "thr", "ctrl", "trig", "out", "reset", "ground"],
        ["low_v", "high_v", "output_r", "discharge_on_r", "discharge_off_r",
         "internal_control", "internal_reset_pullup"],
        defaults={"low_v": 0, "high_v": 3, "output_r": 1e-3,
                  "discharge_on_r": 1e-3, "discharge_off_r": 1e12,
                  "internal_control": 1, "internal_reset_pullup": 1}),
    "spark_gap": component(31, ["A", "B"],
        ["breakdown_v", "arc_r", "holding_current", "off_r", "initial_conducting"],
        defaults={"breakdown_v": 1000, "arc_r": 1, "holding_current": 0,
                  "off_r": 1e12, "initial_conducting": 0}),
    "dc_motor": component(32, ["+", "-"],
        ["resistance", "inductance", "torque_constant", "inertia", "load_torque",
         "back_emf_constant", "viscous_friction", "initial_omega"],
        defaults={"resistance": 1, "inductance": 1e-3, "torque_constant": .01,
                  "inertia": 1e-5, "load_torque": 0, "back_emf_constant": .01,
                  "viscous_friction": 1e-5, "initial_omega": 0}),
    "npn": component(50, ["base", "collector", "emitter"], ["is", "n", "beta", "temp_c", "area"],
        defaults={"is": 1e-14, "n": 1, "beta": 100, "temp_c": 27, "area": 1},
        model_id="Transistor", pl_props={"beta": "放大系数"},
        constant_pl_props={"PNP": 0, "最大功率": PL_MAX_POWER_W, "锁定": 1}),
    "pnp": component(51, ["base", "collector", "emitter"], ["is", "n", "beta", "temp_c", "area"],
        defaults={"is": 1e-14, "n": 1, "beta": 100, "temp_c": 27, "area": 1},
        model_id="Transistor", pl_props={"beta": "放大系数"},
        constant_pl_props={"PNP": 1, "最大功率": PL_MAX_POWER_W, "锁定": 1}),
    "nmos": component(52, ["drain", "gate", "source"], ["kp", "lambda", "vth"], defaults={"kp": 1e-3, "lambda": 0, "vth": 1}),
    "pmos": component(53, ["drain", "gate", "source"], ["kp", "lambda", "vth"], defaults={"kp": 1e-3, "lambda": 0, "vth": -1}),
    "digital_input": component(200, ["out"], ["state"], model_id="Logic Input", pl_props={"state": "开关"}),
    "digital_output": component(201, ["in"], [], model_id="Logic Output"),
}

COMPONENTS["clamped_op_amp"].update({
    "pl_import_model_id": "Operational Amplifier", "pl_pin_order": [1, 0, 2],
    "model_notes": "Finite-gain hard-clamped behavioral amplifier with ideal output; no bandwidth, slew or current limit is inferred.",
})
COMPONENTS["voltage_meter"] = component(25, ["+", "-"], ["r_input"], defaults={"r_input": 1e9})
COMPONENTS["analog_schmitt"].update({
    "pl_import_model_id": "Schmitt Trigger", "pl_pin_order": [0, 1],
    "state_attributes": {"hysteresis_state": 16, "output_v": 17},
    "state_attribute_analysis": ["dc", "op", "tr", "trop"],
    "model_notes": "Analog hysteresis with explicit thresholds and levels. Legacy PhysicsLab slew units are not guessed; imported legacy saves use an explicit instantaneous transition.",
})
COMPONENTS["relay_current_spdt"].update({
    "pl_import_model_id": "Relay Component",
    "state_attributes": {"engaged": 9, "coil_current_a": 10},
    "state_attribute_analysis": ["dc", "op", "tr", "trop"],
    "branch_current_labels": ["COIL+ -> COIL-", "COM -> NC", "COM -> NO"],
    "integer_params": ["initial"], "parameter_ranges": {"initial": [0, 1]},
    "model_notes": "Engineering SPDT relay with a series R-L coil, current hysteresis and finite contact resistances; no bounce, arc or thermal failure model.",
})
COMPONENTS["rated_protection"].update({
    "integer_params": ["initial_broken"], "parameter_ranges": {
        "initial_broken": [0, 1], "trip_temp_c": [-273.15, 1e6],
        "ambient_temp_c": [-273.15, 1e6],
        "rating_reference_temp_c": [-273.15, 1e6],
        "thermal_resistance_k_per_w": [1e-12, 1e12],
        "thermal_capacitance_j_per_k": [1e-12, 1e12]},
    "state_attributes": {"broken": 11, "trip_mask": 12, "current_a": 13,
                         "voltage_v": 14, "power_w": 15, "trip_current_a": 16,
                         "trip_voltage_v": 17, "trip_power_w": 18,
                         "temperature_c": 19, "thermal_energy_j": 20},
    "state_attribute_analysis": ["dc", "op", "tr", "trop", "ac", "acop"],
    "branch_current_labels": ["line_in -> line_out"],
    "model_notes": "Irreversible first-order electrothermal engineering protection. In TR, current and power heat a configurable Rth/Cth state relative to ambient and trip only after accumulated temperature reaches trip_temp_c; temperature cools toward ambient when load falls. Voltage remains an immediate breakdown criterion. DC/OP evaluates the steady-state overload, while AC never mutates damage state. trip_mask bits: 1=current, 2=voltage, 4=power, 8=already broken in the source save. Zero disables a criterion. The model has no spatial heat flow, arc or repair behavior.",
})
COMPONENTS["ne555_timer"].update({
    "integer_params": ["internal_control", "internal_reset_pullup"],
    "parameter_ranges": {"internal_control": [0, 1], "internal_reset_pullup": [0, 1]},
    "state_attributes": {"latched_high": 7, "trigger_threshold_v": 8,
                         "threshold_threshold_v": 9, "output_current_a": 10,
                         "discharge_current_a": 11},
    "state_attribute_analysis": ["dc", "op", "tr", "trop", "ac", "acop"],
    "branch_current_labels": ["OUT -> GND", "DIS -> GND"],
    "model_notes": "Behavioral eight-pin 555: trigger/threshold comparators update a retained SR latch; OUT and DIS use explicit finite resistances. Unwired CTRL uses 2/3 VCC and unwired RESET is pulled high only when the importer marks those pins absent. No propagation delay, output saturation curve, thermal damage or bipolar transistor internals.",
})
COMPONENTS["spark_gap"].update({
    "integer_params": ["initial_conducting"],
    "parameter_ranges": {"initial_conducting": [0, 1]},
    "state_attributes": {"conducting": 5, "current_a": 6,
                         "voltage_v": 7, "power_w": 8},
    "state_attribute_analysis": ["dc", "op", "tr", "trop", "ac", "acop"],
    "branch_current_labels": ["A -> B"],
    "model_notes": "Breakdown closes the arc during nonlinear iteration. Holding-current extinction is evaluated once per transient step to avoid intra-solve chatter. No stochastic ignition, plasma temperature, electrode erosion or RF radiation model.",
})
COMPONENTS["dc_motor"].update({
    "state_attributes": {"angular_velocity_rad_s": 8, "current_a": 9,
                         "voltage_v": 10, "torque_nm": 11,
                         "back_emf_v": 12, "mechanical_power_w": 13},
    "state_attribute_analysis": ["dc", "op", "tr", "trop", "ac", "acop"],
    "branch_current_labels": ["+ -> -"],
    "model_notes": "Lumped permanent-magnet DC motor with armature R/L, torque constant, inertia, load torque, back-EMF and viscous friction. Transient mechanical state uses the previous solved current and an exact first-order friction update; armature inductance uses a trapezoidal companion. AC is locked-rotor electrical small signal.",
})

COMPONENTS["digital_output"].update({
    "extra_params": {
        **COMPONENTS["digital_output"].get("extra_params", {}),
        "setup_time_s": {"native_attribute": "Tsu", "default": 1e-9, "minimum": 0},
        "hold_time_s": {"native_attribute": "Th", "default": 5e-10, "minimum": 0},
    },
    "digital_state_attributes": {"value": 0},
    "digital_state_attribute_analysis": ["dc", "op", "tr", "trop"],
})

# Agent-visible, time-varying controls. These describe physical model
# attributes; they are not synthetic digital ports. Importers may override the
# group kind/role when one PhysicsLab control expands to multiple PE models.
COMPONENTS["switch"].update({
    "interaction": {"kind": "spst", "value_name": "closed",
                    "allowed": [0, 1], "native_attribute": "Cut Through"},
    "state_attributes": {"closed": 0},
    "state_attribute_analysis": ["dc", "op", "tr", "trop"],
    "model_notes": "Controllable analog contact. Native designs default to an ideal closed contact; PhysicsLab imports use the app's finite 1 nOhm closed-contact resistance.",
})
COMPONENTS["resistor"].update({
    "state_attributes": {"resistance_ohm": 0},
    "state_attribute_analysis": ["dc", "op", "tr", "trop"],
})
COMPONENTS["vdc"].update({
    "interaction": {"kind": "voltage_source", "value_name": "voltage_v",
                    "native_attribute": "V", "finite_number": True},
    "state_attributes": {"voltage_v": 0},
    "state_attribute_analysis": ["dc", "op", "tr", "trop"],
})
COMPONENTS["digital_input"].update({
    "interaction": {"kind": "digital_input", "value_name": "state",
                    "allowed": [0, 1, 2, 3], "digital_attribute": 0},
    "digital_state_attributes": {"state": 0},
    "digital_state_attribute_analysis": ["dc", "op", "tr", "trop"],
})

for _kind in ("npn", "pnp"):
    COMPONENTS[_kind].update({
        "model_version": "aurex-ebers-moll-v1",
        "extra_params": {
            "beta_r": {"native_attribute": "BetaR", "default": 1.0, "exclusive_minimum": 0},
            "nr": {"native_attribute": "Nr", "default": 1.0, "exclusive_minimum": 0},
        },
        "pin_current_attributes": [16, 17, 18],
        "pin_current_analysis": ["dc", "op", "tr", "trop"],
        "pin_current_convention": "A, positive into device at base/collector/emitter; DC or transient constitutive current evaluated at solved pin voltages, not an MNA branch.",
        "model_notes": "Quasi-static Ebers-Moll dual-junction BJT, with forward/reverse active, saturation and cutoff. Legacy PE is is the BE base-current scale; transport saturation IS = is * beta, area-scaled. beta_r defaults to 1 and nr independently defaults to 1 (not inherited from n). No junction capacitance/charge storage, Early effect, avalanche breakdown, high-current beta rolloff, parasitic resistances or temperature scaling of is/beta. temp_c affects thermal voltage. PL export maps B/C/E pins 0/1/2, PNP and forward beta; other physical parameters stay in Aurex metadata, with no claim of pointwise equivalence to original PhysicsLab simulation.",
        "pl_mapping_source": "https://github.com/SekaiArendelle/physicslab/blob/v2.0.6/physicsLab/circuit/elements/artificialCircuit.py",
    })

for name, code, model in [("or", 202, "Or"), ("yes", 203, "Yes"), ("and", 204, "And"),
                         ("not", 205, "No"), ("xor", 206, "Xor"), ("xnor", 207, "Xnor"),
                         ("nand", 208, "Nand"), ("nor", 209, "Nor"), ("imp", 211, "Imp"), ("nimp", 212, "Nimp")]:
    COMPONENTS["digital_" + name] = component(code, ["in", "out"] if name in ("not", "yes") else ["a", "b", "out"], [], model_id=model + " Gate")

COMPONENTS.update({
    "digital_tri": component(210, ["in", "enable", "out"], []),
    "digital_half_adder": component(220, ["a", "b", "sum", "carry"], []),
    "digital_full_adder": component(221, ["a", "b", "carry_in", "sum", "carry_out"], []),
    "digital_half_sub": component(222, ["a", "b", "difference", "borrow"], []),
    "digital_full_sub": component(223, ["a", "b", "borrow_in", "difference", "borrow_out"], []),
    "digital_mul2": component(224, ["a0", "a1", "b0", "b1", "p0", "p1", "p2", "p3"], []),
    "digital_dff": component(225, ["d", "clk", "q"], []),
    "digital_tff": component(226, ["t", "clk", "q"], []),
    "digital_t_bar_ff": component(227, ["t_bar", "clk", "q"], []),
    "digital_jkff": component(228, ["j", "k", "clk", "q"], []),
    "digital_counter4": component(229, ["q3", "q2", "q1", "q0", "clk", "enable"], ["initial"], defaults={"initial": 0}),
    "digital_random4": component(230, ["q3", "q2", "q1", "q0", "clk", "reset_n"], ["initial"], defaults={"initial": 1}),
    "digital_input8": component(231, ["b7", "b6", "b5", "b4", "b3", "b2", "b1", "b0"], ["value"], model_id="8bit Input", pl_props={"value": "十进制"}),
    "digital_output8": component(232, ["b7", "b6", "b5", "b4", "b3", "b2", "b1", "b0"], [], model_id="8bit Display"),
})

COMPONENTS["digital_random4"].update({
    "integer_params": ["initial"], "parameter_ranges": {"initial": [0, 15]},
    "state_attributes": {"lfsr_state": 0, "unknown": 1},
    "state_attribute_analysis": ["dc", "op", "tr", "trop"],
    "model_notes": "Deterministic 4-bit PE LFSR surrogate. PhysicsLab saves do not serialize its hidden runtime state; imports use an explicit stable non-zero surrogate seed and preserve saved low/high voltage levels. This supports transition/reset/circuit-response tests but cannot reproduce or certify the original app's exact initial random values or sequence.",
})

# Every native digital primitive has Ll/Hl attributes.  PhysicsLab persists
# those levels on sequential/arithmetic modules too, and real community saves
# connect their outputs to relays, lamps and other analog loads.  Limiting this
# bridge to the basic gates made otherwise supported mixed circuits fail import
# (or tempted callers to silently substitute 0/5 V).  Expose the native
# attributes uniformly so the original saved levels remain authoritative.
for _name, _component in COMPONENTS.items():
    if not _name.startswith("digital_"):
        continue
    _component.setdefault("extra_params", {}).update({
        "low_v": {"native_attribute": "Ll", "default": 0.0},
        "high_v": {"native_attribute": "Hl", "default": 5.0},
    })
    if _component["model_id"]:
        _component["pl_props"].update({"low_v": "低电平", "high_v": "高电平"})
