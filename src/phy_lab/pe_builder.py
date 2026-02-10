from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from pe_sim import ElementCode


class PEBuilderError(RuntimeError):
    pass


_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_NODE_RE = re.compile(r"^[A-Za-z0-9_:+.-]{1,32}$")


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _require_obj(v: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise PEBuilderError(f"{where} must be an object")
    return v


def _require_list(v: Any, *, where: str) -> list[Any]:
    if not isinstance(v, list):
        raise PEBuilderError(f"{where} must be a list")
    return v


def _require_str(v: Any, *, where: str) -> str:
    if not isinstance(v, str):
        raise PEBuilderError(f"{where} must be a string")
    s = v.strip()
    if not s:
        raise PEBuilderError(f"{where} must be a non-empty string")
    return s


def _node_key(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    low = n.casefold()
    if low in ("gnd", "ground", "0"):
        return "gnd"
    return n


@dataclass(frozen=True)
class PEComponent:
    id: str
    type: str
    nodes: tuple[str, ...]
    params: dict[str, float]


@dataclass(frozen=True)
class PEProbe:
    kind: str
    target: str


@dataclass(frozen=True)
class PESimSpec:
    analysis_type: str  # dc|ac|tr
    ac_omega_rad_s: float | None
    tr_t_step_s: float | None
    tr_t_stop_s: float | None
    digital_clk_ticks: int | None
    components: list[PEComponent]
    probes: list[PEProbe]


@dataclass(frozen=True)
class BuiltCircuit:
    element_codes: list[int]
    properties: list[float]
    wires: list[int]
    element_index_by_id: dict[str, int]
    element_index_by_node_pin: dict[tuple[str, int], int]
    node_to_pin: dict[str, tuple[int, int]]  # node -> (element_index, pin)

@dataclass(frozen=True)
class _PropSpec:
    key: str
    synonyms: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class _ModelSpec:
    code: int
    pin_count: int
    props: tuple[_PropSpec, ...] = ()
    counts_as_source: bool = False


def _p(key: str, *synonyms: str, required: bool = True) -> _PropSpec:
    syn = tuple([key] + [s for s in synonyms if isinstance(s, str) and s.strip()])
    return _PropSpec(key=key, synonyms=tuple(s.strip().lower() for s in syn), required=required)


_MODEL_SPECS: dict[str, _ModelSpec] = {
    # Linear sources/passives (2-terminal)
    "resistor": _ModelSpec(
        code=ElementCode.RESISTOR,
        pin_count=2,
        props=(
            _p("r_ohm", "r", "ohm", "resistance", "resistance_ohm", "电阻"),
        ),
    ),
    "capacitor": _ModelSpec(
        code=ElementCode.CAPACITOR,
        pin_count=2,
        props=(
            _p("c_f", "c", "f", "capacitance", "capacitance_f", "电容"),
        ),
    ),
    "inductor": _ModelSpec(
        code=ElementCode.INDUCTOR,
        pin_count=2,
        props=(
            _p("l_h", "l", "h", "inductance", "inductance_h", "电感"),
        ),
    ),
    "vdc": _ModelSpec(
        code=ElementCode.VDC,
        pin_count=2,
        counts_as_source=True,
        props=(
            _p("v_v", "v", "volt", "voltage", "电压"),
        ),
    ),
    "vac": _ModelSpec(
        code=ElementCode.VAC,
        pin_count=2,
        counts_as_source=True,
        props=(
            _p("vp_v", "vp", "v_peak", "vpk"),
            _p("freq_hz", "freq", "freq_hz", "f_hz", "hz"),
            _p("phase_deg", "phase", "phase_deg", "deg"),
        ),
    ),
    "idc": _ModelSpec(
        code=ElementCode.IDC,
        pin_count=2,
        counts_as_source=True,
        props=(
            _p("i_a", "i", "amp", "current", "current_a", "电流"),
        ),
    ),
    "iac": _ModelSpec(
        code=ElementCode.IAC,
        pin_count=2,
        counts_as_source=True,
        props=(
            _p("ip_a", "ip", "i_peak", "ipk"),
            _p("freq_hz", "freq", "freq_hz", "f_hz", "hz"),
            _p("phase_deg", "phase", "phase_deg", "deg"),
        ),
    ),

    # Dependent sources (4-pin): S,T are output; P,Q are control.
    "vccs": _ModelSpec(code=ElementCode.VCCS, pin_count=4, props=(_p("g", "gm", "conductance", "G"),)),
    "vcvs": _ModelSpec(code=ElementCode.VCVS, pin_count=4, props=(_p("mu", "gain"),)),
    "cccs": _ModelSpec(code=ElementCode.CCCS, pin_count=4, props=(_p("alpha", "a"),)),
    "ccvs": _ModelSpec(code=ElementCode.CCVS, pin_count=4, props=(_p("r_ohm", "r", "ohm"),)),

    # Controllers / switches
    "switch_spst": _ModelSpec(code=ElementCode.SWITCH_SPST, pin_count=2, props=(_p("cut_through", "on", "closed", "state"),)),
    "relay": _ModelSpec(code=ElementCode.RELAY, pin_count=4, props=(_p("von_v", "von"), _p("voff_v", "voff"))),
    "comparator": _ModelSpec(code=ElementCode.COMPARATOR, pin_count=3, props=(_p("ll_v", "ll"), _p("hl_v", "hl"))),

    # Non-linear
    "pn_junction": _ModelSpec(
        code=ElementCode.PN_JUNCTION,
        pin_count=2,
        props=(
            _p("is_a", "is"),
            _p("n", "nf"),
            _p("isr_a", "isr"),
            _p("nr",),
            _p("temp_c", "temp"),
            _p("ibv_a", "ibv"),
            _p("bv_v", "bv"),
            _p("bv_set", "bv_set", "bvset"),
            _p("area",),
        ),
    ),
    "bjt_npn": _ModelSpec(
        code=ElementCode.BJT_NPN,
        pin_count=3,
        props=(
            _p("is_a", "is"),
            _p("n",),
            _p("betaf", "beta_f", "beta"),
            _p("temp_c", "temp"),
            _p("area",),
        ),
    ),
    "bjt_pnp": _ModelSpec(
        code=ElementCode.BJT_PNP,
        pin_count=3,
        props=(
            _p("is_a", "is"),
            _p("n",),
            _p("betaf", "beta_f", "beta"),
            _p("temp_c", "temp"),
            _p("area",),
        ),
    ),
    "nmosfet": _ModelSpec(code=ElementCode.NMOSFET, pin_count=3, props=(_p("kp",), _p("lambda", "lam"), _p("vth_v", "vth"))),
    "pmosfet": _ModelSpec(code=ElementCode.PMOSFET, pin_count=3, props=(_p("kp",), _p("lambda", "lam"), _p("vth_v", "vth"))),
    "full_bridge_rectifier": _ModelSpec(code=ElementCode.FULL_BRIDGE_RECTIFIER, pin_count=4),
    "bsim3v32_nmos": _ModelSpec(
        code=ElementCode.BSIM3V32_NMOS,
        pin_count=4,
        props=(
            _p("w_m", "w"),
            _p("l_m", "l"),
            _p("kp",),
            _p("lambda", "lam"),
            _p("vth0_v", "vth0"),
            _p("gamma",),
            _p("phi_v", "phi"),
            _p("cgs_f", "cgs"),
            _p("cgd_f", "cgd"),
            _p("cgb_f", "cgb"),
            _p("diode_is_a", "diode_is"),
            _p("diode_n", "diode_n"),
            _p("temp_c", "temp"),
        ),
    ),
    "bsim3v32_pmos": _ModelSpec(
        code=ElementCode.BSIM3V32_PMOS,
        pin_count=4,
        props=(
            _p("w_m", "w"),
            _p("l_m", "l"),
            _p("kp",),
            _p("lambda", "lam"),
            _p("vth0_v", "vth0"),
            _p("gamma",),
            _p("phi_v", "phi"),
            _p("cgs_f", "cgs"),
            _p("cgd_f", "cgd"),
            _p("cgb_f", "cgb"),
            _p("diode_is_a", "diode_is"),
            _p("diode_n", "diode_n"),
            _p("temp_c", "temp"),
        ),
    ),

    # Coupled devices
    "transformer": _ModelSpec(code=ElementCode.TRANSFORMER, pin_count=4, props=(_p("n",),)),
    "coupled_inductors": _ModelSpec(code=ElementCode.COUPLED_INDUCTORS, pin_count=4, props=(_p("l1_h", "l1"), _p("l2_h", "l2"), _p("k",))),
    "transformer_center_tap": _ModelSpec(code=ElementCode.TRANSFORMER_CENTER_TAP, pin_count=5, props=(_p("n_total",),)),
    "op_amp": _ModelSpec(code=ElementCode.OP_AMP, pin_count=4, props=(_p("mu", "gain"),)),

    # Waveform generators
    "sawtooth": _ModelSpec(code=ElementCode.SAWTOOTH, pin_count=2, counts_as_source=True, props=(_p("vh_v", "vh"), _p("vl_v", "vl"), _p("freq_hz", "freq"), _p("phase_rad", "phase"))),
    "square": _ModelSpec(code=ElementCode.SQUARE, pin_count=2, counts_as_source=True, props=(_p("vh_v", "vh"), _p("vl_v", "vl"), _p("freq_hz", "freq"), _p("duty",), _p("phase_rad", "phase"))),
    "pulse": _ModelSpec(code=ElementCode.PULSE, pin_count=2, counts_as_source=True, props=(_p("vh_v", "vh"), _p("vl_v", "vl"), _p("freq_hz", "freq"), _p("duty",), _p("phase_rad", "phase"), _p("tr_s", "tr"), _p("tf_s", "tf"))),
    "triangle": _ModelSpec(code=ElementCode.TRIANGLE, pin_count=2, counts_as_source=True, props=(_p("vh_v", "vh"), _p("vl_v", "vl"), _p("freq_hz", "freq"), _p("phase_rad", "phase"))),

    # Digital (logic / blocks)
    "digital_input": _ModelSpec(code=ElementCode.DIGITAL_INPUT, pin_count=1, counts_as_source=True, props=(_p("state",),)),
    "digital_output": _ModelSpec(code=ElementCode.DIGITAL_OUTPUT, pin_count=1),
    "digital_or": _ModelSpec(code=ElementCode.DIGITAL_OR, pin_count=3),
    "digital_yes": _ModelSpec(code=ElementCode.DIGITAL_YES, pin_count=2),
    "digital_and": _ModelSpec(code=ElementCode.DIGITAL_AND, pin_count=3),
    "digital_not": _ModelSpec(code=ElementCode.DIGITAL_NOT, pin_count=2),
    "digital_xor": _ModelSpec(code=ElementCode.DIGITAL_XOR, pin_count=3),
    "digital_xnor": _ModelSpec(code=ElementCode.DIGITAL_XNOR, pin_count=3),
    "digital_nand": _ModelSpec(code=ElementCode.DIGITAL_NAND, pin_count=3),
    "digital_nor": _ModelSpec(code=ElementCode.DIGITAL_NOR, pin_count=3),
    "digital_tri": _ModelSpec(code=ElementCode.DIGITAL_TRI, pin_count=3),
    "digital_imp": _ModelSpec(code=ElementCode.DIGITAL_IMP, pin_count=3),
    "digital_nimp": _ModelSpec(code=ElementCode.DIGITAL_NIMP, pin_count=3),

    "digital_half_adder": _ModelSpec(code=ElementCode.DIGITAL_HALF_ADDER, pin_count=4),
    "digital_full_adder": _ModelSpec(code=ElementCode.DIGITAL_FULL_ADDER, pin_count=5),
    "digital_half_subtractor": _ModelSpec(code=ElementCode.DIGITAL_HALF_SUBTRACTOR, pin_count=4),
    "digital_full_subtractor": _ModelSpec(code=ElementCode.DIGITAL_FULL_SUBTRACTOR, pin_count=5),
    "digital_mul2": _ModelSpec(code=ElementCode.DIGITAL_MUL2, pin_count=8),
    "digital_dff": _ModelSpec(code=ElementCode.DIGITAL_DFF, pin_count=3),
    "digital_tff": _ModelSpec(code=ElementCode.DIGITAL_TFF, pin_count=3),
    "digital_t_bar_ff": _ModelSpec(code=ElementCode.DIGITAL_T_BAR_FF, pin_count=3),
    "digital_jkff": _ModelSpec(code=ElementCode.DIGITAL_JKFF, pin_count=4),

    "digital_counter4": _ModelSpec(code=ElementCode.DIGITAL_COUNTER4, pin_count=6, props=(_p("init_value", "value"),)),
    "digital_random_generator4": _ModelSpec(code=ElementCode.DIGITAL_RANDOM_GENERATOR4, pin_count=6, props=(_p("init_state", "state"),)),
    "digital_eight_bit_input": _ModelSpec(code=ElementCode.DIGITAL_EIGHT_BIT_INPUT, pin_count=8, counts_as_source=True, props=(_p("value",),)),
    "digital_eight_bit_display": _ModelSpec(code=ElementCode.DIGITAL_EIGHT_BIT_DISPLAY, pin_count=8),
    "digital_schmitt_trigger": _ModelSpec(
        code=ElementCode.DIGITAL_SCHMITT_TRIGGER,
        pin_count=2,
        props=(
            _p("vth_low_v", "vth_low", "low"),
            _p("vth_high_v", "vth_high", "high"),
            _p("inverted",),
            _p("ll_v", "ll"),
            _p("hl_v", "hl"),
        ),
    ),
}

def _canonical_component_type(t: str) -> str:
    s = (t or "").strip()
    if not s:
        return ""
    low = s.casefold()
    # Normalize common separators so both "digital_input" and "digital input" work.
    low = re.sub(r"[\s_-]+", "_", low).strip("_")

    # Common English aliases
    if low in ("r", "res", "resistor", "resistance", "ohm"):
        return "resistor"
    if low in ("c", "cap", "capacitor", "capacitance"):
        return "capacitor"
    if low in ("l", "ind", "inductor", "inductance"):
        return "inductor"
    if low in ("vdc", "dc", "dc_source", "voltage_source", "battery", "cell"):
        return "vdc"
    if low in ("idc", "dc_current", "current_source"):
        return "idc"
    if low in ("vac", "ac", "ac_source"):
        return "vac"
    if low in ("iac", "ac_current"):
        return "iac"

    # Dependent sources
    if low in ("vccs",):
        return "vccs"
    if low in ("vcvs",):
        return "vcvs"
    if low in ("cccs",):
        return "cccs"
    if low in ("ccvs",):
        return "ccvs"

    # Common controller/nonlinear aliases
    if low in ("switch", "spst", "switch_spst", "single_pole_switch"):
        return "switch_spst"
    if low in ("pn", "diode", "pn_junction"):
        return "pn_junction"
    if low in ("opamp", "op_amp"):
        return "op_amp"
    if low in ("relay",):
        return "relay"
    if low in ("comparator", "cmp"):
        return "comparator"
    if low in ("transformer", "tx"):
        return "transformer"
    if low in ("coupled_inductors", "k"):
        return "coupled_inductors"
    if low in ("transformer_center_tap", "txct", "center_tap_transformer"):
        return "transformer_center_tap"
    if low in ("sawtooth", "saw"):
        return "sawtooth"
    if low in ("square", "sq", "square_gen", "square_wave"):
        return "square"
    if low in ("pulse", "pulse_gen"):
        return "pulse"
    if low in ("triangle", "tri", "triangle_gen"):
        return "triangle"
    if low in ("bjt_npn", "npn"):
        return "bjt_npn"
    if low in ("bjt_pnp", "pnp"):
        return "bjt_pnp"
    if low in ("nmos", "nmosfet", "nmos_fet"):
        return "nmosfet"
    if low in ("pmos", "pmosfet", "pmos_fet"):
        return "pmosfet"
    if low in ("full_bridge_rectifier", "bridge_rectifier", "fbr"):
        return "full_bridge_rectifier"
    if low in ("bsim3v32_nmos", "bsim3_nmos"):
        return "bsim3v32_nmos"
    if low in ("bsim3v32_pmos", "bsim3_pmos"):
        return "bsim3v32_pmos"

    # Digital (prefix or direct names)
    if low in ("digital_input", "d_input", "input"):
        return "digital_input"
    if low in ("digital_output", "d_output", "output"):
        return "digital_output"
    if low in ("digital_and", "d_and", "and"):
        return "digital_and"
    if low in ("digital_or", "d_or", "or"):
        return "digital_or"
    if low in ("digital_not", "d_not", "not"):
        return "digital_not"
    if low in ("digital_xor", "d_xor", "xor"):
        return "digital_xor"
    if low in ("digital_xnor", "d_xnor", "xnor"):
        return "digital_xnor"
    if low in ("digital_nand", "d_nand", "nand"):
        return "digital_nand"
    if low in ("digital_nor", "d_nor", "nor"):
        return "digital_nor"
    if low in ("digital_yes", "d_yes", "yes", "buffer", "buf"):
        return "digital_yes"
    if low in ("digital_tri", "d_tri", "tri_state", "tristate"):
        return "digital_tri"
    if low in ("digital_imp", "d_imp", "imp"):
        return "digital_imp"
    if low in ("digital_nimp", "d_nimp", "nimp"):
        return "digital_nimp"
    if low in ("digital_half_adder", "half_adder"):
        return "digital_half_adder"
    if low in ("digital_full_adder", "full_adder"):
        return "digital_full_adder"
    if low in ("digital_half_subtractor", "half_sub", "half_subtractor"):
        return "digital_half_subtractor"
    if low in ("digital_full_subtractor", "full_sub", "full_subtractor"):
        return "digital_full_subtractor"
    if low in ("digital_mul2", "mul2"):
        return "digital_mul2"
    if low in ("digital_dff", "dff"):
        return "digital_dff"
    if low in ("digital_tff", "tff"):
        return "digital_tff"
    if low in ("digital_t_bar_ff", "t_bar_ff", "tbarff"):
        return "digital_t_bar_ff"
    if low in ("digital_jkff", "jkff"):
        return "digital_jkff"
    if low in ("digital_counter4", "counter4"):
        return "digital_counter4"
    if low in ("digital_random_generator4", "random_generator4", "rng4"):
        return "digital_random_generator4"
    if low in ("digital_eight_bit_input", "eight_bit_input", "8bit_input"):
        return "digital_eight_bit_input"
    if low in ("digital_eight_bit_display", "eight_bit_display", "8bit_display"):
        return "digital_eight_bit_display"
    if low in ("digital_schmitt_trigger", "schmitt_trigger"):
        return "digital_schmitt_trigger"

    # Common CJK aliases (Physics Lab UI terms)
    if low in ("电阻", "电阻器"):
        return "resistor"
    if low in ("电容", "电容器"):
        return "capacitor"
    if low in ("电感", "电感器"):
        return "inductor"
    if low in ("电源", "直流电源", "电压源", "学生电源", "电池"):
        return "vdc"
    if low in ("电流源", "直流电流源"):
        return "idc"
    if low in ("交流电源", "交流电压源"):
        return "vac"
    if low in ("交流电流源",):
        return "iac"

    # Model hallucinations we want to accept gracefully.
    if low in ("student_source", "student_power", "student_powersource"):
        return "vdc"
    return low


def parse_pe_sim_spec(
    obj: Any,
    *,
    max_components: int,
    max_probes: int,
) -> PESimSpec:
    root = _require_obj(obj, where="spec")
    analysis = _require_obj(root.get("analysis", {}), where="spec.analysis")
    a_type = str(analysis.get("type") or "dc").strip().lower()
    if a_type not in ("dc", "ac", "tr"):
        raise PEBuilderError("spec.analysis.type must be one of: dc, ac, tr")

    ac_omega = analysis.get("ac_omega_rad_s")
    tr_step = analysis.get("tr_t_step_s")
    tr_stop = analysis.get("tr_t_stop_s")
    digital_ticks = analysis.get("digital_clk_ticks")
    if ac_omega is not None and not _is_finite_number(ac_omega):
        raise PEBuilderError("spec.analysis.ac_omega_rad_s must be a number")
    if tr_step is not None and not _is_finite_number(tr_step):
        raise PEBuilderError("spec.analysis.tr_t_step_s must be a number")
    if tr_stop is not None and not _is_finite_number(tr_stop):
        raise PEBuilderError("spec.analysis.tr_t_stop_s must be a number")
    if digital_ticks is not None and not isinstance(digital_ticks, int):
        raise PEBuilderError("spec.analysis.digital_clk_ticks must be an int")

    comps_raw = root.get("components")
    # Best-effort normalization: some LLMs emit a single component object instead of a list.
    if isinstance(comps_raw, dict):
        comps_raw = [comps_raw]
    comps_raw = _require_list(comps_raw, where="spec.components")
    if max_components > 0 and len(comps_raw) > max_components:
        raise PEBuilderError(
            f"Too many components (count={len(comps_raw)}, limit={max_components})"
        )

    components: list[PEComponent] = []
    seen_ids: set[str] = set()
    for i, c in enumerate(comps_raw):
        cobj = _require_obj(c, where=f"spec.components[{i}]")
        cid = _require_str(cobj.get("id"), where=f"spec.components[{i}].id")
        if not _ID_RE.match(cid):
            raise PEBuilderError(
                f"spec.components[{i}].id must match {_ID_RE.pattern} (got {cid!r})"
            )
        if cid in seen_ids:
            raise PEBuilderError(f"Duplicate component id: {cid}")
        seen_ids.add(cid)

        ctype_raw = _require_str(cobj.get("type"), where=f"spec.components[{i}].type")
        ctype = _canonical_component_type(ctype_raw)
        ms = _MODEL_SPECS.get(ctype)
        if ms is None:
            raise PEBuilderError(
                f"Unsupported component type: {ctype}. Allowed: {sorted(_MODEL_SPECS.keys())}"
            )

        nodes_raw = _require_list(cobj.get("nodes"), where=f"spec.components[{i}].nodes")
        if len(nodes_raw) != int(ms.pin_count):
            raise PEBuilderError(
                f"spec.components[{i}].nodes must have length {int(ms.pin_count)} for type {ctype}"
            )
        nodes: list[str] = []
        for j, nv in enumerate(nodes_raw):
            n = _node_key(_require_str(nv, where=f"spec.components[{i}].nodes[{j}]"))
            if n != "gnd" and not _NODE_RE.match(n):
                raise PEBuilderError(
                    f"Invalid node name {n!r} in spec.components[{i}].nodes[{j}] (pattern={_NODE_RE.pattern})"
                )
            nodes.append(n)

        params_obj = _require_obj(cobj.get("params", {}), where=f"spec.components[{i}].params")
        raw_params: dict[str, float] = {}
        for k, v in params_obj.items():
            if isinstance(k, str) and _is_finite_number(v):
                raw_params[k.strip().lower()] = float(v)

        params: dict[str, float] = {}
        if ms.props:
            for ps in ms.props:
                val = None
                for syn in ps.synonyms:
                    if syn in raw_params:
                        val = float(raw_params[syn])
                        break
                if val is None:
                    if ps.required:
                        raise PEBuilderError(
                            f"spec.components[{i}] missing required param {ps.key} for {ctype}"
                        )
                    continue

                # Targeted validations for discrete/bounded fields.
                if ctype == "digital_input" and ps.key == "state":
                    iv = int(val)
                    if iv < 0 or iv > 3:
                        raise PEBuilderError("digital_input.state must be in [0,3]")
                    val = float(iv)
                if ctype == "digital_counter4" and ps.key == "init_value":
                    iv = int(val)
                    if iv < 0 or iv > 15:
                        raise PEBuilderError("digital_counter4.init_value must be in [0,15]")
                    val = float(iv)
                if ctype == "digital_random_generator4" and ps.key == "init_state":
                    iv = int(val)
                    if iv < 0 or iv > 15:
                        raise PEBuilderError("digital_random_generator4.init_state must be in [0,15]")
                    val = float(iv)
                if ctype == "digital_eight_bit_input" and ps.key == "value":
                    iv = int(val)
                    if iv < 0 or iv > 255:
                        raise PEBuilderError("digital_eight_bit_input.value must be in [0,255]")
                    val = float(iv)

                params[ps.key] = float(val)
        else:
            # No positional properties for this element code.
            if raw_params:
                raise PEBuilderError(f"spec.components[{i}].params must be empty for type {ctype}")

        components.append(
            PEComponent(id=cid, type=ctype, nodes=tuple(nodes), params=params)
        )

    probes_raw = root.get("probes", [])
    if probes_raw is None:
        probes_raw = []
    if isinstance(probes_raw, dict):
        probes_raw = [probes_raw]
    probes_list = _require_list(probes_raw, where="spec.probes")
    if max_probes > 0 and len(probes_list) > max_probes:
        raise PEBuilderError(f"Too many probes (count={len(probes_list)}, limit={max_probes})")
    probes: list[PEProbe] = []
    for i, p in enumerate(probes_list):
        pobj = _require_obj(p, where=f"spec.probes[{i}]")
        kind = _require_str(pobj.get("kind"), where=f"spec.probes[{i}].kind").lower()
        target = _require_str(pobj.get("target"), where=f"spec.probes[{i}].target")
        if kind not in ("node_voltage", "node_digital", "component_current", "component_vdrop"):
            raise PEBuilderError(
                "spec.probes[].kind must be one of: node_voltage, node_digital, component_current, component_vdrop"
            )
        probes.append(PEProbe(kind=kind, target=target))

    if not components:
        raise PEBuilderError("spec.components must not be empty")

    has_non_digital = any((_MODEL_SPECS.get(c.type) or _ModelSpec(0, 0)).code < 200 for c in components)
    uses_ground = any("gnd" in c.nodes for c in components)
    if has_non_digital and (not uses_ground):
        raise PEBuilderError(
            "Circuit must include a ground node named 'gnd' connected to at least one component"
        )

    has_source = any((_MODEL_SPECS[c.type].counts_as_source) for c in components if c.type in _MODEL_SPECS)
    if not has_source:
        raise PEBuilderError("Circuit must include at least one source/input element")
    return PESimSpec(
        analysis_type=a_type,
        ac_omega_rad_s=float(ac_omega) if ac_omega is not None else None,
        tr_t_step_s=float(tr_step) if tr_step is not None else None,
        tr_t_stop_s=float(tr_stop) if tr_stop is not None else None,
        digital_clk_ticks=int(digital_ticks) if digital_ticks is not None else None,
        components=components,
        probes=probes,
    )


def build_circuit(spec: PESimSpec) -> BuiltCircuit:
    # element index 0 is reserved for a "ground placeholder" as required by Phy-Engine create_circuit().
    element_codes: list[int] = [0]
    properties: list[float] = []
    element_index_by_id: dict[str, int] = {}
    pins_by_node: dict[str, list[tuple[int, int]]] = {"gnd": [(0, 0)]}
    node_to_pin: dict[str, tuple[int, int]] = {"gnd": (0, 0)}

    for comp in spec.components:
        idx = len(element_codes)
        element_index_by_id[comp.id] = idx

        ctype = comp.type.lower()
        ms = _MODEL_SPECS.get(ctype)
        if ms is None:
            raise PEBuilderError(f"Unsupported component type: {ctype}")
        element_codes.append(int(ms.code))

        # Properties stream is positional per element code.
        for ps in ms.props:
            v = comp.params.get(ps.key)
            if v is None:
                raise PEBuilderError(f"Component {comp.id} missing required param {ps.key}")
            properties.append(float(v))

        for pin_idx, node in enumerate(comp.nodes):
            pins_by_node.setdefault(node, []).append((idx, int(pin_idx)))
            node_to_pin.setdefault(node, (idx, int(pin_idx)))

    wires: list[int] = []
    for node, pins in pins_by_node.items():
        if len(pins) <= 1:
            continue
        base = pins[0]
        for p in pins[1:]:
            wires.extend([base[0], base[1], p[0], p[1]])

    element_index_by_node_pin: dict[tuple[str, int], int] = {}
    for node, pins in pins_by_node.items():
        for (ei, pin) in pins:
            element_index_by_node_pin[(node, pin)] = ei

    return BuiltCircuit(
        element_codes=element_codes,
        properties=properties,
        wires=wires,
        element_index_by_id=element_index_by_id,
        element_index_by_node_pin=element_index_by_node_pin,
        node_to_pin=node_to_pin,
    )


def parse_spec_json(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise PEBuilderError(f"Invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise PEBuilderError("Spec JSON must be an object")
    return obj
