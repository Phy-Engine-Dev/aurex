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
    nodes: tuple[str, str]
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


_TYPE_TO_CODE: dict[str, int] = {
    "resistor": ElementCode.RESISTOR,
    "r": ElementCode.RESISTOR,
    "capacitor": ElementCode.CAPACITOR,
    "c": ElementCode.CAPACITOR,
    "inductor": ElementCode.INDUCTOR,
    "l": ElementCode.INDUCTOR,
    "vdc": ElementCode.VDC,
    "idc": ElementCode.IDC,
    "vac": ElementCode.VAC,
    "iac": ElementCode.IAC,
}


_PARAM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "resistor": ("r", "r_ohm", "resistance", "resistance_ohm", "ohm"),
    "capacitor": ("c", "c_f", "capacitance", "capacitance_f", "f"),
    "inductor": ("l", "l_h", "inductance", "inductance_h", "h"),
    "vdc": ("v", "v_v", "voltage", "voltage_v"),
    "idc": ("i", "i_a", "current", "current_a"),
    # VAC/IAC: positional properties: Vp, freq(Hz), phase(deg)
    "vac_vp": ("vp", "vp_v", "v_peak", "vpk"),
    "vac_freq": ("freq", "freq_hz", "f_hz", "hz"),
    "vac_phase": ("phase", "phase_deg", "deg"),
    "iac_ip": ("ip", "ip_a", "i_peak", "ipk"),
    "iac_freq": ("freq", "freq_hz", "f_hz", "hz"),
    "iac_phase": ("phase", "phase_deg", "deg"),
}


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
    if ac_omega is not None and not _is_finite_number(ac_omega):
        raise PEBuilderError("spec.analysis.ac_omega_rad_s must be a number")
    if tr_step is not None and not _is_finite_number(tr_step):
        raise PEBuilderError("spec.analysis.tr_t_step_s must be a number")
    if tr_stop is not None and not _is_finite_number(tr_stop):
        raise PEBuilderError("spec.analysis.tr_t_stop_s must be a number")

    comps_raw = _require_list(root.get("components"), where="spec.components")
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

        ctype = _require_str(cobj.get("type"), where=f"spec.components[{i}].type").lower()
        if ctype not in _TYPE_TO_CODE:
            raise PEBuilderError(
                f"Unsupported component type: {ctype}. Allowed: {sorted(set(_TYPE_TO_CODE.keys()))}"
            )

        nodes_raw = _require_list(cobj.get("nodes"), where=f"spec.components[{i}].nodes")
        if len(nodes_raw) != 2:
            raise PEBuilderError(f"spec.components[{i}].nodes must have length 2")
        n0 = _node_key(_require_str(nodes_raw[0], where=f"spec.components[{i}].nodes[0]"))
        n1 = _node_key(_require_str(nodes_raw[1], where=f"spec.components[{i}].nodes[1]"))
        for n in (n0, n1):
            if n != "gnd" and not _NODE_RE.match(n):
                raise PEBuilderError(
                    f"Invalid node name {n!r} in spec.components[{i}].nodes (pattern={_NODE_RE.pattern})"
                )

        params_obj = _require_obj(cobj.get("params", {}), where=f"spec.components[{i}].params")
        params: dict[str, float] = {}
        for k, v in params_obj.items():
            if isinstance(k, str) and _is_finite_number(v):
                params[k.strip().lower()] = float(v)

        # Validate required params per type.
        base_type = ctype
        if base_type in ("r",):
            base_type = "resistor"
        if base_type in ("c",):
            base_type = "capacitor"
        if base_type in ("l",):
            base_type = "inductor"

        if base_type in ("resistor", "capacitor", "inductor", "vdc", "idc"):
            required = _PARAM_SYNONYMS[base_type]
            if not any(k in params for k in required):
                raise PEBuilderError(
                    f"spec.components[{i}] missing required param for {base_type} (one of {list(required)})"
                )
        elif base_type == "vac":
            for req_key in ("vac_vp", "vac_freq", "vac_phase"):
                if not any(k in params for k in _PARAM_SYNONYMS[req_key]):
                    raise PEBuilderError(
                        f"spec.components[{i}] missing required VAC param (need {req_key})"
                    )
        elif base_type == "iac":
            for req_key in ("iac_ip", "iac_freq", "iac_phase"):
                if not any(k in params for k in _PARAM_SYNONYMS[req_key]):
                    raise PEBuilderError(
                        f"spec.components[{i}] missing required IAC param (need {req_key})"
                    )

        components.append(
            PEComponent(id=cid, type=ctype, nodes=(n0, n1), params=params)
        )

    probes_raw = root.get("probes", [])
    if probes_raw is None:
        probes_raw = []
    probes_list = _require_list(probes_raw, where="spec.probes")
    if max_probes > 0 and len(probes_list) > max_probes:
        raise PEBuilderError(f"Too many probes (count={len(probes_list)}, limit={max_probes})")
    probes: list[PEProbe] = []
    for i, p in enumerate(probes_list):
        pobj = _require_obj(p, where=f"spec.probes[{i}]")
        kind = _require_str(pobj.get("kind"), where=f"spec.probes[{i}].kind").lower()
        target = _require_str(pobj.get("target"), where=f"spec.probes[{i}].target")
        if kind not in ("node_voltage", "component_current", "component_vdrop"):
            raise PEBuilderError(
                "spec.probes[].kind must be one of: node_voltage, component_current, component_vdrop"
            )
        probes.append(PEProbe(kind=kind, target=target))

    if not components:
        raise PEBuilderError("spec.components must not be empty")

    uses_ground = any("gnd" in c.nodes for c in components)
    if not uses_ground:
        raise PEBuilderError("Circuit must include a ground node named 'gnd' connected to at least one component")

    has_source = any(
        c.type.lower() in ("vdc", "idc", "vac", "iac") for c in components
    )
    if not has_source:
        raise PEBuilderError("Circuit must include at least one source (vdc/idc/vac/iac)")
    return PESimSpec(
        analysis_type=a_type,
        ac_omega_rad_s=float(ac_omega) if ac_omega is not None else None,
        tr_t_step_s=float(tr_step) if tr_step is not None else None,
        tr_t_stop_s=float(tr_stop) if tr_stop is not None else None,
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

    def get_param(params: dict[str, float], names: tuple[str, ...]) -> float | None:
        for k in names:
            kk = k.strip().lower()
            if kk in params:
                return float(params[kk])
        return None

    for comp in spec.components:
        idx = len(element_codes)
        element_index_by_id[comp.id] = idx

        ctype = comp.type.lower()
        code = _TYPE_TO_CODE.get(ctype)
        if code is None:
            raise PEBuilderError(f"Unsupported component type: {ctype}")
        element_codes.append(int(code))

        # Properties stream is positional.
        base_type = ctype
        if base_type == "r":
            base_type = "resistor"
        if base_type == "c":
            base_type = "capacitor"
        if base_type == "l":
            base_type = "inductor"

        if base_type in ("resistor", "capacitor", "inductor", "vdc", "idc"):
            v = get_param(comp.params, _PARAM_SYNONYMS[base_type])
            if v is None:
                raise PEBuilderError(f"Component {comp.id} missing required param")
            properties.append(float(v))
        elif base_type == "vac":
            vp = get_param(comp.params, _PARAM_SYNONYMS["vac_vp"])
            freq = get_param(comp.params, _PARAM_SYNONYMS["vac_freq"])
            phase = get_param(comp.params, _PARAM_SYNONYMS["vac_phase"])
            if vp is None or freq is None or phase is None:
                raise PEBuilderError(f"Component {comp.id} missing required VAC params")
            properties.extend([float(vp), float(freq), float(phase)])
        elif base_type == "iac":
            ip = get_param(comp.params, _PARAM_SYNONYMS["iac_ip"])
            freq = get_param(comp.params, _PARAM_SYNONYMS["iac_freq"])
            phase = get_param(comp.params, _PARAM_SYNONYMS["iac_phase"])
            if ip is None or freq is None or phase is None:
                raise PEBuilderError(f"Component {comp.id} missing required IAC params")
            properties.extend([float(ip), float(freq), float(phase)])

        n0, n1 = comp.nodes
        pins_by_node.setdefault(n0, []).append((idx, 0))
        pins_by_node.setdefault(n1, []).append((idx, 1))
        node_to_pin.setdefault(n0, (idx, 0))
        node_to_pin.setdefault(n1, (idx, 1))

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
