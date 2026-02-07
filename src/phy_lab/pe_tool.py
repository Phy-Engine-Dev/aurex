from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pe_builder import BuiltCircuit, PESimSpec
from pe_sim import AnalyzeType, PhyEngineLib


class PEToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class PESample:
    voltage: list[float]
    voltage_ord: list[int]
    current: list[float]
    current_ord: list[int]
    comp_size: int


def run_and_sample(
    *,
    lib_path: str,
    spec: PESimSpec,
    built: BuiltCircuit,
    max_pins_per_comp: int = 16,
    max_branches_per_comp: int = 8,
) -> PESample:
    pe = PhyEngineLib(lib_path)
    circuit, vec_pos, chunk_pos, comp_size = pe.create_circuit(
        element_codes=built.element_codes,
        wires=built.wires,
        properties=built.properties,
    )
    try:
        if spec.analysis_type == "dc":
            pe.set_analyze_type(circuit=circuit, analyze_type=AnalyzeType.DC)
        elif spec.analysis_type == "ac":
            pe.set_analyze_type(circuit=circuit, analyze_type=AnalyzeType.AC)
            omega = float(spec.ac_omega_rad_s or 0.0)
            if omega <= 0.0:
                omega = 2.0 * 3.141592653589793 * 1000.0
            pe.set_ac_omega(circuit=circuit, omega=omega)
        else:
            pe.set_analyze_type(circuit=circuit, analyze_type=AnalyzeType.TR)
            t_step = float(spec.tr_t_step_s or 1e-4)
            t_stop = float(spec.tr_t_stop_s or 5e-3)
            if t_step <= 0.0:
                t_step = 1e-4
            if t_stop <= 0.0:
                t_stop = 5e-3
            if t_step > t_stop:
                t_step = t_stop
            pe.set_tr(circuit=circuit, t_step=t_step, t_stop=t_stop)

        pe.analyze(circuit=circuit)
        voltage, voltage_ord, current, current_ord, _dig, _dig_ord = pe.sample(
            circuit=circuit,
            vec_pos=vec_pos,
            chunk_pos=chunk_pos,
            comp_size=comp_size,
            max_pins_per_comp=max_pins_per_comp,
            max_branches_per_comp=max_branches_per_comp,
        )
        return PESample(
            voltage=voltage,
            voltage_ord=voltage_ord,
            current=current,
            current_ord=current_ord,
            comp_size=comp_size,
        )
    finally:
        pe.destroy_circuit(circuit=circuit, vec_pos=vec_pos, chunk_pos=chunk_pos)


def _comp_index_for_element_index(element_index: int) -> int | None:
    if element_index <= 0:
        return None
    return element_index - 1


def node_voltage(*, built: BuiltCircuit, sample: PESample, node: str) -> float | None:
    n = (node or "").strip()
    if not n:
        return None
    if n.casefold() in ("gnd", "ground", "0"):
        return 0.0
    ref = built.node_to_pin.get(n) or built.node_to_pin.get(n.casefold())
    if not ref:
        return None
    ei, pin = ref
    ci = _comp_index_for_element_index(int(ei))
    if ci is None or ci + 1 >= len(sample.voltage_ord):
        return None
    start = int(sample.voltage_ord[ci])
    end = int(sample.voltage_ord[ci + 1])
    if (end - start) <= int(pin):
        return None
    return float(sample.voltage[start + int(pin)])


def component_current(*, built: BuiltCircuit, sample: PESample, cid: str) -> float | None:
    ei = built.element_index_by_id.get(cid)
    if ei is None:
        return None
    ci = _comp_index_for_element_index(int(ei))
    if ci is None or ci + 1 >= len(sample.current_ord):
        return None
    start = int(sample.current_ord[ci])
    end = int(sample.current_ord[ci + 1])
    if end <= start:
        return None
    return float(sample.current[start])


def component_vdrop(*, built: BuiltCircuit, sample: PESample, cid: str) -> float | None:
    ei = built.element_index_by_id.get(cid)
    if ei is None:
        return None
    ci = _comp_index_for_element_index(int(ei))
    if ci is None or ci + 1 >= len(sample.voltage_ord):
        return None
    start = int(sample.voltage_ord[ci])
    end = int(sample.voltage_ord[ci + 1])
    if (end - start) < 2:
        return None
    v0 = float(sample.voltage[start + 0])
    v1 = float(sample.voltage[start + 1])
    return v0 - v1


def evaluate_probes(
    *,
    built: BuiltCircuit,
    sample: PESample,
    probes: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in probes:
        kind = str(p.get("kind") or "").strip().lower()
        target = str(p.get("target") or "").strip()
        if kind == "node_voltage":
            out.append({"kind": kind, "target": target, "value": node_voltage(built=built, sample=sample, node=target)})
        elif kind == "component_current":
            out.append(
                {
                    "kind": kind,
                    "target": target,
                    "value": component_current(built=built, sample=sample, cid=target),
                }
            )
        elif kind == "component_vdrop":
            out.append(
                {
                    "kind": kind,
                    "target": target,
                    "value": component_vdrop(built=built, sample=sample, cid=target),
                }
            )
        else:
            out.append({"kind": kind, "target": target, "value": None, "error": "unknown_probe_kind"})
    return out

