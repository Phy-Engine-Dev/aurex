from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Iterable


class PESimError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeriesVdcResistorsSpec:
    v_volts: float
    r1_ohm: float
    r2_ohm: float


@dataclass(frozen=True)
class SeriesVdcResistorsResult:
    current_a: float
    v_plus: float
    v_mid: float
    v_ground: float
    r_total_ohm: float


def _as_int_array(values: Iterable[int]) -> ctypes.Array:
    items = list(int(v) for v in values)
    return (ctypes.c_int * len(items))(*items)


def _as_double_array(values: Iterable[float]) -> ctypes.Array:
    items = list(float(v) for v in values)
    return (ctypes.c_double * len(items))(*items)


class PhyEngineLib:
    def __init__(self, lib_path: str):
        if not lib_path:
            raise PESimError("Missing lib_path")
        self._lib = ctypes.CDLL(lib_path)
        self._bind()

    def _bind(self) -> None:
        c_void_p = ctypes.c_void_p
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_size_t = ctypes.c_size_t
        c_size_t_p = ctypes.POINTER(c_size_t)
        c_size_t_pp = ctypes.POINTER(c_size_t_p)
        c_double_p = ctypes.POINTER(ctypes.c_double)
        c_bool_p = ctypes.POINTER(ctypes.c_bool)
        c_uint32 = ctypes.c_uint32

        self._lib.create_circuit.argtypes = [
            c_int_p,
            c_size_t,
            c_int_p,
            c_size_t,
            c_double_p,
            c_size_t_pp,
            c_size_t_pp,
            ctypes.POINTER(c_size_t),
        ]
        self._lib.create_circuit.restype = c_void_p

        self._lib.destroy_circuit.argtypes = [c_void_p, c_size_t_p, c_size_t_p]
        self._lib.destroy_circuit.restype = None

        self._lib.circuit_set_analyze_type.argtypes = [c_void_p, c_uint32]
        self._lib.circuit_set_analyze_type.restype = ctypes.c_int

        self._lib.circuit_analyze.argtypes = [c_void_p]
        self._lib.circuit_analyze.restype = ctypes.c_int

        self._lib.circuit_sample.argtypes = [
            c_void_p,
            c_size_t_p,
            c_size_t_p,
            c_size_t,
            c_double_p,
            c_size_t_p,
            c_double_p,
            c_size_t_p,
            c_bool_p,
            c_size_t_p,
        ]
        self._lib.circuit_sample.restype = ctypes.c_int

    def simulate_series_vdc_two_resistors(
        self,
        *,
        spec: SeriesVdcResistorsSpec,
    ) -> SeriesVdcResistorsResult:
        v = float(spec.v_volts)
        r1 = float(spec.r1_ohm)
        r2 = float(spec.r2_ohm)
        if r1 <= 0.0 or r2 <= 0.0:
            raise PESimError("Resistances must be > 0 ohm")

        # Element codes (see Phy-Engine dll_main.cpp add_model_via_code):
        # 0: ground, 4: VDC, 1: resistor
        elements = _as_int_array([0, 4, 1, 1])
        properties = _as_double_array([v, r1, r2])

        # Wire format: [ele1, pin1, ele2, pin2] repeated.
        # We build a loop: VDC(+)->R1->R2->VDC(-) and tie VDC(-) to ground.
        wires = _as_int_array(
            [
                1,
                0,
                2,
                0,
                2,
                1,
                3,
                0,
                3,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
            ]
        )

        vec_pos = ctypes.POINTER(ctypes.c_size_t)()
        chunk_pos = ctypes.POINTER(ctypes.c_size_t)()
        comp_size = ctypes.c_size_t(0)

        circuit = self._lib.create_circuit(
            ctypes.cast(elements, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_size_t(len(elements)),
            ctypes.cast(wires, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_size_t(len(wires)),
            ctypes.cast(properties, ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(vec_pos),
            ctypes.byref(chunk_pos),
            ctypes.byref(comp_size),
        )
        if not circuit:
            raise PESimError("create_circuit failed")

        try:
            # analyze_type::DC == 1
            if self._lib.circuit_set_analyze_type(circuit, ctypes.c_uint32(1)) != 0:
                raise PESimError("circuit_set_analyze_type failed")
            if self._lib.circuit_analyze(circuit) != 0:
                raise PESimError("circuit_analyze failed")

            n = int(comp_size.value)
            if n != 3:
                raise PESimError(f"Unexpected comp_size={n} (expected 3)")

            # For VDC + 2 resistors, pin views are expected to be small; allocate conservatively.
            voltage = (ctypes.c_double * 64)()
            voltage_ord = (ctypes.c_size_t * (n + 1))()
            current = (ctypes.c_double * 64)()
            current_ord = (ctypes.c_size_t * (n + 1))()
            digital = (ctypes.c_bool * 64)()
            digital_ord = (ctypes.c_size_t * (n + 1))()

            if (
                self._lib.circuit_sample(
                    circuit,
                    vec_pos,
                    chunk_pos,
                    ctypes.c_size_t(n),
                    voltage,
                    voltage_ord,
                    current,
                    current_ord,
                    digital,
                    digital_ord,
                )
                != 0
            ):
                raise PESimError("circuit_sample failed")

            if int(voltage_ord[1] - voltage_ord[0]) < 2:
                raise PESimError("Unexpected VDC pin count (expected >= 2)")
            if int(voltage_ord[2] - voltage_ord[1]) < 2:
                raise PESimError("Unexpected R1 pin count (expected >= 2)")
            if int(current_ord[2] - current_ord[1]) < 1:
                raise PESimError("Unexpected R1 branch count (expected >= 1)")

            # Component order matches non-ground elements in input order: VDC, R1, R2.
            vdc_pin0 = float(voltage[voltage_ord[0] + 0])
            vdc_pin1 = float(voltage[voltage_ord[0] + 1])
            r1_pin1 = float(voltage[voltage_ord[1] + 1])

            i_r1 = float(current[current_ord[1] + 0])
            i = abs(i_r1) if i_r1 != 0.0 else abs(v / (r1 + r2))

            return SeriesVdcResistorsResult(
                current_a=i,
                v_plus=vdc_pin0,
                v_mid=r1_pin1,
                v_ground=vdc_pin1,
                r_total_ohm=(r1 + r2),
            )
        finally:
            self._lib.destroy_circuit(circuit, vec_pos, chunk_pos)
