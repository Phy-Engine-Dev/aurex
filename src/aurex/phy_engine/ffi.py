from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any


class PhyEngineError(RuntimeError):
    pass


class _Lib:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise PhyEngineError(f"phyengine library not found: {path}")
        self._dll = ctypes.CDLL(path)

        self._dll.create_circuit.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._dll.create_circuit.restype = ctypes.c_void_p

        self._dll.destroy_circuit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._dll.destroy_circuit.restype = None

        self._dll.circuit_set_analyze_type.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._dll.circuit_set_analyze_type.restype = ctypes.c_int

        self._dll.circuit_set_tr.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        self._dll.circuit_set_tr.restype = ctypes.c_int

        self._dll.circuit_set_ac_omega.argtypes = [ctypes.c_void_p, ctypes.c_double]
        self._dll.circuit_set_ac_omega.restype = ctypes.c_int

        self._dll.circuit_analyze.argtypes = [ctypes.c_void_p]
        self._dll.circuit_analyze.restype = ctypes.c_int

        self._dll.circuit_digital_clk.argtypes = [ctypes.c_void_p]
        self._dll.circuit_digital_clk.restype = ctypes.c_int

        self._dll.circuit_set_model_digital.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint8,
        ]
        self._dll.circuit_set_model_digital.restype = ctypes.c_int

        self._dll.circuit_sample_u8.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._dll.circuit_sample_u8.restype = ctypes.c_int

    def create_circuit(self, *, elements: list[int], wires: list[int], properties: list[float]) -> "_Circuit":
        ele_arr = (ctypes.c_int * len(elements))(*[int(x) for x in elements])
        wires_arr = (ctypes.c_int * len(wires))(*[int(x) for x in wires]) if wires else None
        props = properties if properties else [0.0]
        prop_arr = (ctypes.c_double * len(props))(*[float(x) for x in props])

        vec_pos_ptr = ctypes.POINTER(ctypes.c_size_t)()
        chunk_pos_ptr = ctypes.POINTER(ctypes.c_size_t)()
        comp_size = ctypes.c_size_t(0)

        cptr = self._dll.create_circuit(
            ele_arr,
            ctypes.c_size_t(len(elements)),
            wires_arr,
            ctypes.c_size_t(len(wires)),
            prop_arr,
            ctypes.byref(vec_pos_ptr),
            ctypes.byref(chunk_pos_ptr),
            ctypes.byref(comp_size),
        )
        if not cptr:
            raise PhyEngineError("create_circuit returned NULL (invalid elements/properties?)")
        return _Circuit(self, cptr, vec_pos_ptr, chunk_pos_ptr, int(comp_size.value))


@dataclass
class _Circuit:
    lib: _Lib
    ptr: int
    vec_pos: Any
    chunk_pos: Any
    comp_size: int

    def close(self) -> None:
        try:
            self.lib._dll.destroy_circuit(self.ptr, self.vec_pos, self.chunk_pos)
        finally:
            self.ptr = 0

    def set_analyze_type(self, analyze_type: int) -> None:
        rc = int(self.lib._dll.circuit_set_analyze_type(self.ptr, ctypes.c_uint32(int(analyze_type))))
        if rc != 0:
            raise PhyEngineError(f"circuit_set_analyze_type failed (rc={rc})")

    def set_tr(self, t_step: float, t_stop: float) -> None:
        rc = int(self.lib._dll.circuit_set_tr(self.ptr, ctypes.c_double(t_step), ctypes.c_double(t_stop)))
        if rc != 0:
            raise PhyEngineError(f"circuit_set_tr failed (rc={rc})")

    def set_ac_omega(self, omega: float) -> None:
        rc = int(self.lib._dll.circuit_set_ac_omega(self.ptr, ctypes.c_double(omega)))
        if rc != 0:
            raise PhyEngineError(f"circuit_set_ac_omega failed (rc={rc})")

    def analyze(self) -> None:
        rc = int(self.lib._dll.circuit_analyze(self.ptr))
        if rc != 0:
            raise PhyEngineError("circuit_analyze failed")

    def digital_clk(self) -> None:
        rc = int(self.lib._dll.circuit_digital_clk(self.ptr))
        if rc != 0:
            raise PhyEngineError(f"circuit_digital_clk failed (rc={rc})")

    def set_model_digital(self, idx: int, attribute_index: int, state: int) -> None:
        if idx < 0 or idx >= self.comp_size:
            raise PhyEngineError("component index out of range")
        vp = int(self.vec_pos[idx])
        cp = int(self.chunk_pos[idx])
        rc = int(
            self.lib._dll.circuit_set_model_digital(
                self.ptr,
                ctypes.c_size_t(vp),
                ctypes.c_size_t(cp),
                ctypes.c_size_t(int(attribute_index)),
                ctypes.c_uint8(int(state) & 0xFF),
            )
        )
        if rc != 0:
            raise PhyEngineError(f"circuit_set_model_digital failed (rc={rc})")

    def sample_u8(self, *, max_pins: int) -> dict[str, Any]:
        comp = self.comp_size
        if comp <= 0:
            return {"comp_size": 0, "voltage": [], "current": [], "digital": []}

        cap = max(1, int(comp) * max(1, int(max_pins)))
        voltage = (ctypes.c_double * cap)()
        current = (ctypes.c_double * cap)()
        digital = (ctypes.c_uint8 * cap)()
        voltage_ord = (ctypes.c_size_t * (comp + 1))()
        current_ord = (ctypes.c_size_t * (comp + 1))()
        digital_ord = (ctypes.c_size_t * (comp + 1))()

        rc = int(
            self.lib._dll.circuit_sample_u8(
                self.ptr,
                self.vec_pos,
                self.chunk_pos,
                ctypes.c_size_t(comp),
                voltage,
                voltage_ord,
                current,
                current_ord,
                digital,
                digital_ord,
            )
        )
        if rc != 0:
            raise PhyEngineError(f"circuit_sample_u8 failed (rc={rc})")

        return {
            "comp_size": comp,
            "voltage": [float(voltage[i]) for i in range(int(voltage_ord[comp]))],
            "current": [float(current[i]) for i in range(int(current_ord[comp]))],
            "digital": [int(digital[i]) for i in range(int(digital_ord[comp]))],
            "voltage_ord": [int(voltage_ord[i]) for i in range(comp + 1)],
            "current_ord": [int(current_ord[i]) for i in range(comp + 1)],
            "digital_ord": [int(digital_ord[i]) for i in range(comp + 1)],
        }


def load_library(path: str) -> _Lib:
    return _Lib(path)

