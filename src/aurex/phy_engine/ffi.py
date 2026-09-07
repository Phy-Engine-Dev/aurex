from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any


_TraceCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_double, ctypes.c_size_t)
_ControlCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_double, ctypes.c_size_t)


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

        if hasattr(self._dll, "circuit_set_gmin"):
            self._dll.circuit_set_gmin.argtypes = [ctypes.c_void_p, ctypes.c_double]
            self._dll.circuit_set_gmin.restype = ctypes.c_int

        self._dll.circuit_analyze.argtypes = [ctypes.c_void_p]
        self._dll.circuit_analyze.restype = ctypes.c_int

        if hasattr(self._dll, "circuit_run_mixed_dc"):
            self._dll.circuit_run_mixed_dc.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            self._dll.circuit_run_mixed_dc.restype = ctypes.c_int

        if hasattr(self._dll, "circuit_transient_digital_propagation_version"):
            self._dll.circuit_transient_digital_propagation_version.argtypes = []
            self._dll.circuit_transient_digital_propagation_version.restype = ctypes.c_int
        if hasattr(self._dll, "circuit_transient_digital_propagation_configured_version"):
            self._dll.circuit_transient_digital_propagation_configured_version.argtypes = []
            self._dll.circuit_transient_digital_propagation_configured_version.restype = ctypes.c_int
        if hasattr(self._dll, "circuit_run_transient_bounded_configured"):
            self._dll.circuit_run_transient_bounded_configured.argtypes = [
                ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            self._dll.circuit_run_transient_bounded_configured.restype = ctypes.c_int
        if hasattr(self._dll, "circuit_run_transient_trace_configured"):
            self._dll.circuit_run_transient_trace_configured.argtypes = [
                ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                _TraceCallback, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            self._dll.circuit_run_transient_trace_configured.restype = ctypes.c_int
        if hasattr(self._dll, "circuit_transient_control_version"):
            self._dll.circuit_transient_control_version.argtypes = []
            self._dll.circuit_transient_control_version.restype = ctypes.c_int
        if hasattr(self._dll, "circuit_run_transient_trace_controlled"):
            self._dll.circuit_run_transient_trace_controlled.argtypes = [
                ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t, _TraceCallback, ctypes.c_void_p,
                _ControlCallback, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t)]
            self._dll.circuit_run_transient_trace_controlled.restype = ctypes.c_int

        if hasattr(self._dll, "circuit_run_transient_bounded"):
            self._dll.circuit_run_transient_bounded.argtypes = [
                ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_size_t),
            ]
            self._dll.circuit_run_transient_bounded.restype = ctypes.c_int

        if hasattr(self._dll, "circuit_get_model_scalar"):
            self._dll.circuit_get_model_scalar.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
            ]
            self._dll.circuit_get_model_scalar.restype = ctypes.c_int

        if hasattr(self._dll, "circuit_get_model_digital"):
            self._dll.circuit_get_model_digital.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8),
            ]
            self._dll.circuit_get_model_digital.restype = ctypes.c_int

        if hasattr(self._dll, "circuit_run_transient_trace"):
            self._dll.circuit_run_transient_trace.argtypes = [
                ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_size_t,
                ctypes.c_size_t, _TraceCallback, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            self._dll.circuit_run_transient_trace.restype = ctypes.c_int

        self._dll.circuit_set_model_double_by_name.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_double,
        ]
        self._dll.circuit_set_model_double_by_name.restype = ctypes.c_int

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

        if hasattr(self._dll, "circuit_sample_complex"):
            d = ctypes.POINTER(ctypes.c_double)
            z = ctypes.POINTER(ctypes.c_size_t)
            self._dll.circuit_sample_complex.argtypes = [
                ctypes.c_void_p, z, z, ctypes.c_size_t, ctypes.c_size_t,
                d, d, z, d, d, z, ctypes.POINTER(ctypes.c_uint8),
            ]
            self._dll.circuit_sample_complex.restype = ctypes.c_int

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

    def set_gmin(self, gmin_siemens: float) -> None:
        entry = getattr(self.lib._dll, "circuit_set_gmin", None)
        if entry is None:
            if gmin_siemens != 0:
                raise PhyEngineError("Rebuild Phy-Engine: explicit GMIN support is unavailable")
            return
        rc = int(entry(self.ptr, ctypes.c_double(gmin_siemens)))
        if rc != 0:
            raise PhyEngineError(f"circuit_set_gmin failed (rc={rc})")

    def analyze(self) -> None:
        rc = int(self.lib._dll.circuit_analyze(self.ptr))
        if rc != 0:
            raise PhyEngineError("circuit_analyze failed")

    def analyze_mixed_dc(self, analyze_type: int) -> None:
        entry = getattr(self.lib._dll, "circuit_run_mixed_dc", None)
        if entry is None:
            raise PhyEngineError("Rebuild Phy-Engine: mixed-signal DC driver is unavailable")
        rc = int(entry(self.ptr, ctypes.c_uint32(analyze_type)))
        if rc:
            raise PhyEngineError(self._mixed_error("Mixed-signal DC solve", rc))

    @staticmethod
    def _mixed_error(prefix: str, rc: int) -> str:
        detail = {
            4: "digital/analog fixed point did not settle within 64 propagations",
            5: "a digital driver produced an invalid target node or nonfinite voltage",
            6: "multiple digital outputs drove one analog node at different voltage levels",
            7: "the coupled analog MNA solve did not converge",
            8: "the coupled solution contained a nonfinite voltage or current",
        }.get(rc, "unknown native mixed-signal failure")
        return f"{prefix} failed (rc={rc}: {detail})"

    def run_transient_bounded(self, step: float, stop: float, *, max_steps: int = 10000,
                              digital_steps_per_tr_step: int = 1) -> dict[str, Any]:
        if not hasattr(self.lib._dll, "circuit_run_transient_bounded"):
            raise PhyEngineError("Rebuild Phy-Engine: exact-endpoint transient driver is unavailable")
        self._validate_digital_steps(digital_steps_per_tr_step)
        actual, steps, digital_steps = ctypes.c_double(), ctypes.c_size_t(), ctypes.c_size_t()
        configured = self._configured_transient_entry('circuit_run_transient_bounded_configured')
        if configured is not None:
            rc = configured(self.ptr, step, stop, max_steps, digital_steps_per_tr_step,
                ctypes.byref(actual), ctypes.byref(steps), ctypes.byref(digital_steps))
        else:
            if digital_steps_per_tr_step != 1:
                raise PhyEngineError('Rebuild Phy-Engine: configured digital steps per TR step are unavailable; requested count was not ignored')
            rc = self.lib._dll.circuit_run_transient_bounded(
                self.ptr, step, stop, max_steps, ctypes.byref(actual), ctypes.byref(steps))
        if 5 <= rc <= 8:
            raise PhyEngineError(self._mixed_error("Transient mixed-signal coupling", rc))
        if rc:
            raise PhyEngineError(f"Transient solve failed (rc={rc}, completed_steps={steps.value}, time_s={actual.value})")
        if configured is not None and digital_steps.value != steps.value * digital_steps_per_tr_step:
            raise PhyEngineError('Native digital propagation count disagrees with completed TR steps')
        return {"actual_stop_s": actual.value, "completed_steps": steps.value,
                "requested_stop_s": stop, "requested_step_s": step,
                "digital_propagation": self.transient_digital_policy(steps.value, digital_steps_per_tr_step,
                    digital_steps.value if configured is not None else None),
                "method": "native bounded transient solve; exact endpoint"}

    @staticmethod
    def _validate_digital_steps(value: int) -> None:
        if type(value) is not int or not 1 <= value <= 64:
            raise PhyEngineError('digital_steps_per_tr_step must be an integer in 1..64; not a digital time interval')

    def _configured_transient_entry(self, name: str):
        entry = getattr(self.lib._dll, name, None)
        if entry is not None:
            query = getattr(self.lib._dll, 'circuit_transient_digital_propagation_configured_version', None)
            if query is None or int(query()) != 1:
                raise PhyEngineError('Unsupported configured transient ABI version; no solve or digital tick was executed')
        return entry

    def transient_digital_policy(self, steps: int, per_step: int = 1, actual_total: int | None = None) -> dict[str, Any]:
        query = getattr(self.lib._dll, "circuit_transient_digital_propagation_version", None)
        version = int(query()) if query is not None else 0
        configured = getattr(self.lib._dll, 'circuit_transient_digital_propagation_configured_version', None)
        configured_version = int(configured()) if configured is not None else 0
        verified = version == 1 and ((actual_total is None and per_step == 1) or
            (configured_version == 1 and actual_total == steps * per_step))
        return {"version": version, "configured_version": configured_version, "verified_per_step": verified,
                "policy": ('once_after_each_native_solve_before_sampling' if per_step == 1 else
                    'configured_after_each_native_solve_before_sampling') if verified else 'unreported_by_native_library',
                "per_tr_step": per_step,
                "completed_propagation_steps": (actual_total if actual_total is not None else steps) if verified else None,
                "count_origin": ('native_counter' if actual_total is not None else 'version1_contract') if verified else 'unreported'}

    def model_scalar(self, idx: int, attribute: int) -> float:
        if not hasattr(self.lib._dll, "circuit_get_model_scalar"):
            raise PhyEngineError("Rebuild Phy-Engine: typed scalar observation is unavailable")
        if not 0 <= idx < self.comp_size or attribute < 0:
            raise PhyEngineError("Model or attribute index out of range")
        result = ctypes.c_double()
        rc = self.lib._dll.circuit_get_model_scalar(
            self.ptr, self.vec_pos[idx], self.chunk_pos[idx], attribute, ctypes.byref(result))
        if rc:
            raise PhyEngineError(f"Model scalar observation failed (rc={rc})")
        return result.value

    def model_digital(self, idx: int, attribute: int) -> int:
        if not hasattr(self.lib._dll, "circuit_get_model_digital"):
            raise PhyEngineError("Rebuild Phy-Engine: typed digital model observation is unavailable")
        if not 0 <= idx < self.comp_size or attribute < 0:
            raise PhyEngineError("Model or attribute index out of range")
        result = ctypes.c_uint8()
        rc = self.lib._dll.circuit_get_model_digital(
            self.ptr, self.vec_pos[idx], self.chunk_pos[idx], attribute, ctypes.byref(result))
        if rc:
            raise PhyEngineError(f"Digital model observation failed (rc={rc})")
        return int(result.value)

    def set_model_scalar(self, idx: int, name: str, value: float) -> None:
        if not 0 <= idx < self.comp_size:
            raise PhyEngineError("Model index out of range")
        raw = name.encode("ascii")
        rc = self.lib._dll.circuit_set_model_double_by_name(
            self.ptr, self.vec_pos[idx], self.chunk_pos[idx], raw, len(raw), value)
        if rc:
            raise PhyEngineError(f"Setting model parameter {name} failed (rc={rc})")

    def run_transient_trace(self, step: float, stop: float, *, sample_every: int, capture, max_steps: int = 10000,
                            digital_steps_per_tr_step: int = 1) -> dict[str, Any]:
        if not hasattr(self.lib._dll, "circuit_run_transient_trace"):
            raise PhyEngineError("Rebuild Phy-Engine: bounded transient trace is unavailable")
        self._validate_digital_steps(digital_steps_per_tr_step)
        samples, errors = [], []

        @_TraceCallback
        def on_sample(_user, actual_time, completed):
            try:
                if len(samples) >= 201:
                    raise PhyEngineError("Transient trace exceeds 201 samples")
                samples.append({"time_s": actual_time, "completed_steps": completed, "sample": capture()})
                return 0
            except Exception as error:
                errors.append(error)
                return 1

        actual, steps, count, digital_steps = ctypes.c_double(), ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        configured = self._configured_transient_entry('circuit_run_transient_trace_configured')
        if configured is not None:
            rc = configured(self.ptr, step, stop, max_steps, sample_every, digital_steps_per_tr_step, on_sample, None,
                ctypes.byref(actual), ctypes.byref(steps), ctypes.byref(count), ctypes.byref(digital_steps))
        else:
            if digital_steps_per_tr_step != 1:
                raise PhyEngineError('Rebuild Phy-Engine: configured digital steps per TR step are unavailable; requested count was not ignored')
            rc = self.lib._dll.circuit_run_transient_trace(
                self.ptr, step, stop, max_steps, sample_every, on_sample, None,
                ctypes.byref(actual), ctypes.byref(steps), ctypes.byref(count))
        if errors:
            raise PhyEngineError("Transient observation failed: " + str(errors[0])) from errors[0]
        if 5 <= rc <= 8:
            raise PhyEngineError(self._mixed_error("Transient mixed-signal coupling", rc))
        if rc or count.value != len(samples):
            raise PhyEngineError(f"Transient trace failed (rc={rc}, completed_steps={steps.value}, time_s={actual.value})")
        if configured is not None and digital_steps.value != steps.value * digital_steps_per_tr_step:
            raise PhyEngineError('Native digital propagation count disagrees with completed TR steps')
        return {"actual_stop_s": actual.value, "completed_steps": steps.value,
                "requested_stop_s": stop, "requested_step_s": step,
                "sample_every": sample_every, "sample_count": len(samples), "samples": samples,
                "digital_propagation": self.transient_digital_policy(steps.value, digital_steps_per_tr_step,
                    digital_steps.value if configured is not None else None),
                "method": "one native transient solve, measurements sampled after completed steps; exact endpoint"}

    def run_transient_controlled(self, step: float, stop: float, *, before_step,
                                 sample_every: int = 0, capture=None,
                                 max_steps: int = 10000,
                                 digital_steps_per_tr_step: int = 1) -> dict[str, Any]:
        entry = getattr(self.lib._dll, "circuit_run_transient_trace_controlled", None)
        version = getattr(self.lib._dll, "circuit_transient_control_version", None)
        if entry is None or version is None or int(version()) != 1:
            raise PhyEngineError(
                "Rebuild Phy-Engine: controlled transient ABI v1 is unavailable; "
                "no interaction was silently ignored")
        self._validate_digital_steps(digital_steps_per_tr_step)
        if type(sample_every) is not int or sample_every < 0:
            raise PhyEngineError("Controlled transient sample_every must be a nonnegative integer")
        if sample_every and capture is None:
            raise PhyEngineError("Controlled transient sampling requires a capture callback")
        samples, trace_errors, control_errors = [], [], []

        @_TraceCallback
        def on_sample(_user, actual_time, completed):
            try:
                if len(samples) >= 201:
                    raise PhyEngineError("Transient trace exceeds 201 samples")
                samples.append({"time_s": actual_time, "completed_steps": completed,
                                "sample": capture()})
                return 0
            except Exception as error:
                trace_errors.append(error)
                return 1

        @_ControlCallback
        def on_control(_user, target_time, step_number):
            try:
                before_step(float(target_time), int(step_number))
                return 0
            except Exception as error:
                control_errors.append(error)
                return 1

        trace_callback = on_sample if sample_every else _TraceCallback()
        actual, steps = ctypes.c_double(), ctypes.c_size_t()
        count, digital_steps = ctypes.c_size_t(), ctypes.c_size_t()
        rc = entry(self.ptr, step, stop, max_steps, sample_every,
            digital_steps_per_tr_step, trace_callback, None, on_control, None,
            ctypes.byref(actual), ctypes.byref(steps), ctypes.byref(count),
            ctypes.byref(digital_steps))
        if control_errors:
            raise PhyEngineError("Transient interaction failed before solve: " + str(control_errors[0])) from control_errors[0]
        if trace_errors:
            raise PhyEngineError("Transient observation failed: " + str(trace_errors[0])) from trace_errors[0]
        if 5 <= rc <= 8:
            raise PhyEngineError(self._mixed_error("Controlled transient mixed-signal coupling", rc))
        if rc or count.value != len(samples):
            raise PhyEngineError(
                f"Controlled transient failed (rc={rc}, completed_steps={steps.value}, "
                f"time_s={actual.value})")
        if digital_steps.value != steps.value * digital_steps_per_tr_step:
            raise PhyEngineError("Native digital propagation count disagrees with completed TR steps")
        return {"actual_stop_s": actual.value, "completed_steps": steps.value,
                "requested_stop_s": stop, "requested_step_s": step,
                "sample_every": sample_every, "sample_count": len(samples),
                **({"samples": samples} if sample_every else {}),
                "digital_propagation": self.transient_digital_policy(
                    steps.value, digital_steps_per_tr_step, digital_steps.value),
                "interaction_abi": {"version": 1,
                    "callback_timing": "before_each_physical_tr_solve",
                    "target_time_and_one_based_step": True},
                "method": ("one native controlled transient solve"
                    + (", measurements sampled after completed steps" if sample_every else "")
                    + "; exact endpoint")}

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

    def sample_complex(self, *, max_pins: int) -> dict[str, Any]:
        if not hasattr(self.lib._dll, "circuit_sample_complex"):
            raise PhyEngineError("Rebuild Phy-Engine: bounded complex sampler is unavailable")
        cap = max(1, self.comp_size * max(max_pins, 8))
        vr, vi, ir, ii = ((ctypes.c_double * cap)() for _ in range(4))
        vo, io = ((ctypes.c_size_t * (self.comp_size + 1))() for _ in range(2))
        digital = (ctypes.c_uint8 * cap)()
        rc = self.lib._dll.circuit_sample_complex(self.ptr, self.vec_pos, self.chunk_pos,
            self.comp_size, cap, vr, vi, vo, ir, ii, io, digital)
        if rc:
            raise PhyEngineError(f"circuit_sample_complex failed (rc={rc})")
        return {
            "voltage": [float(vr[i]) for i in range(vo[self.comp_size])],
            "voltage_imag": [float(vi[i]) for i in range(vo[self.comp_size])],
            "current": [float(ir[i]) for i in range(io[self.comp_size])],
            "current_imag": [float(ii[i]) for i in range(io[self.comp_size])],
            "digital": [int(digital[i]) for i in range(vo[self.comp_size])],
            "voltage_ord": list(vo), "current_ord": list(io), "digital_ord": list(vo),
        }


def load_library(path: str) -> _Lib:
    return _Lib(path)
