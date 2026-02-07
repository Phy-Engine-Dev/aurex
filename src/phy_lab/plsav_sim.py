from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PlSavSimError(RuntimeError):
    pass


@dataclass(frozen=True)
class PEElementMeta:
    index: int
    comp_index: int
    identifier: str
    model_id: str
    label: str | None


@dataclass(frozen=True)
class PECircuitInput:
    element_codes: list[int]
    properties: list[float]
    wires: list[int]
    non_ground_meta: list[PEElementMeta]


_MODEL_TO_CODE: dict[str, int] = {
    "Ground Component": 0,
    "Resistor": 1,
    "Basic Capacitor": 2,
    "Basic Inductor": 3,
    "Battery Source": 4,
    "Current Source": 6,
}


_PROPERTY_KEY_BY_CODE: dict[int, tuple[str, ...]] = {
    1: ("电阻", "Resistance", "R"),
    2: ("电容", "Capacitance", "C"),
    3: ("电感", "Inductance", "L"),
    4: ("电压", "Voltage", "V"),
    6: ("电流", "Current", "I"),
}


def build_pe_circuit_input_from_status_save(
    status_save: dict[str, Any],
    *,
    strict: bool = True,
) -> PECircuitInput:
    if not isinstance(status_save, dict):
        raise PlSavSimError("status_save must be an object")

    elements = status_save.get("Elements")
    wires = status_save.get("Wires")
    if not isinstance(elements, list) or not isinstance(wires, list):
        raise PlSavSimError("status_save must contain Elements and Wires lists")

    identifier_to_index: dict[str, int] = {}
    for idx, el in enumerate(elements):
        if not isinstance(el, dict):
            continue
        ident = el.get("Identifier")
        if isinstance(ident, str) and ident.strip():
            identifier_to_index[ident.strip()] = idx

    unsupported: set[str] = set()
    element_codes: list[int] = [0] * len(elements)
    properties: list[float] = []
    non_ground_meta: list[PEElementMeta] = []

    for idx, el in enumerate(elements):
        if not isinstance(el, dict):
            continue
        model_id = el.get("ModelID")
        if not isinstance(model_id, str) or not model_id.strip():
            if strict:
                unsupported.add("(missing ModelID)")
            continue
        model_id = model_id.strip()
        code = _MODEL_TO_CODE.get(model_id)
        if code is None:
            if strict:
                unsupported.add(model_id)
            continue

        element_codes[idx] = int(code)
        if code == 0:
            continue

        props = el.get("Properties")
        if not isinstance(props, dict):
            raise PlSavSimError(f"Element[{idx}] {model_id}: missing Properties object")
        keys = _PROPERTY_KEY_BY_CODE.get(code, ())
        val = None
        for k in keys:
            v = props.get(k)
            if isinstance(v, (int, float)):
                val = float(v)
                break
        if val is None:
            raise PlSavSimError(
                f"Element[{idx}] {model_id}: missing numeric property (expected one of {list(keys)})"
            )
        properties.append(val)

        ident = el.get("Identifier")
        if not isinstance(ident, str) or not ident.strip():
            ident = f"idx:{idx}"
        label = el.get("Label")
        non_ground_meta.append(
            PEElementMeta(
                index=idx,
                comp_index=len(non_ground_meta),
                identifier=ident.strip(),
                model_id=model_id,
                label=label.strip() if isinstance(label, str) and label.strip() else None,
            )
        )

    if strict and unsupported:
        top = sorted(unsupported)
        shown = top[:30]
        more = "" if len(top) <= len(shown) else f" (+{len(top) - len(shown)} more)"
        raise PlSavSimError(
            "Unsupported Physics Lab elements in this experiment: "
            + ", ".join(shown)
            + more
            + ". Try a smaller circuit using only: "
            + ", ".join(sorted(_MODEL_TO_CODE.keys()))
        )

    pe_wires: list[int] = []
    for w in wires:
        if not isinstance(w, dict):
            continue
        src = w.get("Source")
        tgt = w.get("Target")
        sp = w.get("SourcePin")
        tp = w.get("TargetPin")
        if not (isinstance(src, str) and isinstance(tgt, str) and isinstance(sp, int) and isinstance(tp, int)):
            continue
        if src not in identifier_to_index or tgt not in identifier_to_index:
            if strict:
                raise PlSavSimError("Wire references unknown element Identifier")
            continue
        pe_wires.extend([identifier_to_index[src], int(sp), identifier_to_index[tgt], int(tp)])

    return PECircuitInput(
        element_codes=element_codes,
        properties=properties,
        wires=pe_wires,
        non_ground_meta=non_ground_meta,
    )
