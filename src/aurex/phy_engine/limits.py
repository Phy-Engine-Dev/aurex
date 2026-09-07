"""Server-owned simulation limits, classified by audited native ABI models."""
from __future__ import annotations
from typing import Any
from .catalog import COMPONENTS

ANALOG_COMPONENT_LIMIT = 512
DEFAULT_DIGITAL_COMPONENT_LIMIT = 4096
MAX_DIGITAL_COMPONENT_LIMIT = 16384
# src/dll_main.cpp constructs actual PE digital-device models for these codes.
# Explicit ABI membership, not a model-supplied flag or a 'digital_' prefix.
_DIGITAL_ABI_CODES = frozenset((*range(200, 213), *range(220, 233)))


def configured_digital_limit(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_DIGITAL_COMPONENT_LIMIT:
        raise ValueError(f"phy_engine.digital_component_limit must be an integer in 1..{MAX_DIGITAL_COMPONENT_LIMIT}")
    return value


def native_digital_type(name: Any) -> bool:
    if not isinstance(name, str):
        raise ValueError("Component type must be a known native model name")
    model = COMPONENTS.get(name.strip().lower())
    if model is None:
        raise ValueError(f"Unsupported native component type {name!r}; use circuit_catalog")
    return model["code"] in _DIGITAL_ABI_CODES


def validate_spec_size(spec: Any, digital_component_limit: int = DEFAULT_DIGITAL_COMPONENT_LIMIT) -> int:
    digital_component_limit = configured_digital_limit(digital_component_limit)
    if not isinstance(spec, dict) or not isinstance(spec.get("components"), list) or not spec["components"]:
        raise ValueError("spec.components must be a non-empty array")
    # Validate each actual catalog mapping before granting the larger budget.
    # Even one analog model makes this a mixed/analog circuit, regardless of
    # arbitrary 'device_type', 'digital' or limit keys supplied in the spec.
    digital = True
    for component in spec["components"]:
        if not isinstance(component, dict):
            raise ValueError("Each component must be an object")
        digital = native_digital_type(component.get("type")) and digital
    limit = digital_component_limit if digital else ANALOG_COMPONENT_LIMIT
    if len(spec["components"]) > limit:
        kind = "pure native digital" if digital else "analog/mixed"
        raise ValueError(f"spec.components exceeds {kind} limit {limit} (received {len(spec['components'])})")
    return limit
