"""Parse and merge :class:`~moju.piratio.nondim.NondimScales` for monitor config and Path B."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Literal, Optional

from moju.piratio.nondim import NondimScales

LawScaleMode = Literal["auto", "fixed"]
StateUnits = Literal["nondimensional", "dimensional"]

_VALID_LAW_SCALE_MODES = frozenset({"auto", "fixed"})
_VALID_STATE_UNITS = frozenset({"nondimensional", "dimensional"})


def validate_law_scale_mode(mode: str) -> LawScaleMode:
    m = str(mode).strip().lower()
    if m not in _VALID_LAW_SCALE_MODES:
        raise ValueError(
            f"law_scale_mode must be one of {sorted(_VALID_LAW_SCALE_MODES)!r}; got {mode!r}"
        )
    return m  # type: ignore[return-value]


def validate_state_units(units: str) -> StateUnits:
    u = str(units).strip().lower()
    if u not in _VALID_STATE_UNITS:
        raise ValueError(
            f"state_units must be one of {sorted(_VALID_STATE_UNITS)!r}; got {units!r}"
        )
    return u  # type: ignore[return-value]


def nondim_scales_from_dict(d: Optional[Dict[str, Any]]) -> Optional[NondimScales]:
    """Build :class:`NondimScales` from a JSON-friendly dict; ``None`` if *d* is empty."""
    if not d:
        return None
    field_names = {f.name for f in fields(NondimScales)}
    kwargs: Dict[str, Any] = {}
    for k, v in d.items():
        if k not in field_names:
            raise ValueError(
                f"nondim_scales: unknown field {k!r}; valid fields: {sorted(field_names)}"
            )
        if v is not None:
            kwargs[k] = v
    if "L_ref" not in kwargs:
        raise ValueError("nondim_scales dict must include L_ref when provided explicitly")
    return NondimScales(**kwargs)


def merge_nondim_scales(
    base: NondimScales,
    overrides: Optional[Dict[str, Any]],
) -> NondimScales:
    """Return a new :class:`NondimScales` with *overrides* applied on top of *base*."""
    if not overrides:
        return base
    field_names = {f.name for f in fields(NondimScales)}
    merged: Dict[str, Any] = {}
    for f in fields(NondimScales):
        merged[f.name] = getattr(base, f.name)
    for k, v in overrides.items():
        if k not in field_names:
            raise ValueError(
                f"nondim_scales override: unknown field {k!r}; valid: {sorted(field_names)}"
            )
        if v is not None:
            merged[k] = v
    return NondimScales(**merged)


def nondim_scales_to_dict(scales: NondimScales) -> Dict[str, Any]:
    """JSON-serializable snapshot of reference scales."""
    out: Dict[str, Any] = {}
    for f in fields(NondimScales):
        val = getattr(scales, f.name)
        if val is not None:
            out[f.name] = float(val) if isinstance(val, (int, float)) else val
    return out
