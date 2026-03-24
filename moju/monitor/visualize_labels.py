"""Human-readable labels for monitor visualization (residual keys and categories)."""

from __future__ import annotations

from typing import Dict

_CATEGORY_PREFIXES = frozenset({"laws", "constitutive", "scaling", "data"})

_CLOSURE_SUFFIX_LABELS: Dict[str, str] = {
    "chain_dx": "Spatial Consistency",
    "chain_dy": "Spatial Consistency",
    "chain_dz": "Spatial Consistency",
    "chain_dt": "Temporal Consistency",
    "ref_delta": "Reference Consistency",
    "implied_delta": "Implied Consistency",
    "pi_constant": "Scale Invariance",
}

_CATEGORY_DISPLAY: Dict[str, str] = {
    "laws": "Governing Laws",
    "constitutive": "Constitutive Relations",
    "scaling": "Scaling and Similarity",
    "data": "Data",
}


def _title_token(segment: str) -> str:
    """Replace underscores with spaces and apply title casing."""
    return segment.replace("_", " ").strip().title()


def pretty_residual_key(flat_key: str) -> str:
    """
    Turn a flat residual key (e.g. ``scaling/fo/chain_dx``) into a display string
    (e.g. ``Fo Spatial Consistency``).

    Strips leading category prefixes, maps closure suffixes to readable phrases,
    and removes underscores in favor of spaced words.
    """
    if not flat_key or not str(flat_key).strip():
        return ""
    parts = [p for p in str(flat_key).split("/") if p]
    if parts and parts[0] in _CATEGORY_PREFIXES:
        parts = parts[1:]
    if not parts:
        return _title_token(str(flat_key).replace("/", " "))
    last = parts[-1]
    if last in _CLOSURE_SUFFIX_LABELS:
        suffix = _CLOSURE_SUFFIX_LABELS[last]
        base_parts = parts[:-1]
        if base_parts:
            base = " ".join(_title_token(p) for p in base_parts)
            return f"{base} {suffix}"
        return suffix
    return " ".join(_title_token(p) for p in parts)


def pretty_category_name(category: str) -> str:
    """Display name for a residual category (e.g. for dashboard category labels)."""
    return _CATEGORY_DISPLAY.get(category, _title_token(category))
