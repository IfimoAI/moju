"""
Derivative state keys for constitutive/scaling chain audits.

Centralizes how ``d_<state_key>_<suffix>`` keys are named and enumerated from
``AuditSpec``-shaped dicts (``predicted_spatial``, ``predicted_temporal``,
``chain_spatial_axes``).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

# Spatial directions supported by compute_chain (suffix after state key).
CHAIN_SPATIAL_DERIVS = ("x", "y", "z")

_SUFFIX = {"x": "_dx", "y": "_dy", "z": "_dz", "t": "_dt"}


def deriv_to_state_suffix(deriv: str) -> str:
    if deriv not in _SUFFIX:
        raise ValueError(f"deriv must be one of {tuple(_SUFFIX)}, got {deriv!r}")
    return _SUFFIX[deriv]


def derivative_state_key(state_key: str, deriv: str) -> str:
    """e.g. ('T', 'x') -> 'd_T_dx' (matches closure_registry._deriv_key)."""
    return f"d_{state_key}{deriv_to_state_suffix(deriv)}"


def audit_derivative_keys_for_spec(spec: Dict) -> Tuple[Set[str], Set[str]]:
    """
    Return (spatial_derivative_keys, temporal_derivative_keys) required by one audit spec.
    """
    spatial: Set[str] = set()
    temporal: Set[str] = set()
    output_key = spec.get("output_key")
    pred_x = list(spec.get("predicted_spatial") or [])
    pred_t = list(spec.get("predicted_temporal") or [])
    axes = list(spec.get("chain_spatial_axes") or ["x"])
    for ax in axes:
        if ax not in CHAIN_SPATIAL_DERIVS:
            continue
        if pred_x and output_key:
            spatial.add(derivative_state_key(output_key, ax))
            for k in pred_x:
                spatial.add(derivative_state_key(k, ax))
    if pred_t and output_key:
        temporal.add(derivative_state_key(output_key, "t"))
        for k in pred_t:
            temporal.add(derivative_state_key(k, "t"))
    return spatial, temporal


def collect_audit_derivative_keys(
    constitutive_audit: List[Dict],
    scaling_audit: List[Dict],
) -> Tuple[Set[str], Set[str]]:
    """Union of all spatial and temporal derivative keys required by audit specs."""
    sx: Set[str] = set()
    st: Set[str] = set()
    for spec in constitutive_audit + scaling_audit:
        a, b = audit_derivative_keys_for_spec(spec)
        sx |= a
        st |= b
    return sx, st
