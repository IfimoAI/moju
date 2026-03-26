"""Naming helpers for monitor derivative state keys ``d_<field>_<suffix>``."""

from __future__ import annotations

_SUFFIX = {"x": "_dx", "y": "_dy", "z": "_dz", "t": "_dt"}


def deriv_to_state_suffix(deriv: str) -> str:
    if deriv not in _SUFFIX:
        raise ValueError(f"deriv must be one of {tuple(_SUFFIX)}, got {deriv!r}")
    return _SUFFIX[deriv]


def derivative_state_key(state_key: str, deriv: str) -> str:
    """e.g. ('T', 'x') -> 'd_T_dx'."""
    return f"d_{state_key}{deriv_to_state_suffix(deriv)}"


__all__ = ["deriv_to_state_suffix", "derivative_state_key"]
