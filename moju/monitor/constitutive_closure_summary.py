"""Constitutive closure summary text aligned with fractional bands and admissibility tiers."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from moju.monitor.auditor import (
    CONSTITUTIVE_BAND_FRAC_HIGH,
    CONSTITUTIVE_BAND_FRAC_LOW,
    CONSTITUTIVE_BAND_FRAC_MOD,
    admissibility_level,
)
from moju.monitor.visualize_labels import format_admissibility_pct


def constitutive_band_label(r_eff: float) -> str:
    """
    Human-readable Consistency band for fractional closure error ``r_eff``.

    Bands: green ±0.1%, amber ±0.1–0.5%, red ±0.5–1%, alarm beyond ±1%.
    """
    try:
        r = float(r_eff)
    except (TypeError, ValueError):
        return "unknown band"
    if not math.isfinite(r) or r < 0.0:
        return "unknown band"
    if r <= CONSTITUTIVE_BAND_FRAC_HIGH:
        return "±0.1% green band"
    if r <= CONSTITUTIVE_BAND_FRAC_MOD:
        return "±0.1%–±0.5% amber band"
    if r <= CONSTITUTIVE_BAND_FRAC_LOW:
        return "±0.5%–±1% red band"
    return "beyond ±1% alarm band"


def _short_admissibility_level(score: float) -> str:
    label = admissibility_level(score)
    if label == "Unknown":
        return "N/A"
    return label.split()[0]


def format_constitutive_closure_summary(
    *,
    r_worst: float,
    admissibility_score: float,
    r_rms: Optional[float] = None,
) -> str:
    """
    One-line constitutive closure report, e.g.

    ``Constitutive worst-point fractional error = 0.14% (±0.1%–±0.5% amber band). RMS = 0.05%. Admissibility (worst-point) = 87.2% (Moderate).``
    """
    try:
        r = float(r_worst)
        a = float(admissibility_score)
    except (TypeError, ValueError):
        return "Constitutive closure summary unavailable."
    if not math.isfinite(r) or not math.isfinite(a):
        return "Constitutive closure summary unavailable."
    pct_err = 100.0 * r
    band = constitutive_band_label(r)
    adm_pct = format_admissibility_pct(a)
    level = _short_admissibility_level(a)
    line = (
        f"Constitutive worst-point fractional error = {pct_err:.2f}% "
        f"({band}). "
    )
    if r_rms is not None:
        try:
            rms_f = float(r_rms)
        except (TypeError, ValueError):
            rms_f = float("nan")
        if math.isfinite(rms_f):
            line += f"RMS = {100.0 * rms_f:.2f}%. "
    line += f"Admissibility (worst-point) = {adm_pct} ({level})."
    return line


def _worst_point_score(data: Dict[str, Any]) -> float:
    """Scalar used for ranking worst implied_delta (prefer r_max, else score_for_admissibility)."""
    for field in ("r_max", "score_for_admissibility"):
        try:
            v = float(data.get(field, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            return v
    try:
        return float(data.get("rms", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def select_worst_implied_delta(
    per_key: Dict[str, Any],
) -> Optional[Tuple[str, float, float, float]]:
    """
    Pick the ``…/implied_delta`` key with largest worst-point closure error.

    Returns ``(flat_key, r_worst, r_rms, admissibility_score)`` or ``None``.
    """
    worst_key: Optional[str] = None
    worst_score = -1.0
    worst_rms = float("nan")
    worst_adm = float("nan")
    for key, data in per_key.items():
        if not str(key).endswith("/implied_delta"):
            continue
        if not isinstance(data, dict):
            continue
        score = _worst_point_score(data)
        if not math.isfinite(score):
            continue
        if score <= worst_score:
            continue
        try:
            rms = float(data.get("rms", float("nan")))
        except (TypeError, ValueError):
            rms = float("nan")
        try:
            adm = float(data.get("admissibility_score", float("nan")))
        except (TypeError, ValueError):
            adm = float("nan")
        worst_score = score
        worst_key = key
        worst_rms = rms
        worst_adm = adm
    if worst_key is None:
        return None
    return worst_key, worst_score, worst_rms, worst_adm


def build_constitutive_closure_summary(per_key: Dict[str, Any]) -> Optional[str]:
    """Build summary sentence for worst ``implied_delta`` row, or ``None`` if none present."""
    picked = select_worst_implied_delta(per_key)
    if picked is None:
        return None
    _key, r_worst, r_rms, adm = picked
    return format_constitutive_closure_summary(
        r_worst=r_worst,
        admissibility_score=adm,
        r_rms=r_rms,
    )
