"""
Public, machine-readable admissibility tier definition.

Scores are ``admissibility = 1 / (1 + R_norm)`` with ``R_norm = v_eff / scale_k``. Tier cutoffs are
derived from fractional constitutive bands (±0.1 %, ±0.5 %, ±1 %) at the default nondimensional gauge
``scale_k = 1e-2``. These constants mirror :mod:`moju.monitor.auditor`, which remains the source of
truth used at scoring time.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from moju.monitor.auditor import (
    ADM_HIGH_THRESHOLD,
    ADM_LOW_THRESHOLD,
    ADM_MODERATE_THRESHOLD,
    CONSTITUTIVE_BAND_FRAC_HIGH,
    CONSTITUTIVE_BAND_FRAC_LOW,
    CONSTITUTIVE_BAND_FRAC_MOD,
    DEFAULT_NONDIM_R_NORM_SCALE_K,
    admissibility_level,
)
from moju.monitor.evaluation_coverage import TIER_INCOMPLETE

TIER_HIGH = "High Admissibility"
TIER_MODERATE = "Moderate Admissibility"
TIER_LOW = "Low Admissibility"
TIER_NON_ADMISSIBLE = "Non-Admissible"
TIER_UNKNOWN = "Unknown"

# Score-band tiers only (Incomplete / Not identifiable / Unknown are non-scored).
TIER_ORDER: Tuple[str, ...] = (TIER_HIGH, TIER_MODERATE, TIER_LOW, TIER_NON_ADMISSIBLE)

# Non-scored labels that must not KeyError when mapped alongside TIER_ORDER.
TIER_NON_SCORED: Tuple[str, ...] = (
    TIER_INCOMPLETE,
    TIER_UNKNOWN,
    "Not identifiable",
    "Insufficient points",
)

# Inclusive lower bound on the admissibility score for each tier.
TIER_CUTOFFS: Dict[str, float] = {
    TIER_HIGH: float(ADM_HIGH_THRESHOLD),
    TIER_MODERATE: float(ADM_MODERATE_THRESHOLD),
    TIER_LOW: float(ADM_LOW_THRESHOLD),
    TIER_NON_ADMISSIBLE: 0.0,
}

TIER_DEFINITION: Dict[str, Any] = {
    "score_formula": "admissibility = 1 / (1 + R_norm), R_norm = v_eff / scale_k",
    "default_scale_k": float(DEFAULT_NONDIM_R_NORM_SCALE_K),
    "tiers": [
        {"name": TIER_HIGH, "min_score": TIER_CUTOFFS[TIER_HIGH], "fractional_band": CONSTITUTIVE_BAND_FRAC_HIGH},
        {
            "name": TIER_MODERATE,
            "min_score": TIER_CUTOFFS[TIER_MODERATE],
            "fractional_band": CONSTITUTIVE_BAND_FRAC_MOD,
        },
        {"name": TIER_LOW, "min_score": TIER_CUTOFFS[TIER_LOW], "fractional_band": CONSTITUTIVE_BAND_FRAC_LOW},
        {"name": TIER_NON_ADMISSIBLE, "min_score": 0.0, "fractional_band": None},
    ],
    "metrics": {
        "laws": "rms",
        "constitutive_implied_delta": "worst_point",
        "constitutive_ref_delta": "worst_point",
        "constitutive_bound": "worst_point",
        "data": "rms",
        "declared": "as declared by Scoring(metric=...)",
    },
    "rollup": {
        "laws": "geometric_mean",
        "constitutive": "minimum (when worst-point keys are present, else geometric_mean)",
        "data": "geometric_mean",
        "overall_training": "minimum(laws, constitutive)",
        "overall_eval": "minimum over finite category scores",
    },
    "meaning": (
        "A passing tier indicates consistency with the stated physics within the audited "
        "conditions (declared laws, constitutive models, sample points, and scales). "
        "It is not a guarantee of correctness. "
        "Incomplete (nonfinite_policy='strict') means not assessable due to incomplete "
        "coverage, distinct from Not identifiable."
    ),
    "non_scored_tiers": list(TIER_NON_SCORED),
}


def tier_for_score(score: Optional[float]) -> str:
    """Tier name for an admissibility score (``Incomplete`` if ``None``; ``Unknown`` if non-finite)."""
    return admissibility_level(score)


def tier_rank(level: Optional[str]) -> Optional[int]:
    """
    Index in :data:`TIER_ORDER` for scored bands, or ``None`` for non-scored labels
    (``Incomplete``, ``Not identifiable``, ``Unknown``, …). Safe to call for any tier string.
    """
    if level is None:
        return None
    try:
        return TIER_ORDER.index(str(level))
    except ValueError:
        return None


def tier_definition() -> Dict[str, Any]:
    """Deep copy of :data:`TIER_DEFINITION` (safe to embed in reports)."""
    return copy.deepcopy(TIER_DEFINITION)
