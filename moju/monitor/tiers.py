"""
Public, machine-readable admissibility tier definition.

Scores are ``admissibility = 1 / (1 + R_norm)`` with ``R_norm = v_eff / scale_k``. Tier cutoffs are
derived from fractional constitutive bands (±0.1 %, ±0.5 %, ±1 %) at the default nondimensional gauge
``scale_k = 1e-2``. These constants mirror :mod:`moju.monitor.auditor`, which remains the source of
truth used at scoring time.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

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

TIER_HIGH = "High Admissibility"
TIER_MODERATE = "Moderate Admissibility"
TIER_LOW = "Low Admissibility"
TIER_NON_ADMISSIBLE = "Non-Admissible"
TIER_UNKNOWN = "Unknown"

TIER_ORDER: Tuple[str, ...] = (TIER_HIGH, TIER_MODERATE, TIER_LOW, TIER_NON_ADMISSIBLE)

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
        "It is not a guarantee of correctness."
    ),
}


def tier_for_score(score: float) -> str:
    """Tier name for an admissibility score (``Unknown`` if non-finite)."""
    return admissibility_level(score)


def tier_definition() -> Dict[str, Any]:
    """Deep copy of :data:`TIER_DEFINITION` (safe to embed in reports)."""
    return copy.deepcopy(TIER_DEFINITION)
