"""Human-readable labels for monitor visualization (residual keys and categories)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

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


def _sentence_case_metric_name(title_cased_phrase: str) -> str:
    """First word as given; remaining words lowercased (e.g. ``Thermal Diffusivity`` → ``Thermal diffusivity``)."""
    words = str(title_cased_phrase).split()
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    return words[0] + " " + " ".join(w.lower() for w in words[1:])


def _is_law_slug_segment(segment: str) -> bool:
    return str(segment).lower().startswith("law_")


def truncate_display_label(text: Any, max_len: int = 36) -> str:
    """
    Shorten long metric names for axis tick labels (matplotlib / Plotly).

    Uses a single ellipsis character for a consistent conference-style look.
    """
    s = str(text) if text is not None else ""
    if len(s) <= max_len:
        return s
    keep = max(1, max_len - 1)
    return s[:keep] + "…"


def format_admissibility_pct(score: float) -> str:
    """Format admissibility in [0, 1] as a percentage with two decimal places."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(s):
        return "N/A"
    return f"{100.0 * s:.2f}%"


def category_adm_bar_x_range(vals: List[float]) -> Tuple[float, float]:
    """
    Horizontal axis range for category admissibility bar charts so nearby scores
    (e.g. 0.98 vs 0.995) remain visually distinct. Scores live in [0, 1]; the right
    edge never exceeds 1.0.
    """
    finite: List[float] = []
    for v in vals:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            finite.append(fv)
    if not finite:
        return (0.0, 1.0)
    lo = min(finite)
    hi = max(finite)
    span = hi - lo
    pad = max(0.015, 0.2 * span) if span > 0 else 0.05
    x0 = max(0.0, lo - pad)
    x1 = hi + pad
    x1 = max(x1, hi + 0.04)
    x1 = min(x1, 1.0)
    if x1 - x0 < 0.04:
        mid = (lo + hi) / 2.0 if span > 0 else lo
        x0 = max(0.0, mid - 0.025)
        x1 = min(1.0, mid + 0.025 + 0.06)
    if x1 <= x0:
        x1 = min(1.0, x0 + 0.04)
    return (float(x0), float(x1))


def pretty_residual_key(flat_key: str) -> str:
    """
    Turn a flat residual key (e.g. ``scaling/fo/chain_dx``) into a display string
    (e.g. ``Fo Spatial Consistency``).

    Strips leading category prefixes, maps closure suffixes to readable phrases,
    and removes underscores in favor of spaced words.
    """
    if not flat_key or not str(flat_key).strip():
        return ""
    parts_full = [p for p in str(flat_key).split("/") if p]
    category0 = parts_full[0] if parts_full else ""

    if (
        category0 == "constitutive"
        and len(parts_full) >= 2
        and parts_full[-1] == "implied_delta"
    ):
        middle = parts_full[1:-1]
        stripped = [p for p in middle if not _is_law_slug_segment(p)]
        if not stripped and middle:
            stripped = [middle[0]]
        if stripped:
            titled = " ".join(_title_token(p) for p in stripped)
            base = _sentence_case_metric_name(titled)
            return f"{base} (implied)"

    parts = list(parts_full)
    if parts and parts[0] in _CATEGORY_PREFIXES:
        parts = parts[1:]
    if not parts:
        return _title_token(str(flat_key).replace("/", " "))
    last = parts[-1]
    if last in _CLOSURE_SUFFIX_LABELS:
        base_parts = parts[:-1]
        suffix = _CLOSURE_SUFFIX_LABELS[last]
        if base_parts:
            base = " ".join(_title_token(p) for p in base_parts)
            return f"{base} {suffix}"
        return suffix
    return " ".join(_title_token(p) for p in parts)


def pretty_category_name(category: str) -> str:
    """Display name for a residual category (e.g. for dashboard category labels)."""
    return _CATEGORY_DISPLAY.get(category, _title_token(category))
