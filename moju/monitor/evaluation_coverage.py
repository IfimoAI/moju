"""
Evaluation coverage records for admissibility verdicts.

Non-finite residual samples (NaN / ±inf) are tolerated by NaN-aware reductions so
scoring can proceed on the finite subset. Coverage records make that subset
explicit on every scored item and on the aggregate report.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import jax.numpy as jnp

NONFINITE_POLICIES = ("ignore", "flag", "strict")
DEFAULT_NONFINITE_POLICY = "flag"

_STATUS_RANK = {"ok": 0, "incomplete": 1, "empty": 2}

TIER_INCOMPLETE = "Incomplete"


def validate_nonfinite_policy(policy: Optional[str]) -> str:
    """Return a normalised policy name; default ``flag`` when ``None``/empty."""
    if policy is None or str(policy).strip() == "":
        return DEFAULT_NONFINITE_POLICY
    p = str(policy).strip().lower()
    if p not in NONFINITE_POLICIES:
        raise ValueError(
            f"nonfinite_policy must be one of {NONFINITE_POLICIES!r}; got {policy!r}"
        )
    return p


def coverage_status(*, n_finite: int, n_nonfinite: int) -> str:
    """Map finite / non-finite counts to ``ok`` / ``incomplete`` / ``empty``."""
    nf = int(n_finite)
    nn = int(n_nonfinite)
    if nf <= 0:
        return "empty"
    if nn > 0:
        return "incomplete"
    return "ok"


def worst_coverage_status(statuses: Sequence[str]) -> str:
    """Aggregate status = worst of components (``empty`` > ``incomplete`` > ``ok``)."""
    worst = "ok"
    rank = 0
    for s in statuses:
        key = str(s or "ok")
        r = _STATUS_RANK.get(key, 0)
        if r > rank:
            rank = r
            worst = key
    return worst


def _as_bool_mask(mask: Any) -> Optional[jnp.ndarray]:
    if mask is None:
        return None
    return jnp.asarray(mask).astype(bool)


def _sample_flat(arr: Any) -> jnp.ndarray:
    """Flatten residual values the same way scoring reductions do (``.ravel()``)."""
    return jnp.asarray(arr).ravel()


def count_finite_samples(arr: Any) -> Dict[str, int]:
    """Host counts of finite / non-finite elements in a residual array."""
    a = _sample_flat(arr)
    n = int(a.size)
    if n == 0:
        return {"n_selected": 0, "n_finite": 0, "n_nonfinite": 0}
    finite = jnp.isfinite(a)
    n_finite = int(jnp.sum(finite))
    return {
        "n_selected": n,
        "n_finite": n_finite,
        "n_nonfinite": n - n_finite,
    }


def coverage_record_from_counts(
    *,
    n_total: int,
    n_masked: int,
    n_finite: int,
    n_nonfinite: int,
    eval_mask_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a JSON-serialisable coverage record from integer counts."""
    n_total_i = int(n_total)
    n_masked_i = max(0, int(n_masked))
    n_selected = n_total_i - n_masked_i
    if n_selected < 0:
        n_selected = 0
    # Prefer explicit finite/nonfinite; clamp to selected.
    n_fin = max(0, int(n_finite))
    n_non = max(0, int(n_nonfinite))
    if n_fin + n_non != n_selected and n_selected >= 0:
        # Trust finite/nonfinite when they sum to selected; else re-derive nonfinite.
        if n_fin <= n_selected:
            n_non = n_selected - n_fin
        else:
            n_fin = n_selected
            n_non = 0
    rec: Dict[str, Any] = {
        "n_total": n_total_i,
        "n_masked": n_masked_i,
        "n_selected": int(n_selected),
        "n_finite": int(n_fin),
        "n_nonfinite": int(n_non),
        "coverage_status": coverage_status(n_finite=n_fin, n_nonfinite=n_non),
    }
    if eval_mask_label is not None:
        rec["eval_mask_label"] = str(eval_mask_label)
    return rec


def coverage_from_residual(
    arr: Any,
    *,
    eval_mask: Any = None,
    eval_mask_label: Optional[str] = None,
    n_total: Optional[int] = None,
    n_masked: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Coverage for one residual array.

    When ``eval_mask`` is provided, ``n_total`` / ``n_masked`` default to the mask
    size / excluded count; ``arr`` must already be restricted to selected points
    (as produced by :func:`moju.monitor.eval_mask.mask_flat_residuals`).
    Masked samples never count as non-finite.
    """
    counts = count_finite_samples(arr)
    m = _as_bool_mask(eval_mask)
    if m is not None:
        n_tot = int(n_total) if n_total is not None else int(m.size)
        n_msk = int(n_masked) if n_masked is not None else int(n_tot - int(jnp.sum(m)))
    else:
        n_tot = int(n_total) if n_total is not None else int(counts["n_selected"])
        n_msk = int(n_masked) if n_masked is not None else 0
    return coverage_record_from_counts(
        n_total=n_tot,
        n_masked=n_msk,
        n_finite=counts["n_finite"],
        n_nonfinite=counts["n_nonfinite"],
        eval_mask_label=eval_mask_label,
    )


def coverage_counts_arrays(arr: Any) -> Dict[str, Any]:
    """Tracer-safe finite / size counts for jit/vmap (0-d arrays)."""
    a = _sample_flat(arr)
    n = jnp.asarray(a.size, dtype=jnp.int32)
    n_finite = jnp.sum(jnp.isfinite(a)).astype(jnp.int32)
    n_nonfinite = (n - n_finite).astype(jnp.int32)
    return {"n_selected": n, "n_finite": n_finite, "n_nonfinite": n_nonfinite}


def materialize_coverage_counts(
    nums: Mapping[str, Any],
    *,
    n_total: Optional[int] = None,
    n_masked: int = 0,
    eval_mask_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Host-side coverage record from batched numeric sidecars."""
    n_sel = int(jnp.asarray(nums["n_selected"]).item())
    n_fin = int(jnp.asarray(nums["n_finite"]).item())
    n_non = int(jnp.asarray(nums["n_nonfinite"]).item())
    n_tot = int(n_total) if n_total is not None else n_sel + int(n_masked)
    return coverage_record_from_counts(
        n_total=n_tot,
        n_masked=n_masked,
        n_finite=n_fin,
        n_nonfinite=n_non,
        eval_mask_label=eval_mask_label,
    )


def coverage_map_from_flat(
    flat: Mapping[str, Any],
    *,
    eval_mask: Any = None,
    eval_mask_label: Optional[str] = None,
    mask_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build per-key coverage for a flat residual map (post-mask if applicable)."""
    n_total: Optional[int] = None
    n_masked = 0
    if mask_meta:
        try:
            n_total = int(mask_meta.get("n_points"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            n_total = None
        try:
            n_masked = int(mask_meta.get("n_excluded") or 0)
        except (TypeError, ValueError):
            n_masked = 0
    elif eval_mask is not None:
        m = _as_bool_mask(eval_mask)
        if m is not None:
            n_total = int(m.size)
            n_masked = int(n_total - int(jnp.sum(m)))
    out: Dict[str, Dict[str, Any]] = {}
    for key, arr in flat.items():
        if key == "closure_debug":
            continue
        out[str(key)] = coverage_from_residual(
            arr,
            eval_mask=eval_mask,
            eval_mask_label=eval_mask_label,
            n_total=n_total,
            n_masked=n_masked if (eval_mask is not None or mask_meta) else 0,
        )
    return out


def aggregate_coverage(per_key: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Roll up per-key coverage into one aggregate record."""
    if not per_key:
        return coverage_record_from_counts(
            n_total=0, n_masked=0, n_finite=0, n_nonfinite=0
        )
    n_total = 0
    n_masked = 0
    n_finite = 0
    n_nonfinite = 0
    statuses = []
    label = None
    for rec in per_key.values():
        if not isinstance(rec, Mapping):
            continue
        n_total += int(rec.get("n_total") or 0)
        n_masked += int(rec.get("n_masked") or 0)
        n_finite += int(rec.get("n_finite") or 0)
        n_nonfinite += int(rec.get("n_nonfinite") or 0)
        statuses.append(str(rec.get("coverage_status") or "ok"))
        if label is None and rec.get("eval_mask_label") is not None:
            label = str(rec.get("eval_mask_label"))
    # Aggregate status from component statuses (not re-derived from summed counts),
    # so one incomplete key marks the aggregate incomplete even if totals look ok.
    status = worst_coverage_status(statuses) if statuses else "empty"
    out = coverage_record_from_counts(
        n_total=n_total,
        n_masked=n_masked,
        n_finite=n_finite,
        n_nonfinite=n_nonfinite,
        eval_mask_label=label,
    )
    out["coverage_status"] = status
    return out


def apply_nonfinite_policy_to_key(
    key_report: Dict[str, Any],
    coverage: Optional[Mapping[str, Any]],
    *,
    policy: str,
) -> None:
    """
    Attach coverage and, under ``strict``, null the score when coverage is not ``ok``.

    Does not override an existing ``Not identifiable`` level.
    """
    if coverage:
        key_report["evaluation_coverage"] = dict(coverage)
        key_report["coverage_status"] = coverage.get("coverage_status")
    if policy != "strict" or not coverage:
        return
    if key_report.get("admissibility_level") == "Not identifiable":
        return
    if key_report.get("identifiability_status") == "not_identifiable":
        return
    status = str(coverage.get("coverage_status") or "ok")
    if status == "ok":
        return
    key_report["admissibility_score"] = None
    key_report["admissibility_level"] = TIER_INCOMPLETE
    key_report["coverage_incomplete"] = True


def is_incomplete_key(key_report: Mapping[str, Any]) -> bool:
    """True when a per-key report is marked Incomplete under strict policy."""
    return (
        key_report.get("admissibility_level") == TIER_INCOMPLETE
        or key_report.get("coverage_incomplete") is True
    )


def finite_score_or_none(value: Any) -> Optional[float]:
    """Return a finite float, else ``None``."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


__all__ = [
    "DEFAULT_NONFINITE_POLICY",
    "NONFINITE_POLICIES",
    "TIER_INCOMPLETE",
    "aggregate_coverage",
    "apply_nonfinite_policy_to_key",
    "count_finite_samples",
    "coverage_counts_arrays",
    "coverage_from_residual",
    "coverage_map_from_flat",
    "coverage_record_from_counts",
    "coverage_status",
    "finite_score_or_none",
    "is_incomplete_key",
    "materialize_coverage_counts",
    "validate_nonfinite_policy",
    "worst_coverage_status",
]
