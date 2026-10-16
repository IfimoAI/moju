"""
Named spatial regions for per-region implied-law fitting and scoring (Moju 2.3).

With ``regions=None``, callers must leave the global 2.2 path unchanged.
Regions never change derivatives or the global audit result; they only select
which points enter per-region fits and scores (via 0/1 weights).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np

# Fit minimum only (≥ 2× implied_min_identifiable_count=8; matches
# weighted_safeguard.DEFAULT_MIN_REGION_ABS). Not a percentile-adequacy threshold
# and not tuned to any dataset. See percentile_tail_adequacy().
DEFAULT_REGION_MIN_POINTS: int = 16
RegionsInput = Union[Mapping[str, Any], Tuple[Any, Mapping[Any, str]]]


def percentile_tail_adequacy(
    n_points: int,
    percentile: float,
) -> Dict[str, Any]:
    """
    Expected excluded-tail count for a percentile metric: ``n × (1 − p/100)``.

    Status
    ------
    - ``equivalent_to_worst_point`` when ``tail < 1`` (percentile cannot exclude any point)
    - ``low_tail_count`` when ``1 ≤ tail < 5``
    - ``ok`` otherwise

    For p95, ``tail < 1`` implies ``n < 20``; ``n ≥ 100`` is recommended for a
    stable upper-tail estimate (tail ≥ 5). This is independent of
    ``region_min_points`` (a fit floor, not a percentile threshold).
    """
    n = int(n_points)
    p = float(percentile)
    if n < 0:
        raise ValueError(f"n_points must be >= 0, got {n}")
    if not (0.0 < p <= 100.0):
        raise ValueError(f"percentile must be in (0, 100], got {p}")
    # n × (100 − p) / 100 avoids float drift on exact boundaries (e.g. 10×10/100 = 1).
    tail = (float(n) * (100.0 - p)) / 100.0
    if tail < 1.0:
        status = "equivalent_to_worst_point"
    elif tail < 5.0:
        status = "low_tail_count"
    else:
        status = "ok"
    return {
        "percentile_status": status,
        "percentile_tail_count": tail,
        "n_points": n,
        "percentile": p,
    }


def validate_region_min_points(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"region_min_points must be a positive int, got {value!r}"
        ) from exc
    if n < 1:
        raise ValueError(f"region_min_points must be >= 1, got {n}")
    return n


def _as_bool_mask(
    mask: Any,
    *,
    name: str,
    reference_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    a = np.asarray(mask)
    if a.dtype != bool and not np.issubdtype(a.dtype, np.integer) and not np.issubdtype(
        a.dtype, np.floating
    ):
        raise TypeError(
            f"region {name!r} mask must be boolean or 0/1 numeric, got dtype={a.dtype}"
        )
    if reference_shape is not None:
        ref = tuple(int(s) for s in reference_shape)
        if a.shape != ref and a.size != int(np.prod(ref)):
            raise ValueError(
                f"region {name!r} mask shape {a.shape} incompatible with "
                f"reference shape {ref} ({int(np.prod(ref))} points)"
            )
        if a.shape != ref:
            a = a.reshape(ref)
    uniq = set(np.unique(a).tolist())
    if not uniq.issubset({0, 1, True, False, 0.0, 1.0}):
        bad = [u for u in uniq if u not in (0, 1, True, False, 0.0, 1.0)]
        if bad:
            raise ValueError(
                f"region {name!r} mask must be boolean or 0/1, found values {sorted(bad)[:8]!r}"
            )
    return a.astype(bool)


def _mask_sha256(mask_bool: np.ndarray) -> str:
    # Stable hash of packed bits (endian-independent via uint8 view of bool).
    packed = np.packbits(mask_bool.astype(np.uint8), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def normalize_regions(
    regions: Any,
    *,
    reference_shape: Optional[Sequence[int]] = None,
    region_names: Optional[Mapping[Any, str]] = None,
) -> Dict[str, Any]:
    """
    Normalize ``regions`` to a dict of named boolean flat masks.

    Accepted forms
    --------------
    - ``dict[str, mask]``: boolean or 0/1 arrays
    - ``(labels, region_names)`` or labels with ``region_names=``: integer label
      array; ``region_names`` maps label → name (unmapped labels ignored)

    Returns a provenance-ready dict::

        {
          "names": [...],
          "masks": {name: bool ndarray flat},
          "weights": {name: float32 0/1 ndarray flat},
          "n_points": {name: int},
          "mask_sha256": {name: str},
          "n_unassigned": int,
          "n_total": int,
          "overlaps": bool,
        }
    """
    if regions is None:
        return {
            "names": [],
            "masks": {},
            "weights": {},
            "n_points": {},
            "mask_sha256": {},
            "n_unassigned": 0,
            "n_total": 0,
            "overlaps": False,
        }

    ref_shape = tuple(int(s) for s in reference_shape) if reference_shape is not None else None
    masks: Dict[str, np.ndarray] = {}

    if isinstance(regions, Mapping) and region_names is None:
        if not regions:
            raise ValueError("regions dict must be non-empty when provided")
        for name, mask in regions.items():
            if not isinstance(name, str) or not name.strip():
                raise TypeError(f"region names must be non-empty str, got {name!r}")
            masks[str(name)] = _as_bool_mask(
                mask, name=str(name), reference_shape=ref_shape
            )
    else:
        # Label array form: regions is labels, or (labels, names)
        labels = regions
        names_map = region_names
        if isinstance(regions, tuple) and len(regions) == 2 and region_names is None:
            labels, names_map = regions
        if names_map is None:
            raise TypeError(
                "label-array regions require region_names mapping (label → name)"
            )
        lab = np.asarray(labels)
        if not np.issubdtype(lab.dtype, np.integer):
            raise TypeError(
                f"region labels must be an integer array, got dtype={lab.dtype}"
            )
        if ref_shape is not None:
            if lab.shape != ref_shape and lab.size != int(np.prod(ref_shape)):
                raise ValueError(
                    f"region labels shape {lab.shape} incompatible with {ref_shape}"
                )
            if lab.shape != ref_shape:
                lab = lab.reshape(ref_shape)
        if not names_map:
            raise ValueError("region_names mapping must be non-empty")
        for lab_id, name in names_map.items():
            if not isinstance(name, str) or not name.strip():
                raise TypeError(f"region names must be non-empty str, got {name!r}")
            masks[str(name)] = lab == int(lab_id)

    shapes = {k: v.shape for k, v in masks.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"region masks have inconsistent shapes: {shapes}")
    sample = next(iter(masks.values()))
    n_total = int(sample.size)
    names = sorted(masks.keys())

    empty = [n for n in names if not bool(np.any(masks[n]))]
    if empty:
        raise ValueError(
            f"empty region mask(s): {empty} (all-False / zero coverage is not allowed)"
        )

    assigned = np.zeros(sample.shape, dtype=bool)
    overlap = False
    for n in names:
        m = masks[n]
        if np.any(assigned & m):
            overlap = True
        assigned = assigned | m
    n_unassigned = int(np.sum(~assigned))

    weights = {n: masks[n].astype(np.float32) for n in names}
    return {
        "names": names,
        "masks": masks,
        "weights": weights,
        "n_points": {n: int(np.sum(masks[n])) for n in names},
        "mask_sha256": {n: _mask_sha256(masks[n].reshape(-1)) for n in names},
        "n_unassigned": n_unassigned,
        "n_total": int(n_total),
        "overlaps": bool(overlap),
    }


def regions_provenance(
    normalized: Mapping[str, Any],
    *,
    region_min_points: int,
    moju_version: str,
) -> Dict[str, Any]:
    """Compact provenance block for audit / export reports."""
    return {
        "region_names": list(normalized.get("names") or []),
        "n_points": dict(normalized.get("n_points") or {}),
        "mask_sha256": dict(normalized.get("mask_sha256") or {}),
        "n_unassigned": int(normalized.get("n_unassigned") or 0),
        "n_total": int(normalized.get("n_total") or 0),
        "overlaps": bool(normalized.get("overlaps", False)),
        "region_min_points": int(region_min_points),
        "moju_version": moju_version,
    }


def align_weights_to_residual(weights_flat: Any, residual: Any) -> jnp.ndarray:
    """Broadcast a flat 0/1 weight vector onto a residual array's leading points."""
    w = jnp.asarray(weights_flat, dtype=jnp.float32).reshape(-1)
    a = jnp.asarray(residual)
    n = int(w.shape[0])
    # Leading axis = points (collocation), remaining axes = components.
    if a.size == 0:
        return w
    lead = int(a.shape[0]) if a.ndim >= 1 else 1
    if lead != n and a.size == n:
        # residual is already flat scalar field
        return w
    if lead != n:
        # Try matching total point count on all-but-last for vector fields
        if a.ndim >= 2 and int(np.prod(a.shape[:-1])) == n:
            return jnp.reshape(w, a.shape[:-1])
        if a.ndim >= 1 and int(np.prod(a.shape)) == n:
            return jnp.reshape(w, a.shape)
        raise ValueError(
            f"region weights length {n} incompatible with residual shape {tuple(a.shape)}"
        )
    if a.ndim <= 1:
        return w
    # Broadcast over trailing component axes
    shape = (n,) + (1,) * (a.ndim - 1)
    return jnp.reshape(w, shape)


def weighted_reduce_magnitude_arrays(
    arr: Any,
    weights: Any,
    *,
    metric: str = "worst_point",
    percentile: float = 95.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Jit-safe masked magnitude reduce.

    Returns ``(v_eff, worst_flat_index)`` as arrays. ``worst_flat_index`` is ``-1``
    when the region has no finite in-mask points. Percentile uses NaN-masking
    (``jnp.nanpercentile``) so out-of-region points are excluded.
    """
    a = jnp.asarray(arr)
    if a.ndim >= 1 and a.shape[-1] in (2, 3) and a.ndim >= 2:
        mag = jnp.sqrt(jnp.sum(jnp.square(a), axis=-1) + 1e-30)
    else:
        mag = jnp.abs(a)
    mag_flat = jnp.reshape(mag, (-1,))
    w = jnp.asarray(weights, dtype=jnp.float32).reshape(-1)
    # NaN-mask out-of-region and non-finite so percentile/max ignore them.
    scored = jnp.where((w > 0.5) & jnp.isfinite(mag_flat), mag_flat, jnp.nan)
    has = jnp.any(jnp.isfinite(scored))
    if metric == "percentile":
        v = jnp.nanpercentile(scored, float(percentile))
    else:
        v = jnp.nanmax(scored)
    for_argmax = jnp.where(jnp.isfinite(scored), scored, -jnp.inf)
    worst_idx = jnp.argmax(for_argmax).astype(jnp.int32)
    v = jnp.where(has, v, jnp.asarray(jnp.nan, dtype=mag_flat.dtype))
    worst_idx = jnp.where(has, worst_idx, jnp.asarray(-1, dtype=jnp.int32))
    return v, worst_idx


def weighted_reduce_magnitude(
    arr: Any,
    weights: Any,
    *,
    metric: str = "worst_point",
    percentile: float = 95.0,
) -> Tuple[float, Optional[int]]:
    """
    Aggregate ``|arr|`` with 0/1 weights (out-of-region → excluded).

    Returns ``(v_eff, worst_flat_index)``. ``worst_flat_index`` is the flat index
    of the maximum in-region magnitude (or None when empty / non-finite).
    """
    a = jnp.asarray(arr)
    mag = a
    if a.ndim >= 1 and a.shape[-1] in (2, 3) and a.ndim >= 2:
        mag = jnp.sqrt(jnp.sum(jnp.square(a), axis=-1) + 1e-30)
    else:
        mag = jnp.abs(a)
    mag_flat = jnp.reshape(mag, (-1,))
    w = jnp.asarray(weights, dtype=jnp.float32).reshape(-1)
    if int(w.shape[0]) != int(mag_flat.shape[0]):
        w = align_weights_to_residual(weights, mag).reshape(-1)
    if int(w.shape[0]) != int(mag_flat.shape[0]):
        raise ValueError(
            f"weight length {int(w.shape[0])} != magnitude length {int(mag_flat.shape[0])}"
        )
    v, worst_idx = weighted_reduce_magnitude_arrays(
        a, w, metric=metric, percentile=percentile
    )
    v_f = float(v)
    wi = int(worst_idx)
    if wi < 0 or not math.isfinite(v_f):
        return (float("nan") if not math.isfinite(v_f) else v_f), None
    return v_f, wi


def weighted_reduce_magnitude_numpy_reference(
    arr: Any,
    weights: Any,
    *,
    metric: str = "worst_point",
    percentile: float = 95.0,
) -> float:
    """NumPy reference for masked percentile/max (exact check in tests)."""
    a = np.asarray(arr, dtype=float)
    if a.ndim >= 1 and a.shape[-1] in (2, 3) and a.ndim >= 2:
        mag = np.sqrt(np.sum(np.square(a), axis=-1) + 1e-30)
    else:
        mag = np.abs(a)
    mag_flat = mag.reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != mag_flat.size:
        raise ValueError("weight/array size mismatch in numpy reference")
    keep = (w > 0.5) & np.isfinite(mag_flat)
    vals = mag_flat[keep]
    if vals.size == 0:
        return float("nan")
    if metric == "percentile":
        return float(np.percentile(vals, float(percentile)))
    return float(np.max(vals))


def _spatial_shape_for_field(field: Any) -> Tuple[int, ...]:
    a = jnp.asarray(field)
    if a.ndim >= 2 and a.shape[-1] in (2, 3):
        return tuple(int(s) for s in a.shape[:-1])
    return tuple(int(s) for s in a.shape)


def regional_viscosity_fit_arrays(
    *,
    rhs: Any,
    basis: Any,
    region_weights: Any,
    term_fraction: Any,
    fit_quality: Any,
    term_threshold: float,
    fit_threshold: float,
    min_frac: float,
    min_count: int,
    region_min_points: int,
    scale: Optional[Any] = None,
) -> Dict[str, jnp.ndarray]:
    """
    Jit-safe regional viscosity LSQ. Returns only numeric arrays:

    ``n_points``, ``n_identifiable``, ``identifiability_ratio``, ``nu_tilde``,
    ``mu_tilde``, ``fit_quality_median``, ``insufficient`` (0/1), ``identifiable`` (bool).
    """
    from moju.monitor.law_implied_diagnostics import _global_scalar_coefficient

    spatial = _spatial_shape_for_field(rhs)
    w = jnp.reshape(jnp.asarray(region_weights, dtype=jnp.float32), spatial)
    n_region = jnp.sum((w > 0.5).astype(jnp.int32))
    insufficient = n_region < int(region_min_points)

    tf = jnp.reshape(jnp.asarray(term_fraction), spatial)
    fq = jnp.reshape(jnp.asarray(fit_quality), spatial)
    gate = (
        (w > 0.5)
        & jnp.isfinite(tf)
        & jnp.isfinite(fq)
        & (tf >= float(term_threshold))
        & (fq >= float(fit_threshold))
    )
    n_ok = jnp.sum(gate.astype(jnp.int32))
    # eff_min uses static Python ints from min_count and region_min_points floor only
    # when n_region is known statically for the insufficient branch; under jit we use
    # jnp.minimum against the traced n_region.
    eff_min = jnp.minimum(
        jnp.asarray(int(min_count), dtype=jnp.int32),
        jnp.maximum(jnp.asarray(1, dtype=jnp.int32), n_region),
    )
    identifiable = (~insufficient) & (n_ok >= eff_min) & (
        n_ok.astype(jnp.float32) / jnp.maximum(n_region.astype(jnp.float32), 1.0)
        >= float(min_frac)
    )
    fq_in = jnp.where(w > 0.5, fq, jnp.nan)
    fq_med = jnp.nanmedian(fq_in)
    nu = _global_scalar_coefficient(rhs, basis, gate)
    nu = jnp.where(identifiable, nu, jnp.asarray(jnp.nan, dtype=jnp.result_type(nu)))
    if scale is not None:
        mu = jnp.asarray(scale) * nu
    else:
        mu = jnp.asarray(jnp.nan, dtype=jnp.result_type(nu))
    ratio = n_ok.astype(jnp.float32) / jnp.maximum(n_region.astype(jnp.float32), 1.0)
    return {
        "n_points": n_region,
        "n_identifiable": n_ok,
        "identifiability_ratio": ratio,
        "nu_tilde": nu,
        "mu_tilde": mu,
        "fit_quality_median": fq_med,
        "insufficient": insufficient.astype(jnp.int32),
        "identifiable": identifiable,
    }


def regional_viscosity_fit(
    *,
    rhs: Any,
    basis: Any,
    region_weights: Any,
    term_fraction: Any,
    fit_quality: Any,
    term_threshold: float,
    fit_threshold: float,
    min_frac: float,
    min_count: int,
    region_min_points: int,
    scale: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Region-restricted LSQ viscosity coefficient with the same gate as 2.1+.

    ``scale`` optional (``ρ U L``) to convert ν̃* → μ̃. Returns status / ν̃ / counts.
    """
    nums = regional_viscosity_fit_arrays(
        rhs=rhs,
        basis=basis,
        region_weights=region_weights,
        term_fraction=term_fraction,
        fit_quality=fit_quality,
        term_threshold=term_threshold,
        fit_threshold=fit_threshold,
        min_frac=min_frac,
        min_count=min_count,
        region_min_points=region_min_points,
        scale=scale,
    )
    n_region = int(jnp.asarray(nums["n_points"]).item())
    n_ok = int(jnp.asarray(nums["n_identifiable"]).item())
    insufficient = bool(jnp.asarray(nums["insufficient"]).item())
    identifiable = bool(jnp.asarray(nums["identifiable"]).item())
    fq_med = float(jnp.asarray(nums["fit_quality_median"]).item())
    nu_f = float(jnp.asarray(nums["nu_tilde"]).item())
    mu_f = float(jnp.asarray(nums["mu_tilde"]).item())
    if insufficient:
        return {
            "status": "insufficient_points",
            "n_points": n_region,
            "n_identifiable": 0,
            "nu_tilde": float("nan"),
            "mu_tilde": float("nan"),
            "fit_quality_median": float("nan"),
            "reason": (
                f"Region has {n_region} points; region_min_points={int(region_min_points)}."
            ),
        }
    if not identifiable:
        return {
            "status": "not_identifiable",
            "n_points": n_region,
            "n_identifiable": n_ok,
            "nu_tilde": float("nan"),
            "mu_tilde": float("nan"),
            "fit_quality_median": fq_med,
            "reason": (
                "Constitutive viscosity check could not be evaluated in this region: "
                "too few points satisfy the term-fraction and fit-quality gates."
            ),
        }
    return {
        "status": "ok",
        "n_points": n_region,
        "n_identifiable": n_ok,
        "nu_tilde": nu_f if math.isfinite(nu_f) else float("nan"),
        "mu_tilde": mu_f if math.isfinite(mu_f) else float("nan"),
        "fit_quality_median": fq_med if math.isfinite(fq_med) else float("nan"),
        "reason": None,
    }


def status_from_region_numerics(
    *,
    n_points: int,
    region_min_points: int,
    identifiable: bool,
    n_identifiable: int,
    metric: str,
    percentile: float,
) -> Dict[str, Any]:
    """Host-side status / percentile flags from batched numeric outputs."""
    out: Dict[str, Any] = {"n_points": int(n_points)}
    if int(n_points) < int(region_min_points):
        out["status"] = "insufficient_points"
        out["reason"] = (
            f"Region has {int(n_points)} points; region_min_points={int(region_min_points)}."
        )
        out["viscosity_status"] = "insufficient_points"
    elif not identifiable:
        out["status"] = "ok"
        out["reason"] = None
        out["viscosity_status"] = "not_identifiable"
        out["viscosity_reason"] = (
            "Constitutive viscosity check could not be evaluated in this region: "
            "too few points satisfy the term-fraction and fit-quality gates."
        )
    else:
        out["status"] = "ok"
        out["reason"] = None
        out["viscosity_status"] = "ok"
    out["n_identifiable"] = int(n_identifiable)
    if metric == "percentile":
        out.update(percentile_tail_adequacy(int(n_points), float(percentile)))
    return out


def _scale_k_for_key(entry: Mapping[str, Any], key: str, *, default: float = 1.0e-2) -> float:
    scale = entry.get("scale") or {}
    try:
        s = float(scale.get(key, default))
    except (TypeError, ValueError):
        s = default
    if not (s > 0.0) or not math.isfinite(s):
        return default
    return s


def _flatten_residuals(residual_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Flat ``category/name`` residual map (skips ``closure_debug`` sidecars)."""
    flat: Dict[str, Any] = {}
    for category, content in residual_dict.items():
        if category == "closure_debug":
            continue
        if not isinstance(content, dict):
            if hasattr(content, "shape"):
                flat[str(category)] = jnp.asarray(content)
            continue
        for name, arr in content.items():
            flat[f"{category}/{name}"] = jnp.asarray(arr)
    return flat


def compute_region_results(
    *,
    residual_dict: Mapping[str, Any],
    entry: Mapping[str, Any],
    normalized_regions: Mapping[str, Any],
    region_min_points: int,
    implied_scoring_opts: Mapping[str, Any],
    moju_version: str,
) -> Dict[str, Any]:
    """
    Build per-region fit/score tables. Does not modify global scores.

    Returns ``{"provenance": ..., "regions": {name: {...}}}``.
    """
    from moju.monitor.auditor import admissibility_level
    from moju.monitor.implied_scoring import (
        DEFAULT_IMPLIED_METRIC,
        DEFAULT_IMPLIED_PERCENTILE,
        noise_normalized_delta,
        score_from_r_max,
    )
    from moju.monitor.law_implied_diagnostics import _gate_params

    flat = _flatten_residuals(residual_dict)
    closure_debug = residual_dict.get("closure_debug") or {}
    metric = str(implied_scoring_opts.get("implied_metric", DEFAULT_IMPLIED_METRIC))
    pct = float(implied_scoring_opts.get("implied_percentile", DEFAULT_IMPLIED_PERCENTILE))
    use_noise = bool(implied_scoring_opts.get("implied_noise_normalization", False))
    # Gate thresholds from constants stored on entry when present.
    constants = dict(entry.get("constants") or {})
    term_thr, fit_thr, min_frac, min_count = _gate_params(constants)

    prov = regions_provenance(
        normalized_regions,
        region_min_points=region_min_points,
        moju_version=moju_version,
    )
    out_regions: Dict[str, Any] = {}
    for name in normalized_regions.get("names") or []:
        w = normalized_regions["weights"][name]
        n_pts = int(normalized_regions["n_points"][name])
        per_key: Dict[str, Any] = {}
        region_status = "ok"
        region_reason = None
        if n_pts < int(region_min_points):
            region_status = "insufficient_points"
            region_reason = (
                f"Region has {n_pts} points; region_min_points={int(region_min_points)}."
            )

        # Viscosity fit (if any implied viscosity debug sidecars present).
        viscosity_meta: Optional[Dict[str, Any]] = None
        for bn, dbg in closure_debug.items():
            if not isinstance(dbg, dict):
                continue
            if dbg.get("noise_rhs") is None or dbg.get("noise_basis") is None:
                continue
            if dbg.get("term_fraction") is None or dbg.get("fit_quality") is None:
                continue
            scale = dbg.get("noise_scale")
            fit = regional_viscosity_fit(
                rhs=dbg["noise_rhs"],
                basis=dbg["noise_basis"],
                region_weights=w,
                term_fraction=dbg["term_fraction"],
                fit_quality=dbg["fit_quality"],
                term_threshold=term_thr,
                fit_threshold=fit_thr,
                min_frac=min_frac,
                min_count=min_count,
                region_min_points=region_min_points,
                scale=scale,
            )
            viscosity_meta = {
                "basename": str(bn),
                **fit,
            }
            break  # first viscosity-like sidecar

        for k, arr in flat.items():
            key_row: Dict[str, Any] = {
                "n_points": n_pts,
            }
            if region_status == "insufficient_points":
                key_row.update(
                    {
                        "status": "insufficient_points",
                        "admissibility_score": float("nan"),
                        "admissibility_level": "Insufficient points",
                        "score_for_admissibility": float("nan"),
                        "worst_point_index": None,
                        "reason": region_reason,
                    }
                )
                per_key[k] = key_row
                continue

            # Implied viscosity keys: attach fit metadata; may be not_identifiable.
            is_implied = str(k).startswith("constitutive/") and str(k).endswith(
                "/implied_delta"
            )
            if is_implied and viscosity_meta is not None and str(k).find(
                viscosity_meta["basename"]
            ) >= 0:
                if viscosity_meta.get("status") == "insufficient_points":
                    key_row.update(
                        {
                            "status": "insufficient_points",
                            "admissibility_score": float("nan"),
                            "admissibility_level": "Insufficient points",
                            "nu_tilde": viscosity_meta.get("nu_tilde"),
                            "mu_tilde": viscosity_meta.get("mu_tilde"),
                            "fit_quality": viscosity_meta.get("fit_quality_median"),
                            "n_identifiable": viscosity_meta.get("n_identifiable"),
                            "reason": viscosity_meta.get("reason"),
                            "worst_point_index": None,
                        }
                    )
                    per_key[k] = key_row
                    continue
                if viscosity_meta.get("status") == "not_identifiable":
                    key_row.update(
                        {
                            "status": "not_identifiable",
                            "admissibility_score": float("nan"),
                            "admissibility_level": "Not identifiable",
                            "nu_tilde": viscosity_meta.get("nu_tilde"),
                            "mu_tilde": viscosity_meta.get("mu_tilde"),
                            "fit_quality": viscosity_meta.get("fit_quality_median"),
                            "n_identifiable": viscosity_meta.get("n_identifiable"),
                            "reason": viscosity_meta.get("reason"),
                            "worst_point_index": None,
                        }
                    )
                    per_key[k] = key_row
                    continue
                key_row["nu_tilde"] = viscosity_meta.get("nu_tilde")
                key_row["mu_tilde"] = viscosity_meta.get("mu_tilde")
                key_row["fit_quality"] = viscosity_meta.get("fit_quality_median")
                key_row["n_identifiable"] = viscosity_meta.get("n_identifiable")

            scored_arr = arr
            if is_implied and use_noise:
                bn = str(k)[len("constitutive/") : -len("/implied_delta")]
                dbg = closure_debug.get(bn) if isinstance(closure_debug.get(bn), dict) else {}
                pred = dbg.get("pred")
                raw = dbg.get("raw")
                if pred is not None and raw is not None:
                    scored_arr = noise_normalized_delta(
                        pred=pred, implied=dbg.get("implied"), raw=raw, debug=dbg
                    )

            # Laws use same metric settings as global implied audit when worst/percentile;
            # for RMS-style law keys under worst_point, use max |r| in region (consistent
            # with constitutive worst-point). Percentile uses the configured percentile.
            law_metric = metric
            try:
                v_eff, worst_idx = weighted_reduce_magnitude(
                    scored_arr, w, metric=law_metric, percentile=pct
                )
            except ValueError:
                # Shape mismatch — skip key rather than fail the global audit.
                continue
            scale_k = _scale_k_for_key(entry, k)
            if not math.isfinite(v_eff):
                score = float("nan")
            else:
                score = score_from_r_max(v_eff, scale_k=scale_k)
            key_row.update(
                {
                    "status": "ok",
                    "score_for_admissibility": float(v_eff) if math.isfinite(v_eff) else float("nan"),
                    "admissibility_score": score,
                    "admissibility_level": admissibility_level(score),
                    "admissibility_metric": (
                        f"noise_norm+p{pct:g}"
                        if use_noise and is_implied and metric == "percentile"
                        else (
                            f"noise_norm+max"
                            if use_noise and is_implied
                            else (f"p{pct:g}" if metric == "percentile" else "max")
                        )
                    ),
                    "worst_point_index": worst_idx,
                    "scale_k": scale_k,
                }
            )
            per_key[k] = key_row

        region_block: Dict[str, Any] = {
            "n_points": n_pts,
            "status": region_status,
            "reason": region_reason,
            "mask_sha256": normalized_regions["mask_sha256"][name],
            "per_key": per_key,
            "viscosity": viscosity_meta,
        }
        if metric == "percentile":
            region_block.update(percentile_tail_adequacy(n_pts, pct))
        out_regions[name] = region_block

    return {"provenance": prov, "regions": out_regions}


__all__ = [
    "DEFAULT_REGION_MIN_POINTS",
    "percentile_tail_adequacy",
    "validate_region_min_points",
    "normalize_regions",
    "regions_provenance",
    "align_weights_to_residual",
    "weighted_reduce_magnitude",
    "weighted_reduce_magnitude_arrays",
    "weighted_reduce_magnitude_numpy_reference",
    "regional_viscosity_fit",
    "regional_viscosity_fit_arrays",
    "status_from_region_numerics",
    "compute_region_results",
]


