"""
Safeguard statistics for weighted residual norms (Moju 2.0).

Effective scoring residual is ``max(weighted_R_eff, safeguard_stat)``, which is
equivalent to ``min(score(weighted_R_eff), score(safeguard_stat))`` under the
standard ``1/(1+R/scale)`` map.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp

from moju.monitor.weighted_norms import weighted_rms

# Default safeguard: 95th percentile (percentile comparison in
# docs/phase_a_fd_impact_report.md; configurable via safeguard_percentile).
DEFAULT_SAFEGUARD = "percentile"
DEFAULT_PERCENTILE = 95.0
DEFAULT_TILES: Tuple[int, int] = (4, 4)
DEFAULT_BOUNDARY_FRAC = 0.1
DEFAULT_MIN_REGION_FRAC = 0.05
DEFAULT_MIN_REGION_ABS = 16


def residual_magnitude(arr: Any) -> jnp.ndarray:
    a = jnp.asarray(arr)
    if a.ndim >= 1 and a.shape[-1] in (2, 3) and a.ndim >= 2:
        return jnp.sqrt(jnp.sum(jnp.square(a), axis=-1) + 1e-30)
    return jnp.abs(a)


def percentile_stat(arr: Any, q: float = DEFAULT_PERCENTILE) -> float:
    mag = residual_magnitude(arr)
    return float(jnp.percentile(jnp.reshape(mag, (-1,)), float(q)))


def smooth_error_field(
    eps: Any,
    *,
    radius: int = 1,
) -> Any:
    """
    Local median smoother for a pointwise error field (2D or 1D).

    Isolates sharp residual spikes from being fully excused by a coincidentally
    large collocated ε sample. ``radius`` is the half-width of the neighbourhood
    in index space (default 1 → 3×3 in 2D / 3-point in 1D).
    """
    e = jnp.abs(jnp.asarray(eps, dtype=float))
    r = max(0, int(radius))
    if r == 0:
        return e
    if e.ndim == 1:
        n = int(e.shape[0])
        pads = []
        for k in range(-r, r + 1):
            idx = jnp.clip(jnp.arange(n) + k, 0, n - 1)
            pads.append(e[idx])
        return jnp.median(jnp.stack(pads, axis=0), axis=0)
    if e.ndim == 2:
        n0, n1 = int(e.shape[0]), int(e.shape[1])
        pads = []
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ii = jnp.clip(jnp.arange(n0)[:, None] + di, 0, n0 - 1)
                jj = jnp.clip(jnp.arange(n1)[None, :] + dj, 0, n1 - 1)
                pads.append(e[ii, jj])
        return jnp.median(jnp.stack(pads, axis=0), axis=0)
    # Higher-D: flatten-safe no-op smooth
    return e


def error_aware_percentile_stat(
    residual: Any,
    deriv_error_field: Any,
    q: float = DEFAULT_PERCENTILE,
    *,
    scale_k: float = 1.0,
    delta_frac: float = 1e-12,
    smooth_radius: int = 0,
) -> Dict[str, float]:
    """
    Percentile safeguard on residuals normalized by a pointwise derivative-error estimate.

    Definition (Phase A4, opt-in)::

        ε_i = |e_i|          # pointwise error field (e.g. |r_hi - r_lo|)
        ε̃_i = local_median(ε) if smooth_radius > 0 else ε
        δ   = max(delta_frac * scale_k, 1e-30)
        ρ_i = |r_i| / (ε̃_i + δ)
        R_sg = percentile(ρ, q) * median(ε̃ + δ)

    Scoring still uses ``max(weighted_R_eff, R_sg)`` so the statistic stays in residual
    units. Points with |r| ≈ ε contribute ρ ≈ 1 and do not inflate the percentile.
    """
    mag = residual_magnitude(residual)
    eps = jnp.abs(jnp.asarray(deriv_error_field, dtype=float))
    if mag.shape != eps.shape:
        # Broadcast / reshape when possible
        eps = jnp.reshape(eps, mag.shape)
    if int(smooth_radius) > 0:
        eps = smooth_error_field(eps, radius=int(smooth_radius))
    delta = max(float(delta_frac) * max(float(scale_k), 0.0), 1e-30)
    denom = eps + delta
    rho = mag / denom
    sg_rho = float(jnp.percentile(jnp.reshape(rho, (-1,)), float(q)))
    eps_ref = float(jnp.median(jnp.reshape(denom, (-1,))))
    return {
        "stat": sg_rho * eps_ref,
        "percentile_rho": sg_rho,
        "eps_ref": eps_ref,
        "delta": delta,
        "percentile": float(q),
        "smooth_radius": int(smooth_radius),
    }


def max_stat(arr: Any) -> float:
    return float(jnp.max(residual_magnitude(arr)))


def worst_region_weighted_rms(
    arr: Any,
    weights: Any,
    *,
    tiles: Tuple[int, int] = DEFAULT_TILES,
    boundary_frac: float = DEFAULT_BOUNDARY_FRAC,
    min_region_frac: float = DEFAULT_MIN_REGION_FRAC,
    min_region_abs: int = DEFAULT_MIN_REGION_ABS,
) -> Dict[str, Any]:
    """Worst area-weighted RMS over tiles and boundary bands (2D spatial)."""
    mag = residual_magnitude(arr)
    w = jnp.asarray(weights, dtype=float)
    if mag.ndim == 1:
        # 1D: split into tiles along the line + end bands
        n = int(mag.shape[0])
        min_n = max(int(min_region_abs), int(math.ceil(float(min_region_frac) * n)))
        t0 = int(tiles[0])
        best = -1.0
        best_name = None
        regions = []
        for i in range(t0):
            a0, a1 = (i * n) // t0, ((i + 1) * n) // t0
            mask = jnp.zeros((n,), dtype=bool).at[a0:a1].set(True)
            if int(jnp.sum(mask)) < min_n:
                continue
            val = float(weighted_rms(mag, jnp.where(mask, w, 0.0)))
            regions.append({"name": f"tile_{i}", "n": int(jnp.sum(mask)), "weighted_rms": val})
            if val > best:
                best, best_name = val, f"tile_{i}"
        b = max(1, int(math.ceil(boundary_frac * n)))
        for name, sl in (("bound_lo", slice(0, b)), ("bound_hi", slice(n - b, n))):
            mask = jnp.zeros((n,), dtype=bool).at[sl].set(True)
            if int(jnp.sum(mask)) < min_n:
                continue
            val = float(weighted_rms(mag, jnp.where(mask, w, 0.0)))
            regions.append({"name": name, "n": int(jnp.sum(mask)), "weighted_rms": val})
            if val > best:
                best, best_name = val, name
        return {
            "worst_region": best_name,
            "worst_weighted_rms": best if best_name else float("nan"),
            "min_region_points": min_n,
            "regions": regions,
        }

    if mag.ndim != 2:
        # Fall back to global weighted RMS for higher-D / odd shapes
        val = float(weighted_rms(mag, w))
        return {
            "worst_region": "global",
            "worst_weighted_rms": val,
            "min_region_points": int(mag.size),
            "regions": [{"name": "global", "n": int(mag.size), "weighted_rms": val}],
        }

    n0, n1 = int(mag.shape[0]), int(mag.shape[1])
    n_tot = n0 * n1
    min_n = max(int(min_region_abs), int(math.ceil(float(min_region_frac) * n_tot)))
    t0, t1 = int(tiles[0]), int(tiles[1])
    best = -1.0
    best_name = None
    regions: List[Dict[str, Any]] = []
    for i in range(t0):
        i0, i1 = (i * n0) // t0, ((i + 1) * n0) // t0
        for j in range(t1):
            j0, j1 = (j * n1) // t1, ((j + 1) * n1) // t1
            mask = jnp.zeros((n0, n1), dtype=bool).at[i0:i1, j0:j1].set(True)
            if int(jnp.sum(mask)) < min_n:
                continue
            val = float(weighted_rms(mag, jnp.where(mask, w, 0.0)))
            regions.append(
                {"name": f"tile_{i}_{j}", "n": int(jnp.sum(mask)), "weighted_rms": val}
            )
            if val > best:
                best, best_name = val, f"tile_{i}_{j}"
    b0 = max(1, int(math.ceil(boundary_frac * n0)))
    b1 = max(1, int(math.ceil(boundary_frac * n1)))
    for name, sl in (("bound_lo_0", slice(0, b0)), ("bound_hi_0", slice(n0 - b0, n0))):
        mask = jnp.zeros((n0, n1), dtype=bool).at[sl, :].set(True)
        if int(jnp.sum(mask)) < min_n:
            continue
        val = float(weighted_rms(mag, jnp.where(mask, w, 0.0)))
        regions.append({"name": name, "n": int(jnp.sum(mask)), "weighted_rms": val})
        if val > best:
            best, best_name = val, name
    for name, sl in (("bound_lo_1", slice(0, b1)), ("bound_hi_1", slice(n1 - b1, n1))):
        mask = jnp.zeros((n0, n1), dtype=bool).at[:, sl].set(True)
        if int(jnp.sum(mask)) < min_n:
            continue
        val = float(weighted_rms(mag, jnp.where(mask, w, 0.0)))
        regions.append({"name": name, "n": int(jnp.sum(mask)), "weighted_rms": val})
        if val > best:
            best, best_name = val, name
    return {
        "worst_region": best_name,
        "worst_weighted_rms": best if best_name else float("nan"),
        "min_region_points": min_n,
        "regions": regions,
        "params": {
            "tiles": list(tiles),
            "boundary_frac": boundary_frac,
            "min_region_frac": min_region_frac,
            "min_region_abs": min_region_abs,
        },
    }


def compute_safeguard_stat(
    arr: Any,
    weights: Any,
    *,
    method: str = DEFAULT_SAFEGUARD,
    percentile: float = DEFAULT_PERCENTILE,
    tiles: Tuple[int, int] = DEFAULT_TILES,
    boundary_frac: float = DEFAULT_BOUNDARY_FRAC,
    min_region_frac: float = DEFAULT_MIN_REGION_FRAC,
    min_region_abs: int = DEFAULT_MIN_REGION_ABS,
    error_aware: bool = False,
    deriv_error_field: Any = None,
    scale_k: float = 1.0,
    delta_frac: float = 1e-12,
    smooth_radius: int = 0,
) -> Dict[str, Any]:
    """
    Return ``{"stat": float, "method": str, ...}`` for one residual array.

    When ``error_aware=True`` and ``deriv_error_field`` is provided with a percentile
    method, uses :func:`error_aware_percentile_stat` (opt-in; default off).
    """
    method = (method or DEFAULT_SAFEGUARD).strip().lower()
    if method in ("max", "residual_max"):
        return {"stat": max_stat(arr), "method": "max"}
    if method in ("percentile", "p", "pct"):
        if error_aware and deriv_error_field is not None:
            ea = error_aware_percentile_stat(
                arr,
                deriv_error_field,
                percentile,
                scale_k=scale_k,
                delta_frac=delta_frac,
                smooth_radius=smooth_radius,
            )
            return {
                "stat": float(ea["stat"]),
                "method": "percentile_error_aware",
                "percentile": float(percentile),
                "percentile_rho": ea["percentile_rho"],
                "eps_ref": ea["eps_ref"],
                "delta": ea["delta"],
                "smooth_radius": ea["smooth_radius"],
            }
        return {
            "stat": percentile_stat(arr, percentile),
            "method": "percentile",
            "percentile": float(percentile),
        }
    if method in ("worst_region", "region", "tiles"):
        detail = worst_region_weighted_rms(
            arr,
            weights,
            tiles=tiles,
            boundary_frac=boundary_frac,
            min_region_frac=min_region_frac,
            min_region_abs=min_region_abs,
        )
        return {
            "stat": float(detail["worst_weighted_rms"]),
            "method": "worst_region",
            "detail": detail,
        }
    raise ValueError(
        "safeguard method must be 'percentile', 'max', or 'worst_region', "
        f"got {method!r}"
    )


def combine_weighted_and_safeguard(
    weighted_r_eff: float, safeguard_stat: float
) -> float:
    """Effective residual for scoring: max(weighted R_eff, safeguard)."""
    return float(max(float(weighted_r_eff), float(safeguard_stat)))


def _score_r(stat: float, scale_k: float) -> float:
    if not math.isfinite(stat) or not math.isfinite(scale_k) or scale_k <= 0:
        return float("nan")
    return 1.0 / (1.0 + float(stat) / float(scale_k))


def boundary_layer_mask(shape: Tuple[int, ...], width: int) -> Any:
    """Boolean mask for the outer ``width`` index rows/cols of a 2D array."""
    if len(shape) < 2:
        raise ValueError("boundary_layer_mask expects at least 2D shape")
    n0, n1 = int(shape[0]), int(shape[1])
    w = max(1, int(width))
    mask = jnp.zeros((n0, n1), dtype=bool)
    mask = mask.at[:w, :].set(True)
    mask = mask.at[n0 - w :, :].set(True)
    mask = mask.at[:, :w].set(True)
    mask = mask.at[:, n1 - w :].set(True)
    return mask


def boundary_spike_stats(
    mag: Any,
    scale_k: float,
    *,
    fd_order: int = 4,
    high_threshold: float,
) -> Dict[str, Any]:
    """
    Points with |r| above the High-tier residual threshold inside FD boundary layers.

    ``high_threshold`` is the admissibility score cutoff (e.g. ``ADM_HIGH_THRESHOLD``);
    ``r_high = scale_k * (1/high_threshold - 1)``.
    """
    m = residual_magnitude(mag)
    if m.ndim != 2:
        return {
            "fd_order": int(fd_order),
            "frac_boundary_spikes": 0.0,
            "n_boundary_spikes": 0,
            "n_total": int(m.size),
            "applicable": False,
        }
    r_high = (
        float(scale_k) * (1.0 / float(high_threshold) - 1.0)
        if scale_k > 0 and high_threshold > 0
        else float("inf")
    )
    spike = m > r_high
    width = max(1, (int(fd_order) + 1) // 2)
    bmask = boundary_layer_mask((int(m.shape[0]), int(m.shape[1])), width)
    n_tot = int(m.size)
    n_bound = int(jnp.sum(bmask))
    n_bspike = int(jnp.sum(spike & bmask))
    return {
        "fd_order": int(fd_order),
        "boundary_stencil_width": width,
        "r_high": r_high,
        "frac_above_r_high": float(jnp.mean(spike.astype(float))),
        "frac_boundary_spikes": n_bspike / max(n_tot, 1),
        "frac_of_boundary_layer_spiking": n_bspike / max(n_bound, 1),
        "n_boundary_spikes": n_bspike,
        "n_boundary_layer": n_bound,
        "n_total": n_tot,
        "applicable": True,
    }


# Warn when ≥ this fraction of all points are High-threshold spikes in the FD boundary band.
DEFAULT_BOUNDARY_SPIKE_WARN_FRAC = 0.02


def build_safeguard_diagnostic_warnings(
    residual: Any,
    weights: Any,
    *,
    scale_k: float,
    scoring_r: float,
    key: str,
    high_threshold: float,
    fd_order: int = 4,
    boundary_spike_warn_frac: float = DEFAULT_BOUNDARY_SPIKE_WARN_FRAC,
    derivative_error_flagged: bool = False,
    resolution_sensor_warned: bool = False,
) -> List[str]:
    """
    Diagnostic warnings when percentile (or other) safeguards mask hot spots.

    Never changes tiers. Emits warnings when residual max or worst-region weighted
    RMS would drop the score below High while the safeguarded scoring residual
    keeps the score at High. Also warns on material boundary-stencil spike
    fractions, noting coincidence with derivative-error / resolution-sensor flags.
    """
    warnings: List[str] = []
    sk = float(scale_k)
    score_sg = _score_r(float(scoring_r), sk)
    if not math.isfinite(score_sg):
        return warnings

    r_max = max_stat(residual)
    score_max = _score_r(r_max, sk)
    wr = worst_region_weighted_rms(residual, weights)
    r_wr = float(wr.get("worst_weighted_rms", float("nan")))
    score_wr = _score_r(r_wr, sk)
    region_name = wr.get("worst_region")

    sg_keeps_high = score_sg >= float(high_threshold)
    if sg_keeps_high and math.isfinite(score_max) and score_max < float(high_threshold):
        warnings.append(
            f"safeguard_diagnostic: key {key!r} residual max would drop below High "
            f"(score={score_max:.4f}) while the safeguarded score stays High "
            f"(score={score_sg:.4f}). Inspect the hot spot or use eval_mask / "
            f"region scoring for critical subdomains."
        )
    if (
        sg_keeps_high
        and math.isfinite(score_wr)
        and score_wr < float(high_threshold)
        and region_name is not None
    ):
        warnings.append(
            f"safeguard_diagnostic: key {key!r} worst-region {region_name!r} "
            f"would drop below High (score={score_wr:.4f}) while the safeguarded "
            f"score stays High (score={score_sg:.4f}). Consider eval_mask or "
            f"region inspection for that tile/boundary band."
        )

    spikes = boundary_spike_stats(
        residual, sk, fd_order=fd_order, high_threshold=high_threshold
    )
    if spikes.get("applicable") and float(spikes["frac_boundary_spikes"]) >= float(
        boundary_spike_warn_frac
    ):
        msg = (
            f"safeguard_diagnostic: key {key!r} boundary-stencil spike fraction "
            f"{spikes['frac_boundary_spikes']:.3f} "
            f"(width={spikes['boundary_stencil_width']}, fd_order={fd_order}). "
            f"On coarse bounded grids, wall FD truncation can approach the p95 "
            f"tail; prefer a finer grid, supplied derivatives, or a boundary-band "
            f"eval_mask."
        )
        coincidences = []
        if derivative_error_flagged:
            coincidences.append("derivative-error flag")
        if resolution_sensor_warned:
            coincidences.append("resolution-sensor warning")
        if coincidences:
            msg += " Coincides with " + " and ".join(coincidences) + "."
        warnings.append(msg)
    return warnings
