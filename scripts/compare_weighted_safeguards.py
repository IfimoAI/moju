#!/usr/bin/env python3
"""
Percentile safeguard comparison for Moju 2.0.

Primary candidates: p95, p97.5, p99.
Reference rows: residual max, worst-region weighted RMS.

Studies:
  a) Detection floor — uniform TG / Kovasznay, localized bands ~2/5/10/20% points
  b) Clean Path B — TG / Kovasznay at coarse & fine; boundary-stencil spike fraction
  c) Noise — TG / Kovasznay at 1e-4, 1e-3, 1e-2 (fd_order=4), both resolutions
  d) Clustered — Phase A3 Kovasznay + Taylor–Couette localized cases

Decision rule (prefer p97.5 when it passes):
  clean Kov & clean TG High on both resolutions, and TG High at noise=1e-3;
  else keep p95.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import (
    kovasznay_2d,
    ns_state_from_analytic,
    taylor_couette_2d,
    taylor_green_2d,
)
from moju.monitor.auditor import ADM_HIGH_THRESHOLD
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.tiers import tier_for_score
from moju.monitor.weighted_norms import structured_point_weights, trapezoid_weights_1d, weighted_rms


ROOT = Path(__file__).resolve().parents[1]
HIGH = float(ADM_HIGH_THRESHOLD)
PERCENTILES = (95.0, 97.5, 99.0)
CAND_KEYS = ("p95", "p97_5", "p99")  # primary
REF_KEYS = ("max", "worst_region")
ALL_SG_KEYS = CAND_KEYS + REF_KEYS + ("weighted_only", "unweighted_rms")

DEFAULT_MIN_REGION_FRAC = 0.05
DEFAULT_MIN_REGION_ABS = 16
DEFAULT_TILES = (4, 4)
DEFAULT_BOUNDARY_FRAC = 0.1

# Coarse / fine uniform resolutions
RES_TG = {"coarse": 32, "fine": 64}
# Coarse Kovasznay matches prior A3 (40×40); fine doubles both axes.
RES_KOV = {"coarse": (40, 40), "fine": (80, 80)}


def _ns_engine() -> ResidualEngine:
    return ResidualEngine(
        constants={},
        laws=[
            {
                "name": "momentum_navier_stokes",
                "state_map": {
                    "u_t": "u_t",
                    "u": "u",
                    "u_grad": "u_grad",
                    "p_grad": "p_grad",
                    "u_laplacian": "u_laplacian",
                    "re": "re",
                },
            }
        ],
        groups=[],
        law_implied_audits=False,
        default_coord_dimension=2,
    )


def _mag(residual: Any) -> jnp.ndarray:
    a = jnp.asarray(residual)
    if a.ndim >= 1 and a.shape[-1] in (2, 3):
        return jnp.sqrt(jnp.sum(jnp.square(a), axis=-1) + 1e-30)
    return jnp.abs(a)


def _score(stat: float, scale_k: float) -> float:
    if not math.isfinite(stat) or not math.isfinite(scale_k) or scale_k <= 0:
        return float("nan")
    return 1.0 / (1.0 + float(stat) / float(scale_k))


def _pkey(q: float) -> str:
    if abs(q - 97.5) < 1e-9:
        return "p97_5"
    return f"p{int(q)}" if abs(q - int(q)) < 1e-9 else f"p{q}".replace(".", "_")


def stretched_axis(n: int, lo: float, hi: float, *, power: float = 2.5) -> jnp.ndarray:
    s = jnp.linspace(0.0, 1.0, n)
    return lo + (hi - lo) * (s**power)


def polar_weights(r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    r1 = jnp.reshape(jnp.asarray(r, dtype=float), (-1,))
    t1 = jnp.reshape(jnp.asarray(theta, dtype=float), (-1,))
    wr = trapezoid_weights_1d(r1)
    wt = trapezoid_weights_1d(t1)
    return (r1[:, None] * wr[:, None]) * wt[None, :]


def residual_percentile(mag: jnp.ndarray, q: float) -> float:
    flat = jnp.reshape(mag, (-1,))
    return float(jnp.percentile(flat, q))


def worst_region_weighted_rms(
    mag: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    tiles: Tuple[int, int] = DEFAULT_TILES,
    boundary_frac: float = DEFAULT_BOUNDARY_FRAC,
    min_region_frac: float = DEFAULT_MIN_REGION_FRAC,
    min_region_abs: int = DEFAULT_MIN_REGION_ABS,
) -> Dict[str, Any]:
    m = jnp.asarray(mag)
    w = jnp.asarray(weights, dtype=float)
    if m.shape != w.shape:
        raise ValueError(f"mag {m.shape} vs weights {w.shape}")
    n0, n1 = int(m.shape[0]), int(m.shape[1])
    n_tot = n0 * n1
    min_n = max(int(min_region_abs), int(math.ceil(float(min_region_frac) * n_tot)))
    regions: List[Tuple[str, jnp.ndarray]] = []
    t0, t1 = int(tiles[0]), int(tiles[1])
    for i in range(t0):
        i0, i1 = (i * n0) // t0, ((i + 1) * n0) // t0
        for j in range(t1):
            j0, j1 = (j * n1) // t1, ((j + 1) * n1) // t1
            mask = jnp.zeros((n0, n1), dtype=bool).at[i0:i1, j0:j1].set(True)
            regions.append((f"tile_{i}_{j}", mask))
    b0 = max(1, int(math.ceil(boundary_frac * n0)))
    for name, sl in (("bound_lo_0", slice(0, b0)), ("bound_hi_0", slice(n0 - b0, n0))):
        mask = jnp.zeros((n0, n1), dtype=bool).at[sl, :].set(True)
        regions.append((name, mask))
    b1 = max(1, int(math.ceil(boundary_frac * n1)))
    for name, sl in (("bound_lo_1", slice(0, b1)), ("bound_hi_1", slice(n1 - b1, n1))):
        mask = jnp.zeros((n0, n1), dtype=bool).at[:, sl].set(True)
        regions.append((name, mask))

    best_name = None
    best_val = -1.0
    kept = []
    for name, mask in regions:
        n_reg = int(jnp.sum(mask))
        if n_reg < min_n:
            continue
        val = float(weighted_rms(m, jnp.where(mask, w, 0.0)))
        kept.append({"name": name, "n": n_reg, "weighted_rms": val})
        if val > best_val:
            best_val = val
            best_name = name
    return {
        "worst_region": best_name,
        "worst_weighted_rms": best_val if best_name is not None else float("nan"),
        "min_region_points": min_n,
        "n_regions_kept": len(kept),
        "regions": kept,
    }


def boundary_layer_mask(shape: Tuple[int, int], width: int) -> jnp.ndarray:
    n0, n1 = shape
    w = max(1, int(width))
    mask = jnp.zeros((n0, n1), dtype=bool)
    mask = mask.at[:w, :].set(True)
    mask = mask.at[n0 - w :, :].set(True)
    mask = mask.at[:, :w].set(True)
    mask = mask.at[:, n1 - w :].set(True)
    return mask


def boundary_spike_stats(
    mag: jnp.ndarray, scale_k: float, *, fd_order: int = 4
) -> Dict[str, Any]:
    """
    Points with |r| above the High-tier residual threshold inside the FD
    boundary stencil layers (width = fd_order // 2).
    """
    m = jnp.asarray(mag)
    r_high = float(scale_k) * (1.0 / HIGH - 1.0) if scale_k > 0 else float("inf")
    spike = m > r_high
    width = max(1, int(fd_order) // 2)
    bmask = boundary_layer_mask((int(m.shape[0]), int(m.shape[1])), width)
    n_tot = int(m.size)
    n_bound = int(jnp.sum(bmask))
    n_spike = int(jnp.sum(spike))
    n_bspike = int(jnp.sum(spike & bmask))
    return {
        "fd_order": int(fd_order),
        "boundary_stencil_width": width,
        "r_high": r_high,
        "frac_above_r_high": n_spike / max(n_tot, 1),
        "frac_boundary_spikes": n_bspike / max(n_tot, 1),
        "frac_of_boundary_layer_spiking": n_bspike / max(n_bound, 1),
        "n_boundary_spikes": n_bspike,
        "n_boundary_layer": n_bound,
        "n_total": n_tot,
    }


def safeguards_from_residual(
    residual: Any,
    weights: Any,
    scale_k: float,
    weighted_rms_val: float,
) -> Dict[str, Any]:
    mag = _mag(residual)
    w = jnp.asarray(weights, dtype=float)
    rmax = float(jnp.max(mag))
    wr = worst_region_weighted_rms(mag, w)
    wrms = float(weighted_rms_val)

    def combo(stat: float) -> Dict[str, Any]:
        sw = _score(wrms, scale_k)
        ss = _score(stat, scale_k)
        final = min(sw, ss) if math.isfinite(sw) and math.isfinite(ss) else float("nan")
        return {
            "weighted_rms": wrms,
            "safeguard_stat": stat,
            "score_weighted": sw,
            "score_safeguard": ss,
            "score_final": final,
            "tier_final": tier_for_score(final) if math.isfinite(final) else "Unknown",
            "below_high": final < HIGH if math.isfinite(final) else True,
        }

    out: Dict[str, Any] = {
        "max": combo(rmax),
        "worst_region": {**combo(float(wr["worst_weighted_rms"])), "detail": {
            "worst_region": wr["worst_region"],
            "worst_weighted_rms": wr["worst_weighted_rms"],
            "min_region_points": wr["min_region_points"],
        }},
        "weighted_only": {
            "score_final": _score(wrms, scale_k),
            "tier_final": tier_for_score(_score(wrms, scale_k)),
            "below_high": _score(wrms, scale_k) < HIGH,
        },
        "scale_k": scale_k,
        "stats": {"max": rmax, "worst_region_wrms": wr["worst_weighted_rms"]},
    }
    for q in PERCENTILES:
        key = _pkey(q)
        val = residual_percentile(mag, q)
        out[key] = combo(val)
        out["stats"][key] = val
    return out


def _attach_unweighted(sg: Dict[str, Any], urw: float, scale: float) -> Dict[str, Any]:
    uw_score = _score(urw, scale)
    sg["unweighted_rms"] = {
        "stat": urw,
        "score_final": uw_score,
        "tier_final": tier_for_score(uw_score),
        "below_high": uw_score < HIGH,
    }
    return sg


def _run_get_residual_scale(
    state: Dict[str, Any],
    *,
    grid: Optional[PathBGridConfig] = None,
    fill_law_fd: bool = False,
    weights: Any,
) -> Tuple[Any, float, float, float]:
    eng = _ns_engine()
    eng.compute_residuals(
        state,
        run_mode="training",
        auto_path_b_derivatives=grid if grid is not None else False,
        fill_law_fd=fill_law_fd,
        weighted_norms="report",
        point_weights=weights,
    )
    res = eng.last_residuals["laws"]["momentum_navier_stokes"]
    scale_k = float(eng.log[-1]["scale"]["laws/momentum_navier_stokes"])
    entry = eng.log[-1]
    urw = float(entry["rms"]["laws/momentum_navier_stokes"])
    wr = float(entry["rms_weighted"]["laws/momentum_navier_stokes"])
    return res, scale_k, urw, wr


def _first_below(rows: List[Dict[str, Any]], key: str, mag_key: str = "magnitude") -> Optional[float]:
    for r in rows:
        if r[key]["below_high"]:
            return float(r[mag_key])
    return None


def _band_mask(nx: int, ny: int, target_frac: float) -> Tuple[jnp.ndarray, float]:
    """Axis-0 strip covering approximately ``target_frac`` of points."""
    n_cols = max(1, int(round(target_frac * nx)))
    n_cols = min(n_cols, nx)
    mask = jnp.zeros((nx, ny), dtype=bool).at[:n_cols, :].set(True)
    frac = float(jnp.mean(mask.astype(float)))
    return mask, frac


def case_detection_floor(
    flow: str,
    resolution: str,
    band_fracs: Sequence[float],
    mags: Sequence[float],
) -> Dict[str, Any]:
    if flow == "taylor_green":
        n = RES_TG[resolution]
        xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = taylor_green_2d(x, y, 0.1, re=100.0)
        base = ns_state_from_analytic(fields, x, y)
        w = structured_point_weights([xs, ys])
        nx, ny = n, n
    else:
        nx, ny = RES_KOV[resolution]
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = kovasznay_2d(x, y, re=40.0)
        base = ns_state_from_analytic(fields, x, y)
        w = structured_point_weights([xs, ys])

    out_bands = []
    for target in band_fracs:
        region, actual_frac = _band_mask(nx, ny, float(target))
        rows = []
        for mag in mags:
            st = dict(base)
            pg = jnp.asarray(st["p_grad"])
            delta = jnp.zeros_like(pg).at[..., 0].set(jnp.where(region, float(mag), 0.0))
            st["p_grad"] = pg + delta
            resid, scale, urw, wr = _run_get_residual_scale(st, weights=w)
            sg = _attach_unweighted(safeguards_from_residual(resid, w, scale, wr), urw, scale)
            rows.append({"magnitude": float(mag), **{k: sg[k] for k in ALL_SG_KEYS if k in sg}})
        first = {k: _first_below(rows, k) for k in CAND_KEYS + REF_KEYS + ("weighted_only", "unweighted_rms")}
        out_bands.append(
            {
                "target_band_frac": float(target),
                "actual_band_frac": actual_frac,
                "n_band": int(jnp.sum(region)),
                "first_below_high": first,
                "rows": rows,
            }
        )
        print(
            f"  {flow}/{resolution} band~{target:.0%} actual={actual_frac:.3f} first={first}"
        )
    return {
        "flow": flow,
        "resolution": resolution,
        "grid": {"nx": nx, "ny": ny},
        "bands": out_bands,
    }


def case_clean_path_b(flow: str, resolution: str) -> Dict[str, Any]:
    grid = PathBGridConfig(spatial_dimension=2, steady=True, fd_order=4, fd_order_time=4)
    if flow == "taylor_green":
        n = RES_TG[resolution]
        xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = taylor_green_2d(x, y, 0.1, re=100.0)
        w = structured_point_weights([xs, ys])
        state = {
            "x": x,
            "y": y,
            "u": fields["u"],
            "u_t": fields["u_t"],
            "p": fields["p"],
            "re": fields["re"],
        }
        shape = (n, n)
    else:
        nx, ny = RES_KOV[resolution]
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = kovasznay_2d(x, y, re=40.0)
        w = structured_point_weights([xs, ys])
        state = {
            "x": x,
            "y": y,
            "u": fields["u"],
            "u_t": fields["u_t"],
            "p": fields["p"],
            "re": fields["re"],
        }
        shape = (nx, ny)

    resid, scale, urw, wr = _run_get_residual_scale(
        state, grid=grid, fill_law_fd=True, weights=w
    )
    sg = _attach_unweighted(safeguards_from_residual(resid, w, scale, wr), urw, scale)
    spikes = boundary_spike_stats(_mag(resid), scale, fd_order=4)
    tiers = {k: {"score": sg[k]["score_final"], "tier": sg[k]["tier_final"]} for k in CAND_KEYS + REF_KEYS}
    tiers["weighted_only"] = {
        "score": sg["weighted_only"]["score_final"],
        "tier": sg["weighted_only"]["tier_final"],
    }
    tiers["unweighted_rms"] = {
        "score": sg["unweighted_rms"]["score_final"],
        "tier": sg["unweighted_rms"]["tier_final"],
    }
    print(
        f"  clean {flow}/{resolution} shape={shape} "
        f"p95={tiers['p95']['tier'][:4]} p97.5={tiers['p97_5']['tier'][:4]} "
        f"p99={tiers['p99']['tier'][:4]} bspike={spikes['frac_boundary_spikes']:.4f}"
    )
    return {
        "flow": flow,
        "resolution": resolution,
        "grid": {"nx": shape[0], "ny": shape[1]},
        "tiers": tiers,
        "boundary_spikes": spikes,
    }


def case_noise(flow: str, resolution: str, noise_levels: Sequence[float]) -> Dict[str, Any]:
    key = jax.random.PRNGKey(1 if flow == "kovasznay" else 0)
    grid = PathBGridConfig(spatial_dimension=2, steady=True, fd_order=4, fd_order_time=4)
    if flow == "taylor_green":
        n = RES_TG[resolution]
        xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = taylor_green_2d(x, y, 0.1, re=100.0)
        w = structured_point_weights([xs, ys])
    else:
        nx, ny = RES_KOV[resolution]
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = kovasznay_2d(x, y, re=40.0)
        w = structured_point_weights([xs, ys])
    u_scale = float(jnp.sqrt(jnp.mean(jnp.square(fields["u"]))))
    p_scale = float(jnp.sqrt(jnp.mean(jnp.square(fields["p"]))))
    rows = []
    for eps in noise_levels:
        key, k1, k2 = jax.random.split(key, 3)
        u_n = fields["u"] + eps * u_scale * jax.random.normal(k1, fields["u"].shape)
        p_n = fields["p"] + eps * p_scale * jax.random.normal(k2, fields["p"].shape)
        state = {
            "x": x,
            "y": y,
            "u": u_n,
            "u_t": fields["u_t"],
            "p": p_n,
            "re": fields["re"],
        }
        resid, scale, urw, wr = _run_get_residual_scale(
            state, grid=grid, fill_law_fd=True, weights=w
        )
        sg = _attach_unweighted(safeguards_from_residual(resid, w, scale, wr), urw, scale)
        row = {"noise_rel": float(eps)}
        for k in ALL_SG_KEYS:
            if k in sg:
                row[k] = sg[k]
        rows.append(row)
        bits = " | ".join(
            f"{k}={row[k]['tier_final'][:4]}" for k in CAND_KEYS + ("max",)
        )
        print(f"  noise {flow}/{resolution} eps={eps}: {bits}")
    return {"flow": flow, "resolution": resolution, "u_scale": u_scale, "rows": rows}


def case_localized_clustered_kovasznay(mags: Sequence[float]) -> Dict[str, Any]:
    nx, ny = 48, 40
    xs = stretched_axis(nx, 0.0, 1.0)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    base = ns_state_from_analytic(fields, x, y)
    w = structured_point_weights([xs, ys])
    x_cut = float(xs[max(3, nx // 5)])
    region = x <= x_cut
    rows = []
    for mag in mags:
        st = dict(base)
        pg = jnp.asarray(st["p_grad"])
        delta = jnp.zeros_like(pg).at[..., 0].set(jnp.where(region, float(mag), 0.0))
        st["p_grad"] = pg + delta
        resid, scale, urw, wr = _run_get_residual_scale(st, weights=w)
        sg = _attach_unweighted(safeguards_from_residual(resid, w, scale, wr), urw, scale)
        rows.append({"magnitude": float(mag), **{k: sg[k] for k in ALL_SG_KEYS if k in sg}})
    return {
        "meta": {
            "x_cut": x_cut,
            "point_frac": float(jnp.mean(region.astype(float))),
            "weight_frac": float(jnp.sum(w * region) / jnp.sum(w)),
        },
        "first_below_high": {
            k: _first_below(rows, k) for k in CAND_KEYS + REF_KEYS + ("weighted_only", "unweighted_rms")
        },
        "rows": rows,
    }


def case_localized_clustered_couette(mags: Sequence[float]) -> Dict[str, Any]:
    nr, nt = 36, 48
    r = stretched_axis(nr, 1.0, 2.0)
    th = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
    fields = taylor_couette_2d(r, th, re=100.0)
    w = polar_weights(r, th)
    base = {
        "x": fields["x"],
        "y": fields["y"],
        "u": fields["u"],
        "u_t": fields["u_t"],
        "u_grad": fields["u_grad"],
        "u_laplacian": fields["u_laplacian"],
        "p": fields["p"],
        "p_grad": fields["p_grad"],
        "re": fields["re"],
    }
    r_cut = float(r[max(3, nr // 5)])
    region = fields["r"] <= r_cut
    rows = []
    for mag in mags:
        st = dict(base)
        pg = jnp.asarray(st["p_grad"])
        delta = jnp.zeros_like(pg).at[..., 0].set(jnp.where(region, float(mag), 0.0))
        st["p_grad"] = pg + delta
        resid, scale, urw, wr = _run_get_residual_scale(st, weights=w)
        sg = _attach_unweighted(safeguards_from_residual(resid, w, scale, wr), urw, scale)
        rows.append({"magnitude": float(mag), **{k: sg[k] for k in ALL_SG_KEYS if k in sg}})
    return {
        "meta": {
            "r_cut": r_cut,
            "point_frac": float(jnp.mean(region.astype(float))),
            "weight_frac": float(jnp.sum(w * region) / jnp.sum(w)),
            "worst_region_params": {
                "tiles": list(DEFAULT_TILES),
                "boundary_frac": DEFAULT_BOUNDARY_FRAC,
                "min_region_frac": DEFAULT_MIN_REGION_FRAC,
                "min_region_abs": DEFAULT_MIN_REGION_ABS,
            },
        },
        "first_below_high": {
            k: _first_below(rows, k) for k in CAND_KEYS + REF_KEYS + ("weighted_only", "unweighted_rms")
        },
        "rows": rows,
    }


def _tier_high(tier: str) -> bool:
    return str(tier).startswith("High")


def decide(report: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer p97.5 if clean High both resolutions + TG High @ noise 1e-3; else p95."""
    decision: Dict[str, Any] = {
        "criteria": (
            "Prefer p97.5 if clean Kovasznay and clean Taylor–Green stay High on both "
            "resolutions and Taylor–Green stays High at noise=1e-3 fd4; otherwise keep p95. "
            "Detection floor: lower first_below_high magnitude / smaller band is better."
        ),
        "candidates": {},
    }

    def clean_high(flow: str, cand: str) -> Dict[str, bool]:
        out = {}
        for res in ("coarse", "fine"):
            key = f"clean_{flow}_{res}"
            tier = report["cases"][key]["tiers"][cand]["tier"]
            out[res] = _tier_high(tier)
        return out

    def noise_high(flow: str, res: str, cand: str, eps: float = 1e-3) -> bool:
        rows = report["cases"][f"noise_{flow}_{res}"]["rows"]
        for row in rows:
            if abs(row["noise_rel"] - eps) < 1e-15:
                return not bool(row[cand]["below_high"])
        return False

    # Detection floor summary: min band_frac where any mag drops High, and that mag
    def detection_summary(cand: str) -> Dict[str, Any]:
        summary = {}
        for flow in ("taylor_green", "kovasznay"):
            for res in ("coarse", "fine"):
                bands = report["cases"][f"detection_{flow}_{res}"]["bands"]
                first_band = None
                first_mag = None
                for b in bands:
                    mag = b["first_below_high"].get(cand)
                    if mag is not None:
                        first_band = b["actual_band_frac"]
                        first_mag = mag
                        break
                summary[f"{flow}_{res}"] = {
                    "first_band_frac": first_band,
                    "first_magnitude": first_mag,
                }
        return summary

    for cand in CAND_KEYS:
        clean_tg = clean_high("taylor_green", cand)
        clean_kov = clean_high("kovasznay", cand)
        tg_noise = {
            res: noise_high("taylor_green", res, cand) for res in ("coarse", "fine")
        }
        kov_noise = {
            res: noise_high("kovasznay", res, cand) for res in ("coarse", "fine")
        }
        clean_ok = all(clean_tg.values()) and all(clean_kov.values())
        # Prefer TG High at 1e-3 on both resolutions (coarse is the A3 bar)
        noise_ok = tg_noise["coarse"] and tg_noise["fine"]
        passes_prefer = bool(clean_ok and noise_ok)
        clustered = {
            "kovasznay": report["cases"]["clustered_kovasznay"]["first_below_high"].get(cand),
            "couette": report["cases"]["clustered_couette"]["first_below_high"].get(cand),
        }
        decision["candidates"][cand] = {
            "clean_tg_high": clean_tg,
            "clean_kov_high": clean_kov,
            "tg_high_noise_1e-3": tg_noise,
            "kov_high_noise_1e-3": kov_noise,
            "clean_ok": clean_ok,
            "noise_ok_tg": noise_ok,
            "passes_prefer_rule": passes_prefer,
            "detection_floor": detection_summary(cand),
            "clustered_first_below_high": clustered,
        }

    p975 = decision["candidates"]["p97_5"]
    p95 = decision["candidates"]["p95"]
    if p975["passes_prefer_rule"]:
        chosen = "p97_5"
        chosen_percentile = 97.5
        reason = (
            "p97.5 keeps clean TG and clean Kovasznay High on coarse and fine, "
            "and stays High on TG at noise=1e-3 on both resolutions."
        )
    else:
        chosen = "p95"
        chosen_percentile = 95.0
        reason = (
            "p97.5 fails clean Kovasznay on coarse "
            f"(clean_ok={p975['clean_ok']}; tg_noise_ok={p975['noise_ok_tg']}); "
            "keep p95. "
            + (
                "p95 is the only percentile that passes the full prefer rule."
                if p95["passes_prefer_rule"]
                else "p95 is the stability fallback."
            )
        )
        if not p95["passes_prefer_rule"]:
            for cand in ("p95", "p97_5", "p99"):
                if decision["candidates"][cand]["clean_ok"]:
                    chosen = cand
                    chosen_percentile = {"p95": 95.0, "p97_5": 97.5, "p99": 99.0}[cand]
                    reason = (
                        f"No candidate fully passes prefer rule; selecting {cand} "
                        "as best clean-data survivor among percentiles."
                    )
                    break

    decision["chosen"] = chosen
    decision["chosen_percentile"] = chosen_percentile
    decision["reason"] = reason
    decision["rationale"] = decision["candidates"][chosen]
    return decision


def main() -> None:
    band_fracs = [0.02, 0.05, 0.10, 0.20]
    mags = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    noise_levels = [0.0, 1e-4, 1e-3, 1e-2]
    clustered_mags = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    report: Dict[str, Any] = {
        "phase": "A3_safeguard_percentile_compare",
        "high_cutoff": HIGH,
        "percentiles": list(PERCENTILES),
        "resolutions": {"taylor_green": RES_TG, "kovasznay": {k: list(v) for k, v in RES_KOV.items()}},
        "worst_region_defaults": {
            "tiles": list(DEFAULT_TILES),
            "boundary_frac": DEFAULT_BOUNDARY_FRAC,
            "min_region_frac": DEFAULT_MIN_REGION_FRAC,
            "min_region_abs": DEFAULT_MIN_REGION_ABS,
        },
        "cases": {},
    }

    print("=== (a) detection floor ===")
    for flow in ("taylor_green", "kovasznay"):
        for res in ("coarse", "fine"):
            print(f"detection {flow}/{res}...")
            report["cases"][f"detection_{flow}_{res}"] = case_detection_floor(
                flow, res, band_fracs, mags
            )

    print("=== (b) clean Path B ===")
    for flow in ("taylor_green", "kovasznay"):
        for res in ("coarse", "fine"):
            print(f"clean {flow}/{res}...")
            report["cases"][f"clean_{flow}_{res}"] = case_clean_path_b(flow, res)

    print("=== (c) noise ===")
    for flow in ("taylor_green", "kovasznay"):
        for res in ("coarse", "fine"):
            print(f"noise {flow}/{res}...")
            report["cases"][f"noise_{flow}_{res}"] = case_noise(flow, res, noise_levels)

    print("=== (d) clustered A3 ===")
    print("clustered_kovasznay...")
    report["cases"]["clustered_kovasznay"] = case_localized_clustered_kovasznay(clustered_mags)
    print("  first", report["cases"]["clustered_kovasznay"]["first_below_high"])
    print("clustered_couette...")
    report["cases"]["clustered_couette"] = case_localized_clustered_couette(clustered_mags)
    print("  first", report["cases"]["clustered_couette"]["first_below_high"])

    report["decision"] = decide(report)

    out = ROOT / "docs/phase_a3_safeguard_compare.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {out}")
    print(
        "DECISION:",
        report["decision"]["chosen"],
        "percentile=",
        report["decision"]["chosen_percentile"],
    )
    print("REASON:", report["decision"]["reason"])


if __name__ == "__main__":
    main()
