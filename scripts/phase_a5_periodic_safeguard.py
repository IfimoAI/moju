#!/usr/bin/env python3
"""
Phase A5: periodic-axis FD + percentile rerun (before switching 2.0 defaults).

1) Verify non-periodic d2 / mixed boundary orders (poly-exp, Couette radial).
2) Rerun percentile comparison with Kovasznay y-periodic and Taylor–Green both-periodic.
3) Resolution-sensor grid-scale warning smoke check.
4) Decision rule (a) plain p97.5 → (b) error-aware p97.5 → (c) plain p95.
   Do NOT switch defaults — report and stop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from moju.monitor import ResidualEngine
from moju.monitor.analytic_flows import (
    kovasznay_2d,
    ns_state_from_analytic,
    taylor_couette_2d,
    taylor_green_2d,
)
from moju.monitor.auditor import ADM_HIGH_THRESHOLD
from moju.monitor.fd_stencils import diff2_uniform, mixed_diff_uniform
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.resolution_sensor import build_resolution_sensor_report
from moju.monitor.tiers import tier_for_score
from moju.monitor.weighted_norms import structured_point_weights, trapezoid_weights_1d, weighted_rms
from moju.monitor.weighted_safeguard import (
    error_aware_percentile_stat,
    residual_magnitude,
)


ROOT = Path(__file__).resolve().parents[1]
HIGH = float(ADM_HIGH_THRESHOLD)
PERCENTILES = (95.0, 97.5, 99.0)
RES_TG = {"coarse": 32, "fine": 64}
RES_KOV = {"coarse": (40, 40), "fine": (80, 80)}
NS_CONV = (24, 48, 96)


def _pkey(q: float) -> str:
    return "p97_5" if abs(q - 97.5) < 1e-9 else f"p{int(q)}"


def _score(stat: float, scale_k: float) -> float:
    if not math.isfinite(stat) or not math.isfinite(scale_k) or scale_k <= 0:
        return float("nan")
    return 1.0 / (1.0 + float(stat) / float(scale_k))


def _obs_order(e_c: float, e_f: float, ratio: float = 2.0) -> float:
    if e_c <= 0 or e_f <= 0:
        return float("nan")
    return math.log(e_c / e_f) / math.log(ratio)


def _ns_engine() -> ResidualEngine:
    return ResidualEngine(
        constants={},
        laws=[{
            "name": "momentum_navier_stokes",
            "state_map": {
                "u_t": "u_t", "u": "u", "u_grad": "u_grad",
                "p_grad": "p_grad", "u_laplacian": "u_laplacian", "re": "re",
            },
        }],
        groups=[],
        law_implied_audits=False,
        default_coord_dimension=2,
    )


def _grid_for(flow: str) -> PathBGridConfig:
    if flow == "taylor_green":
        return PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order=4, periodic=(True, True)
        )
    if flow == "kovasznay":
        return PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order=4, periodic=(False, True)
        )
    raise ValueError(flow)


def _run(state, weights, *, grid=None, fill=False):
    eng = _ns_engine()
    eng.compute_residuals(
        state,
        auto_path_b_derivatives=grid if grid is not None else False,
        fill_law_fd=fill,
        weighted_norms="report",
        point_weights=weights,
    )
    res = eng.last_residuals["laws"]["momentum_navier_stokes"]
    scale = float(eng.log[-1]["scale"]["laws/momentum_navier_stokes"])
    wr = float(eng.log[-1]["rms_weighted"]["laws/momentum_navier_stokes"])
    return res, scale, wr


def _combo(residual, weights, scale, wr, *, q=95.0, eps=None, error_aware=False):
    mag = residual_magnitude(residual)
    if error_aware and eps is not None:
        ea = error_aware_percentile_stat(residual, eps, q, scale_k=scale)
        sg = float(ea["stat"])
    else:
        sg = float(jnp.percentile(jnp.reshape(mag, (-1,)), q))
    final = min(_score(wr, scale), _score(sg, scale))
    return {
        "score": final,
        "tier": tier_for_score(final),
        "below_high": final < HIGH,
        "sg": sg,
    }


def boundary_spike_frac(mag, scale_k, *, fd_order=4) -> Dict[str, float]:
    m = residual_magnitude(mag)
    r_high = float(scale_k) * (1.0 / HIGH - 1.0)
    width = (fd_order + 1) // 2
    nx, ny = int(m.shape[0]), int(m.shape[1])
    mask = jnp.zeros((nx, ny), dtype=bool)
    mask = mask.at[:width, :].set(True)
    mask = mask.at[-width:, :].set(True)
    mask = mask.at[:, :width].set(True)
    mask = mask.at[:, -width:].set(True)
    spike = m > r_high
    n_tot = int(m.size)
    n_bspike = int(jnp.sum(spike & mask))
    return {
        "r_high": r_high,
        "frac_boundary_spikes": n_bspike / max(n_tot, 1),
        "frac_above_r_high": float(jnp.mean(spike.astype(float))),
    }


# ---------------------------------------------------------------------------
# 1) Closure order verification
# ---------------------------------------------------------------------------


def verify_d2_mixed_orders() -> Dict[str, Any]:
    half = 2
    out: Dict[str, Any] = {}

    # poly-exp on [0,1]
    rows = []
    for n in NS_CONV:
        x = jnp.linspace(0.0, 1.0, n)
        h = float(x[1] - x[0])
        f = jnp.exp(x) * (x**3)
        exact = jnp.exp(x) * (x**3 + 6.0 * x**2 + 6.0 * x)
        e = jnp.abs(diff2_uniform(f, h, order=4) - exact)
        band = jnp.concatenate([e[:half], e[-half:]])
        rows.append({
            "n": n,
            "boundary_rms": float(jnp.sqrt(jnp.mean(jnp.square(band)))),
            "interior_rms": float(jnp.sqrt(jnp.mean(jnp.square(e[half:-half])))),
        })
    out["poly_exp_d2"] = {
        "rows": rows,
        "obs_order_boundary": _obs_order(rows[-2]["boundary_rms"], rows[-1]["boundary_rms"]),
        "obs_order_interior": _obs_order(rows[-2]["interior_rms"], rows[-1]["interior_rms"]),
        "expected": 4.0,
        "pass": _obs_order(rows[-2]["boundary_rms"], rows[-1]["boundary_rms"]) > 3.5,
    }

    # mixed corners
    rows_m = []
    for n in NS_CONV:
        xs = jnp.linspace(0.0, 1.0, n)
        ys = jnp.linspace(0.0, 1.0, n)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        hx = float(xs[1] - xs[0])
        hy = float(ys[1] - ys[0])
        f = jnp.exp(x + y) * x * y
        exact = jnp.exp(x + y) * (x * y + x + y + 1.0)
        e = mixed_diff_uniform(f, hx, hy, order=4) - exact
        mask = jnp.zeros_like(e, dtype=bool)
        mask = mask.at[:half, :].set(True)
        mask = mask.at[-half:, :].set(True)
        mask = mask.at[:, :half].set(True)
        mask = mask.at[:, -half:].set(True)
        rows_m.append({
            "n": n,
            "boundary_rms": float(jnp.sqrt(jnp.mean(jnp.square(e[mask])))),
            "interior_rms": float(jnp.sqrt(jnp.mean(jnp.square(e[~mask])))),
        })
    out["mixed_poly_exp"] = {
        "rows": rows_m,
        "obs_order_boundary": _obs_order(rows_m[-2]["boundary_rms"], rows_m[-1]["boundary_rms"]),
        "obs_order_interior": _obs_order(rows_m[-2]["interior_rms"], rows_m[-1]["interior_rms"]),
        "expected": 4.0,
        "pass": _obs_order(rows_m[-2]["boundary_rms"], rows_m[-1]["boundary_rms"]) > 3.5,
    }

    # Taylor–Couette radial (1D circumferential mean of u_θ(r))
    rows_c = []
    for nr in NS_CONV:
        r = jnp.linspace(1.0, 2.0, nr)
        h = float(r[1] - r[0])
        # Couette: u_θ = A r + B / r  → d²u/dr² = 2 B / r³
        ri, ro, omega = 1.0, 2.0, 1.0
        A = -omega * (ro**2) / (ro**2 - ri**2)
        B = omega * (ri**2) * (ro**2) / (ro**2 - ri**2)
        u = A * r + B / r
        exact = 2.0 * B / (r**3)
        e = jnp.abs(diff2_uniform(u, h, order=4) - exact)
        band = jnp.concatenate([e[:half], e[-half:]])
        rows_c.append({
            "n": nr,
            "boundary_rms": float(jnp.sqrt(jnp.mean(jnp.square(band)))),
            "interior_rms": float(jnp.sqrt(jnp.mean(jnp.square(e[half:-half])))),
        })
    out["couette_radial_d2"] = {
        "rows": rows_c,
        "obs_order_boundary": _obs_order(rows_c[-2]["boundary_rms"], rows_c[-1]["boundary_rms"]),
        "obs_order_interior": _obs_order(rows_c[-2]["interior_rms"], rows_c[-1]["interior_rms"]),
        "expected": 4.0,
        "pass": _obs_order(rows_c[-2]["boundary_rms"], rows_c[-1]["boundary_rms"]) > 3.5,
    }
    print(
        "  d2/mixed orders:",
        {k: (v["obs_order_boundary"], v["pass"]) for k, v in out.items()},
    )
    return out


# ---------------------------------------------------------------------------
# 2) Percentile suite with correct periodic declarations
# ---------------------------------------------------------------------------


def _eps_order(state, weights, grid_hi: PathBGridConfig):
    g4 = grid_hi
    g2 = PathBGridConfig(
        spatial_dimension=2,
        steady=True,
        fd_order=2,
        periodic=grid_hi.periodic,
    )
    r4, scale, wr = _run(state, weights, grid=g4, fill=True)
    r2, _, _ = _run(state, weights, grid=g2, fill=True)
    eps = residual_magnitude(jnp.asarray(r4) - jnp.asarray(r2))
    return r4, eps, scale, wr


def _eps_dual(state, weights, grid: PathBGridConfig):
    r_h, scale, wr = _run(state, weights, grid=grid, fill=True)
    st_c = {}
    for k, v in state.items():
        a = jnp.asarray(v)
        if a.ndim >= 2 and a.shape[:2] == jnp.asarray(state["x"]).shape[:2]:
            st_c[k] = a[::2, ::2, ...]
        else:
            st_c[k] = v
    st_c["x"] = jnp.asarray(state["x"])[::2, ::2]
    st_c["y"] = jnp.asarray(state["y"])[::2, ::2]
    w_c = weights[::2, ::2]
    r_c, _, _ = _run(st_c, w_c, grid=grid, fill=True)
    rh = jnp.asarray(r_h)
    rc = jnp.asarray(r_c)
    up = jnp.repeat(jnp.repeat(rc, 2, axis=0), 2, axis=1)
    slices = tuple(slice(0, s) for s in rh.shape[: up.ndim])
    rc_up = up[slices]
    eps = residual_magnitude(rh - rc_up) / float(2**4 - 1)
    return r_h, eps, scale, wr


def case_clean(flow: str, resolution: str) -> Dict[str, Any]:
    grid = _grid_for(flow)
    if flow == "taylor_green":
        n = RES_TG[resolution]
        xs = jnp.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        ys = jnp.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = taylor_green_2d(x, y, 0.1, re=100.0)
    else:
        nx, ny = RES_KOV[resolution]
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 2.0 * math.pi, ny, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = kovasznay_2d(x, y, re=40.0)
    w = structured_point_weights([xs, ys])
    state = {
        "x": x, "y": y, "u": fields["u"], "u_t": fields["u_t"],
        "p": fields["p"], "re": fields["re"],
    }
    r, eps_ord, scale, wr = _eps_order(state, w, grid)
    _, eps_dual, _, _ = _eps_dual(state, w, grid)
    spikes = boundary_spike_frac(r, scale)
    scores: Dict[str, Any] = {"plain": {}, "ea_order": {}, "ea_dual": {}}
    for q in PERCENTILES:
        scores["plain"][_pkey(q)] = _combo(r, w, scale, wr, q=q)
        scores["ea_order"][_pkey(q)] = _combo(
            r, w, scale, wr, q=q, eps=eps_ord, error_aware=True
        )
        scores["ea_dual"][_pkey(q)] = _combo(
            r, w, scale, wr, q=q, eps=eps_dual, error_aware=True
        )
    print(
        f"  clean {flow}/{resolution} periodic={grid.periodic} "
        f"plain97.5={scores['plain']['p97_5']['tier'][:4]} "
        f"bspike={spikes['frac_boundary_spikes']:.4f}"
    )
    return {
        "flow": flow,
        "resolution": resolution,
        "periodic": list(grid.spatial_periodic_flags(2)),
        "spikes": spikes,
        "scores": scores,
    }


def case_noise(flow: str, resolution: str, levels=(1e-4, 1e-3, 1e-2)) -> Dict[str, Any]:
    grid = _grid_for(flow)
    key = jax.random.PRNGKey(0 if flow == "taylor_green" else 1)
    if flow == "taylor_green":
        n = RES_TG[resolution]
        xs = jnp.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        ys = jnp.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = taylor_green_2d(x, y, 0.1, re=100.0)
    else:
        nx, ny = RES_KOV[resolution]
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 2.0 * math.pi, ny, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = kovasznay_2d(x, y, re=40.0)
    w = structured_point_weights([xs, ys])
    u_scale = float(jnp.sqrt(jnp.mean(jnp.square(fields["u"]))))
    p_scale = float(jnp.sqrt(jnp.mean(jnp.square(fields["p"]))))
    rows = []
    for eps_n in levels:
        key, k1, k2 = jax.random.split(key, 3)
        state = {
            "x": x, "y": y,
            "u": fields["u"] + eps_n * u_scale * jax.random.normal(k1, fields["u"].shape),
            "u_t": fields["u_t"],
            "p": fields["p"] + eps_n * p_scale * jax.random.normal(k2, fields["p"].shape),
            "re": fields["re"],
        }
        r, eps_ord, scale, wr = _eps_order(state, w, grid)
        row = {"noise_rel": float(eps_n), "plain": {}, "ea_order": {}}
        for q in PERCENTILES:
            row["plain"][_pkey(q)] = _combo(r, w, scale, wr, q=q)
            row["ea_order"][_pkey(q)] = _combo(
                r, w, scale, wr, q=q, eps=eps_ord, error_aware=True
            )
        rows.append(row)
        print(
            f"  noise {flow}/{resolution} eps={eps_n}: "
            f"p95={row['plain']['p95']['tier'][:4]} "
            f"p97.5={row['plain']['p97_5']['tier'][:4]}"
        )
    return {"flow": flow, "resolution": resolution, "rows": rows}


def _apply_pert(base, kind: str, mag: float):
    st = dict(base)
    pg = jnp.asarray(st["p_grad"])
    nx, ny = int(pg.shape[0]), int(pg.shape[1])
    if kind == "smooth_band":
        n_cols = max(1, int(round(0.1 * nx)))
        delta = jnp.zeros((nx, ny)).at[:n_cols, :].set(mag)
    elif kind == "gaussian":
        i0, j0 = nx // 2, ny // 2
        ii = jnp.arange(nx)[:, None]
        jj = jnp.arange(ny)[None, :]
        delta = mag * jnp.exp(-0.5 * (((ii - i0) / 1.5) ** 2 + ((jj - j0) / 1.5) ** 2))
    elif kind == "checker":
        i0, j0 = nx // 3, ny // 3
        patch = jnp.zeros((nx, ny))
        for i in range(i0, min(i0 + 4, nx)):
            for j in range(j0, min(j0 + 4, ny)):
                patch = patch.at[i, j].set(mag * (1.0 if (i + j) % 2 == 0 else -1.0))
        delta = patch
    else:
        raise ValueError(kind)
    st["p_grad"] = pg + jnp.zeros_like(pg).at[..., 0].set(delta)
    return st


def case_sharp() -> Dict[str, Any]:
    n = 40
    xs = jnp.linspace(0.0, 1.0, n)
    ys = jnp.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    base = ns_state_from_analytic(fields, x, y)
    w = structured_point_weights([xs, ys])
    grid = _grid_for("kovasznay")
    out: Dict[str, Any] = {}
    for kind in ("smooth_band", "gaussian", "checker"):
        out[kind] = {"by_mag": []}
        for mag in (0.1, 0.2, 0.5, 1.0):
            st = _apply_pert(base, kind, mag)
            # exact-∂ residual for detection; ε from Path B on primitives
            r, scale, wr = _run(st, w)
            st_pb = {
                "x": x, "y": y, "u": st["u"], "u_t": st["u_t"],
                "p": st["p"], "re": st["re"],
            }
            _, eps_ord, _, _ = _eps_order(st_pb, w, grid)
            _, eps_dual, _, _ = _eps_dual(st_pb, w, grid)
            row = {"magnitude": mag, "configs": {}}
            for q in PERCENTILES:
                row["configs"][f"plain_{_pkey(q)}"] = _combo(r, w, scale, wr, q=q)
                row["configs"][f"ea_order_{_pkey(q)}"] = _combo(
                    r, w, scale, wr, q=q, eps=eps_ord, error_aware=True
                )
                row["configs"][f"ea_dual_{_pkey(q)}"] = _combo(
                    r, w, scale, wr, q=q, eps=eps_dual, error_aware=True
                )
            out[kind]["by_mag"].append(row)
            print(
                f"  sharp {kind} mag={mag}: "
                f"p95={row['configs']['plain_p95']['tier'][:4]} "
                f"p97.5={row['configs']['plain_p97_5']['tier'][:4]} "
                f"ea97.5={row['configs']['ea_order_p97_5']['tier'][:4]}"
            )
    return out


def case_clustered_a3() -> Dict[str, Any]:
    """A3-style clustered mag=0.1 on Kov (periodic y) + Couette exact-∂."""
    out: Dict[str, Any] = {}
    # Kovasznay clustered
    nx, ny = 48, 40
    s = jnp.linspace(0.0, 1.0, nx)
    xs = 0.0 + 1.0 * (s**2.5)
    ys = jnp.linspace(0.0, 2.0 * math.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    base = ns_state_from_analytic(fields, x, y)
    w = structured_point_weights([xs, ys])
    x_cut = float(xs[max(3, nx // 5)])
    region = x <= x_cut
    st = dict(base)
    pg = jnp.asarray(st["p_grad"])
    st["p_grad"] = pg + jnp.zeros_like(pg).at[..., 0].set(jnp.where(region, 0.1, 0.0))
    r, scale, wr = _run(st, w)
    out["kovasznay_mag0.1"] = {
        "plain": { _pkey(q): _combo(r, w, scale, wr, q=q) for q in PERCENTILES },
        "point_frac": float(jnp.mean(region.astype(float))),
    }
    # Couette
    nr, nt = 36, 48
    s = jnp.linspace(0.0, 1.0, nr)
    r1 = 1.0 + 1.0 * (s**2.5)
    th = jnp.linspace(0.0, 2.0 * math.pi, nt, endpoint=False)
    fields = taylor_couette_2d(r1, th, re=100.0)
    wr_r = trapezoid_weights_1d(r1)
    wt = trapezoid_weights_1d(th)
    w = (r1[:, None] * wr_r[:, None]) * wt[None, :]
    base = {
        "x": fields["x"], "y": fields["y"], "u": fields["u"], "u_t": fields["u_t"],
        "u_grad": fields["u_grad"], "u_laplacian": fields["u_laplacian"],
        "p": fields["p"], "p_grad": fields["p_grad"], "re": fields["re"],
    }
    r_cut = float(r1[max(3, nr // 5)])
    region = fields["r"] <= r_cut
    st = dict(base)
    pg = jnp.asarray(st["p_grad"])
    st["p_grad"] = pg + jnp.zeros_like(pg).at[..., 0].set(jnp.where(region, 0.1, 0.0))
    r, scale, wr = _run(st, w)
    out["couette_mag0.1"] = {
        "plain": { _pkey(q): _combo(r, w, scale, wr, q=q) for q in PERCENTILES },
        "point_frac": float(jnp.mean(region.astype(float))),
    }
    print(
        "  clustered A3:",
        {k: v["plain"]["p95"]["tier"][:4] for k, v in out.items()},
    )
    return out


def case_sensor() -> Dict[str, Any]:
    n = 32
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 1.0, n)
    checker = ((jnp.arange(n)[:, None] + jnp.arange(n)[None, :]) % 2).astype(float)
    ii = jnp.arange(n)[:, None]
    jj = jnp.arange(n)[None, :]
    gauss = jnp.exp(-0.5 * (((ii - n // 2) / 1.2) ** 2 + ((jj - n // 2) / 1.2) ** 2))
    rc = build_resolution_sensor_report({"x": x, "y": y, "T": checker}, field_keys=["T"])
    rg = build_resolution_sensor_report({"x": x, "y": y, "T": gauss}, field_keys=["T"])
    return {
        "checker": {
            "grid_scale_energy": rc["per_field"]["T"].get("grid_scale_energy"),
            "grid_scale_flag": rc["per_field"]["T"].get("grid_scale_flag"),
            "n_warnings": len(rc["warnings"]),
        },
        "narrow_gaussian": {
            "grid_scale_energy": rg["per_field"]["T"].get("grid_scale_energy"),
            "grid_scale_flag": rg["per_field"]["T"].get("grid_scale_flag"),
            "n_warnings": len(rg["warnings"]),
        },
        "note": rc["note"],
    }


def _is_high(tier: str) -> bool:
    return str(tier).startswith("High")


def decide(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    a) plain p97.5 if prefer rule passes with correct periodic handling
    b) else error-aware p97.5 if it passes and detects sharp ≥ plain p95
    c) else plain p95
    """
    prefer = {
        "clean_tg_both_high": True,
        "clean_kov_both_high": True,
        "tg_noise_1e-3_high": True,
        "a3_detect_mag0.1": True,
    }
    for res in ("coarse", "fine"):
        tg = report["cases"][f"clean_taylor_green_{res}"]["scores"]["plain"]["p97_5"]
        kov = report["cases"][f"clean_kovasznay_{res}"]["scores"]["plain"]["p97_5"]
        prefer["clean_tg_both_high"] &= _is_high(tg["tier"])
        prefer["clean_kov_both_high"] &= _is_high(kov["tier"])
        # noise
        for row in report["cases"][f"noise_taylor_green_{res}"]["rows"]:
            if abs(row["noise_rel"] - 1e-3) < 1e-15:
                prefer["tg_noise_1e-3_high"] &= _is_high(row["plain"]["p97_5"]["tier"])
    a3 = report["cases"]["clustered_a3"]
    prefer["a3_detect_mag0.1"] = (
        a3["kovasznay_mag0.1"]["plain"]["p97_5"]["below_high"]
        and a3["couette_mag0.1"]["plain"]["p97_5"]["below_high"]
    )
    plain97_ok = all(prefer.values())

    # Sharp detection: first mag below High for gaussian/checker
    def first_detect(cfg: str) -> Dict[str, Optional[float]]:
        out = {}
        for kind in ("gaussian", "checker"):
            out[kind] = None
            for row in report["cases"]["sharp"][kind]["by_mag"]:
                if row["configs"][cfg]["below_high"]:
                    out[kind] = row["magnitude"]
                    break
        return out

    plain95_first = first_detect("plain_p95")
    ea_order_first = first_detect("ea_order_p97_5")
    ea_dual_first = first_detect("ea_dual_p97_5")

    def as_good(cand: Dict[str, Optional[float]], ref: Dict[str, Optional[float]]) -> bool:
        for k in ("gaussian", "checker"):
            if ref[k] is None:
                continue
            if cand[k] is None or cand[k] > ref[k] + 1e-12:
                return False
        return True

    # Error-aware p97.5 prefer-rule on clean+noise
    ea_prefer = {
        "clean_tg_both_high": True,
        "clean_kov_both_high": True,
        "tg_noise_1e-3_high": True,
    }
    for res in ("coarse", "fine"):
        tg = report["cases"][f"clean_taylor_green_{res}"]["scores"]["ea_order"]["p97_5"]
        kov = report["cases"][f"clean_kovasznay_{res}"]["scores"]["ea_order"]["p97_5"]
        ea_prefer["clean_tg_both_high"] &= _is_high(tg["tier"])
        ea_prefer["clean_kov_both_high"] &= _is_high(kov["tier"])
        for row in report["cases"][f"noise_taylor_green_{res}"]["rows"]:
            if abs(row["noise_rel"] - 1e-3) < 1e-15:
                ea_prefer["tg_noise_1e-3_high"] &= _is_high(
                    row["ea_order"]["p97_5"]["tier"]
                )

    if plain97_ok:
        chosen = {
            "rule": "a",
            "safeguard_percentile": 97.5,
            "safeguard_error_aware": False,
            "reason": (
                "Plain p97.5 passes prefer rule with Kovasznay y-periodic and "
                "Taylor–Green both-periodic; keep error-aware opt-in."
            ),
        }
    elif all(ea_prefer.values()) and as_good(ea_order_first, plain95_first):
        chosen = {
            "rule": "b",
            "safeguard_percentile": 97.5,
            "safeguard_error_aware": True,
            "epsilon": "order_difference",
            "reason": (
                "Error-aware p97.5 with order-difference ε passes prefer rule and "
                "detects sharp perturbations at least as well as plain p95."
            ),
        }
    elif all(ea_prefer.values()) and as_good(ea_dual_first, plain95_first):
        # dual as fallback under rule b
        ea_dual_prefer = True
        for res in ("coarse", "fine"):
            tg = report["cases"][f"clean_taylor_green_{res}"]["scores"]["ea_dual"]["p97_5"]
            kov = report["cases"][f"clean_kovasznay_{res}"]["scores"]["ea_dual"]["p97_5"]
            ea_dual_prefer &= _is_high(tg["tier"]) and _is_high(kov["tier"])
        if ea_dual_prefer:
            chosen = {
                "rule": "b",
                "safeguard_percentile": 97.5,
                "safeguard_error_aware": True,
                "epsilon": "dual_spacing",
                "reason": (
                    "Error-aware p97.5 with dual-spacing ε passes and detects sharp "
                    "perturbations at least as well as plain p95."
                ),
            }
        else:
            chosen = {
                "rule": "c",
                "safeguard_percentile": 95.0,
                "safeguard_error_aware": False,
                "reason": "Neither plain nor error-aware p97.5 clears the prefer rule; keep plain p95.",
            }
    else:
        chosen = {
            "rule": "c",
            "safeguard_percentile": 95.0,
            "safeguard_error_aware": False,
            "reason": "Neither plain nor error-aware p97.5 clears the prefer rule; keep plain p95.",
        }

    return {
        "prefer_plain_p97_5": prefer,
        "plain_p97_5_ok": plain97_ok,
        "ea_order_p97_5_prefer": ea_prefer,
        "sharp_first_mag": {
            "plain_p95": plain95_first,
            "ea_order_p97_5": ea_order_first,
            "ea_dual_p97_5": ea_dual_first,
        },
        "recommendation": chosen,
        "note_defaults_not_switched": True,
    }


def main() -> None:
    report: Dict[str, Any] = {
        "phase": "A5_periodic_safeguard",
        "high_cutoff": HIGH,
        "d2_mixed_verification": {},
        "cases": {},
        "decision": {},
    }
    print("=== (1) d2 / mixed boundary orders ===")
    report["d2_mixed_verification"] = verify_d2_mixed_orders()

    print("=== (2) clean / noise / sharp / clustered with periodic axes ===")
    for flow in ("taylor_green", "kovasznay"):
        for res in ("coarse", "fine"):
            report["cases"][f"clean_{flow}_{res}"] = case_clean(flow, res)
            report["cases"][f"noise_{flow}_{res}"] = case_noise(flow, res)
    report["cases"]["sharp"] = case_sharp()
    report["cases"]["clustered_a3"] = case_clustered_a3()

    print("=== (3) resolution sensor grid-scale ===")
    report["cases"]["resolution_sensor"] = case_sensor()

    print("=== (4) decision ===")
    report["decision"] = decide(report)
    print(json.dumps(report["decision"]["recommendation"], indent=2))

    out_json = ROOT / "docs" / "phase_a5_periodic_safeguard.json"
    out_md = ROOT / "docs" / "phase_a5_periodic_safeguard.md"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    rec = report["decision"]["recommendation"]
    d2 = report["d2_mixed_verification"]
    md = [
        "# Phase A5: periodic axes + percentile rerun",
        "",
        "Defaults are **not** switched in this step.",
        "",
        "## Periodic-axis declaration",
        "",
        "- `PathBGridConfig.periodic` may be `False`, `True`, or a per-axis sequence "
        "such as `(False, True)` for Kovasznay (`x` bounded, `y` periodic).",
        "- Never inferred. Misdeclaring a periodic physical axis as bounded inflates "
        "boundary residuals (one-sided closures on a wrap-around field).",
        "- Spectral differentiation still requires **all** spatial axes declared periodic.",
        "",
        "## Second-derivative / mixed boundary orders",
        "",
        f"- poly-exp d2 boundary observed order: "
        f"{d2['poly_exp_d2']['obs_order_boundary']:.3f} "
        f"(pass={d2['poly_exp_d2']['pass']})",
        f"- mixed poly-exp boundary observed order: "
        f"{d2['mixed_poly_exp']['obs_order_boundary']:.3f} "
        f"(pass={d2['mixed_poly_exp']['pass']})",
        f"- Couette radial d2 boundary observed order: "
        f"{d2['couette_radial_d2']['obs_order_boundary']:.3f} "
        f"(pass={d2['couette_radial_d2']['pass']})",
        "",
        "Matching Fornberg boundary bands for second derivatives now use width "
        "`order + 2` so formal order matches the interior.",
        "",
        "## Grid-scale resolution-sensor warning",
        "",
        "- Warns when Nyquist-bin spectral energy exceeds threshold `0.05` "
        "(checkerboard / grid-scale artifacts).",
        "- Diagnostic only; never changes admissibility tiers.",
        "",
        f"- Checkerboard flag: {report['cases']['resolution_sensor']['checker']}",
        f"- Narrow Gaussian: {report['cases']['resolution_sensor']['narrow_gaussian']}",
        "",
        "## Recommendation (not applied)",
        "",
        f"- Rule: **{rec['rule']}**",
        f"- `safeguard_percentile` = {rec['safeguard_percentile']}",
        f"- `safeguard_error_aware` = {rec['safeguard_error_aware']}",
    ]
    if rec.get("epsilon"):
        md.append(f"- ε mode: `{rec['epsilon']}`")
    md.extend(["", f"- Reason: {rec['reason']}", ""])
    # Spike summary table
    md.append("## Spike fractions (clean Path B)")
    md.append("")
    md.append("| case | boundary spike frac | plain p95 | plain p97.5 | plain p99 |")
    md.append("|---|---|---|---|---|")
    for flow in ("taylor_green", "kovasznay"):
        for res in ("coarse", "fine"):
            c = report["cases"][f"clean_{flow}_{res}"]
            sp = c["spikes"]["frac_boundary_spikes"]
            p = c["scores"]["plain"]
            md.append(
                f"| {flow}/{res} | {sp:.4f} | {p['p95']['tier']} | "
                f"{p['p97_5']['tier']} | {p['p99']['tier']} |"
            )
    md.append("")
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
