#!/usr/bin/env python3
"""
Phase A4 verification before approving defaults.

1) Closure operator verification + fix confirmation
2) Spike source on clean coarse Kovasznay (matching)
3) Sharp-violation tests (Gaussian / checker / kink vs smooth)
4) Dual-spacing ε and smoothed ε vs order-difference ε
5) Recommend defaults (error-aware default only if ≥ plain p95 on sharp tests)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic, taylor_green_2d
from moju.monitor.auditor import ADM_HIGH_THRESHOLD
from moju.monitor.fd_stencils import diff1_uniform, diff2_uniform, mixed_diff_uniform
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.tiers import tier_for_score
from moju.monitor.weighted_norms import structured_point_weights, weighted_rms
from moju.monitor.weighted_safeguard import (
    error_aware_percentile_stat,
    residual_magnitude,
    smooth_error_field,
)


ROOT = Path(__file__).resolve().parents[1]
HIGH = float(ADM_HIGH_THRESHOLD)
CLOSURES = ("matching", "order_reduced", "sbp", "bc_aware")
# Resolutions for convergence (n points per axis)
NS = (24, 48, 96)


def _pkey(q: float) -> str:
    return "p97_5" if abs(q - 97.5) < 1e-9 else f"p{int(q)}"


def _score(stat: float, scale_k: float) -> float:
    if not math.isfinite(stat) or not math.isfinite(scale_k) or scale_k <= 0:
        return float("nan")
    return 1.0 / (1.0 + float(stat) / float(scale_k))


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


def _obs_order(e_coarse: float, e_fine: float, ratio: float = 2.0) -> float:
    if e_coarse <= 0 or e_fine <= 0 or not math.isfinite(e_coarse) or not math.isfinite(e_fine):
        return float("nan")
    return math.log(e_coarse / e_fine) / math.log(ratio)


def _band_rms(err: jnp.ndarray, *, half: int, region: str) -> float:
    e = jnp.abs(err)
    if e.ndim == 1:
        n = int(e.shape[0])
        if region == "boundary":
            band = jnp.concatenate([e[:half], e[n - half :]])
        else:
            band = e[half : n - half]
        return float(jnp.sqrt(jnp.mean(jnp.square(band))))
    # 2D: boundary = outer half-index frame; interior = complement
    n0, n1 = int(e.shape[0]), int(e.shape[1])
    mask = jnp.zeros((n0, n1), dtype=bool)
    mask = mask.at[:half, :].set(True)
    mask = mask.at[n0 - half :, :].set(True)
    mask = mask.at[:, :half].set(True)
    mask = mask.at[:, n1 - half :].set(True)
    if region == "boundary":
        sel = e[mask]
    else:
        sel = e[~mask]
    if sel.size == 0:
        return float("nan")
    return float(jnp.sqrt(jnp.mean(jnp.square(sel))))


# ---------------------------------------------------------------------------
# 1) Closure verification
# ---------------------------------------------------------------------------


def _diff_kw(closure: str) -> Dict[str, Any]:
    return {
        "boundary_closure": closure,
        "use_field_boundary_values": closure == "bc_aware",
    }


def verify_closure_1d(closure: str, order: int = 4) -> Dict[str, Any]:
    """Polynomial / sine / exp on [0,1]; report boundary & interior errors + observed order."""
    half = (order + 1) // 2
    # Expected: interior ~ order; boundary matching~order, order_reduced/sbp~order-2 or order//2
    cases = {}

    def run_case(name: str, f_fn, d1_fn, d2_fn):
        rows = []
        for n in NS:
            x = jnp.linspace(0.0, 1.0, n)
            h = float(x[1] - x[0])
            f = f_fn(x)
            kw = _diff_kw(closure)
            a1 = diff1_uniform(f, h, order=order, axis=0, **kw)
            a2 = diff2_uniform(f, h, order=order, axis=0, **kw)
            e1 = a1 - d1_fn(x)
            e2 = a2 - d2_fn(x)
            rows.append({
                "n": n,
                "h": h,
                "d1_boundary_rms": _band_rms(e1, half=half, region="boundary"),
                "d1_interior_rms": _band_rms(e1, half=half, region="interior"),
                "d2_boundary_rms": _band_rms(e2, half=half, region="boundary"),
                "d2_interior_rms": _band_rms(e2, half=half, region="interior"),
            })
        # orders between last two refinements
        def ord_of(key):
            return _obs_order(rows[-2][key], rows[-1][key], ratio=NS[-1] / NS[-2])

        cases[name] = {
            "rows": rows,
            "obs_order": {
                "d1_boundary": ord_of("d1_boundary_rms"),
                "d1_interior": ord_of("d1_interior_rms"),
                "d2_boundary": ord_of("d2_boundary_rms"),
                "d2_interior": ord_of("d2_interior_rms"),
            },
        }

    # Polynomials up to degree `order` (exact for FD of that order in interior)
    # deg 3: nonzero d1,d2; deg 4 for order-4 interior identity checks
    run_case(
        "poly3",
        lambda x: x**3,
        lambda x: 3 * x**2,
        lambda x: 6 * x,
    )
    run_case(
        "poly4",
        lambda x: x**4,
        lambda x: 4 * x**3,
        lambda x: 12 * x**2,
    )
    run_case(
        "sine",
        lambda x: jnp.sin(jnp.pi * x),
        lambda x: jnp.pi * jnp.cos(jnp.pi * x),
        lambda x: -(jnp.pi**2) * jnp.sin(jnp.pi * x),
    )
    run_case(
        "exp",
        lambda x: jnp.exp(x),
        lambda x: jnp.exp(x),
        lambda x: jnp.exp(x),
    )
    return cases


def verify_closure_2d_mixed(closure: str, order: int = 4) -> Dict[str, Any]:
    half = (order + 1) // 2
    rows = []
    for n in NS:
        xs = jnp.linspace(0.0, 1.0, n)
        ys = jnp.linspace(0.0, 1.0, n)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        hx = float(xs[1] - xs[0])
        hy = float(ys[1] - ys[0])
        f = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
        exact = (jnp.pi**2) * jnp.cos(jnp.pi * x) * (-jnp.sin(jnp.pi * y))  # ∂²/∂x∂y
        kw = _diff_kw(closure)
        # mixed via successive d1
        ddx = diff1_uniform(f, hx, order=order, axis=0, **kw)
        approx = diff1_uniform(ddx, hy, order=order, axis=1, **kw)
        # also API
        approx2 = mixed_diff_uniform(f, hx, hy, order=order, axis_x=0, axis_y=1)
        # mixed_diff_uniform doesn't pass closure — use successive for closure test
        e = approx - exact
        rows.append({
            "n": n,
            "mixed_boundary_rms": _band_rms(e, half=half, region="boundary"),
            "mixed_interior_rms": _band_rms(e, half=half, region="interior"),
            "mixed_default_api_max_abs": float(jnp.max(jnp.abs(approx2 - exact))),
        })
    return {
        "rows": rows,
        "obs_order_boundary": _obs_order(
            rows[-2]["mixed_boundary_rms"], rows[-1]["mixed_boundary_rms"],
            ratio=NS[-1] / NS[-2],
        ),
        "obs_order_interior": _obs_order(
            rows[-2]["mixed_interior_rms"], rows[-1]["mixed_interior_rms"],
            ratio=NS[-1] / NS[-2],
        ),
    }


def verify_closure_kovasznay(closure: str, order: int = 4) -> Dict[str, Any]:
    half = (order + 1) // 2
    rows = []
    for n in NS:
        xs = jnp.linspace(0.0, 1.0, n)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = kovasznay_2d(x, y, re=40.0)
        hx = float(xs[1] - xs[0])
        u0 = fields["u"][..., 0]
        exact = fields["u_grad"][..., 0, 0]
        approx = diff1_uniform(u0, hx, order=order, axis=0, **_diff_kw(closure))
        e = approx - exact
        rows.append({
            "n": n,
            "boundary_rms": _band_rms(e, half=half, region="boundary"),
            "interior_rms": _band_rms(e, half=half, region="interior"),
        })
    return {
        "rows": rows,
        "obs_order_boundary": _obs_order(
            rows[-2]["boundary_rms"], rows[-1]["boundary_rms"], ratio=NS[-1] / NS[-2]
        ),
        "obs_order_interior": _obs_order(
            rows[-2]["interior_rms"], rows[-1]["interior_rms"], ratio=NS[-1] / NS[-2]
        ),
    }


def expected_orders(closure: str, order: int = 4) -> Dict[str, float]:
    if closure == "matching":
        return {"boundary": float(order), "interior": float(order)}
    if closure in ("order_reduced", "sbp"):
        return {"boundary": float(order // 2 if closure == "sbp" and order == 4 else (2 if order == 4 else 4)),
                "interior": float(order)}
    # bc_aware: centred → same formal order as interior when Dirichlet exact
    return {"boundary": float(order), "interior": float(order)}


def closure_passes(obs: Dict[str, float], expected: Dict[str, float], *, tol: float = 0.75) -> bool:
    """Require observed order ≥ expected - tol (loose for coarse grids / SBP ends)."""
    ok = True
    for key in ("d1_boundary", "d1_interior", "d2_interior"):
        if key not in obs:
            continue
        exp = expected["boundary"] if "boundary" in key else expected["interior"]
        # d2 boundary often one order lower even for matching Fornberg of width order+1
        if key == "d2_boundary":
            exp = max(1.0, expected["boundary"] - 1.0)
        o = obs[key]
        if not math.isfinite(o) or o < exp - tol:
            # Allow poly exactness → inf/huge orders
            if math.isfinite(o) and o < exp - tol:
                ok = False
    return ok


# ---------------------------------------------------------------------------
# 2) Spike source
# ---------------------------------------------------------------------------


def spike_source_kovasznay(n: int = 40) -> Dict[str, Any]:
    xs = jnp.linspace(0.0, 1.0, n)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    w = structured_point_weights([xs, ys])
    state = {
        "x": x, "y": y, "u": fields["u"], "u_t": fields["u_t"],
        "p": fields["p"], "re": fields["re"],
    }
    grid = PathBGridConfig(
        spatial_dimension=2, steady=True, fd_order=4, boundary_closure="matching"
    )
    eng = _ns_engine()
    eng.compute_residuals(
        state, auto_path_b_derivatives=grid, fill_law_fd=True,
        weighted_norms="report", point_weights=w,
    )
    entry = eng.log[-1]
    scale = float(entry["scale"]["laws/momentum_navier_stokes"])
    r_high = scale * (1.0 / HIGH - 1.0)
    res = eng.last_residuals["laws"]["momentum_navier_stokes"]
    mag = residual_magnitude(res)
    half = 2
    # Term breakdown if present
    tb = entry.get("term_breakdown", {}).get("laws/momentum_navier_stokes", {})

    # Per-component residual on boundary bands / corners
    def frame_stats(arr):
        a = residual_magnitude(arr) if jnp.asarray(arr).ndim >= 2 and jnp.asarray(arr).shape[-1] in (2, 3) else jnp.abs(jnp.asarray(arr))
        if a.ndim > 2:
            a = residual_magnitude(a)
        n0, n1 = int(a.shape[0]), int(a.shape[1])
        edge = {
            "lo_0": float(jnp.sqrt(jnp.mean(jnp.square(a[:half, :])))),
            "hi_0": float(jnp.sqrt(jnp.mean(jnp.square(a[n0 - half :, :])))),
            "lo_1": float(jnp.sqrt(jnp.mean(jnp.square(a[:, :half])))),
            "hi_1": float(jnp.sqrt(jnp.mean(jnp.square(a[:, n1 - half :])))),
        }
        corners = float(jnp.sqrt(jnp.mean(jnp.square(jnp.array([
            a[0, 0], a[0, -1], a[-1, 0], a[-1, -1],
            a[1, 0], a[1, -1], a[-2, 0], a[-2, -1],
            a[0, 1], a[0, -2], a[-1, 1], a[-1, -2],
        ])))))
        return {"edges": edge, "corners_rms": corners, "max": float(jnp.max(a)),
                "frac_above_r_high": float(jnp.mean((a > r_high).astype(float)))}

    # Compare FD vs analytic for each derivative key
    from moju.monitor.path_b_derivatives import fill_path_b_derivatives

    st2 = dict(state)
    st2, _ = fill_path_b_derivatives(
        st2,
        laws_spec=eng.laws_spec,
        constants={},
        grid=grid,
        fill_law_recipes=True,
    )
    deriv_errs = {}
    for key, exact_key in (
        ("u_grad", "u_grad"),
        ("u_laplacian", "u_laplacian"),
        ("p_grad", "p_grad"),
    ):
        if key not in st2 or exact_key not in fields:
            continue
        fd = jnp.asarray(st2[key])
        ex = jnp.asarray(fields[exact_key])
        err = residual_magnitude(fd - ex) if fd.shape == ex.shape else jnp.abs(fd - ex)
        if err.ndim > 2:
            err = residual_magnitude(err)
        deriv_errs[key] = {
            "boundary_rms": _band_rms(err, half=half, region="boundary"),
            "interior_rms": _band_rms(err, half=half, region="interior"),
            "max": float(jnp.max(err)),
            "edges": frame_stats(err)["edges"],
            "corners_rms": frame_stats(err)["corners_rms"],
        }

    # Also split u_grad into ∂/∂x vs ∂/∂y components
    ug_fd = jnp.asarray(st2["u_grad"])
    ug_ex = jnp.asarray(fields["u_grad"])
    # shape (nx,ny,2,2): [comp, deriv_axis]
    for comp in (0, 1):
        for ax in (0, 1):
            e = jnp.abs(ug_fd[..., comp, ax] - ug_ex[..., comp, ax])
            deriv_errs[f"u{comp}_d{ax}"] = {
                "boundary_rms": _band_rms(e, half=half, region="boundary"),
                "interior_rms": _band_rms(e, half=half, region="interior"),
                "max": float(jnp.max(e)),
            }

    return {
        "n": n,
        "r_high": r_high,
        "residual_frame": frame_stats(mag),
        "term_breakdown": tb,
        "derivative_errors_fd_vs_analytic": deriv_errs,
        "dominant": max(
            ((k, v["boundary_rms"]) for k, v in deriv_errs.items() if "boundary_rms" in v),
            key=lambda kv: kv[1],
        ),
    }


# ---------------------------------------------------------------------------
# 3–4) Sharp violations + ε variants
# ---------------------------------------------------------------------------


def _combo(residual, weights, scale, wr, *, eps=None, q=95.0, error_aware=False, smooth_radius=0):
    mag = residual_magnitude(residual)
    if error_aware and eps is not None:
        ea = error_aware_percentile_stat(
            residual, eps, q, scale_k=scale, smooth_radius=smooth_radius
        )
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


def _run_res(state, weights, *, grid=None, fill=False):
    eng = _ns_engine()
    eng.compute_residuals(
        state, auto_path_b_derivatives=grid if grid is not None else False,
        fill_law_fd=fill, weighted_norms="report", point_weights=weights,
    )
    res = eng.last_residuals["laws"]["momentum_navier_stokes"]
    scale = float(eng.log[-1]["scale"]["laws/momentum_navier_stokes"])
    wr = float(eng.log[-1]["rms_weighted"]["laws/momentum_navier_stokes"])
    return res, scale, wr


def _eps_order_diff(state, weights) -> Tuple[Any, Any, float, float]:
    g4 = PathBGridConfig(spatial_dimension=2, steady=True, fd_order=4)
    g2 = PathBGridConfig(spatial_dimension=2, steady=True, fd_order=2)
    r4, scale, wr = _run_res(state, weights, grid=g4, fill=True)
    r2, _, _ = _run_res(state, weights, grid=g2, fill=True)
    eps = residual_magnitude(jnp.asarray(r4) - jnp.asarray(r2))
    return r4, eps, scale, wr


def _eps_dual_spacing(state, weights, *, order: int = 4) -> Tuple[Any, Any, float, float]:
    """
    ε from same fd_order at h and ~2h (every-other-point), Richardson-scaled to fine.

    ε_fine ≈ |r_h - r_{2h}| / (2^p - 1), with r_{2h} prolongated by repeat-nearest.
    """
    g = PathBGridConfig(spatial_dimension=2, steady=True, fd_order=order)
    r_h, scale, wr = _run_res(state, weights, grid=g, fill=True)
    # Build coarse state by subsampling
    st_c = {}
    for k, v in state.items():
        a = jnp.asarray(v)
        if a.ndim >= 2 and a.shape[0] == jnp.asarray(state["x"]).shape[0]:
            st_c[k] = a[::2, ::2, ...] if a.ndim >= 2 else a
        else:
            st_c[k] = v
    # Fix coords
    st_c["x"] = jnp.asarray(state["x"])[::2, ::2]
    st_c["y"] = jnp.asarray(state["y"])[::2, ::2]
    w_c = weights[::2, ::2]
    r_c, _, _ = _run_res(st_c, w_c, grid=g, fill=True)
    # Prolongate coarse residual to fine by repeat
    rh = jnp.asarray(r_h)
    rc = jnp.asarray(r_c)
    # Simple broadcast prolong: each coarse sample covers 2x2
    def prolong(c, fine_shape):
        c = jnp.asarray(c)
        # repeat along first two axes
        up = jnp.repeat(jnp.repeat(c, 2, axis=0), 2, axis=1)
        # trim
        slices = tuple(slice(0, s) for s in fine_shape[: up.ndim])
        return up[slices]

    if rh.shape != rc.shape:
        rc_up = prolong(rc, rh.shape)
    else:
        rc_up = rc
    p = int(order)
    denom = float(2**p - 1)
    eps = residual_magnitude(rh - rc_up) / denom
    return r_h, eps, scale, wr


def _base_kov(n: int = 40):
    xs = jnp.linspace(0.0, 1.0, n)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    base = ns_state_from_analytic(fields, x, y)
    w = structured_point_weights([xs, ys])
    return base, w, x, y, xs, ys


def _apply_pert(base, x, y, kind: str, mag: float, xs, ys):
    st = dict(base)
    pg = jnp.asarray(st["p_grad"])
    nx, ny = int(x.shape[0]), int(x.shape[1])
    hx = float(xs[1] - xs[0])
    hy = float(ys[1] - ys[0])
    if kind == "smooth_band":
        n_cols = max(1, int(round(0.1 * nx)))
        region = jnp.zeros((nx, ny), dtype=bool).at[:n_cols, :].set(True)
        delta = jnp.where(region, mag, 0.0)
    elif kind == "gaussian":
        # width 1–2 cells
        i0, j0 = nx // 2, ny // 2
        ii = jnp.arange(nx)[:, None]
        jj = jnp.arange(ny)[None, :]
        sig = 1.5
        delta = mag * jnp.exp(-0.5 * (((ii - i0) / sig) ** 2 + ((jj - j0) / sig) ** 2))
    elif kind == "checker":
        i0, j0 = nx // 3, ny // 3
        wdt = 4
        patch = jnp.zeros((nx, ny))
        for i in range(i0, min(i0 + wdt, nx)):
            for j in range(j0, min(j0 + wdt, ny)):
                patch = patch.at[i, j].set(mag * (1.0 if (i + j) % 2 == 0 else -1.0))
        delta = patch
    elif kind == "kink":
        # jump across a vertical line
        i_line = nx // 2
        delta = jnp.where(jnp.arange(nx)[:, None] >= i_line, mag, 0.0) + jnp.zeros((nx, ny))
    else:
        raise ValueError(kind)
    st["p_grad"] = pg + jnp.zeros_like(pg).at[..., 0].set(delta)
    return st


def sharp_violation_suite() -> Dict[str, Any]:
    base, w, x, y, xs, ys = _base_kov(40)
    mags = [0.1, 0.2, 0.5, 1.0]
    kinds = ("smooth_band", "gaussian", "checker", "kink")
    configs = (
        ("plain_p95", dict(error_aware=False, q=95.0, eps_mode=None, smooth=0)),
        ("ea_order_p95", dict(error_aware=True, q=95.0, eps_mode="order", smooth=0)),
        ("ea_order_p97_5", dict(error_aware=True, q=97.5, eps_mode="order", smooth=0)),
        ("ea_order_p99", dict(error_aware=True, q=99.0, eps_mode="order", smooth=0)),
        ("ea_dual_p95", dict(error_aware=True, q=95.0, eps_mode="dual", smooth=0)),
        ("ea_dual_p97_5", dict(error_aware=True, q=97.5, eps_mode="dual", smooth=0)),
        ("ea_order_smooth_p95", dict(error_aware=True, q=95.0, eps_mode="order", smooth=1)),
        ("ea_order_smooth_p97_5", dict(error_aware=True, q=97.5, eps_mode="order", smooth=1)),
        ("ea_dual_smooth_p97_5", dict(error_aware=True, q=97.5, eps_mode="dual", smooth=1)),
    )
    out: Dict[str, Any] = {}
    for kind in kinds:
        out[kind] = {"by_mag": []}
        for mag in mags:
            st = _apply_pert(base, x, y, kind, mag, xs, ys)
            # exact-∂ residual (no Path B) for sharp tests — same as A3 localized
            r, scale, wr = _run_res(st, w)
            # For error-aware need an ε; on exact-∂ state dual-order Path B differs
            # Use Path B dual estimates on the perturbed primitive fields
            st_pb = {
                "x": x, "y": y, "u": st["u"], "u_t": st["u_t"],
                "p": st["p"], "re": st["re"],
            }
            # Actually keep exact derivatives in st; for ε use Path B on primitives only
            r_ord, eps_ord, _, _ = _eps_order_diff(st_pb, w)
            r_dual, eps_dual, _, _ = _eps_dual_spacing(st_pb, w)
            # Score the exact residual (law with supplied ∂) with each safeguard
            row = {"magnitude": mag, "configs": {}}
            for cname, cfg in configs:
                if not cfg["error_aware"]:
                    row["configs"][cname] = _combo(r, w, scale, wr, q=cfg["q"])
                else:
                    eps = eps_ord if cfg["eps_mode"] == "order" else eps_dual
                    # Align shapes: Path B residual may differ from exact residual shape
                    # Use ε from Path B magnitude field shape matched to exact residual
                    em = residual_magnitude(r)
                    if eps.shape != em.shape:
                        # fallback: use |r| * 0 relative — take eps from order on exact by
                        # comparing nothing; use smoothed zeros + small
                        eps_use = jnp.zeros_like(em)
                    else:
                        eps_use = eps
                    row["configs"][cname] = _combo(
                        r, w, scale, wr, eps=eps_use, q=cfg["q"],
                        error_aware=True, smooth_radius=cfg["smooth"],
                    )
            out[kind]["by_mag"].append(row)
            print(
                f"  {kind} mag={mag}: plain95={row['configs']['plain_p95']['tier'][:4]} "
                f"ea97.5={row['configs']['ea_order_p97_5']['tier'][:4]} "
                f"eaS97.5={row['configs']['ea_order_smooth_p97_5']['tier'][:4]}"
            )
    return out


def detect_at_least_as_well(sharp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Error-aware may default only if it detects sharp perts at least as well as plain p95.

    'Detect' = below_high at a given mag. Compare first mag detected and count of
    (kind, mag) pairs detected.
    """
    kinds = ("gaussian", "checker", "kink")
    configs = (
        "plain_p95",
        "ea_order_p95",
        "ea_order_p97_5",
        "ea_order_smooth_p97_5",
        "ea_dual_p97_5",
        "ea_dual_smooth_p97_5",
    )
    summary = {}
    for cname in configs:
        first = {}
        n_detect = 0
        n_total = 0
        for kind in kinds:
            first[kind] = None
            for row in sharp[kind]["by_mag"]:
                n_total += 1
                if row["configs"][cname]["below_high"]:
                    n_detect += 1
                    if first[kind] is None:
                        first[kind] = row["magnitude"]
        summary[cname] = {
            "first_mag": first,
            "n_detect": n_detect,
            "n_total": n_total,
        }
    plain = summary["plain_p95"]
    # Candidates for default error-aware
    verdicts = {}
    for cname in configs:
        if cname == "plain_p95":
            continue
        ok = True
        for kind in kinds:
            pf, cf = plain["first_mag"][kind], summary[cname]["first_mag"][kind]
            # at least as sensitive: first mag ≤ plain (or both None)
            if pf is None and cf is None:
                continue
            if pf is None and cf is not None:
                continue  # better
            if cf is None and pf is not None:
                ok = False
            elif cf is not None and pf is not None and cf > pf + 1e-12:
                ok = False
        # also require at least as many detections
        if summary[cname]["n_detect"] < plain["n_detect"]:
            ok = False
        verdicts[cname] = {"at_least_as_well_as_plain_p95": ok, **summary[cname]}
    return {"plain_p95": plain, "verdicts": verdicts}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def clean_percentile_with_eps(eps_mode: str, *, q: float = 97.5, smooth: int = 0) -> Dict[str, Any]:
    """Clean TG/Kov coarse+fine tiers under error-aware percentile."""
    out = {}
    for flow in ("taylor_green", "kovasznay"):
        for res, n in (("coarse", 32 if flow == "taylor_green" else 40),
                       ("fine", 64 if flow == "taylor_green" else 80)):
            if flow == "taylor_green":
                xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
                ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
                x, y = jnp.meshgrid(xs, ys, indexing="ij")
                fields = taylor_green_2d(x, y, 0.1, re=100.0)
            else:
                xs = jnp.linspace(0.0, 1.0, n)
                ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
                x, y = jnp.meshgrid(xs, ys, indexing="ij")
                fields = kovasznay_2d(x, y, re=40.0)
            w = structured_point_weights([xs, ys])
            state = {
                "x": x, "y": y, "u": fields["u"], "u_t": fields["u_t"],
                "p": fields["p"], "re": fields["re"],
            }
            if eps_mode == "order":
                r, eps, scale, wr = _eps_order_diff(state, w)
            else:
                r, eps, scale, wr = _eps_dual_spacing(state, w)
            scores = {}
            for qq in (95.0, 97.5, 99.0):
                scores[_pkey(qq)] = _combo(
                    r, w, scale, wr, eps=eps, q=qq, error_aware=True, smooth_radius=smooth
                )
            scores["plain_p95"] = _combo(r, w, scale, wr, q=95.0)
            out[f"{flow}_{res}"] = {
                "n": n,
                "scores": scores,
                "spike_frac": float(
                    jnp.mean(
                        (residual_magnitude(r) > scale * (1.0 / HIGH - 1.0)).astype(float)
                    )
                ),
            }
            print(
                f"  clean {eps_mode}/s{smooth} {flow}/{res}:",
                {k: scores[k]["tier"][:4] for k in ("plain_p95", "p95", "p97_5", "p99")},
            )
    return out


def main() -> None:
    report: Dict[str, Any] = {
        "phase": "A4_verification",
        "high_cutoff": HIGH,
        "dtype": "float64",
        "closure_verification": {},
        "closure_kovasznay": {},
        "spike_source": {},
        "sharp_violations": {},
        "detection_verdict": {},
        "clean_eps_suites": {},
        "recommendation": {},
    }

    print("=== (1) closure verification ===")
    for closure in CLOSURES:
        print(f"  {closure}")
        v1 = verify_closure_1d(closure)
        v2 = verify_closure_2d_mixed(closure)
        vk = verify_closure_kovasznay(closure)
        exp = expected_orders(closure)
        # Summarize sine obs orders as representative
        sine_obs = v1["sine"]["obs_order"]
        passes = closure_passes(sine_obs, exp)
        # Kov boundary error at n=48
        kov_b = vk["rows"][1]["boundary_rms"]
        report["closure_verification"][closure] = {
            "expected": exp,
            "sine_obs_order": sine_obs,
            "poly3_obs_order": v1["poly3"]["obs_order"],
            "exp_obs_order": v1["exp"]["obs_order"],
            "mixed_obs_order": {
                "boundary": v2["obs_order_boundary"],
                "interior": v2["obs_order_interior"],
            },
            "kovasznay": vk,
            "passes_sine_order_check": passes,
            "kov_boundary_rms_n48": kov_b,
        }
        print(
            f"    sine orders {sine_obs} passes={passes} "
            f"kov_brow@48={kov_b:.3e} kov_ord_b={vk['obs_order_boundary']:.2f}"
        )

    print("=== (2) spike source ===")
    report["spike_source"] = spike_source_kovasznay(40)
    print("  dominant", report["spike_source"]["dominant"])
    print("  residual edges", report["spike_source"]["residual_frame"]["edges"])
    for k, v in report["spike_source"]["derivative_errors_fd_vs_analytic"].items():
        if "boundary_rms" in v:
            print(f"  {k}: brow={v['boundary_rms']:.3e} int={v['interior_rms']:.3e} max={v['max']:.3e}")

    print("=== (3–4) sharp violations + ε variants ===")
    report["sharp_violations"] = sharp_violation_suite()
    report["detection_verdict"] = detect_at_least_as_well(report["sharp_violations"])
    print(json.dumps(report["detection_verdict"], indent=2, default=str))

    print("=== clean suites: order vs dual ε ===")
    for mode in ("order", "dual"):
        for sm in (0, 1):
            key = f"{mode}_smooth{sm}"
            report["clean_eps_suites"][key] = clean_percentile_with_eps(mode, smooth=sm)

    def _clean_high(suite: Dict[str, Any], cand: str) -> bool:
        for k, block in suite.items():
            if not str(block["scores"][cand]["tier"]).startswith("High"):
                return False
        return True

    # Recommendation
    verdicts = report["detection_verdict"]["verdicts"]
    # Prefer error-aware default only if some ea config ≥ plain p95 on sharp tests
    # AND clean TG+Kov High on both resolutions at that percentile.
    candidates = []
    for cname, v in verdicts.items():
        if not v["at_least_as_well_as_plain_p95"]:
            continue
        eps_mode = "dual" if "dual" in cname else "order"
        smooth = 1 if "smooth" in cname else 0
        suite = report["clean_eps_suites"][f"{eps_mode}_smooth{smooth}"]
        if "p97" in cname:
            pct, cand = 97.5, "p97_5"
        elif cname.endswith("p99"):
            pct, cand = 99.0, "p99"
        else:
            pct, cand = 95.0, "p95"
        if _clean_high(suite, cand) and _clean_high(suite, "p95"):  # p95 always as sanity
            # also require TG noise not rechecked here; clean High is the gate from A4
            # Prefer higher percentile, then order-diff ε (unlocks p99 on clean Kov),
            # then unsmoothed ε (smoothed missed a 10% smooth band at mag 0.1).
            candidates.append((
                -pct,
                0 if eps_mode == "order" else 1,
                0 if smooth == 0 else 1,
                cname,
                pct,
                eps_mode,
                smooth,
            ))
    if candidates:
        candidates.sort()
        _, _, _, chosen, pct, eps_mode, smooth = candidates[0]
        rec = {
            "boundary_closure": "matching",
            "safeguard_error_aware": True,
            "safeguard_percentile": pct,
            "eps_mode": eps_mode,
            "smooth_radius": smooth,
            "reason": (
                f"{chosen} detects sharp perts ≥ plain p95 and keeps clean TG/Kov High "
                f"on coarse+fine; prefer highest such percentile."
            ),
            "winning_config": chosen,
        }
    else:
        # Fall back: any ea p95 that passes sharp+clean, else plain p95
        ea_p95 = [
            k for k, v in verdicts.items()
            if v["at_least_as_well_as_plain_p95"] and k.endswith("p95")
        ]
        ok_clean_p95 = False
        chosen = None
        for cname in ea_p95:
            eps_mode = "dual" if "dual" in cname else "order"
            smooth = 1 if "smooth" in cname else 0
            suite = report["clean_eps_suites"][f"{eps_mode}_smooth{smooth}"]
            if _clean_high(suite, "p95"):
                ok_clean_p95 = True
                chosen = cname
                break
        if ok_clean_p95 and chosen:
            rec = {
                "boundary_closure": "matching",
                "safeguard_error_aware": True,
                "safeguard_percentile": 95.0,
                "eps_mode": "dual" if "dual" in chosen else "order",
                "smooth_radius": 1 if "smooth" in chosen else 0,
                "reason": (
                    f"No p97.5+ config clears sharp+clean gates; {chosen} does. "
                    "Keep percentile 95 with error-aware."
                ),
                "winning_config": chosen,
            }
        else:
            rec = {
                "boundary_closure": "matching",
                "safeguard_error_aware": False,
                "safeguard_percentile": 95.0,
                "eps_mode": None,
                "smooth_radius": 0,
                "reason": (
                    "No error-aware config clears both sharp-detection (≥ plain p95) "
                    "and clean High gates; keep plain p95 as default, error-aware opt-in."
                ),
                "winning_config": "plain_p95",
            }
    # bc_aware fix confirmation
    bc = report["closure_verification"]["bc_aware"]["kov_boundary_rms_n48"]
    rec["bc_aware_kov_boundary_rms_n48"] = bc
    rec["bc_aware_fixed"] = bool(bc < 1.0)
    report["recommendation"] = rec
    print("\nRECOMMENDATION", json.dumps(rec, indent=2))

    out = ROOT / "docs/phase_a4_verification.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
