#!/usr/bin/env python3
"""
Phase A4: boundary closures + error-aware safeguard vs percentile selection.

1) Compare boundary closures (matching, order_reduced, sbp, bc_aware) on TG / Kov /
   Couette at coarse & fine: boundary-row derivative error and spike fraction.
2) Confirm error-aware safeguard does not raise tiers on known nonzero residuals.
3) Re-run percentile comparison (p95/p97.5/p99) for:
     - current closures (matching)
     - best closure from (1)
     - best closure + error-aware safeguard
4) Prefer highest percentile that keeps clean TG & Kov High on both resolutions and
   TG High at noise 1e-3, while detecting clustered A3 violations at mag 0.1.
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
from moju.monitor.fd_stencils import diff1_uniform
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.tiers import tier_for_score
from moju.monitor.weighted_norms import structured_point_weights, trapezoid_weights_1d, weighted_rms
from moju.monitor.weighted_safeguard import (
    error_aware_percentile_stat,
    residual_magnitude,
)


ROOT = Path(__file__).resolve().parents[1]
HIGH = float(ADM_HIGH_THRESHOLD)
PERCENTILES = (95.0, 97.5, 99.0)
CLOSURES = ("matching", "order_reduced", "sbp", "bc_aware")
RES_TG = {"coarse": 32, "fine": 64}
RES_KOV = {"coarse": (40, 40), "fine": (80, 80)}
RES_COU = {"coarse": (32, 48), "fine": (64, 96)}


def _pkey(q: float) -> str:
    if abs(q - 97.5) < 1e-9:
        return "p97_5"
    return f"p{int(q)}"


def _score(stat: float, scale_k: float) -> float:
    if not math.isfinite(stat) or not math.isfinite(scale_k) or scale_k <= 0:
        return float("nan")
    return 1.0 / (1.0 + float(stat) / float(scale_k))


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


def stretched_axis(n: int, lo: float, hi: float, *, power: float = 2.5) -> jnp.ndarray:
    s = jnp.linspace(0.0, 1.0, n)
    return lo + (hi - lo) * (s**power)


def polar_weights(r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    r1 = jnp.reshape(jnp.asarray(r, dtype=float), (-1,))
    t1 = jnp.reshape(jnp.asarray(theta, dtype=float), (-1,))
    wr = trapezoid_weights_1d(r1)
    wt = trapezoid_weights_1d(t1)
    return (r1[:, None] * wr[:, None]) * wt[None, :]


def boundary_spike_frac(mag: jnp.ndarray, scale_k: float, *, fd_order: int = 4) -> float:
    r_high = float(scale_k) * (1.0 / HIGH - 1.0) if scale_k > 0 else float("inf")
    width = max(1, int(fd_order) // 2)
    n0, n1 = int(mag.shape[0]), int(mag.shape[1])
    bmask = jnp.zeros((n0, n1), dtype=bool)
    bmask = bmask.at[:width, :].set(True)
    bmask = bmask.at[n0 - width :, :].set(True)
    bmask = bmask.at[:, :width].set(True)
    bmask = bmask.at[:, n1 - width :].set(True)
    spike = mag > r_high
    return float(jnp.sum(spike & bmask) / max(mag.size, 1))


def _grid(
    closure: str,
    *,
    fd_order: int = 4,
    bc_aware: bool = False,
) -> PathBGridConfig:
    return PathBGridConfig(
        spatial_dimension=2,
        steady=True,
        fd_order=fd_order,
        fd_order_time=fd_order,
        boundary_closure=closure,
        use_field_boundary_values=bool(bc_aware or closure == "bc_aware"),
    )


# ---------------------------------------------------------------------------
# 1) Closure accuracy
# ---------------------------------------------------------------------------


def _boundary_row_error_1d(
    exact: jnp.ndarray, approx: jnp.ndarray, *, half: int = 2
) -> float:
    e = jnp.abs(approx - exact)
    n = int(e.shape[-1])
    band = jnp.concatenate([e[..., :half].reshape(-1), e[..., n - half :].reshape(-1)])
    return float(jnp.sqrt(jnp.mean(jnp.square(band))))


def closure_derivative_errors(closure: str, flow: str, resolution: str) -> Dict[str, Any]:
    """Boundary-row RMS error of ∂u/∂x (or ∂u_θ/∂r) and spike frac on Path B residual."""
    use_bc = closure == "bc_aware"
    if flow == "taylor_green":
        n = RES_TG[resolution]
        xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        t, re = 0.1, 100.0
        fields = taylor_green_2d(x, y, t, re=re)
        hx = float(xs[1] - xs[0])
        # TG is periodic — boundary closures are less meaningful; still report FD error
        u0 = fields["u"][..., 0]
        exact = fields["u_grad"][..., 0, 0]
        approx = diff1_uniform(
            u0, hx, order=4, axis=0, boundary_closure=closure, use_field_boundary_values=use_bc
        )
        brow = _boundary_row_error_1d(exact, approx, half=2)
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
    elif flow == "kovasznay":
        nx, ny = RES_KOV[resolution]
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = kovasznay_2d(x, y, re=40.0)
        hx = float(xs[1] - xs[0])
        u0 = fields["u"][..., 0]
        exact = fields["u_grad"][..., 0, 0]
        approx = diff1_uniform(
            u0, hx, order=4, axis=0, boundary_closure=closure, use_field_boundary_values=use_bc
        )
        brow = _boundary_row_error_1d(exact, approx, half=2)
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
    else:  # couette — uniform polar for closure modes
        nr, nt = RES_COU[resolution]
        r = jnp.linspace(1.0, 2.0, nr)
        th = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
        fields = taylor_couette_2d(r, th, re=100.0)
        hr = float(r[1] - r[0])
        # Physical azimuthal velocity magnitude ~ |u|; use u_theta analytic via Cartesian ux
        # Differentiating ux along r on the polar mesh (axis 0)
        ux = fields["u"][..., 0]
        # Finite-difference vs analytic: use polar gradient of Cartesian component
        # Analytic ∂ux/∂r from supplied u_grad projected — approximate via central on interior
        # Compare FD of ux along r to a high-order reference (order-6 matching)
        ref = diff1_uniform(ux, hr, order=6, axis=0, boundary_closure="matching")
        approx = diff1_uniform(
            ux, hr, order=4, axis=0, boundary_closure=closure, use_field_boundary_values=use_bc
        )
        brow = _boundary_row_error_1d(ref, approx, half=2)
        w = polar_weights(r, th)
        state = {
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
        shape = (nr, nt)

    # Path B residual spike fraction (skip Couette Path B — curvilinear; use supplied ∂)
    if flow == "couette":
        eng = _ns_engine()
        eng.compute_residuals(state, weighted_norms="report", point_weights=w)
        res = eng.last_residuals["laws"]["momentum_navier_stokes"]
        scale = float(eng.log[-1]["scale"]["laws/momentum_navier_stokes"])
        mag = residual_magnitude(res)
        # Couette analytic residual ~0; spike frac from FD not applicable — report 0
        spike = 0.0
        tier_p95 = tier_for_score(
            _score(float(weighted_rms(mag, w)), scale)
        )
    else:
        eng = _ns_engine()
        eng.compute_residuals(
            state,
            auto_path_b_derivatives=_grid(closure),
            fill_law_fd=True,
            weighted_norms="report",
            point_weights=w,
        )
        res = eng.last_residuals["laws"]["momentum_navier_stokes"]
        scale = float(eng.log[-1]["scale"]["laws/momentum_navier_stokes"])
        mag = residual_magnitude(res)
        spike = boundary_spike_frac(mag, scale, fd_order=4)
        # p95 / p97.5 / p99 tiers for this closure on clean data
        wr = float(eng.log[-1]["rms_weighted"]["laws/momentum_navier_stokes"])
        tiers = {}
        for q in PERCENTILES:
            sg = float(jnp.percentile(jnp.reshape(mag, (-1,)), q))
            final = min(_score(wr, scale), _score(sg, scale))
            tiers[_pkey(q)] = {
                "score": final,
                "tier": tier_for_score(final),
            }
        return {
            "flow": flow,
            "resolution": resolution,
            "closure": closure,
            "grid": {"n0": shape[0], "n1": shape[1]},
            "boundary_row_deriv_rms": brow,
            "frac_boundary_spikes": spike,
            "clean_tiers": tiers,
        }

    return {
        "flow": flow,
        "resolution": resolution,
        "closure": closure,
        "grid": {"n0": shape[0], "n1": shape[1]},
        "boundary_row_deriv_rms": brow,
        "frac_boundary_spikes": spike,
        "note": "couette uses supplied derivatives for residual; brow vs order-6 ref",
    }


# ---------------------------------------------------------------------------
# Residual helpers for percentile study
# ---------------------------------------------------------------------------


def _run_residual(
    state: Dict[str, Any],
    weights: Any,
    *,
    grid: Optional[PathBGridConfig] = None,
    fill_law_fd: bool = False,
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
    scale = float(eng.log[-1]["scale"]["laws/momentum_navier_stokes"])
    urw = float(eng.log[-1]["rms"]["laws/momentum_navier_stokes"])
    wr = float(eng.log[-1]["rms_weighted"]["laws/momentum_navier_stokes"])
    return res, scale, urw, wr


def _dual_order_eps(
    state: Dict[str, Any],
    weights: Any,
    *,
    closure: str,
    fd_order: int = 4,
) -> Tuple[Any, Any, float, float]:
    """Return (r_hi, eps=|r_hi-r_lo|, scale, wr_hi)."""
    g_hi = _grid(closure, fd_order=fd_order)
    g_lo = _grid(closure, fd_order=2)
    r_hi, scale, _, wr = _run_residual(state, weights, grid=g_hi, fill_law_fd=True)
    r_lo, _, _, _ = _run_residual(state, weights, grid=g_lo, fill_law_fd=True)
    eps = jnp.abs(residual_magnitude(r_hi) - residual_magnitude(r_lo))
    # When shapes match for vector residuals, use magnitude difference of residuals
    # Recompute as magnitude of difference for vector fields
    a = jnp.asarray(r_hi)
    b = jnp.asarray(r_lo)
    if a.shape == b.shape:
        eps = residual_magnitude(a - b)
    return r_hi, eps, scale, wr


def _combo_scores(
    residual: Any,
    weights: Any,
    scale: float,
    wr: float,
    *,
    eps: Optional[Any] = None,
    error_aware: bool = False,
) -> Dict[str, Any]:
    mag = residual_magnitude(residual)
    w = jnp.asarray(weights, dtype=float)
    out: Dict[str, Any] = {}
    for q in PERCENTILES:
        key = _pkey(q)
        if error_aware and eps is not None:
            ea = error_aware_percentile_stat(residual, eps, q, scale_k=scale)
            sg = float(ea["stat"])
        else:
            sg = float(jnp.percentile(jnp.reshape(mag, (-1,)), q))
        final = min(_score(wr, scale), _score(sg, scale))
        out[key] = {
            "score_final": final,
            "tier_final": tier_for_score(final),
            "below_high": final < HIGH,
            "safeguard_stat": sg,
        }
    out["weighted_only"] = {
        "score_final": _score(wr, scale),
        "tier_final": tier_for_score(_score(wr, scale)),
        "below_high": _score(wr, scale) < HIGH,
    }
    return out


def _band_mask(nx: int, ny: int, target_frac: float) -> Tuple[jnp.ndarray, float]:
    n_cols = max(1, min(nx, int(round(target_frac * nx))))
    mask = jnp.zeros((nx, ny), dtype=bool).at[:n_cols, :].set(True)
    return mask, float(jnp.mean(mask.astype(float)))


def case_clean(flow: str, resolution: str, closure: str, *, error_aware: bool) -> Dict[str, Any]:
    if flow == "taylor_green":
        n = RES_TG[resolution]
        xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = taylor_green_2d(x, y, 0.1, re=100.0)
        w = structured_point_weights([xs, ys])
        state = {
            "x": x, "y": y, "u": fields["u"], "u_t": fields["u_t"],
            "p": fields["p"], "re": fields["re"],
        }
    else:
        nx, ny = RES_KOV[resolution]
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        fields = kovasznay_2d(x, y, re=40.0)
        w = structured_point_weights([xs, ys])
        state = {
            "x": x, "y": y, "u": fields["u"], "u_t": fields["u_t"],
            "p": fields["p"], "re": fields["re"],
        }
    if error_aware:
        r, eps, scale, wr = _dual_order_eps(state, w, closure=closure)
        spike = boundary_spike_frac(residual_magnitude(r), scale)
        scores = _combo_scores(r, w, scale, wr, eps=eps, error_aware=True)
    else:
        r, scale, _, wr = _run_residual(
            state, w, grid=_grid(closure), fill_law_fd=True
        )
        spike = boundary_spike_frac(residual_magnitude(r), scale)
        scores = _combo_scores(r, w, scale, wr, error_aware=False)
    return {"flow": flow, "resolution": resolution, "spike_frac": spike, "scores": scores}


def case_noise(
    flow: str, resolution: str, closure: str, noise_levels: Sequence[float], *, error_aware: bool
) -> Dict[str, Any]:
    key = jax.random.PRNGKey(1 if flow == "kovasznay" else 0)
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
    for eps_n in noise_levels:
        key, k1, k2 = jax.random.split(key, 3)
        state = {
            "x": x,
            "y": y,
            "u": fields["u"] + eps_n * u_scale * jax.random.normal(k1, fields["u"].shape),
            "u_t": fields["u_t"],
            "p": fields["p"] + eps_n * p_scale * jax.random.normal(k2, fields["p"].shape),
            "re": fields["re"],
        }
        if error_aware:
            r, eps, scale, wr = _dual_order_eps(state, w, closure=closure)
            scores = _combo_scores(r, w, scale, wr, eps=eps, error_aware=True)
        else:
            r, scale, _, wr = _run_residual(
                state, w, grid=_grid(closure), fill_law_fd=True
            )
            scores = _combo_scores(r, w, scale, wr)
        rows.append({"noise_rel": float(eps_n), **scores})
    return {"rows": rows}


def case_detection(
    flow: str, resolution: str, closure: str, band_fracs: Sequence[float], mags: Sequence[float],
    *, error_aware: bool,
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
    bands = []
    for target in band_fracs:
        region, actual = _band_mask(nx, ny, float(target))
        first = { _pkey(q): None for q in PERCENTILES }
        for mag in mags:
            st = dict(base)
            pg = jnp.asarray(st["p_grad"])
            st["p_grad"] = pg + jnp.zeros_like(pg).at[..., 0].set(
                jnp.where(region, float(mag), 0.0)
            )
            # Exact ∂ — no Path B; error-aware uses a tiny floor eps so ρ≈|r|/δ
            r, scale, _, wr = _run_residual(st, w)
            if error_aware:
                # Dual-order on exact state is ~0; use |r|*0 + scale floor via δ only
                eps = jnp.zeros_like(residual_magnitude(r))
                scores = _combo_scores(r, w, scale, wr, eps=eps, error_aware=True)
            else:
                scores = _combo_scores(r, w, scale, wr)
            for q in PERCENTILES:
                k = _pkey(q)
                if first[k] is None and scores[k]["below_high"]:
                    first[k] = float(mag)
        bands.append({
            "target_band_frac": float(target),
            "actual_band_frac": actual,
            "first_below_high": first,
        })
    return {"bands": bands}


def case_clustered(closure: str, mags: Sequence[float], *, error_aware: bool) -> Dict[str, Any]:
    # Clustered Kov — exact ∂ + localized p_grad (closure unused for exact ∂)
    nx, ny = 48, 40
    xs = stretched_axis(nx, 0.0, 1.0)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    base = ns_state_from_analytic(fields, x, y)
    w = structured_point_weights([xs, ys])
    region = x <= float(xs[max(3, nx // 5)])
    first = {_pkey(q): None for q in PERCENTILES}
    for mag in mags:
        st = dict(base)
        pg = jnp.asarray(st["p_grad"])
        st["p_grad"] = pg + jnp.zeros_like(pg).at[..., 0].set(
            jnp.where(region, float(mag), 0.0)
        )
        r, scale, _, wr = _run_residual(st, w)
        if error_aware:
            scores = _combo_scores(
                r, w, scale, wr, eps=jnp.zeros_like(residual_magnitude(r)), error_aware=True
            )
        else:
            scores = _combo_scores(r, w, scale, wr)
        for q in PERCENTILES:
            k = _pkey(q)
            if first[k] is None and scores[k]["below_high"]:
                first[k] = float(mag)
    # Couette clustered
    nr, nt = 36, 48
    r = stretched_axis(nr, 1.0, 2.0)
    th = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
    fields_c = taylor_couette_2d(r, th, re=100.0)
    w_c = polar_weights(r, th)
    base_c = {
        "x": fields_c["x"], "y": fields_c["y"], "u": fields_c["u"],
        "u_t": fields_c["u_t"], "u_grad": fields_c["u_grad"],
        "u_laplacian": fields_c["u_laplacian"], "p": fields_c["p"],
        "p_grad": fields_c["p_grad"], "re": fields_c["re"],
    }
    region_c = fields_c["r"] <= float(r[max(3, nr // 5)])
    first_c = {_pkey(q): None for q in PERCENTILES}
    for mag in mags:
        st = dict(base_c)
        pg = jnp.asarray(st["p_grad"])
        st["p_grad"] = pg + jnp.zeros_like(pg).at[..., 0].set(
            jnp.where(region_c, float(mag), 0.0)
        )
        resid, scale, _, wr = _run_residual(st, w_c)
        if error_aware:
            scores = _combo_scores(
                resid, w_c, scale, wr,
                eps=jnp.zeros_like(residual_magnitude(resid)), error_aware=True,
            )
        else:
            scores = _combo_scores(resid, w_c, scale, wr)
        for q in PERCENTILES:
            k = _pkey(q)
            if first_c[k] is None and scores[k]["below_high"]:
                first_c[k] = float(mag)
    return {"kovasznay": first, "couette": first_c}


def case_error_aware_no_raise(closure: str) -> Dict[str, Any]:
    """Confirm error-aware does not raise tier vs plain on known nonzero residuals."""
    results = {}
    # Laplace sine: Δφ + π²φ = 0 ⇒ residual of laplace_equation should be small but
    # we audit a manufactured nonzero residual via perturbed TG.
    for name, builder in (
        ("perturbed_taylor_green", "tg"),
        ("perturbed_kovasznay", "kov"),
    ):
        if builder == "tg":
            n = 32
            xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
            ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
            x, y = jnp.meshgrid(xs, ys, indexing="ij")
            fields = taylor_green_2d(x, y, 0.1, re=100.0)
            base = ns_state_from_analytic(fields, x, y)
            w = structured_point_weights([xs, ys])
        else:
            nx, ny = 40, 40
            xs = jnp.linspace(0.0, 1.0, nx)
            ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
            x, y = jnp.meshgrid(xs, ys, indexing="ij")
            fields = kovasznay_2d(x, y, re=40.0)
            base = ns_state_from_analytic(fields, x, y)
            w = structured_point_weights([xs, ys])
        st = dict(base)
        pg = jnp.asarray(st["p_grad"])
        st["p_grad"] = pg + 0.3  # global perturbation → nonzero residual
        r, scale, _, wr = _run_residual(st, w)
        plain = _combo_scores(r, w, scale, wr)
        ea = _combo_scores(
            r, w, scale, wr, eps=jnp.zeros_like(residual_magnitude(r)), error_aware=True
        )
        # With eps=0, δ floor → ρ large → R_sg can be large; use dual-order eps instead
        # For exact-∂ residual, dual-order Path B eps is unrelated; use |r|*0.05 as
        # a generous error estimate that still leaves violation visible.
        eps_gen = 0.05 * residual_magnitude(r) + 1e-12
        ea2 = _combo_scores(r, w, scale, wr, eps=eps_gen, error_aware=True)
        ok = True
        detail = {}
        for q in PERCENTILES:
            k = _pkey(q)
            # Tier must not improve (High > Moderate > Low > Non-Admissible)
            tier_rank = {
                "High Admissibility": 3,
                "Moderate Admissibility": 2,
                "Low Admissibility": 1,
                "Non-Admissible": 0,
            }
            raised = tier_rank.get(ea2[k]["tier_final"], -1) > tier_rank.get(
                plain[k]["tier_final"], -1
            )
            detail[k] = {
                "plain_score": plain[k]["score_final"],
                "plain_tier": plain[k]["tier_final"],
                "ea_score": ea2[k]["score_final"],
                "ea_tier": ea2[k]["tier_final"],
                "raised_tier": raised,
            }
            if raised:
                ok = False
        results[name] = {"ok_no_raise": ok, "detail": detail}

    # Laplace sine (1D): residual of φ_xx + π² φ
    n = 64
    x = jnp.linspace(0.0, 1.0, n)
    phi = jnp.sin(jnp.pi * x)
    eng = ResidualEngine(
        constants={},
        laws=[{
            "name": "laplace_equation",
            "state_map": {"phi_laplacian": "phi_laplacian"},
        }],
        groups=[],
        law_implied_audits=False,
        default_coord_dimension=1,
    )
    # Manufactured nonzero: wrong laplacian
    wrong_lap = -0.5 * (jnp.pi**2) * phi  # should be -π² φ
    w = trapezoid_weights_1d(x)
    eng.compute_residuals(
        {"x": x, "phi": phi, "phi_laplacian": wrong_lap},
        weighted_norms=True,
        weighted_safeguard=True,
        safeguard_percentile=95.0,
        point_weights=w,
    )
    score_plain = float(audit(eng.log)["overall_admissibility_score"])
    eng.clear_log()
    r_field = wrong_lap  # residual ~ lap + π²φ for the law — engine computes it
    # Use engine residual
    eng.compute_residuals(
        {"x": x, "phi": phi, "phi_laplacian": wrong_lap},
        weighted_norms="report",
        point_weights=w,
    )
    res = eng.last_residuals["laws"]["laplace_equation"]
    scale = float(eng.log[-1]["scale"]["laws/laplace_equation"])
    wr = float(eng.log[-1]["rms_weighted"]["laws/laplace_equation"])
    plain_s = _combo_scores(res, w, scale, wr)
    ea_s = _combo_scores(
        res, w, scale, wr, eps=0.05 * residual_magnitude(res) + 1e-12, error_aware=True
    )
    tier_rank = {
        "High Admissibility": 3,
        "Moderate Admissibility": 2,
        "Low Admissibility": 1,
        "Non-Admissible": 0,
    }
    raised = tier_rank.get(ea_s["p95"]["tier_final"], -1) > tier_rank.get(
        plain_s["p95"]["tier_final"], -1
    )
    results["laplace_sine"] = {
        "ok_no_raise": not raised,
        "plain_score": plain_s["p95"]["score_final"],
        "ea_score": ea_s["p95"]["score_final"],
        "plain_tier": plain_s["p95"]["tier_final"],
        "ea_tier": ea_s["p95"]["tier_final"],
        "plain_engine_score": score_plain,
    }
    return results


def decide_for_config(cfg_block: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer highest percentile among p99, p97.5, p95 that passes the rule."""
    def clean_high(cand: str) -> bool:
        for flow in ("taylor_green", "kovasznay"):
            for res in ("coarse", "fine"):
                tier = cfg_block["clean"][f"{flow}_{res}"]["scores"][cand]["tier_final"]
                if not str(tier).startswith("High"):
                    return False
        return True

    def tg_noise_high(cand: str) -> bool:
        for res in ("coarse", "fine"):
            rows = cfg_block["noise"][f"taylor_green_{res}"]["rows"]
            for row in rows:
                if abs(row["noise_rel"] - 1e-3) < 1e-15:
                    if row[cand]["below_high"]:
                        return False
        return True

    def detect_a3(cand: str) -> bool:
        kov = cfg_block["clustered"]["kovasznay"].get(cand)
        cou = cfg_block["clustered"]["couette"].get(cand)
        return kov is not None and cou is not None and kov <= 0.1 + 1e-12 and cou <= 0.1 + 1e-12

    cands = {}
    for cand, pct in (("p99", 99.0), ("p97_5", 97.5), ("p95", 95.0)):
        ch = clean_high(cand)
        nh = tg_noise_high(cand)
        det = detect_a3(cand)
        cands[cand] = {
            "percentile": pct,
            "clean_high_both": ch,
            "tg_noise_1e-3_high": nh,
            "detect_a3_mag_0.1": det,
            "passes": bool(ch and nh and det),
        }
    chosen = "p95"
    for cand in ("p99", "p97_5", "p95"):
        if cands[cand]["passes"]:
            chosen = cand
            break
    return {
        "candidates": cands,
        "chosen": chosen,
        "chosen_percentile": cands[chosen]["percentile"],
    }


def run_percentile_suite(closure: str, *, error_aware: bool) -> Dict[str, Any]:
    label = f"{closure}+ea" if error_aware else closure
    print(f"\n=== percentile suite: {label} ===")
    band_fracs = [0.02, 0.05, 0.10, 0.20]
    mags = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    noise_levels = [0.0, 1e-4, 1e-3, 1e-2]
    out: Dict[str, Any] = {
        "closure": closure,
        "error_aware": error_aware,
        "clean": {},
        "noise": {},
        "detection": {},
        "clustered": {},
    }
    for flow in ("taylor_green", "kovasznay"):
        for res in ("coarse", "fine"):
            print(f"  clean {flow}/{res}")
            out["clean"][f"{flow}_{res}"] = case_clean(
                flow, res, closure, error_aware=error_aware
            )
            print(
                "   ",
                {k: out["clean"][f"{flow}_{res}"]["scores"][k]["tier_final"][:4]
                 for k in ("p95", "p97_5", "p99")},
                "spike",
                f"{out['clean'][f'{flow}_{res}']['spike_frac']:.4f}",
            )
    for flow in ("taylor_green", "kovasznay"):
        for res in ("coarse", "fine"):
            print(f"  noise {flow}/{res}")
            out["noise"][f"{flow}_{res}"] = case_noise(
                flow, res, closure, noise_levels, error_aware=error_aware
            )
    for flow in ("taylor_green", "kovasznay"):
        for res in ("coarse", "fine"):
            print(f"  detection {flow}/{res}")
            out["detection"][f"{flow}_{res}"] = case_detection(
                flow, res, closure, band_fracs, mags, error_aware=error_aware
            )
    print("  clustered A3")
    out["clustered"] = case_clustered(closure, [0.02, 0.05, 0.1, 0.2, 0.5, 1.0], error_aware=error_aware)
    print("   ", out["clustered"])
    out["decision"] = decide_for_config(out)
    print("  DECISION", out["decision"]["chosen"], out["decision"]["candidates"])
    return out


def pick_best_closure(closure_table: List[Dict[str, Any]]) -> str:
    """
    Best closure: minimize clean Kov coarse spike frac among those that keep
    p95 High on clean Kov coarse; break ties by boundary-row error on Kov coarse.
    """
    # Index clean tiers from closure_table entries that have clean_tiers
    scored = []
    for row in closure_table:
        if row["flow"] != "kovasznay" or row["resolution"] != "coarse":
            continue
        tiers = row.get("clean_tiers") or {}
        p95_high = str(tiers.get("p95", {}).get("tier", "")).startswith("High")
        scored.append((
            0 if p95_high else 1,
            float(row["frac_boundary_spikes"]),
            float(row["boundary_row_deriv_rms"]),
            row["closure"],
        ))
    scored.sort()
    return scored[0][3] if scored else "matching"


def main() -> None:
    report: Dict[str, Any] = {
        "phase": "A4_boundary_safeguard",
        "high_cutoff": HIGH,
        "closures": list(CLOSURES),
        "closure_accuracy": [],
        "error_aware_no_raise": {},
        "percentile_suites": {},
        "recommendation": {},
    }

    print("=== (1) closure accuracy ===")
    for closure in CLOSURES:
        for flow in ("taylor_green", "kovasznay", "couette"):
            for res in ("coarse", "fine"):
                print(f"  {closure} {flow}/{res}")
                row = closure_derivative_errors(closure, flow, res)
                report["closure_accuracy"].append(row)
                print(
                    f"    brow={row['boundary_row_deriv_rms']:.4e} "
                    f"spike={row['frac_boundary_spikes']:.4f}"
                )

    best_closure = pick_best_closure(report["closure_accuracy"])
    # Also prefer a closure that makes p97.5 High on Kov coarse if any does
    for closure in ("order_reduced", "sbp", "bc_aware", "matching"):
        for row in report["closure_accuracy"]:
            if (
                row["closure"] == closure
                and row["flow"] == "kovasznay"
                and row["resolution"] == "coarse"
                and row.get("clean_tiers")
                and str(row["clean_tiers"].get("p97_5", {}).get("tier", "")).startswith("High")
            ):
                best_closure = closure
                break
        else:
            continue
        break
    report["best_closure"] = best_closure
    print("BEST_CLOSURE", best_closure)

    print("=== (2) error-aware no-raise checks ===")
    report["error_aware_no_raise"] = case_error_aware_no_raise(best_closure)
    print(json.dumps(report["error_aware_no_raise"], indent=2, default=str)[:1500])

    print("=== (3) percentile suites ===")
    report["percentile_suites"]["matching"] = run_percentile_suite(
        "matching", error_aware=False
    )
    if best_closure == "matching":
        report["percentile_suites"]["best_closure"] = report["percentile_suites"][
            "matching"
        ]
    else:
        report["percentile_suites"]["best_closure"] = run_percentile_suite(
            best_closure, error_aware=False
        )
    report["percentile_suites"]["best_closure_error_aware"] = run_percentile_suite(
        best_closure, error_aware=True
    )

    # Recommendation: among the three suites, pick the one whose decision has the
    # highest chosen percentile; tie-break prefer error_aware then best_closure.
    ranking = []
    for key, block in report["percentile_suites"].items():
        if block is None:
            continue
        d = block["decision"]
        ranking.append((
            -float(d["chosen_percentile"]),
            0 if "error_aware" in key else 1,
            key,
            d,
        ))
    ranking.sort()
    best_key = ranking[0][2]
    best_dec = ranking[0][3]
    report["recommendation"] = {
        "suite": best_key,
        "closure": report["percentile_suites"][best_key]["closure"],
        "error_aware": report["percentile_suites"][best_key]["error_aware"],
        "safeguard_percentile": best_dec["chosen_percentile"],
        "decision": best_dec,
        "note": (
            "Prefer highest percentile that keeps clean TG & Kov High on both "
            "resolutions, TG High at noise 1e-3, and A3 detection at mag 0.1."
        ),
    }
    print("\nRECOMMENDATION", json.dumps(report["recommendation"], indent=2, default=str))

    out = ROOT / "docs/phase_a4_boundary_safeguard.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
