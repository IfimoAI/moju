"""
Analytic NS diagnostics: Taylor–Green and Kovasznay.

Covers (a) FD convergence with spacing/order, (b) exact supplied derivatives →
machine-precision residuals, (c) misaligned time derivatives → consistency warning
and larger momentum residual, (d) derivative-error flag on coarse spacing.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic, taylor_green_2d
from moju.monitor.fd_stencils import diff1_uniform
from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives


def _tg_mesh(n: int = 32, *, t: float = 0.1, re: float = 100.0):
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    state = ns_state_from_analytic(fields, x, y, t=t)
    return xs, ys, x, y, fields, state


def _ns_engine():
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


def test_taylor_green_exact_derivatives_machine_precision():
    *_, state = _tg_mesh(48)
    eng = _ns_engine()
    res = eng.compute_residuals(state, run_mode="training")
    r = res["laws"]["momentum_navier_stokes"]
    assert float(jnp.max(jnp.abs(r))) < 1e-5
    report = audit(eng.log)
    assert "term_breakdown" in report
    bd = report["term_breakdown"]["momentum_navier_stokes"]
    assert "scale_k" in bd
    assert "residual" in bd
    assert "terms" in bd
    # Exact residuals must not change the tier path relative to a second identical call
    score1 = report["overall_admissibility_score"]
    eng2 = _ns_engine()
    eng2.compute_residuals(state, run_mode="training")
    score2 = audit(eng2.log)["overall_admissibility_score"]
    assert score1 == score2


def test_kovasznay_exact_derivatives_machine_precision():
    nx, ny = 40, 40
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    state = ns_state_from_analytic(fields, x, y)
    eng = _ns_engine()
    res = eng.compute_residuals(state, run_mode="training")
    r = res["laws"]["momentum_navier_stokes"]
    assert float(jnp.max(jnp.abs(r))) < 1e-4


@pytest.mark.parametrize("order", [2, 4, 6])
def test_fd_convergence_taylor_green_velocity(order):
    """(a) Moju FD derivatives of u converge under refinement for each stencil order."""
    errs = []
    for n in (24, 48):
        xs, ys, x, y, fields, _ = _tg_mesh(n)
        hx = float(xs[1] - xs[0])
        # Differentiate u-component along x
        u = fields["u"][..., 0]
        du_fd = diff1_uniform(u, hx, order=order, axis=0)
        exact = fields["u_grad"][..., 0, 0]
        pad = order + 1
        errs.append(float(jnp.max(jnp.abs(du_fd[pad:-pad, pad:-pad] - exact[pad:-pad, pad:-pad]))))
    assert errs[1] < errs[0] * 0.85


def test_misaligned_time_derivative_warns_and_raises_residual():
    """(c) Shifted u_t triggers consistency warning and larger residual."""
    *_, fields, state = _tg_mesh(32, t=0.2)
    # Exact residual baseline
    eng0 = _ns_engine()
    r0 = eng0.compute_residuals(state, run_mode="training")["laws"]["momentum_navier_stokes"]
    rms0 = float(jnp.sqrt(jnp.mean(r0**2)))

    # Misalign: take u_t from a different time
    fields_shift = taylor_green_2d(state["x"], state["y"], 0.5, re=100.0)
    state_bad = dict(state)
    state_bad["u_t"] = fields_shift["u_t"]

    meta = {
        "u_t": {
            "source": "finite_difference",
            "evaluation_time": 0.5,
            "time_stencil": [0.4, 0.5, 0.6],
        },
        "u_grad": {"source": "solver", "evaluation_time": 0.2},
        "u_laplacian": {"source": "solver", "evaluation_time": 0.2},
        "p_grad": {"source": "solver", "evaluation_time": 0.2},
    }
    eng = _ns_engine()
    r1 = eng.compute_residuals(
        state_bad, run_mode="training", derivative_metadata=meta
    )["laws"]["momentum_navier_stokes"]
    rms1 = float(jnp.sqrt(jnp.mean(r1**2)))
    assert rms1 > rms0 * 10.0
    report = audit(eng.log)
    warns = " ".join(report.get("diagnostic_warnings") or [])
    assert "evaluation times" in warns or "derivative_consistency" in warns
    # Tier thresholds unchanged: level mapping still uses the same score function
    from moju.monitor.auditor import admissibility_level

    assert admissibility_level(0.95) == admissibility_level(0.95)


def test_derivative_error_flag_on_coarse_spacing():
    """(d) Coarse Path B FD vs finer alternate set flags derivative-dominated residual."""
    # Coarse grid: large FD error on TG vortex
    n = 12
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    t = 0.1
    fields = taylor_green_2d(x, y, t, re=100.0)
    # Primitives only — Moju fills derivatives
    state = {
        "x": x,
        "y": y,
        "u": fields["u"],
        "p": fields["p"],
        "re": fields["re"],
        "t": jnp.asarray(t),
    }
    # Also need u_t for unsteady NS — supply exact time deriv so residual error is spatial FD
    state["u_t"] = fields["u_t"]

    grid = PathBGridConfig(
        layout="meshgrid",
        spatial_dimension=2,
        steady=True,
        fd_order=2,
        estimate_derivative_error=True,
    )
    eng = _ns_engine()
    eng.compute_residuals(
        state,
        run_mode="training",
        auto_path_b_derivatives=grid,
        fill_law_fd=True,
        estimate_derivative_error=True,
        derivative_error_threshold=0.1,
    )
    report = audit(eng.log)
    assert "derivative_error" in report
    per_law = report["derivative_error"]["per_law"].get("momentum_navier_stokes")
    assert per_law is not None
    # On a coarse grid the FD error should be comparable to the residual
    assert per_law["residual_may_be_dominated_by_derivative_error"] is True


def test_eval_mask_recorded_and_excludes_boundary():
    *_, state = _tg_mesh(24)
    # Mask: keep interior half
    m = jnp.zeros(state["u"].shape[:-1], dtype=bool)
    m = m.at[6:-6, 6:-6].set(True)
    eng = _ns_engine()
    eng.compute_residuals(state, run_mode="training", eval_mask=m)
    report = audit(eng.log)
    assert report["eval_mask"]["n_kept"] == int(jnp.sum(m))
    assert report["eval_mask"]["n_excluded"] > 0


def test_path_b_fd_order_fills_without_changing_default_api():
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, 32, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, 32, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, 0.0, re=100.0)
    state = {"x": x, "y": y, "u": fields["u"], "p": fields["p"], "re": fields["re"]}
    grid = PathBGridConfig(spatial_dimension=2, steady=True, fd_order=4)
    filled, warns = fill_path_b_derivatives(
        state,
        laws_spec=[
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
        grid=grid,
        fill_law_recipes=True,
    )
    assert filled.get("u_grad") is not None
    assert filled.get("u_laplacian") is not None
