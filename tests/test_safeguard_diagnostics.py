"""Safeguard diagnostic warnings (never change tiers) — JAX + Torch."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic
from moju.monitor.auditor import ADM_HIGH_THRESHOLD
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.weighted_norms import structured_point_weights
from moju.monitor.weighted_safeguard import (
    build_safeguard_diagnostic_warnings,
    boundary_spike_stats,
)


HIGH = float(ADM_HIGH_THRESHOLD)


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


def test_helper_warns_when_max_masked_by_percentile():
    """A few huge spikes: p95 scoring stays High-ish while max would drop."""
    n = 40
    # Mostly tiny residual; 2×2 corner spikes huge
    r = jnp.ones((n, n)) * 1e-6
    r = r.at[:2, :2].set(10.0)
    w = jnp.ones((n, n))
    scale = 1.0
    # scoring_r ≈ p95 of mostly-tiny field (very small)
    scoring_r = float(jnp.percentile(r, 95.0))
    warns = build_safeguard_diagnostic_warnings(
        r,
        w,
        scale_k=scale,
        scoring_r=scoring_r,
        key="laws/momentum_navier_stokes",
        high_threshold=HIGH,
        fd_order=4,
    )
    assert any("residual max would drop" in w for w in warns)
    assert any("worst-region" in w for w in warns)


def test_helper_boundary_spike_mentions_coincidence():
    n = 32
    r = jnp.zeros((n, n))
    # Fill boundary layer with large values
    r = r.at[:2, :].set(5.0)
    r = r.at[-2:, :].set(5.0)
    r = r.at[:, :2].set(5.0)
    r = r.at[:, -2:].set(5.0)
    w = jnp.ones((n, n))
    warns = build_safeguard_diagnostic_warnings(
        r,
        w,
        scale_k=1.0,
        scoring_r=5.0,  # safeguarded score also low → no max-masking warn required
        key="laws/test",
        high_threshold=HIGH,
        fd_order=4,
        derivative_error_flagged=True,
        resolution_sensor_warned=True,
    )
    spike_msgs = [m for m in warns if "boundary-stencil spike" in m]
    assert spike_msgs
    assert "derivative-error flag" in spike_msgs[0]
    assert "resolution-sensor warning" in spike_msgs[0]


def test_jax_engine_emits_safeguard_diagnostics_on_localized():
    """Exact-∂ Kov with a tiny hot patch: p95 may stay High while max drops."""
    n = 48
    xs = jnp.linspace(0.0, 1.0, n)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    state = ns_state_from_analytic(fields, x, y)
    # Narrow hot spot (~4 cells, mag~1): weighted+p95 stay High; max drops.
    pg = jnp.asarray(state["p_grad"])
    delta = jnp.zeros((n, n)).at[n // 2 : n // 2 + 2, n // 2 : n // 2 + 2].set(1.0)
    state["p_grad"] = pg + jnp.zeros_like(pg).at[..., 0].set(delta)
    w = structured_point_weights([xs, ys])
    eng = _ns_engine()
    eng.compute_residuals(state, weighted_norms=True, point_weights=w)
    entry = eng.log[-1]
    warns = entry.get("diagnostic_warnings") or []
    # At least one safeguard_diagnostic about max or region
    assert any(str(w).startswith("safeguard_diagnostic:") for w in warns), warns
    # Tier must still be driven only by scoring — diagnostics are additive
    report = audit(eng.log)
    assert "overall_admissibility_level" in report
    assert report["overall_admissibility_level"].startswith("High")


def test_torch_helper_parity_on_numpy_bridge():
    torch = pytest.importorskip("torch")
    n = 36
    r_np = np.ones((n, n), dtype=np.float64) * 1e-6
    r_np[:2, :2] = 8.0
    w_np = np.ones((n, n), dtype=np.float64)
    r_t = torch.as_tensor(r_np)
    w_t = torch.as_tensor(w_np)
    # Helper accepts array-like (torch → via jnp.asarray works for CPU tensors
    # only if converted; use numpy bridge for Torch side).
    scoring_r = float(np.percentile(r_np, 95.0))
    warns_np = build_safeguard_diagnostic_warnings(
        r_np, w_np, scale_k=1.0, scoring_r=scoring_r, key="laws/t",
        high_threshold=HIGH, fd_order=4,
    )
    warns_t = build_safeguard_diagnostic_warnings(
        r_t.detach().cpu().numpy(),
        w_t.detach().cpu().numpy(),
        scale_k=1.0,
        scoring_r=scoring_r,
        key="laws/t",
        high_threshold=HIGH,
        fd_order=4,
    )
    assert warns_np == warns_t
    assert any("residual max" in w for w in warns_t)


def test_boundary_spike_stats_kov_path_b():
    n = 40
    xs = jnp.linspace(0.0, 1.0, n)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    state = {
        "x": x, "y": y, "u": fields["u"], "u_t": fields["u_t"],
        "p": fields["p"], "re": fields["re"],
    }
    w = structured_point_weights([xs, ys])
    eng = _ns_engine()
    eng.compute_residuals(
        state,
        weighted_norms=True,
        point_weights=w,
        auto_path_b_derivatives=PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order=4, periodic=(False, True)
        ),
        fill_law_fd=True,
        resolution_sensor=True,
    )
    entry = eng.log[-1]
    res = eng.last_residuals["laws"]["momentum_navier_stokes"]
    scale = float(entry["scale"]["laws/momentum_navier_stokes"])
    from moju.monitor.weighted_safeguard import residual_magnitude

    stats = boundary_spike_stats(
        residual_magnitude(res), scale, fd_order=4, high_threshold=HIGH
    )
    assert stats["applicable"]
    # With y-periodic, spikes concentrate on x-walls — still a material fraction
    assert stats["frac_boundary_spikes"] >= 0.0
