"""Compatibility: 1.x legacy golden path + final 2.0 defaults JAX/Torch parity."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import ns_state_from_analytic, taylor_green_2d
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.weighted_norms import structured_point_weights


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


def test_legacy_fd_unweighted_matches_1x_style_high_on_tg_supplied():
    """1.3.1-style: supplied derivatives + unweighted scoring stays High."""
    n = 32
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, 0.1, re=100.0)
    state = ns_state_from_analytic(fields, x, y, t=0.1)
    eng = _ns_engine()
    eng.compute_residuals(
        state,
        weighted_norms=False,
        auto_path_b_derivatives=PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order="legacy", fd_order_time="legacy"
        ),
        fill_law_fd=False,
    )
    report = audit(eng.log)
    assert report["overall_admissibility_level"].startswith("High")


def test_final_defaults_path_b_periodic_tg_high():
    n = 32
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, 0.1, re=100.0)
    w = structured_point_weights([xs, ys])
    state = {
        "x": x, "y": y, "u": fields["u"], "u_t": fields["u_t"],
        "p": fields["p"], "re": fields["re"],
    }
    eng = _ns_engine()
    eng.compute_residuals(
        state,
        weighted_norms=True,
        point_weights=w,
        auto_path_b_derivatives=PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order=4, periodic=(True, True)
        ),
        fill_law_fd=True,
    )
    report = audit(eng.log)
    assert report["overall_admissibility_level"].startswith("High")
    entry = eng.log[-1]
    assert entry["weighted_norms"]["safeguard"]["percentile"] == 95.0


def test_jax_torch_parity_final_defaults():
    torch = pytest.importorskip("torch")
    from moju.torch._path_b import fill_path_b_derivatives_torch
    from moju.monitor.path_b_derivatives import fill_path_b_derivatives

    n = 32
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, 0.1, re=100.0)
    state = {
        "x": x, "y": y, "u": fields["u"], "u_t": fields["u_t"],
        "p": fields["p"], "re": fields["re"],
    }
    laws = [{
        "name": "momentum_navier_stokes",
        "state_map": {
            "u_t": "u_t", "u": "u", "u_grad": "u_grad",
            "p_grad": "p_grad", "u_laplacian": "u_laplacian", "re": "re",
        },
    }]
    grid = PathBGridConfig(
        spatial_dimension=2, steady=True, fd_order=4, periodic=(True, True)
    )
    filled_j, _, _ = fill_path_b_derivatives(
        dict(state), laws_spec=laws, grid=grid, fill_law_recipes=True, return_provenance=True
    )
    state_t = {k: torch.tensor(jnp.asarray(v).tolist()) for k, v in state.items()}
    filled_t, _ = fill_path_b_derivatives_torch(
        state_t, fd_order=4, periodic=(True, True)
    )
    g_err = float(
        jnp.max(
            jnp.abs(
                jnp.asarray(filled_j["u_grad"])
                - jnp.asarray(filled_t["u_grad"].detach().numpy())
            )
        )
    )
    assert g_err < 1e-4
