"""JAX ↔ Torch parity for FD order-4 Path B fill and residual scores."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
torch = pytest.importorskip("torch", reason="torch not installed")

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import taylor_green_2d
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.torch import TorchResidualEngine
from moju.torch._path_b import fill_path_b_derivatives_torch


def test_fd4_grad_parity_sine():
    from moju.monitor.fd_stencils import diff1_uniform
    from moju.torch._fd_stencils import diff1_uniform_torch

    n = 48
    L = 2.0 * jnp.pi
    x = jnp.linspace(0.0, L, n, endpoint=False)
    h = float(L / n)
    f = jnp.sin(x)
    dj = diff1_uniform(f, h, order=4, axis=0)
    dt = diff1_uniform_torch(torch.tensor(f.tolist()), h, order=4, axis=0)
    err = float(jnp.max(jnp.abs(dj - jnp.asarray(dt.detach().numpy()))))
    assert err < 1e-5


def test_path_b_fd4_residual_parity_taylor_green():
    n, t, re = 24, 0.1, 100.0
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    state = {
        "x": x,
        "y": y,
        "u": fields["u"],
        "u_t": fields["u_t"],
        "p": fields["p"],
        "re": fields["re"],
    }
    laws = [
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
    ]
    grid = PathBGridConfig(spatial_dimension=2, steady=True, fd_order=4, fd_order_time=4)
    eng = ResidualEngine(
        constants={},
        laws=laws,
        groups=[],
        law_implied_audits=False,
        default_coord_dimension=2,
    )
    eng.compute_residuals(
        state,
        auto_path_b_derivatives=grid,
        fill_law_fd=True,
        weighted_norms=False,
    )
    r_jax = eng.last_residuals["laws"]["momentum_navier_stokes"]
    report_j = audit(eng.log)

    state_t = {k: torch.tensor(jnp.asarray(v).tolist()) for k, v in state.items()}
    filled, _ = fill_path_b_derivatives_torch(state_t, fd_order=4, fd_order_time=4)
    # Compare u_grad
    ug_j = eng.log  # use filled via recompute
    from moju.monitor.path_b_derivatives import fill_path_b_derivatives

    filled_j, _, _ = fill_path_b_derivatives(
        {k: v for k, v in state.items()},
        laws_spec=laws,
        grid=grid,
        fill_law_recipes=True,
        return_provenance=True,
    )
    g_err = float(
        jnp.max(
            jnp.abs(
                jnp.asarray(filled_j["u_grad"])
                - jnp.asarray(filled["u_grad"].detach().numpy())
            )
        )
    )
    assert g_err < 5e-5

    torch_eng = TorchResidualEngine(
        laws=laws,
        constants={},
        law_implied_audits=False,
        path_b_fill=True,
        path_b_fd_order=4,
        path_b_fd_order_time=4,
    )
    tres = torch_eng.compute_residuals_torch(state_t)
    r_t = tres["laws"]["momentum_navier_stokes"]
    r_err = float(
        jnp.max(jnp.abs(jnp.asarray(r_jax) - jnp.asarray(r_t.detach().numpy())))
    )
    assert r_err < 1e-4
    assert report_j["overall_admissibility_level"].startswith("High")
