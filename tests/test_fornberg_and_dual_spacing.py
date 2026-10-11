"""Fornberg non-uniform FD and dual-spacing derivative-error estimation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from moju.monitor.derivative_error import (
    estimate_errors_dual_spacing,
    fill_dual_spacing_alternate,
)
from moju.monitor.fd_stencils import (
    diff1_nonuniform,
    diff1_uniform,
    fornberg_weights,
)
from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives


def test_fornberg_uniform_matches_order2_interior():
    import numpy as np

    x = np.linspace(0.0, 1.0, 5)
    w = fornberg_weights(x[2], x[1:4], 1)
    # Central 3-point first derivative: [-1/(2h), 0, 1/(2h)]
    h = x[1] - x[0]
    assert abs(w[1, 0] - (-0.5 / h)) < 1e-10
    assert abs(w[1, 1]) < 1e-10
    assert abs(w[1, 2] - (0.5 / h)) < 1e-10


def test_nonuniform_diff1_on_stretched_grid():
    # Chebyshev-like clustering near ends
    n = 40
    xi = jnp.linspace(-1.0, 1.0, n)
    x = jnp.sin(0.5 * jnp.pi * xi)  # non-uniform in physical x on [-1,1] mapped
    # Use x = cumsum of varying spacing on [0, 2π]
    h = 0.05 + 0.04 * jnp.sin(jnp.linspace(0, jnp.pi, n - 1))
    x = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(h)])
    x = x * (2.0 * jnp.pi / float(x[-1]))
    f = jnp.sin(x)
    df = diff1_nonuniform(f, x, order=4, axis=0)
    err = float(jnp.max(jnp.abs(df[4:-4] - jnp.cos(x[4:-4]))))
    assert err < 5e-3


def test_uniform_fornberg_path_via_fd_order_matches_explicit():
    n = 32
    L = 2.0 * jnp.pi
    x = jnp.linspace(0.0, L, n, endpoint=False)
    h = float(L / n)
    f = jnp.sin(x)
    a = diff1_uniform(f, h, order=4, axis=0)
    b = diff1_nonuniform(f, x, order=4, axis=0)
    # Interior agreement (boundaries differ: periodic-like linspace vs one-sided)
    assert float(jnp.max(jnp.abs(a[5:-5] - b[5:-5]))) < 1e-4


def test_dual_spacing_fill_and_errors():
    n = 32
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    u = jnp.stack([jnp.sin(x) * jnp.cos(y), -jnp.cos(x) * jnp.sin(y)], axis=-1)
    state = {"x": x, "y": y, "u": u, "p": 0.25 * (jnp.cos(2 * x) + jnp.cos(2 * y)), "re": 100.0}
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
    grid = PathBGridConfig(spatial_dimension=2, steady=True, fd_order=4)
    fine, _ = fill_path_b_derivatives(
        {**state, "u_t": jnp.zeros_like(u)},
        laws_spec=laws,
        grid=grid,
        fill_law_recipes=True,
    )
    coarse = fill_dual_spacing_alternate(
        {k: v for k, v in state.items()},
        laws_spec=laws,
        grid=grid,
        fill_law_recipes=True,
    )
    assert coarse["u"].shape[0] == n // 2
    errs = estimate_errors_dual_spacing(fine, coarse, spatial_ndim=2, order=2)
    assert "u_grad" in errs or "u_laplacian" in errs or "p_grad" in errs


def test_spectral_non_periodic_raises():
    from moju.monitor.path_b_derivatives import fill_path_b_derivatives

    n = 16
    x = jnp.linspace(0, 1, n)
    state = {"x": x, "phi": jnp.sin(2 * jnp.pi * x)}
    grid = PathBGridConfig(
        spatial_dimension=1, steady=True, diff_method="spectral", periodic=False
    )
    with pytest.raises(ValueError, match="periodic"):
        fill_path_b_derivatives(
            state,
            laws_spec=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
            grid=grid,
            fill_law_recipes=True,
        )
