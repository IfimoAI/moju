"""Tests for Path B finite-difference law-input fill and derivative key naming."""

import jax.numpy as jnp

from moju.monitor.derivative_keys import derivative_state_key
from moju.monitor.path_b_derivatives import PathBGridConfig, _uniform_1d_spacing, fill_path_b_derivatives


class TestDerivativeKeys:
    def test_derivative_state_key(self):
        assert derivative_state_key("T", "x") == "d_T_dx"
        assert derivative_state_key("T", "y") == "d_T_dy"
        assert derivative_state_key("mu", "t") == "d_mu_dt"


class TestFillPathBLawFD:
    def test_law_fd_fills_laplacian_1d(self):
        x = jnp.linspace(0.0, 1.0, 16)
        phi = x**2
        state = {"phi": phi, "x": x}
        laws = [{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}]
        out, w = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
            fill_law_recipes=True,
        )
        assert not w
        assert "phi_laplacian" in out
        hx = _uniform_1d_spacing(x)
        assert hx is not None
        expect = jnp.asarray(jnp.gradient(jnp.gradient(phi, hx), hx))
        assert jnp.allclose(out["phi_laplacian"], expect, rtol=1e-4, atol=1e-4)

    def test_skip_existing_non_none_law_fd(self):
        x = jnp.linspace(0.0, 1.0, 8)
        phi = x**2
        sentinel = jnp.full_like(phi, 99.0)
        state = {"phi": phi, "x": x, "phi_laplacian": sentinel}
        laws = [{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}]
        out, _ = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
            fill_law_recipes=True,
        )
        assert jnp.allclose(out["phi_laplacian"], sentinel)
