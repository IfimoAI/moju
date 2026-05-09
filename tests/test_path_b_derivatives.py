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
        # phi = x^2 => d^2 phi / dx^2 = 2 (4th-order FD should match closely on uniform grid)
        assert jnp.allclose(out["phi_laplacian"], 2.0, rtol=2e-3, atol=2e-3)

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

    def test_nonuniform_spacing_warns_and_uses_second_order(self):
        x = jnp.array([0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
        phi = x**2
        state = {"phi": phi, "x": x}
        laws = [{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}]
        out, w = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
            fill_law_recipes=True,
        )
        assert "phi_laplacian" in out
        assert any("uniform" in s.lower() or "2nd-order" in s for s in w)
