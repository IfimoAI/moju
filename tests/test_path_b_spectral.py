"""Tests for opt-in periodic Fourier Path B spatial derivatives."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from moju.monitor import ResidualEngine, fill_path_b_spectral
from moju.monitor.law_fd_recipes import fill_law_fd_from_primitives
from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives
from moju.monitor.path_b_spectral import (
    period_length_1d,
    spectral_diff_along_axis,
    validate_spectral_grid_config,
)
from moju.piratio.laws import Laws


def _periodic_1d_grid(n: int = 64, L: float = 2.0 * math.pi):
    x = jnp.linspace(0.0, L, n, endpoint=False)
    return x, L


class TestSpectralPrimitives:
    def test_sin_mode_first_and_second_derivative(self):
        x, L = _periodic_1d_grid()
        u = jnp.sin(x)
        ux = spectral_diff_along_axis(u, 0, L, order=1)
        uxx = spectral_diff_along_axis(u, 0, L, order=2)
        assert float(jnp.max(jnp.abs(ux - jnp.cos(x)))) < 5e-4
        assert float(jnp.max(jnp.abs(uxx + jnp.sin(x)))) < 5e-4

    def test_period_length_uniform(self):
        x, L = _periodic_1d_grid(n=32, L=4.0)
        assert abs(period_length_1d(x, 32) - L) < 1e-12

    def test_period_length_rejects_nonuniform(self):
        x = jnp.asarray([0.0, 0.1, 0.3, 0.6, 1.0])
        with pytest.raises(ValueError, match="uniform"):
            period_length_1d(x, 5)

    def test_validate_requires_periodic(self):
        with pytest.raises(ValueError, match="periodic=True"):
            validate_spectral_grid_config(
                PathBGridConfig(diff_method="spectral", periodic=False)
            )


class TestSpectralFillRecipes:
    def test_burgers_fills_grad_and_laplacian(self):
        x, L = _periodic_1d_grid()
        u = jnp.sin(x)[:, None]
        laws = [
            {
                "name": "burgers_equation",
                "state_map": {
                    "u_t": "u_t",
                    "u": "u",
                    "u_grad": "u_grad",
                    "u_laplacian": "u_laplacian",
                    "re": "re",
                    "U": "U",
                    "L": "L",
                },
                "fn": Laws.burgers_equation,
            }
        ]
        grid = PathBGridConfig(
            diff_method="spectral",
            periodic=True,
            spatial_dimension=1,
            steady=True,
            layout="separable",
        )
        state, warns = fill_law_fd_from_primitives(
            {"u": u, "x": x},
            laws,
            constants={"re": 100.0, "U": 1.0, "L": L},
            grid=grid,
        )
        assert state["u_grad"] is not None
        assert state["u_laplacian"] is not None
        # Jacobian (..., 1, 1); laplacian (..., 1)
        ux = state["u_grad"][..., 0, 0]
        uxx = state["u_laplacian"][..., 0]
        assert float(jnp.max(jnp.abs(ux - jnp.cos(x)))) < 5e-4
        assert float(jnp.max(jnp.abs(uxx + jnp.sin(x)))) < 5e-4

    def test_fill_path_b_spectral_alias(self):
        x, L = _periodic_1d_grid(n=32)
        u = jnp.sin(2.0 * x)[:, None]
        laws = [
            {
                "name": "burgers_equation",
                "state_map": {
                    "u": "u",
                    "u_grad": "u_grad",
                    "u_laplacian": "u_laplacian",
                    "u_t": "u_t",
                    "re": "re",
                    "U": "U",
                    "L": "L",
                },
            }
        ]
        state, _ = fill_path_b_spectral(
            {"u": u, "x": x},
            laws_spec=laws,
            constants={"re": 50.0, "U": 1.0, "L": L},
            grid=PathBGridConfig(spatial_dimension=1, layout="separable"),
        )
        assert "u_grad" in state and "u_laplacian" in state

    def test_does_not_overwrite_existing(self):
        x, L = _periodic_1d_grid(n=32)
        u = jnp.sin(x)[:, None]
        sentinel = jnp.ones_like(u)
        laws = [
            {
                "name": "burgers_equation",
                "state_map": {
                    "u": "u",
                    "u_grad": "u_grad",
                    "u_laplacian": "u_laplacian",
                    "u_t": "u_t",
                    "re": "re",
                    "U": "U",
                    "L": "L",
                },
            }
        ]
        grid = PathBGridConfig(
            diff_method="spectral",
            periodic=True,
            spatial_dimension=1,
            layout="separable",
        )
        state, _ = fill_path_b_derivatives(
            {"u": u, "x": x, "u_laplacian": sentinel},
            laws_spec=laws,
            constants={"re": 10.0, "U": 1.0, "L": L},
            grid=grid,
            fill_law_recipes=True,
        )
        assert jnp.allclose(state["u_laplacian"], sentinel)

    def test_rejects_spectral_without_periodic(self):
        x, L = _periodic_1d_grid(n=16)
        with pytest.raises(ValueError, match="periodic=True"):
            fill_path_b_derivatives(
                {"u": jnp.sin(x)[:, None], "x": x},
                laws_spec=[{"name": "burgers_equation", "state_map": {"u": "u", "u_laplacian": "u_laplacian"}}],
                grid=PathBGridConfig(diff_method="spectral", periodic=False, spatial_dimension=1),
                fill_law_recipes=True,
            )

    def test_2d_scalar_laplacian(self):
        nx = ny = 48
        Lx = Ly = 2.0 * math.pi
        xs = jnp.linspace(0.0, Lx, nx, endpoint=False)
        ys = jnp.linspace(0.0, Ly, ny, endpoint=False)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        phi = jnp.sin(X) * jnp.cos(Y)
        # ∇² phi = -sin(x)cos(y) - sin(x)cos(y) = -2 sin(x)cos(y)
        grid = PathBGridConfig(
            diff_method="spectral",
            periodic=True,
            spatial_dimension=2,
            layout="separable",
        )
        state, _ = fill_law_fd_from_primitives(
            {"phi": phi, "x": xs, "y": ys},
            [{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian", "phi": "phi"}}],
            grid=grid,
        )
        expected = -2.0 * phi
        assert float(jnp.max(jnp.abs(state["phi_laplacian"] - expected))) < 5e-4

    def test_2d_vector_burgers_shapes(self):
        nx = ny = 32
        L = 2.0 * math.pi
        xs = jnp.linspace(0.0, L, nx, endpoint=False)
        ys = jnp.linspace(0.0, L, ny, endpoint=False)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        u = jnp.stack([jnp.sin(X), jnp.cos(Y)], axis=-1)
        grid = PathBGridConfig(
            diff_method="spectral",
            periodic=True,
            spatial_dimension=2,
            layout="separable",
        )
        laws = [
            {
                "name": "burgers_equation",
                "state_map": {
                    "u": "u",
                    "u_grad": "u_grad",
                    "u_laplacian": "u_laplacian",
                    "u_t": "u_t",
                    "re": "re",
                    "U": "U",
                    "L": "L",
                },
            }
        ]
        state, _ = fill_law_fd_from_primitives(
            {"u": u, "x": xs, "y": ys},
            laws,
            constants={"re": 100.0, "U": 1.0, "L": L},
            grid=grid,
        )
        assert state["u_grad"].shape == (nx, ny, 2, 2)
        assert state["u_laplacian"].shape == (nx, ny, 2)
        # ∂u0/∂x = cos(x), ∂²u0/∂x² = -sin(x); u1 laplacian = -cos(y)
        assert float(jnp.max(jnp.abs(state["u_grad"][..., 0, 0] - jnp.cos(X)))) < 5e-4
        assert float(jnp.max(jnp.abs(state["u_laplacian"][..., 0] + jnp.sin(X)))) < 5e-4
        assert float(jnp.max(jnp.abs(state["u_laplacian"][..., 1] + jnp.cos(Y)))) < 5e-4


class TestSpectralEngineSmoke:
    def test_residual_engine_spectral_path_b(self):
        x, L = _periodic_1d_grid(n=64)
        u = jnp.sin(x)[:, None]
        # Steady Burgers-like: need u_t — provide zeros so law can run
        u_t = jnp.zeros_like(u)
        engine = ResidualEngine(
            laws=[
                {
                    "name": "burgers_equation",
                    "state_map": {
                        "u_t": "u_t",
                        "u": "u",
                        "u_grad": "u_grad",
                        "u_laplacian": "u_laplacian",
                        "re": "Re",
                        "U": "U",
                        "L": "L",
                    },
                    "fn": Laws.burgers_equation,
                }
            ],
            constants={"Re": 100.0, "U": 1.0, "L": L},
            default_coord_dimension=1,
            law_implied_audits=False,
        )
        grid = PathBGridConfig(
            diff_method="spectral",
            periodic=True,
            spatial_dimension=1,
            layout="separable",
            steady=True,
        )
        residuals = engine.compute_residuals(
            {"u": u, "u_t": u_t, "x": x},
            auto_path_b_derivatives=grid,
            fill_law_fd=True,
            log_to_python=False,
        )
        assert "burgers_equation" in residuals["laws"]
        r = residuals["laws"]["burgers_equation"]
        assert jnp.asarray(r).shape == u.shape
