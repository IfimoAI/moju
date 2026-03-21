"""Tests for optional finite-difference fill of Laws.* inputs (law_fd_recipes)."""

import jax.numpy as jnp
import pytest

from moju.monitor import ResidualEngine
from moju.monitor.law_fd_recipes import (
    fill_law_fd_from_primitives,
    list_law_fd_supported_laws,
)
from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives
from moju.piratio import Laws


class TestListSupported:
    def test_list_includes_laplace(self):
        names = list_law_fd_supported_laws()
        assert "laplace_equation" in names
        assert "mass_incompressible" in names


class TestFillLawFdLaplace:
    def test_linear_phi_laplacian_near_zero_meshgrid_1d(self, rtol, atol):
        x = jnp.linspace(0.0, 1.0, 33)
        phi = x
        state = {"phi": phi, "x": x}
        laws = [
            {"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}
        ]
        out, w = fill_law_fd_from_primitives(
            state,
            laws,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert "phi_laplacian" in out
        # Interior FD for linear field ~ 0
        lap = out["phi_laplacian"]
        assert jnp.allclose(lap[1:-1], 0.0, rtol=rtol, atol=1e-4)
        r = Laws.laplace_equation(out["phi_laplacian"])
        assert jnp.allclose(r[1:-1], 0.0, rtol=rtol, atol=1e-4)

    def test_sine_matches_analytic_laplacian_1d(self, rtol, atol):
        x = jnp.linspace(0.0, 1.0, 65)
        phi = jnp.sin(jnp.pi * x)
        state = {"phi": phi, "x": x}
        laws = [
            {"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}
        ]
        out, _ = fill_law_fd_from_primitives(
            state,
            laws,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        expect = -(jnp.pi**2) * phi
        assert jnp.allclose(out["phi_laplacian"][1:-1], expect[1:-1], rtol=1e-3, atol=1e-3)


class TestFillPathBWithLawRecipes:
    def test_fill_path_b_derivatives_law_flag(self, rtol, atol):
        x = jnp.linspace(0.0, 1.0, 33)
        phi = x
        state = {"phi": phi, "x": x}
        laws = [
            {"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}
        ]
        out, _ = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert jnp.allclose(out["phi_laplacian"][1:-1], 0.0, rtol=rtol, atol=1e-4)


class TestMassIncompressibleFd:
    def test_solenoidal_field_residual_small(self, rtol, atol):
        nx, ny = 24, 22
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 1.0, ny)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        # u = (y, -x) => div = 0
        u = jnp.stack([Y, -X], axis=-1)
        state = {"u": u, "x": X, "y": Y}
        laws = [
            {
                "name": "mass_incompressible",
                "state_map": {"u_grad": "u_grad"},
            }
        ]
        out, _ = fill_law_fd_from_primitives(
            state,
            laws,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=2, steady=True),
        )
        assert "u_grad" in out
        r = Laws.mass_incompressible(out["u_grad"])
        assert jnp.allclose(r[1:-1, 1:-1], 0.0, rtol=1e-2, atol=1e-2)


class TestResidualEngineFillLawFd:
    def test_engine_laplace_with_auto_fd(self, rtol, atol):
        x = jnp.linspace(0.0, 1.0, 33)
        phi = x
        state = {"phi": phi, "x": x}
        eng = ResidualEngine(
            laws=[
                {
                    "name": "laplace_equation",
                    "state_map": {"phi_laplacian": "phi_laplacian"},
                }
            ],
        )
        res = eng.compute_residuals(
            state,
            auto_path_b_derivatives=PathBGridConfig(
                layout="meshgrid", spatial_dimension=1, steady=True
            ),
            fill_law_fd=True,
        )
        lap_r = res["laws"]["laplace_equation"]
        assert jnp.allclose(lap_r[1:-1], 0.0, rtol=rtol, atol=1e-4)

    def test_fill_law_fd_requires_auto_path_b(self):
        eng = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        with pytest.raises(ValueError, match="fill_law_fd"):
            eng.compute_residuals(
                {"phi": jnp.ones(4), "x": jnp.linspace(0, 1, 4)},
                fill_law_fd=True,
            )
