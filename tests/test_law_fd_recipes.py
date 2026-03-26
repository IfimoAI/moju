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


class TestSchrodingerLawFd:
    def test_psi_laplacian_includes_L2_factor(self, rtol, atol):
        """FD Laplacian is multiplied by L**2 for schrodinger_steady (L from merged state)."""
        L = jnp.array(2.0)
        x = jnp.linspace(0.0, 1.0, 65)
        psi = jnp.sin(jnp.pi * x)
        state = {"psi": psi, "x": x, "L": L}
        laws = [
            {"name": "schrodinger_steady", "state_map": {"psi_laplacian": "psi_laplacian"}}
        ]
        out, w = fill_law_fd_from_primitives(
            state,
            laws,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert "psi_laplacian" in out, w
        expect = (L**2) * (-(jnp.pi**2)) * psi
        assert jnp.allclose(out["psi_laplacian"][1:-1], expect[1:-1], rtol=1e-2, atol=1e-2)

    def test_psi_laplacian_skipped_without_L(self):
        x = jnp.linspace(0.0, 1.0, 33)
        psi = jnp.sin(jnp.pi * x)
        state = {"psi": psi, "x": x}
        laws = [
            {"name": "schrodinger_steady", "state_map": {"psi_laplacian": "psi_laplacian"}}
        ]
        out, w = fill_law_fd_from_primitives(
            state,
            laws,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert out.get("psi_laplacian") is None
        assert any("need L" in s for s in w)


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

    def test_fourier_conduction_t_laplacian_from_t_and_x(self, rtol, atol):
        """Studio / Path B: T + x can produce T_laplacian before the law runs (plan doc)."""
        x = jnp.linspace(0.0, 1.0, 33)
        T = x
        state = {"T": T, "x": x}
        laws = [
            {"name": "fourier_conduction", "state_map": {"T_laplacian": "T_laplacian"}}
        ]
        out, _ = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert "T_laplacian" in out
        assert jnp.allclose(out["T_laplacian"][1:-1], 0.0, rtol=rtol, atol=1e-3)

    def test_fourier_conduction_t_laplacian_t_1d_x_column(self, rtol, atol):
        """Common NPZ layout: T (N,) and x (N, 1) — same size, different shape."""
        x1d = jnp.linspace(0.0, 1.0, 33)
        T = x1d
        x = x1d.reshape(-1, 1)
        state = {"T": T, "x": x}
        laws = [
            {"name": "fourier_conduction", "state_map": {"T_laplacian": "T_laplacian"}}
        ]
        out, w = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert not w or all("need x same shape" not in s for s in w)
        assert "T_laplacian" in out
        assert out["T_laplacian"].shape == T.shape
        assert jnp.allclose(out["T_laplacian"][1:-1], 0.0, rtol=rtol, atol=1e-3)

    def test_fourier_conduction_t_laplacian_meshgrid_with_1d_axes(self, rtol, atol):
        """Default Studio layout is meshgrid; NPZ often has T (nx,ny) + x (nx,) + y (ny,)."""
        nx, ny = 24, 22
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 1.0, ny)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        T = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        state = {"T": T, "x": xs, "y": ys}
        laws = [
            {"name": "fourier_conduction", "state_map": {"T_laplacian": "T_laplacian"}}
        ]
        out, w = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension="auto", steady=True),
        )
        assert "T_laplacian" in out, w
        lap = out["T_laplacian"]
        expect = -2 * (jnp.pi**2) * T
        assert jnp.allclose(lap[1:-1, 1:-1], expect[1:-1, 1:-1], rtol=1e-2, atol=1e-2)

    def test_fourier_conduction_t_column_vector_infers_1d(self, rtol, atol):
        """T (N, 1) must not require a fake y axis when using default meshgrid + x."""
        x1d = jnp.linspace(0.0, 1.0, 33)
        T = x1d.reshape(-1, 1)
        state = {"T": T, "x": x1d.reshape(-1, 1)}
        laws = [
            {"name": "fourier_conduction", "state_map": {"T_laplacian": "T_laplacian"}}
        ]
        out, _ = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension="auto", steady=True),
        )
        assert "T_laplacian" in out
        assert out["T_laplacian"].shape == T.shape
        assert jnp.allclose(out["T_laplacian"][1:-1, 0], 0.0, rtol=rtol, atol=1e-3)

    def test_fourier_conduction_fills_t_laplacian_when_constants_has_null_placeholder(self, rtol, atol):
        """Constants JSON often includes T_laplacian: null — must not block law-FD fill."""
        x1d = jnp.linspace(0.0, 1.0, 17)
        T = x1d
        state = {"T": T, "x": x1d}
        constants = {"T_laplacian": None}
        laws = [
            {"name": "fourier_conduction", "state_map": {"T_laplacian": "T_laplacian"}}
        ]
        out, w = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            constants=constants,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert "T_laplacian" in out, w

    def test_fourier_conduction_t_row_vector_auto_spatial_dim_1d(self, rtol, atol):
        """T (1, N) row layout with spatial_dimension=auto must infer 1D Laplacian (not 2D + y)."""
        x1d = jnp.linspace(0.0, 1.0, 33)
        T = x1d.reshape(1, -1)
        state = {"T": T, "x": x1d}
        laws = [
            {"name": "fourier_conduction", "state_map": {"T_laplacian": "T_laplacian"}}
        ]
        out, w = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension="auto", steady=True),
        )
        assert "T_laplacian" in out, w
        assert out["T_laplacian"].shape == T.shape
        assert jnp.allclose(out["T_laplacian"][0, 1:-1], 0.0, rtol=rtol, atol=1e-3)

    def test_fourier_conduction_t_laplacian_steady_true_time_stack(self, rtol, atol):
        """Default steady=True + T (nt,nx,ny) + t(nt,) must fill T_laplacian per time slice."""
        nt, nx, ny = 5, 20, 18
        t = jnp.linspace(0.0, 1.0, nt)
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 1.0, ny)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        g = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        fac = 1.0 + 0.1 * t.reshape(-1, 1, 1)
        T = fac * g.reshape(1, nx, ny)
        state = {"T": T, "t": t, "x": xs, "y": ys}
        laws = [
            {"name": "fourier_conduction", "state_map": {"T_laplacian": "T_laplacian"}}
        ]
        out, w = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension="auto", steady=True),
        )
        assert "T_laplacian" in out, w
        assert out["T_laplacian"].shape == T.shape
        lap_g = -2 * (jnp.pi**2) * g
        for k in range(nt):
            expect_k = float(fac[k, 0, 0]) * lap_g
            assert jnp.allclose(
                out["T_laplacian"][k, 1:-1, 1:-1],
                expect_k[1:-1, 1:-1],
                rtol=1e-2,
                atol=1e-2,
            )

    def test_fourier_conduction_t_laplacian_square_nt_equals_nx_time_stack(self, rtol, atol):
        """Regression: square (n,n) with t(n,) was misread as 2D spatial and broke jnp.gradient."""
        n = 33  # enough points for stable 1D FD Laplacian along x
        t = jnp.linspace(0.0, 1.0, n)
        x = jnp.linspace(0.0, 1.0, n)
        T = (1.0 + 0.1 * t[:, None]) * jnp.sin(jnp.pi * x[None, :])
        state = {"T": T, "t": t, "x": x}
        laws = [
            {"name": "fourier_conduction", "state_map": {"T_laplacian": "T_laplacian"}}
        ]
        out, w = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert "T_laplacian" in out, w
        assert not any("list index out of range" in s for s in w)
        assert out["T_laplacian"].shape == T.shape
        lap_row = -(jnp.pi**2) * jnp.sin(jnp.pi * x)
        for k in range(n):
            fac = float(1.0 + 0.1 * t[k])
            assert jnp.allclose(
                out["T_laplacian"][k, 1:-1],
                fac * lap_row[1:-1],
                rtol=1e-2,
                atol=1e-2,
            )

    def test_fourier_conduction_t_laplacian_snapshot_stack_without_t_coord(self, rtol, atol):
        """NPZ stacks often omit ``t``; (nt,nx,ny) with nt > max(nx,ny) still gets spatial FD."""
        nt, nx, ny = 12, 20, 18
        xs = jnp.linspace(0.0, 1.0, nx)
        ys = jnp.linspace(0.0, 1.0, ny)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        g = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        T = jnp.stack([g * (1.0 + 0.05 * float(k)) for k in range(nt)], axis=0)
        state = {"T": T, "x": xs, "y": ys}
        laws = [
            {"name": "fourier_conduction", "state_map": {"T_laplacian": "T_laplacian"}}
        ]
        out, w = fill_path_b_derivatives(
            state,
            laws_spec=laws,
            fill_law_recipes=True,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension="auto", steady=True),
        )
        assert "T_laplacian" in out, w
        lap_g = -2 * (jnp.pi**2) * g
        for k in range(nt):
            fk = 1.0 + 0.05 * float(k)
            assert jnp.allclose(
                out["T_laplacian"][k, 1:-1, 1:-1],
                (fk * lap_g)[1:-1, 1:-1],
                rtol=1e-2,
                atol=1e-2,
            )


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

    def test_laplace_fd_survives_constants_placeholder_phi_laplacian_none(self, rtol, atol):
        """Constants must not overwrite FD-filled ``phi_laplacian`` (Studio constants dict issue)."""
        from moju.monitor.config import MonitorConfig

        x = jnp.linspace(0.0, 1.0, 33)
        phi = x
        cfg = MonitorConfig(
            laws=[
                {
                    "name": "laplace_equation",
                    "state_map": {"phi_laplacian": "phi_laplacian"},
                }
            ],
            constants={"phi_laplacian": None},
        )
        eng = ResidualEngine(config=cfg)
        res = eng.compute_residuals(
            {"phi": phi, "x": x},
            auto_path_b_derivatives=PathBGridConfig(
                layout="meshgrid", spatial_dimension=1, steady=True
            ),
            fill_law_fd=True,
        )
        r = res["laws"]["laplace_equation"]
        assert jnp.allclose(r[1:-1], 0.0, rtol=rtol, atol=1e-4)
