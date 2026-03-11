"""Tests for moju.piratio.Laws (conservation-law residuals)."""

import pytest
import jax.numpy as jnp
from moju.piratio import Laws


class TestLawsMassIncompressible:
    """div(u) = 0 residual."""

    def test_zero_gradient_gives_zero_residual(self, rtol, atol):
        """Constant velocity field: u_grad = 0 => div u = 0."""
        u_grad = jnp.zeros((2, 2))
        residual = Laws.mass_incompressible(u_grad)
        assert jnp.allclose(residual, 0.0, rtol=rtol, atol=atol)

    def test_trace_zero_gives_zero_residual(self, rtol, atol):
        """Solenoidal field: trace(u_grad) = 0 => div u = 0."""
        # e.g. u = (y, -x) => u_grad = [[0,1],[-1,0]], trace = 0
        u_grad = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        residual = Laws.mass_incompressible(u_grad)
        assert jnp.allclose(residual, 0.0, rtol=rtol, atol=atol)

    def test_non_solenoidal_gives_nonzero_residual(self, rtol, atol):
        """trace != 0 => residual != 0."""
        u_grad = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # trace = 2
        residual = Laws.mass_incompressible(u_grad)
        assert not jnp.allclose(residual, 0.0, atol=1e-6)
        assert jnp.allclose(residual, 2.0, rtol=rtol, atol=atol)

    def test_batch_mass_incompressible(self, rtol, atol):
        """Batch of velocity gradients returns 1D residual."""
        u_grad = jnp.zeros((3, 2, 2))  # 3 points, 2x2 gradient each
        residual = Laws.mass_incompressible(u_grad)
        assert residual.shape == (3,)
        assert jnp.allclose(residual, 0.0, rtol=rtol, atol=atol)


class TestLawsLaplaceAndWave:
    """Simple PDE residuals."""

    def test_laplace_equation_residual(self, rtol, atol):
        """Laplace equation: residual = Laplacian(phi). Zero when Laplacian is 0."""
        residual = Laws.laplace_equation(0.0)
        assert jnp.allclose(residual, 0.0, rtol=rtol, atol=atol)

    def test_wave_equation_residual_zero(self, rtol, atol):
        """Wave equation: phi_tt - c^2 Laplacian(phi) = 0. Zero when both terms cancel."""
        phi_tt = 1.0
        phi_laplacian = 1.0
        c = 1.0
        residual = Laws.wave_equation(phi_tt, phi_laplacian, c)
        expected = phi_tt - (c ** 2) * phi_laplacian
        assert jnp.allclose(residual, expected, rtol=rtol, atol=atol)
        assert jnp.allclose(residual, 0.0, rtol=rtol, atol=atol)


class TestLawsStokesFlow:
    """Stokes flow residual: grad p - (1/Re) Laplacian(u) = 0."""

    def test_stokes_flow_residual_shape(self):
        """Stokes residual is vector of same dimension as p_grad."""
        p_grad = jnp.array([1.0, 0.0])
        u_laplacian = jnp.array([0.0, 0.0])
        re = 100.0
        residual = Laws.stokes_flow(p_grad, u_laplacian, re)
        assert residual.shape == (2,)
