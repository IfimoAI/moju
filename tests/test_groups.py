"""Tests for moju.piratio.Groups (dimensionless numbers)."""

import pytest
import jax.numpy as jnp
from moju.piratio import Groups


class TestGroupsScalar:
    """Known-value and formula consistency for scalar inputs."""

    def test_re_formula(self, rtol, atol):
        """Re = rho * u * L / mu."""
        Re = Groups.re(u=1.0, L=0.1, rho=1000.0, mu=1e-3)
        expected = 1000.0 * 1.0 * 0.1 / 1e-3
        assert jnp.allclose(Re, expected, rtol=rtol, atol=atol)

    def test_pr_formula(self, rtol, atol):
        """Pr = cp * mu / k."""
        Pr = Groups.pr(mu=1e-3, cp=4186.0, k=0.6)
        expected = 4186.0 * 1e-3 / 0.6
        assert jnp.allclose(Pr, expected, rtol=rtol, atol=atol)

    def test_nu_formula(self, rtol, atol):
        """Nu = h * L / k."""
        Nu = Groups.nu(h=100.0, L=0.1, k=0.6)
        expected = 100.0 * 0.1 / 0.6
        assert jnp.allclose(Nu, expected, rtol=rtol, atol=atol)

    def test_ma_formula(self, rtol, atol):
        """Ma = u / a."""
        Ma = Groups.ma(u=100.0, a=343.0)
        expected = 100.0 / 343.0
        assert jnp.allclose(Ma, expected, rtol=rtol, atol=atol)

    def test_pe_from_re_pr(self, rtol, atol):
        """Pe = Re * Pr."""
        Re = Groups.re(u=1.0, L=0.1, rho=1000.0, mu=1e-3)
        Pr = Groups.pr(mu=1e-3, cp=4186.0, k=0.6)
        Pe = Groups.pe(re=Re, pr=Pr)
        assert jnp.allclose(Pe, Re * Pr, rtol=rtol, atol=atol)

    def test_da_formula(self, rtol, atol):
        """Da = K / L^2."""
        Da = Groups.da(K=1e-10, L=0.01)
        expected = 1e-10 / (0.01 ** 2)
        assert jnp.allclose(Da, expected, rtol=rtol, atol=atol)


class TestGroupsBatch:
    """Batch evaluation returns correct shape and matches scalar."""

    def test_re_batch_same_as_scalar(self, rtol, atol):
        """Batched Re matches scalar Re for same inputs."""
        u = jnp.array([1.0, 2.0])
        L = jnp.array([0.1, 0.2])
        rho = jnp.array([1000.0, 1000.0])
        mu = jnp.array([1e-3, 1e-3])
        Re_batch = Groups.re(u=u, L=L, rho=rho, mu=mu)
        assert Re_batch.shape == (2,)
        for i in range(2):
            Re_i = Groups.re(u=float(u[i]), L=float(L[i]), rho=float(rho[i]), mu=float(mu[i]))
            assert jnp.allclose(Re_batch[i], Re_i, rtol=rtol, atol=atol)

    def test_ma_batch_shape(self):
        """Batched Ma returns 1D array of same length as inputs."""
        u = jnp.array([10.0, 100.0, 200.0])
        a = jnp.array([343.0, 343.0, 343.0])
        Ma = Groups.ma(u=u, a=a)
        assert Ma.shape == (3,)


class TestGroupsNew:
    """New dimensionless groups: Fo_mass, wavenumber, pe_mass, st_wave."""

    def test_fo_mass_formula(self, rtol, atol):
        """Fo_mass = D * t / L^2."""
        D = 1e-9
        t = 1.0
        L = 0.01
        Fo_mass = Groups.fo_mass(D=D, t=t, L=L)
        expected = (D * t) / (L**2)
        assert jnp.allclose(Fo_mass, expected, rtol=rtol, atol=atol)

    def test_wavenumber_formula(self, rtol, atol):
        """wavenumber = k * L."""
        k = 2.0
        L = 0.5
        kL = Groups.wavenumber(k=k, L=L)
        assert jnp.allclose(kL, 1.0, rtol=rtol, atol=atol)

    def test_pe_mass_formula(self, rtol, atol):
        """pe_mass = Re * Sc."""
        re = 100.0
        sc = 0.7
        Pe_mass = Groups.pe_mass(re=re, sc=sc)
        assert jnp.allclose(Pe_mass, 70.0, rtol=rtol, atol=atol)

    def test_st_wave_formula(self, rtol, atol):
        """st_wave = omega * L / c."""
        omega = 2.0
        L = 1.0
        c = 343.0
        St_wave = Groups.st_wave(omega=omega, L=L, c=c)
        expected = (omega * L) / c
        assert jnp.allclose(St_wave, expected, rtol=rtol, atol=atol)
