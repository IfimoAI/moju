"""Tests for moju.piratio.Models (physical constitutive models)."""

import pytest
import jax.numpy as jnp
from moju.piratio import Models


class TestModelsKnownValues:
    """Known-value and formula consistency."""

    def test_ideal_gas_rho_at_stp(self, rtol, atol):
        """Ideal gas: rho = P / (R*T). Air at 1 bar, 300 K."""
        P = 101325.0
        R = 287.0  # J/(kg·K) for air
        T = 300.0
        rho = Models.ideal_gas_rho(P=P, R=R, T=T)
        expected = P / (R * T)
        assert jnp.allclose(rho, expected, rtol=rtol, atol=atol)
        assert 1.1 < float(rho) < 1.3  # approximate air density

    def test_sutherland_at_reference_temperature(self, rtol, atol):
        """Sutherland: at T=T0, mu = mu0 (from formula)."""
        mu0 = 1.8e-5
        T0 = 273.0
        S = 110.4
        mu = Models.sutherland_mu(T=T0, mu0=mu0, T0=T0, S=S)
        assert jnp.allclose(mu, mu0, rtol=rtol, atol=atol)

    def test_stefan_boltzmann_flux_formula(self, rtol, atol):
        """q = epsilon * sigma * T^4. sigma ≈ 5.67e-8."""
        epsilon = 1.0
        T = 400.0
        q = Models.stefan_boltzmann_flux(epsilon=epsilon, T=T)
        sigma = 5.670374419e-8
        expected = epsilon * sigma * (T ** 4)
        assert jnp.allclose(q, expected, rtol=rtol, atol=atol)

    def test_boussinesq_rho_at_dT_zero(self, rtol, atol):
        """Boussinesq: at dT=0, rho = rho0."""
        rho0 = 1000.0
        beta = 2e-4
        rho = Models.boussinesq_rho(rho0=rho0, beta=beta, dT=0.0)
        assert jnp.allclose(rho, rho0, rtol=rtol, atol=atol)

    def test_kinematic_viscosity(self, rtol, atol):
        """nu = mu / rho."""
        mu = 1e-3
        rho = 1000.0
        nu = Models.kinematic_viscosity(mu=mu, rho=rho)
        assert jnp.allclose(nu, 1e-6, rtol=rtol, atol=atol)

    def test_thermal_diffusivity(self, rtol, atol):
        """alpha = k / (rho * cp)."""
        k = 0.6
        rho = 1000.0
        cp = 4186.0
        alpha = Models.thermal_diffusivity(k=k, rho=rho, cp=cp)
        expected = k / (rho * cp)
        assert jnp.allclose(alpha, expected, rtol=rtol, atol=atol)

    def test_speed_of_sound_air(self, rtol, atol):
        """a = sqrt(gamma * R * T). Air gamma≈1.4, R=287, T=300."""
        gamma = 1.4
        R = 287.0
        T = 300.0
        a = Models.speed_of_sound(gamma=gamma, R=R, T=T)
        expected = (gamma * R * T) ** 0.5
        assert jnp.allclose(a, expected, rtol=rtol, atol=atol)
        assert 340 < float(a) < 350

    def test_dynamic_pressure(self, rtol, atol):
        """q = 0.5 * rho * u^2."""
        rho = 1.2
        u = 10.0
        q = Models.dynamic_pressure(rho=rho, u=u)
        assert jnp.allclose(q, 0.5 * rho * u ** 2, rtol=rtol, atol=atol)

    def test_power_law_mu_n_equal_one(self, rtol, atol):
        """Power-law at n=1 gives Newtonian: mu_app = K."""
        K = 0.001
        n = 1.0
        gamma_dot = 100.0
        mu = Models.power_law_mu(gamma_dot=gamma_dot, K=K, n=n)
        assert jnp.allclose(mu, K, rtol=rtol, atol=atol)
