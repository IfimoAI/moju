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

    def test_smagorinsky_nu_t(self, rtol, atol):
        """nu_t = (Cs*Delta)^2 * |S|."""
        Cs = jnp.array(0.17)
        Delta = jnp.array(0.01)
        S = jnp.array(10.0)
        nu = Models.smagorinsky_nu_t(Cs=Cs, Delta=Delta, strain_rate_magnitude=S)
        expected = (0.17 * 0.01) ** 2 * 10.0
        assert jnp.allclose(nu, expected, rtol=rtol, atol=atol)

    def test_k_epsilon_nu_t(self, rtol, atol):
        """nu_t = C_mu * k^2 / (epsilon + eps0)."""
        C_mu = jnp.array(0.09)
        k = jnp.array(4.0)
        epsilon = jnp.array(2.0)
        eps0 = jnp.array(1e-9)
        nu = Models.k_epsilon_nu_t(C_mu=C_mu, k=k, epsilon=epsilon, eps0=eps0)
        expected = 0.09 * 16.0 / (2.0 + 1e-9)
        assert jnp.allclose(nu, expected, rtol=rtol, atol=atol)

    def test_k_epsilon_nu_t_small_epsilon_uses_floor(self, rtol, atol):
        """Floor eps0 keeps nu_t finite when epsilon -> 0."""
        C_mu = jnp.array(0.09)
        k = jnp.array(1.0)
        epsilon = jnp.array(0.0)
        eps0 = jnp.array(1e-12)
        nu = Models.k_epsilon_nu_t(C_mu=C_mu, k=k, epsilon=epsilon, eps0=eps0)
        expected = 0.09 * 1.0 / 1e-12
        assert jnp.allclose(nu, expected, rtol=rtol, atol=atol)

    def test_k_omega_nu_t(self, rtol, atol):
        """nu_t = k / (omega + omega0)."""
        k = jnp.array(0.5)
        omega = jnp.array(2.0)
        omega0 = jnp.array(1e-9)
        nu = Models.k_omega_nu_t(k=k, omega=omega, omega0=omega0)
        expected = 0.5 / (2.0 + 1e-9)
        assert jnp.allclose(nu, expected, rtol=rtol, atol=atol)

    def test_k_omega_nu_t_small_omega_uses_floor(self, rtol, atol):
        omega0 = jnp.array(1e-12)
        nu = Models.k_omega_nu_t(k=jnp.array(3.0), omega=jnp.array(0.0), omega0=omega0)
        assert jnp.allclose(nu, 3.0 / 1e-12, rtol=rtol, atol=atol)


class TestIsotropicLinearStress:
    """Known-value tests for Models.isotropic_linear_stress (Voigt d=1,3,6)."""

    def test_1d_axial_stress(self, rtol, atol):
        """d=1: σ = E·ε."""
        E = jnp.array(2.0)
        nu = jnp.array(0.0)
        strain = jnp.array([0.5])
        stress = Models.isotropic_linear_stress(E=E, nu=nu, strain=strain)
        assert stress.shape == (1,)
        assert jnp.allclose(stress, jnp.array([1.0]), rtol=rtol, atol=atol)

    def test_1d_batch_stress(self, rtol, atol):
        """d=1 with leading batch dimension."""
        E = jnp.array(3.0)
        nu = jnp.array(0.0)
        strain = jnp.array([[1.0], [2.0], [3.0]])  # (3, 1)
        stress = Models.isotropic_linear_stress(E=E, nu=nu, strain=strain)
        assert stress.shape == (3, 1)
        expected = 3.0 * strain
        assert jnp.allclose(stress, expected, rtol=rtol, atol=atol)

    def test_2d_plane_stress_nu_zero(self, rtol, atol):
        """d=3 plane stress with nu=0: C = E*I_block (diagonal)."""
        E = jnp.array(2.0)
        nu = jnp.array(0.0)
        # C = 2 * [[1,0,0],[0,1,0],[0,0,0.5]]
        strain = jnp.array([1.0, 2.0, 4.0])
        stress = Models.isotropic_linear_stress(E=E, nu=nu, strain=strain)
        expected = jnp.array([2.0, 4.0, 4.0])  # [2*1, 2*2, 2*0.5*4]
        assert stress.shape == (3,)
        assert jnp.allclose(stress, expected, rtol=rtol, atol=atol)

    def test_2d_plane_stress_analytic(self, rtol, atol):
        """d=3 plane stress: σ_xx = E/(1-ν²)*(ε_xx + ν*ε_yy)."""
        E = 1.0
        nu = 0.3
        ex, ey, exy = 0.001, -0.0003, 0.0
        fac = E / (1.0 - nu ** 2)
        sx_exp = fac * (ex + nu * ey)
        sy_exp = fac * (nu * ex + ey)
        sxy_exp = fac * ((1.0 - nu) / 2.0) * exy
        stress = Models.isotropic_linear_stress(
            E=jnp.array(E), nu=jnp.array(nu), strain=jnp.array([ex, ey, exy])
        )
        assert jnp.allclose(stress[0], sx_exp, rtol=rtol, atol=1e-7)
        assert jnp.allclose(stress[1], sy_exp, rtol=rtol, atol=1e-7)
        assert jnp.allclose(stress[2], sxy_exp, rtol=rtol, atol=atol)

    def test_3d_voigt_nu_zero(self, rtol, atol):
        """d=6 with nu=0: λ=0, C = diag(2G, 2G, 2G, G, G, G) with G=E/2."""
        E = jnp.array(2.0)
        nu = jnp.array(0.0)
        # G = E/(2*(1+0)) = 1.0, lam = 0
        # C = diag(2,2,2,1,1,1)
        strain = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        stress = Models.isotropic_linear_stress(E=E, nu=nu, strain=strain)
        expected = jnp.array([2.0, 4.0, 6.0, 4.0, 5.0, 6.0])
        assert stress.shape == (6,)
        assert jnp.allclose(stress, expected, rtol=rtol, atol=atol)

    def test_3d_voigt_symmetric_stiffness(self, rtol, atol):
        """d=6 isotropic: stress = lam*(eps_kk)*I + 2G*eps (Voigt notation)."""
        E = 200e9
        nu = 0.3
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        eps = jnp.array([1e-3, -3e-4, 0.0, 0.0, 0.0, 0.0])
        # Analytic: σ_xx = λ(ε_xx+ε_yy+ε_zz) + 2G*ε_xx
        eps_kk = float(eps[0]) + float(eps[1]) + float(eps[2])
        sx_exp = lam * eps_kk + 2 * G * float(eps[0])
        sy_exp = lam * eps_kk + 2 * G * float(eps[1])
        sz_exp = lam * eps_kk + 2 * G * float(eps[2])
        stress = Models.isotropic_linear_stress(E=jnp.array(E), nu=jnp.array(nu), strain=eps)
        assert jnp.allclose(stress[0], sx_exp, rtol=1e-4, atol=1.0)
        assert jnp.allclose(stress[1], sy_exp, rtol=1e-4, atol=1.0)
        assert jnp.allclose(stress[2], sz_exp, rtol=1e-4, atol=1.0)
        assert jnp.allclose(stress[3], 0.0, atol=1.0)

    def test_invalid_voigt_dim_raises(self):
        """d not in {1,3,6} raises ValueError."""
        import pytest
        with pytest.raises(ValueError, match="unsupported Voigt dimension"):
            Models.isotropic_linear_stress(
                E=jnp.array(1.0), nu=jnp.array(0.0), strain=jnp.array([1.0, 2.0])
            )


class TestTurbulentViscousAcceleration:
    def test_k_omega_matches_nu_times_laplacian(self, rtol, atol):
        u_lap = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        nu_m = jnp.array(1e-6)
        k = jnp.array(0.4)
        omega = jnp.array(2.0)
        omega0 = jnp.array(1e-12)
        out = Models.turbulent_viscous_acceleration_k_omega(
            u_laplacian=u_lap, nu_molecular=nu_m, k=k, omega=omega, omega0=omega0
        )
        nu_t = Models.k_omega_nu_t(k, omega, omega0)
        exp = (nu_m + nu_t) * u_lap
        assert jnp.allclose(out, exp, rtol=rtol, atol=atol)
