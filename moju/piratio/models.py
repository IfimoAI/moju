import jax
import jax.numpy as jnp

# Stefan-Boltzmann constant [W/(m^2 K^4)]; as JAX scalar for tracing
_STEFAN_BOLTZMANN = 5.670374419e-8

class Models:
    """
    Differentiable physical models for anchoring AI predictions to reality.
    Inputs may have leading batch dimensions; all operations broadcast and support
    both single-point and batched evaluation.
    """

    @staticmethod
    @jax.jit
    def sutherland_mu(T, mu0, T0, S):
        """
        Sutherland's Viscosity Law for Gases.
        
        :param T: Local temperature [K].
        :param mu0: Reference viscosity at T0 [Pa*s].
        :param T0: Reference temperature [K].
        :param S: Sutherland constant for the specific gas [K].
        :return: Dynamic viscosity at temperature T [Pa*s].
        """
        return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)

    @staticmethod
    @jax.jit
    def vft_mu(T, A, B, T0_v):
        """
        Vogel-Fulcher-Tammann (VFT) Viscosity Law for Liquids.
        
        :param T: Local temperature [K].
        :param A: Pre-exponential factor [Pa*s].
        :param B: Activation-related constant [K].
        :param T0_v: Vogel temperature (theoretical glass transition) [K].
        :return: Dynamic viscosity at temperature T [Pa*s].
        """
        return A * jnp.exp(B / (T - T0_v))

    @staticmethod
    @jax.jit
    def ideal_gas_rho(P, R, T):
        """
        Ideal Gas Law for Density.
        
        :param P: Absolute pressure [Pa].
        :param R: Specific gas constant [J/kg*K].
        :param T: Absolute temperature [K].
        :return: Density [kg/m^3].
        """
        return P / (R * T)

    @staticmethod
    @jax.jit
    def stefan_boltzmann_flux(epsilon, T):
        """
        Stefan-Boltzmann Radiative Heat Flux.
        
        :param epsilon: Surface emissivity (0 to 1).
        :param T: Absolute temperature [K].
        :return: Radiative heat flux [W/m^2].
        """
        return epsilon * _STEFAN_BOLTZMANN * T**4

    @staticmethod
    @jax.jit
    def boussinesq_rho(rho0, beta, dT):
        """
        Boussinesq Approximation for density variation.
        
        :param rho0: Reference density [kg/m^3].
        :param beta: Thermal expansion coefficient [1/K].
        :param dT: Temperature difference from reference [K].
        :return: Approximated density [kg/m^3].
        """
        return rho0 * (1 - beta * dT)

    @staticmethod
    @jax.jit
    def specific_heat_nasa(T, coeffs):
        """
        NASA 7-coefficient polynomial for specific heat (Cp).
        
        :param T: Temperature [K].
        :param coeffs: Array of 7 coefficients [a0, a1, a2, a3, a4, ...]; first 5 used for Cp/R polynomial.
        :return: Specific heat capacity [J/kg*K].
        """
        a = jnp.asarray(coeffs)
        return a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4

    @staticmethod
    @jax.jit
    def power_law_mu(gamma_dot, K, n):
        """
        Non-Newtonian Power-Law viscosity.
        
        :param gamma_dot: Shear rate [1/s].
        :param K: Consistency index [Pa*s^n].
        :param n: Flow behavior index (dimensionless); n < 1 shear-thinning, n > 1 shear-thickening.
        :return: Apparent viscosity [Pa*s].
        
        Use case: Blood (n < 1), slurries (n > 1), shear-thinning fluids.
        """
        return K * gamma_dot ** (n - 1)

    @staticmethod
    @jax.jit
    def speed_of_sound(gamma, R, T):
        """
        Speed of sound in an ideal gas.
        
        :param gamma: Ratio of specific heats (dimensionless).
        :param R: Specific gas constant [J/kg*K].
        :param T: Absolute temperature [K].
        :return: Speed of sound [m/s].
        """
        return jnp.sqrt(gamma * R * T)

    @staticmethod
    @jax.jit
    def dynamic_pressure(rho, u):
        """
        Dynamic pressure (kinetic energy per unit volume).
        
        :param rho: Fluid density [kg/m^3].
        :param u: Flow velocity [m/s].
        :return: Dynamic pressure [Pa].
        """
        return 0.5 * rho * u**2

    @staticmethod
    @jax.jit
    def hydraulic_diameter(area, perimeter):
        """
        Hydraulic diameter for non-circular ducts.
        
        :param area: Cross-sectional flow area [m^2].
        :param perimeter: Wetted perimeter [m].
        :return: Hydraulic diameter [m].
        
        Use case: Reynolds number and friction in non-circular pipes.
        """
        return (4 * area) / perimeter

    @staticmethod
    @jax.jit
    def darcy_weisbach_dp(f, L, D, rho, u):
        """
        Darcy-Weisbach pressure drop in a pipe.
        
        :param f: Darcy friction factor (dimensionless).
        :param L: Pipe length [m].
        :param D: Pipe diameter [m].
        :param rho: Fluid density [kg/m^3].
        :param u: Mean flow velocity [m/s].
        :return: Pressure drop [Pa].
        """
        return f * (L / D) * (rho * u**2 / 2)

    @staticmethod
    @jax.jit
    def colebrook_friction(re, epsilon_d):
        """
        Haaland approximation for Darcy friction factor.
        
        :param re: Reynolds number (dimensionless).
        :param epsilon_d: Relative roughness (epsilon/D) (dimensionless).
        :return: Darcy friction factor (dimensionless).
        """
        return 1.0 / (-1.8 * jnp.log10((epsilon_d / 3.7) ** 1.11 + 6.9 / re)) ** 2

    @staticmethod
    @jax.jit
    def stokes_drag_force(mu, r, u):
        """
        Stokes drag force on a sphere (creeping flow, Re < 1).
        
        :param mu: Dynamic viscosity [Pa*s].
        :param r: Sphere radius [m].
        :param u: Relative flow velocity [m/s].
        :return: Drag force [N].
        """
        return 6 * jnp.pi * mu * r * u

    @staticmethod
    @jax.jit
    def kinematic_viscosity(mu, rho):
        """
        Kinematic viscosity (momentum diffusivity).
        
        :param mu: Dynamic viscosity [Pa*s].
        :param rho: Fluid density [kg/m^3].
        :return: Kinematic viscosity [m^2/s].
        """
        return mu / rho

    @staticmethod
    @jax.jit
    def thermal_diffusivity(k, rho, cp):
        """
        Thermal diffusivity of a material.
        
        :param k: Thermal conductivity [W/m*K].
        :param rho: Density [kg/m^3].
        :param cp: Specific heat capacity at constant pressure [J/kg*K].
        :return: Thermal diffusivity [m^2/s].
        """
        return k / (rho * cp)

    @staticmethod
    @jax.jit
    def mass_diffusivity(fo_mass, t, L):
        """
        Mass diffusivity recovered from Fourier number for mass.

        :param fo_mass: Mass Fourier number from Groups.fo_mass(D, t, L).
        :param t: Elapsed time [s].
        :param L: Characteristic length [m].
        :return: Mass diffusivity D [m^2/s].
        """
        return fo_mass * (L**2) / t

    @staticmethod
    @jax.jit
    def wave_speed_from_st(omega, L, st_wave):
        """
        Wave speed from wave Strouhal definition.

        :param omega: Angular frequency [rad/s].
        :param L: Characteristic length [m].
        :param st_wave: Wave Strouhal number from Groups.st_wave(omega, L, c).
        :return: Wave speed c [m/s].
        """
        return (omega * L) / st_wave

    @staticmethod
    @jax.jit
    def dynamic_viscosity_from_re(rho, u, L, re):
        """
        Dynamic viscosity from Reynolds number definition using local speed magnitude.

        :param rho: Density [kg/m^3].
        :param u: Velocity vector [m/s] or scalar speed.
        :param L: Characteristic length [m].
        :param re: Reynolds number.
        :return: Dynamic viscosity mu [Pa*s].
        """
        u_arr = jnp.asarray(u)
        u_mag = jnp.abs(u_arr) if u_arr.ndim == 0 else jnp.sqrt(jnp.sum(u_arr**2, axis=-1))
        return (jnp.asarray(rho) * u_mag * jnp.asarray(L)) / (jnp.asarray(re) + 1e-30)

    @staticmethod
    @jax.jit
    def kinematic_viscosity_from_re(U, L, re):
        """
        Kinematic viscosity from Reynolds number: :math:`\\nu = U L / \\mathrm{Re}`.

        Same coefficient as :func:`moju.piratio.laws.Laws.burgers_equation`.

        :param U: Characteristic velocity [m/s].
        :param L: Characteristic length [m].
        :param re: Reynolds number.
        :return: Kinematic viscosity nu [m^2/s].
        """
        return (jnp.asarray(U) * jnp.asarray(L)) / (jnp.asarray(re) + 1e-30)

    @staticmethod
    @jax.jit
    def scalar_diffusivity_from_pe(u, L, pe):
        """
        Effective scalar diffusivity from Peclet number definition.

        :param u: Velocity vector [m/s] or scalar speed.
        :param L: Characteristic length [m].
        :param pe: Peclet number.
        :return: Effective diffusivity kappa [m^2/s].
        """
        u_arr = jnp.asarray(u)
        u_mag = jnp.abs(u_arr) if u_arr.ndim == 0 else jnp.sqrt(jnp.sum(u_arr**2, axis=-1))
        return (u_mag * jnp.asarray(L)) / (jnp.asarray(pe) + 1e-30)

    @staticmethod
    @jax.jit
    def arrhenius_rate(A, Ea, T, R=8.314):
        """
        Arrhenius reaction rate constant vs temperature.
        
        :param A: Pre-exponential factor [1/s or appropriate units].
        :param Ea: Activation energy [J/mol].
        :param T: Absolute temperature [K].
        :param R: Universal gas constant [J/mol*K]; default 8.314.
        :return: Rate constant (same units as A).
        """
        return A * jnp.exp(-Ea / (R * T))

    @staticmethod
    @jax.jit
    def law_of_the_wall(y_plus):
        """
        Dimensionless velocity in the log-law region (turbulent boundary layer).
        
        :param y_plus: Dimensionless wall distance y_+ = y*u_tau/nu.
        :return: Dimensionless velocity u_+ = u/u_tau.
        
        Use case: Wall functions and boundary-layer modeling.
        """
        return 2.5 * jnp.log(y_plus) + 5.0

    @staticmethod
    @jax.jit
    def smagorinsky_nu_t(Cs, Delta, strain_rate_magnitude):
        """
        Smagorinsky subgrid eddy viscosity (scalar |S| form).

        :param Cs: Smagorinsky constant (dimensionless), typically ~0.1–0.2.
        :param Delta: Filter width [m].
        :param strain_rate_magnitude: Resolved strain-rate magnitude |S| [1/s]
            (often sqrt(2 S_ij S_ij) from the velocity field).
        :return: Eddy kinematic viscosity nu_t [m^2/s].

        Use case: LES-style closure auditing; combine with ``constitutive_audit`` chain
        residuals when ``strain_rate_magnitude`` varies in space or time.
        """
        return (Cs * Delta) ** 2 * strain_rate_magnitude

    @staticmethod
    @jax.jit
    def k_epsilon_nu_t(C_mu, k, epsilon, eps0):
        """
        Standard k–ε eddy viscosity (kinematic): νₜ = C_μ k² / (ε + ε₀).

        :param C_mu: Model constant (dimensionless), often ~0.09.
        :param k: Turbulent kinetic energy [m²/s²].
        :param epsilon: Turbulent dissipation rate [m²/s³].
        :param eps0: Positive floor on ε [same as epsilon] for numerical stability and
            stable JAX autodiff when ε → 0.
        :return: Eddy kinematic viscosity νₜ [m²/s].

        This closure is **algebraic νₜ only**. Full k–ε transport PDE residuals belong
        in ``Laws.*`` or a custom law, not in this helper.
        """
        return C_mu * k**2 / (epsilon + eps0)

    @staticmethod
    @jax.jit
    def k_omega_nu_t(k, omega, omega0):
        """
        Standard k–ω (Wilcox-style) eddy viscosity: νₜ = k / (ω + ω₀).

        :param k: Turbulent kinetic energy [m²/s²].
        :param omega: Specific dissipation rate [1/s].
        :param omega0: Positive floor on ω [same as omega] for stability and AD near ω → 0.
        :return: Eddy kinematic viscosity νₜ [m²/s].

        Algebraic νₜ only; ω and k transport equations are separate (``Laws.*`` / custom).
        """
        return k / (omega + omega0)

    @staticmethod
    @jax.jit
    def turbulent_viscous_acceleration_k_omega(u_laplacian, nu_molecular, k, omega, omega0):
        r"""Newtonian viscous acceleration :math:`(\nu_m+\nu_t)\nabla^2\mathbf{u}` with :math:`\nu_t=k/(\omega+\omega_0)`."""
        nu_t = Models.k_omega_nu_t(k, omega, omega0)
        nu_tot = jnp.asarray(nu_molecular) + nu_t
        ul = jnp.asarray(u_laplacian)
        if nu_tot.ndim < ul.ndim:
            nu_tot = nu_tot[..., jnp.newaxis]
        return nu_tot * ul

    @staticmethod
    @jax.jit
    def turbulent_viscous_acceleration_k_epsilon(u_laplacian, nu_molecular, C_mu, k, epsilon, eps0):
        r"""Same as :meth:`turbulent_viscous_acceleration_k_omega` but :math:`\nu_t=C_\mu k^2/(\varepsilon+\varepsilon_0)`."""
        nu_t = Models.k_epsilon_nu_t(C_mu, k, epsilon, eps0)
        nu_tot = jnp.asarray(nu_molecular) + nu_t
        ul = jnp.asarray(u_laplacian)
        if nu_tot.ndim < ul.ndim:
            nu_tot = nu_tot[..., jnp.newaxis]
        return nu_tot * ul

    @staticmethod
    @jax.jit
    def turbulent_viscous_acceleration_smagorinsky(u_laplacian, nu_molecular, Cs, Delta, strain_rate_magnitude):
        """Same with Smagorinsky :math:`\nu_t`."""
        nu_t = Models.smagorinsky_nu_t(Cs, Delta, strain_rate_magnitude)
        nu_tot = jnp.asarray(nu_molecular) + nu_t
        ul = jnp.asarray(u_laplacian)
        if nu_tot.ndim < ul.ndim:
            nu_tot = nu_tot[..., jnp.newaxis]
        return nu_tot * ul

    @staticmethod
    @jax.jit
    def turbulent_viscous_acceleration_compressible_k_omega(rho, u_laplacian, nu_molecular, k, omega, omega0):
        return jnp.asarray(rho)[..., jnp.newaxis] * Models.turbulent_viscous_acceleration_k_omega(
            u_laplacian, nu_molecular, k, omega, omega0
        )

    @staticmethod
    @jax.jit
    def turbulent_viscous_acceleration_compressible_k_epsilon(rho, u_laplacian, nu_molecular, C_mu, k, epsilon, eps0):
        return jnp.asarray(rho)[..., jnp.newaxis] * Models.turbulent_viscous_acceleration_k_epsilon(
            u_laplacian, nu_molecular, C_mu, k, epsilon, eps0
        )

    @staticmethod
    @jax.jit
    def turbulent_viscous_acceleration_compressible_smagorinsky(
        rho, u_laplacian, nu_molecular, Cs, Delta, strain_rate_magnitude
    ):
        return jnp.asarray(rho)[..., jnp.newaxis] * Models.turbulent_viscous_acceleration_smagorinsky(
            u_laplacian, nu_molecular, Cs, Delta, strain_rate_magnitude
        )

    @staticmethod
    @jax.jit
    def orifice_flow(Cd, A, dp, rho):
        """
        Volumetric flow rate through an orifice or restriction.
        
        :param Cd: Discharge coefficient (dimensionless).
        :param A: Orifice cross-sectional area [m^2].
        :param dp: Pressure drop across orifice [Pa].
        :param rho: Fluid density [kg/m^3].
        :return: Volumetric flow rate [m^3/s].
        """
        return Cd * A * jnp.sqrt(2 * dp / rho)

    @staticmethod
    @jax.jit
    def heat_flux_conduction(k, dT, dx):
        """
        Fourier's law: conductive heat flux.
        
        :param k: Thermal conductivity [W/m*K].
        :param dT: Temperature difference (positive for heat flow in +x) [K].
        :param dx: Distance over which dT is measured [m].
        :return: Heat flux in the x-direction [W/m^2].
        """
        return -k * (dT / dx)

    @staticmethod
    @jax.jit
    def isotropic_linear_stress(E, nu, strain):
        r"""
        Isotropic linear-elastic stress via the Voigt-form stiffness :math:`\mathbf{C}(E,\nu)\cdot\boldsymbol{\varepsilon}`.

        Dispatches on the last dimension ``d`` of ``strain``:

        * ``d = 1``: 1-D axial — :math:`\sigma = E\,\varepsilon`.
        * ``d = 3``: 2-D plane-stress Voigt components
          :math:`[\sigma_{xx},\sigma_{yy},\sigma_{xy}]`;
          :math:`\mathbf{C} = \frac{E}{1-\nu^2}\begin{bmatrix}1&\nu&0\\\nu&1&0\\0&0&\frac{1-\nu}{2}\end{bmatrix}`.
        * ``d = 6``: 3-D isotropic Voigt components
          :math:`[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{yz},\sigma_{xz}]`;
          :math:`\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}`, :math:`G = \frac{E}{2(1+\nu)}`.

        :param E: Young's modulus (dimensional [Pa] or nondimensional :math:`E^* = E/E_\mathrm{ref}`).
        :param nu: Poisson ratio (dimensionless).
        :param strain: Voigt strain vector. Shape (..., d), ``d`` ∈ {1, 3, 6}.
        :return: Voigt stress vector. Shape (..., d).

        Use case: Implied constitutive audit for :meth:`hookes_law_residual`; checks that predicted
        stress and strain fields are consistent with isotropic material constants ``E`` and ``nu``.
        """
        eps = jnp.asarray(strain)
        d = eps.shape[-1]
        E_s = jnp.asarray(E, dtype=eps.dtype)
        nu_s = jnp.asarray(nu, dtype=eps.dtype)
        if d == 1:
            # σ = E ε  (1-D axial)
            return E_s * eps
        elif d == 3:
            # 2-D plane stress Voigt: [σ_xx, σ_yy, σ_xy]
            fac = E_s / (1.0 - nu_s ** 2)
            z = jnp.zeros_like(E_s)
            o = jnp.ones_like(E_s)
            sh = (1.0 - nu_s) / 2.0
            C = jnp.stack([
                jnp.stack([o,    nu_s, z ]),
                jnp.stack([nu_s, o,    z ]),
                jnp.stack([z,    z,    sh]),
            ]) * fac  # shape (3, 3)
        elif d == 6:
            # 3-D isotropic Voigt: [σ_xx,σ_yy,σ_zz,σ_xy,σ_yz,σ_xz]
            lam = E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s))
            G = E_s / (2.0 * (1.0 + nu_s))
            l2g = lam + 2.0 * G
            z = jnp.zeros_like(E_s)
            C = jnp.stack([
                jnp.stack([l2g, lam, lam, z, z, z]),
                jnp.stack([lam, l2g, lam, z, z, z]),
                jnp.stack([lam, lam, l2g, z, z, z]),
                jnp.stack([z,   z,   z,   G, z, z]),
                jnp.stack([z,   z,   z,   z, G, z]),
                jnp.stack([z,   z,   z,   z, z, G]),
            ])  # shape (6, 6)
        else:
            raise ValueError(
                f"isotropic_linear_stress: unsupported Voigt dimension d={d}; expected 1, 3, or 6."
            )
        return jnp.einsum("ij,...j->...i", C, eps)

    @staticmethod
    @jax.jit
    def surface_tension_eotvos(gamma0, T, Tc):
        """
        Eötvös rule: surface tension vs temperature (up to critical point).
        
        :param gamma0: Reference surface tension at T=0 [N/m].
        :param T: Temperature [K].
        :param Tc: Critical temperature [K].
        :return: Surface tension [N/m].
        
        Use case: Liquids; valid for T < Tc.
        """
        return gamma0 * (1 - T / Tc) ** (11 / 9)
