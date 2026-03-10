import jax
import jax.numpy as jnp

# Stefan-Boltzmann constant [W/(m^2 K^4)]; as JAX scalar for tracing
_STEFAN_BOLTZMANN = 5.670374419e-8

class Models:
    """
    Differentiable physical models for anchoring AI predictions to reality.
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
