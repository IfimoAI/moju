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
