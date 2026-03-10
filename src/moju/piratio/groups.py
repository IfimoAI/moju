import jax
import jax.numpy as jnp

class Groups:
    """
    JAX-accelerated suite of dimensionless groups for scaling Physics AI models.
    """

    @staticmethod
    @jax.jit
    def re(u, L, rho, mu):
        """
        Reynolds Number (Re): Ratio of Inertial to Viscous forces.
        
        :param u: Flow velocity [m/s].
        :param L: Characteristic length scale (e.g., pipe diameter) [m].
        :param rho: Fluid density [kg/m^3].
        :param mu: Dynamic viscosity [Pa*s].
        :return: Dimensionless Reynolds number.
        
        SciML: Scales the diffusive term in Navier-Stokes loss functions.
        """
        return (rho * u * L) / mu

    @staticmethod
    @jax.jit
    def pr(mu, cp, k):
        """
        Prandtl Number (Pr): Ratio of Momentum to Thermal diffusivity.
        
        :param mu: Dynamic viscosity [Pa*s].
        :param cp: Specific heat capacity at constant pressure [J/kg*K].
        :param k: Thermal conductivity [W/m*K].
        :return: Dimensionless Prandtl number.
        
        SciML: Bridges the velocity and temperature fields in multi-physics PINNs.
        """
        return (cp * mu) / k

    @staticmethod
    @jax.jit
    def nu(h, L, k):
        """
        Nusselt Number (Nu): Ratio of Convective to Conductive heat transfer.
        
        :param h: Convective heat transfer coefficient [W/m^2*K].
        :param L: Characteristic length [m].
        :param k: Fluid thermal conductivity [W/m*K].
        :return: Dimensionless Nusselt number.
        
        SciML: Acts as a target metric for AI-driven thermal optimization.
        """
        return (h * L) / k

    @staticmethod
    @jax.jit
    def gr(beta, dT, L, nu):
        """
        Grashof Number (Gr): Ratio of Buoyancy to Viscous forces.
        
        :param beta: Volumetric thermal expansion coefficient [1/K].
        :param dT: Temperature difference (T_surface - T_ambient) [K].
        :param L: Characteristic length [m].
        :param nu: Kinematic viscosity [m^2/s].
        :return: Dimensionless Grashof number.
        
        SciML: Drives the buoyancy source term in natural convection models.
        """
        g = 9.81
        return (g * beta * dT * L**3) / (nu**2)

    @staticmethod
    @jax.jit
    def ma(u, a):
        """
        Mach Number (Ma): Ratio of Flow velocity to speed of sound.
        
        :param u: Flow velocity [m/s].
        :param a: Local speed of sound [m/s].
        :return: Dimensionless Mach number.
        """
        return u / a

    @staticmethod
    @jax.jit
    def we(rho, u, L, sigma):
        """
        Weber Number (We): Ratio of Inertia to Surface Tension.
        
        :param rho: Fluid density [kg/m^3].
        :param u: Flow velocity [m/s].
        :param L: Characteristic length (e.g., droplet diameter) [m].
        :param sigma: Surface tension [N/m].
        :return: Dimensionless Weber number.
        """
        return (rho * u**2 * L) / sigma
