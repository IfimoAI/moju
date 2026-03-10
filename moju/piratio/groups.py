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

    @staticmethod
    @jax.jit
    def pe(re, pr):
        """
        Peclet Number (Pe): Ratio of Advection to Diffusion.
        
        :param re: Reynolds number (dimensionless).
        :param pr: Prandtl number (dimensionless).
        :return: Dimensionless Peclet number.
        
        Use case: Heat and mass transfer in flowing fluids.
        SciML: Determines if the AI should prioritize the advective or diffusive term.
        """
        return re * pr

    @staticmethod
    @jax.jit
    def st(f, u, L):
        """
        Strouhal Number (St): Ratio of Oscillatory to Inertial mechanisms.
        
        :param f: Characteristic frequency (e.g., vortex shedding) [Hz].
        :param u: Flow velocity [m/s].
        :param L: Characteristic length [m].
        :return: Dimensionless Strouhal number.
        
        Use case: Vortex shedding and unsteady aerodynamics.
        SciML: Helps AI learn temporal frequencies in periodic flow signatures.
        """
        return (f * L) / u

    @staticmethod
    @jax.jit
    def bi(h, L, k_solid):
        """
        Biot Number (Bi): Internal vs External thermal resistance.
        
        :param h: Convective heat transfer coefficient [W/m^2*K].
        :param L: Characteristic length (e.g., half-thickness of solid) [m].
        :param k_solid: Thermal conductivity of the solid [W/m*K].
        :return: Dimensionless Biot number.
        
        Use case: Transient conduction in solids.
        SciML: Decides if a solid's internal temperature gradient can be ignored (lumped mass).
        """
        return (h * L) / k_solid

    @staticmethod
    @jax.jit
    def fo(alpha, t, L):
        """
        Fourier Number (Fo): Dimensionless time for heat conduction.
        
        :param alpha: Thermal diffusivity of the solid [m^2/s].
        :param t: Elapsed time [s].
        :param L: Characteristic length [m].
        :return: Dimensionless Fourier number.
        
        Use case: Measuring how far heat has penetrated a solid.
        SciML: Provides a normalized time-scale for unsteady thermal PINNs.
        """
        return (alpha * t) / L**2

    @staticmethod
    @jax.jit
    def sc(nu, D):
        """
        Schmidt Number (Sc): Momentum vs Mass diffusivity.
        
        :param nu: Kinematic viscosity [m^2/s].
        :param D: Mass diffusivity [m^2/s].
        :return: Dimensionless Schmidt number.
        
        Use case: Mass transfer and chemical vapor deposition.
        SciML: Controls the coupling between velocity and species concentration fields.
        """
        return nu / D

    @staticmethod
    @jax.jit
    def le(sc, pr):
        """
        Lewis Number (Le): Thermal vs Mass diffusivity.
        
        :param sc: Schmidt number (dimensionless).
        :param pr: Prandtl number (dimensionless).
        :return: Dimensionless Lewis number.
        
        Use case: Combined heat and mass transfer (e.g., combustion).
        SciML: Regulates the relative widths of thermal and concentration boundary layers.
        """
        return sc / pr

    @staticmethod
    @jax.jit
    def kn(lambda_mfp, L):
        """
        Knudsen Number (Kn): Mean free path vs Length scale.
        
        :param lambda_mfp: Mean free path of the gas [m].
        :param L: Characteristic length scale [m].
        :return: Dimensionless Knudsen number.
        
        Use case: Rarefied gas dynamics (microchannels, high-altitude flight).
        SciML: Indicates when Navier-Stokes breaks down (AI may need kinetic models).
        """
        return lambda_mfp / L

    @staticmethod
    @jax.jit
    def bo(rho, g, L, sigma):
        """
        Bond Number (Bo): Gravity vs Surface tension.
        
        :param rho: Fluid density [kg/m^3].
        :param g: Gravitational acceleration [m/s^2].
        :param L: Characteristic length [m].
        :param sigma: Surface tension [N/m].
        :return: Dimensionless Bond number.
        
        Use case: Shape of bubbles and capillary action.
        SciML: Critical for AI-driven microfluidics or ink-jet simulation.
        """
        return (rho * g * L**2) / sigma

    @staticmethod
    @jax.jit
    def ca(mu, u, sigma):
        """
        Capillary Number (Ca): Viscous forces vs Surface tension.
        
        :param mu: Dynamic viscosity [Pa*s].
        :param u: Flow velocity [m/s].
        :param sigma: Surface tension [N/m].
        :return: Dimensionless Capillary number.
        
        Use case: Porous media flow or thin-film coating.
        SciML: Scales the pressure-jump condition across interfaces in the AI loss.
        """
        return (mu * u) / sigma

    @staticmethod
    @jax.jit
    def eu(dp, rho, u):
        """
        Euler Number (Eu): Pressure vs Inertial forces.
        
        :param dp: Pressure difference [Pa].
        :param rho: Fluid density [kg/m^3].
        :param u: Characteristic flow velocity [m/s].
        :return: Dimensionless Euler number.
        
        Use case: Flow through valves, orifices, or pumps.
        SciML: Used to check the balance of the pressure-gradient term in momentum residuals.
        """
        return dp / (rho * u**2)

    @staticmethod
    @jax.jit
    def da(K, L):
        """
        Darcy Number (Da): Permeability vs Characteristic area.
        
        :param K: Permeability of the porous medium [m^2].
        :param L: Characteristic length scale [m].
        :return: Dimensionless Darcy number.
        
        Use case: Flow through sponges, soils, or filters.
        SciML: Scales the Darcy resistance term in porous-media-informed networks.
        """
        return K / L**2

    @staticmethod
    @jax.jit
    def ec(u, cp, dT):
        """
        Eckert Number (Ec): Kinetic energy vs Enthalpy difference.
        
        :param u: Flow velocity [m/s].
        :param cp: Specific heat capacity at constant pressure [J/kg*K].
        :param dT: Characteristic temperature difference [K].
        :return: Dimensionless Eckert number.
        
        Use case: High-speed flow where viscous dissipation causes heating.
        SciML: Controls the strength of the viscous dissipation term in the energy equation.
        """
        return u**2 / (cp * dT)
