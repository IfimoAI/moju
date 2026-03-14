import jax
import jax.numpy as jnp


class Laws:
    """
    JIT-differentiable conservation and constitutive laws for Physics AI.
    All functions return a residual (R). R=0 implies physical consistency.
    Inputs may have leading batch dimensions; all operations support both
    single-point and batched evaluation.
    """

    # --- FLUID DYNAMICS ---

    @staticmethod
    @jax.jit
    def mass_incompressible(u_grad):
        """
        Continuity equation (incompressible): div(u) = 0.

        :param u_grad: Velocity gradient tensor (Jacobian) [1/s]. Shape (..., d, d).
        :return: Scalar residual of div(u). Shape (...).

        Use case: Low-speed liquid or gas flow (Ma < 0.3).
        """
        return jnp.trace(u_grad, axis1=-2, axis2=-1)

    @staticmethod
    @jax.jit
    def mass_compressible(rho, rho_t, u, rho_grad, u_grad):
        """
        Continuity equation (compressible): d(rho)/dt + div(rho*u) = 0.

        :param rho: Fluid density [kg/m^3].
        :param rho_t: Time derivative of density [kg/m^3*s].
        :param u: Velocity vector [m/s]. Shape (..., d).
        :param rho_grad: Spatial gradient of density [kg/m^4]. Shape (..., d).
        :param u_grad: Velocity gradient tensor [1/s]. Shape (..., d, d).
        :return: Scalar residual. Shape (...).

        Use case: High-speed gas flow where density changes over time or space.
        """
        return rho_t + jnp.sum(u * rho_grad, axis=-1) + rho * jnp.trace(u_grad, axis1=-2, axis2=-1)

    @staticmethod
    @jax.jit
    def momentum_navier_stokes(u_t, u, u_grad, p_grad, u_laplacian, re):
        """
        Navier-Stokes momentum equation (dimensionless residual).

        :param u_t: Time derivative of velocity [m/s^2]. Shape (..., d).
        :param u: Velocity vector [m/s]. Shape (..., d).
        :param u_grad: Velocity gradient tensor [1/s]. Shape (..., d, d).
        :param p_grad: Pressure gradient vector [Pa/m]. Shape (..., d).
        :param u_laplacian: Laplacian of velocity [1/m*s]. Shape (..., d).
        :param re: Reynolds number from Groups.re().
        :return: Vector residual of momentum balance. Shape (..., d).

        Use case: Standard Newtonian fluid flow (water, air, oils).
        """
        advection = u_t + jnp.einsum("...ij,...j->...i", u_grad, u)
        viscous = (1.0 / re) * u_laplacian
        return advection + p_grad - viscous

    @staticmethod
    @jax.jit
    def stokes_flow(p_grad, u_laplacian, re):
        """
        Stokes (creeping) flow: inertia neglected.

        :param p_grad: Pressure gradient vector [Pa/m]. Shape (..., d).
        :param u_laplacian: Laplacian of velocity [1/m*s]. Shape (..., d).
        :param re: Reynolds number from Groups.re().
        :return: Vector residual. Shape (..., d).

        Use case: Very slow flows where inertia is negligible (Re << 1).
        """
        return p_grad - (1.0 / re) * u_laplacian

    @staticmethod
    @jax.jit
    def euler_momentum(u_t, u, u_grad, p_grad, eu):
        """
        Euler equations (inviscid momentum residual) using Euler number.

        :param u_t: Time derivative of velocity [m/s^2]. Shape (..., d).
        :param u: Velocity vector [m/s]. Shape (..., d).
        :param u_grad: Velocity gradient tensor [1/s]. Shape (..., d, d).
        :param p_grad: Dimensionless pressure gradient (scale p by rho*U^2 via Groups.eu()). Shape (..., d).
        :param eu: Euler number from Groups.eu(dp, rho, u).
        :return: Vector residual. Shape (..., d).

        Use case: High-speed aerodynamics where viscosity is ignored.
        """
        return u_t + jnp.einsum("...ij,...j->...i", u_grad, u) + p_grad

    # --- HEAT & MASS TRANSPORT ---

    @staticmethod
    @jax.jit
    def fourier_conduction(T_t, T_laplacian, fo, t, L):
        """
        Heat diffusion (Fourier's law) using Fourier number: Fo·T_t - T_laplacian (scaled) = 0.

        Uses Groups.fo(alpha, t, L); alpha = fo*L^2/t so residual is T_t - alpha*laplacian(T).

        :param T_t: Time derivative of temperature [K/s].
        :param T_laplacian: Laplacian of temperature [K/m^2].
        :param fo: Fourier number from Groups.fo(alpha, t, L).
        :param t: Elapsed time [s].
        :param L: Characteristic length [m].
        :return: Scalar residual.

        Use case: Pure heat conduction in solids or static fluids.
        """
        alpha = fo * (L**2) / t
        return T_t - alpha * T_laplacian

    @staticmethod
    @jax.jit
    def advection_diffusion(phi_t, u, phi_grad, phi_laplacian, pe):
        """
        Scalar transport (energy or species): d(phi)/dt + u·grad(phi) - (1/Pe) laplacian(phi) = 0.

        :param phi_t: Time derivative of scalar field.
        :param u: Velocity vector [m/s]. Shape (..., d).
        :param phi_grad: Gradient of scalar field. Shape (..., d).
        :param phi_laplacian: Laplacian of scalar field.
        :param pe: Peclet number (Re*Pr or Re*Sc) from Groups.pe().
        :return: Scalar residual.

        Use case: Temperature or chemical concentration in a moving flow.
        """
        return phi_t + jnp.sum(phi_grad * u, axis=-1) - (1.0 / pe) * phi_laplacian

    @staticmethod
    @jax.jit
    def viscous_dissipation(u_grad, re, ec, U, L):
        """
        Viscous dissipation source (dimensionless) using Re and Ec.

        Dimensionless source = (Ec/Re) * 2 * sum(strain_star^2); strain_star = strain_rate * L/U.

        :param u_grad: Velocity gradient tensor [1/s]. Shape (..., d, d).
        :param re: Reynolds number from Groups.re().
        :param ec: Eckert number from Groups.ec().
        :param U: Characteristic velocity [m/s].
        :param L: Characteristic length [m].
        :return: Dimensionless heat source (scale by rho*U^3/L for dimensional).

        Use case: High-speed or very viscous flows where flow friction heats the fluid.
        """
        strain_rate = 0.5 * (u_grad + jnp.swapaxes(u_grad, -2, -1))
        strain_star = strain_rate * (L / U)
        return (ec / re) * 2.0 * jnp.sum(strain_star**2, axis=(-2, -1))

    # --- SOLID MECHANICS & POROUS MEDIA ---

    @staticmethod
    @jax.jit
    def hookes_law_residual(stress, strain, stiffness_tensor):
        """
        Linear elasticity constitutive residual: stress - C : strain.

        :param stress: Stress tensor (Voigt or full) [Pa]. Shape (..., d).
        :param strain: Strain tensor (dimensionless). Shape (..., d).
        :param stiffness_tensor: Material stiffness matrix C. Shape (..., d, d).
        :return: Tensor residual. Shape (..., d).

        Use case: Structural AI predicting deformation of metals or polymers.
        """
        return stress - jnp.einsum("...ij,...j->...i", stiffness_tensor, strain)

    @staticmethod
    @jax.jit
    def darcy_flow(u, p_grad, da, L, mu):
        """
        Darcy's law for porous media using Darcy number: u + (Da*L^2/mu) grad(p) = 0.

        :param u: Superficial velocity vector [m/s]. Shape (..., d).
        :param p_grad: Pressure gradient [Pa/m]. Shape (..., d).
        :param da: Darcy number from Groups.da(K, L).
        :param L: Characteristic length [m].
        :param mu: Fluid viscosity [Pa*s].
        :return: Vector residual. Shape (..., d).

        Use case: Ground water flow or oil reservoir modeling.
        """
        return u + (da * (L**2) / mu) * p_grad

    @staticmethod
    @jax.jit
    def brinkman_extension(u, u_laplacian, p_grad, re, da, mu, L):
        """
        Brinkman equations using Re and Da: -p_grad + (mu/Re) laplacian(u) - (mu/(Da*L^2)) u = 0.

        :param u: Velocity vector [m/s]. Shape (..., d).
        :param u_laplacian: Laplacian of velocity. Shape (..., d).
        :param p_grad: Pressure gradient [Pa/m]. Shape (..., d).
        :param re: Reynolds number from Groups.re().
        :param da: Darcy number from Groups.da(K, L).
        :param mu: Dynamic viscosity [Pa*s].
        :param L: Characteristic length [m].
        :return: Vector residual. Shape (..., d).

        Use case: Flow in high-porosity media where viscous shear near walls matters.
        """
        shear_term = (mu / re) * u_laplacian
        darcy_term = (mu / (da * L**2)) * u
        return -p_grad + shear_term - darcy_term

    # --- ELECTROMAGNETICS ---

    @staticmethod
    @jax.jit
    def poisson_equation(phi_laplacian, source, epsilon):
        """
        Poisson equation: laplacian(phi) + source/epsilon = 0.

        :param phi_laplacian: Laplacian of the potential field.
        :param source: Source term (e.g., charge density or mass).
        :param epsilon: Permittivity or field constant.
        :return: Scalar residual.

        Use case: Electrostatics, gravity, or pressure-Poisson in CFD.
        """
        return phi_laplacian + (source / epsilon)

    @staticmethod
    @jax.jit
    def faraday_law(E_curl, B_t):
        """
        Faraday's law of induction: curl(E) + dB/dt = 0.

        :param E_curl: Curl of the electric field. Shape (..., 3).
        :param B_t: Time derivative of the magnetic field. Shape (..., 3).
        :return: Vector residual. Shape (..., 3).

        Use case: Electromagnetic induction simulations.
        """
        return E_curl + B_t

    # --- VIBRATIONS & WAVES ---

    @staticmethod
    @jax.jit
    def wave_equation(phi_tt, phi_laplacian, st_wave, omega, L):
        """
        Classical wave equation using wave Strouhal: phi_tt - c^2 laplacian(phi) = 0 with c = omega*L/st_wave.

        :param phi_tt: Second time derivative of field.
        :param phi_laplacian: Spatial Laplacian of field.
        :param st_wave: Wave Strouhal from Groups.st_wave(omega, L, c).
        :param omega: Angular frequency [rad/s].
        :param L: Characteristic length [m].
        :return: Scalar residual.

        Use case: Acoustics, seismic waves, or string vibrations.
        """
        c = (omega * L) / st_wave
        return phi_tt - (c**2) * phi_laplacian

    @staticmethod
    @jax.jit
    def helmholtz_equation(phi, phi_laplacian, kL, L):
        """
        Helmholtz equation using dimensionless wavenumber: laplacian(phi) + (kL/L)^2 phi = 0.

        :param phi: Field amplitude.
        :param phi_laplacian: Laplacian of field.
        :param kL: Dimensionless wavenumber from Groups.wavenumber(k, L).
        :param L: Characteristic length [m].
        :return: Scalar residual.

        Use case: Steady-state frequency-domain wave problems (e.g., resonance).
        """
        k_sq = (kL / L) ** 2
        return phi_laplacian + k_sq * phi

    # --- CHEMICAL & KINETIC ---

    @staticmethod
    @jax.jit
    def fick_diffusion(phi_t, phi_laplacian, fo_mass, t, L):
        """
        Fick's second law of diffusion using Fourier number for mass: d(phi)/dt - D laplacian(phi) = 0.

        Uses Groups.fo_mass(D, t, L); D = fo_mass*L^2/t so residual is phi_t - D*laplacian(phi).

        :param phi_t: Rate of change of concentration.
        :param phi_laplacian: Laplacian of concentration.
        :param fo_mass: Fourier number for mass from Groups.fo_mass(D, t, L).
        :param t: Elapsed time [s].
        :param L: Characteristic length [m].
        :return: Scalar residual.

        Use case: Mixing of chemicals or heat diffusion in a static medium.
        """
        D = fo_mass * (L**2) / t
        return phi_t - D * phi_laplacian

    @staticmethod
    @jax.jit
    def burgers_equation(u_t, u, u_grad, u_laplacian, re, U, L):
        """
        Viscous Burgers equation using Reynolds number: u_t + (u·grad)u - nu laplacian(u) = 0 with nu = U*L/Re.

        :param u_t: Time derivative of velocity [m/s^2]. Shape (..., d).
        :param u: Velocity vector [m/s]. Shape (..., d).
        :param u_grad: Velocity gradient tensor [1/s]. Shape (..., d, d).
        :param u_laplacian: Laplacian of velocity. Shape (..., d).
        :param re: Reynolds number from Groups.re().
        :param U: Characteristic velocity [m/s].
        :param L: Characteristic length [m].
        :return: Vector residual. Shape (..., d).

        Use case: Simplified turbulence modeling or shockwave propagation.
        """
        nu = (U * L) / re
        return u_t + jnp.einsum("...ij,...j->...i", u_grad, u) - nu * u_laplacian

    @staticmethod
    @jax.jit
    def laplace_equation(phi_laplacian):
        """
        Laplace equation: laplacian(phi) = 0.

        :param phi_laplacian: Laplacian of the potential field.
        :return: Scalar residual.

        Use case: Steady-state potential flow or steady heat conduction without sources.
        """
        return phi_laplacian

    @staticmethod
    @jax.jit
    def schrodinger_steady(psi_laplacian, V, E, psi, m, h_bar=1.054571817e-34):
        """
        Time-independent Schrödinger equation residual: -(hbar^2/2m) laplacian(psi) + (V - E) psi = 0.

        :param psi_laplacian: Laplacian of the wavefunction.
        :param V: Potential energy [J].
        :param E: Total energy [J].
        :param psi: Wavefunction value.
        :param m: Particle mass [kg].
        :param h_bar: Reduced Planck constant [J*s]; default 1.054571817e-34.
        :return: Scalar residual.

        Use case: AI modeling of quantum states or electron density.
        """
        coeff = -(h_bar**2) / (2 * m)
        return coeff * psi_laplacian + (V - E) * psi

    @staticmethod
    @jax.jit
    def laplace_beltrami(phi_laplacian_g):
        """
        Laplace-Beltrami on a manifold: residual of the Laplacian on metric g.

        :param phi_laplacian_g: Laplacian of the field on the manifold (metric g).
        :return: Scalar residual.

        Use case: Diffusion or geometric deep learning on curved surfaces/manifolds.
        """
        return phi_laplacian_g
