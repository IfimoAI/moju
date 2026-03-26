import jax
import jax.numpy as jnp


class Laws:
    """
    JIT-differentiable **nondimensional** governing-law residuals for Physics AI.

    Moju treats governing laws as **inherently nondimensional**: each residual is written in
    reference-scaled coordinates and/or dimensionless groups from :class:`moju.piratio.Groups`
    (and, where useful, coefficient recovery from :class:`moju.piratio.Models`). Callers supply
    fields and derivatives in the documented nondimensional sense—typically
    :math:`u^* = u/U`, :math:`x^* = x/L`, :math:`t^* = t U/L` (or Fourier time :math:`\\alpha t/L^2`),
    and pressure gradients scaled like :math:`\\nabla^* p^*` with :math:`p^* = p/(\\rho_\\mathrm{ref} U^2)`,
    unless a law states otherwise.

    All functions return a residual ``R``; ``R = 0`` implies physical consistency in that
    nondimensionalization. Inputs may have leading batch dimensions; operations broadcast for
    single-point and batched evaluation.
    """

    # --- FLUID DYNAMICS ---

    @staticmethod
    @jax.jit
    def mass_incompressible(u_grad):
        """
        Continuity (incompressible): :math:`\\nabla^*\\cdot\\mathbf{u}^* = 0`.

        :param u_grad: Nondimensional Jacobian :math:`\\partial u^*_i/\\partial x^*_j`. Shape (..., d, d).
        :return: Scalar residual ``trace(u_grad)``, i.e. :math:`\\nabla^*\\cdot\\mathbf{u}^*`. Shape (...).

        Use case: Low-speed liquid or gas flow (Ma < 0.3).
        """
        return jnp.trace(u_grad, axis1=-2, axis2=-1)

    @staticmethod
    @jax.jit
    def mass_compressible(rho, rho_t, u, rho_grad, u_grad):
        """
        Continuity (compressible) in nondimensional form:
        :math:`\\partial\\rho^*/\\partial t^* + \\mathbf{u}^*\\cdot\\nabla^*\\rho^* + \\rho^*\\nabla^*\\cdot\\mathbf{u}^* = 0`.

        :param rho: Nondimensional density :math:`\\rho^* = \\rho/\\rho_\\mathrm{ref}`.
        :param rho_t: :math:`\\partial\\rho^*/\\partial t^*` with :math:`t^* = t U_\\mathrm{ref}/L`.
        :param u: :math:`\\mathbf{u}^* = \\mathbf{u}/U_\\mathrm{ref}`. Shape (..., d).
        :param rho_grad: :math:`\\nabla^*\\rho^* = (L/\\rho_\\mathrm{ref})\\nabla\\rho`. Shape (..., d).
        :param u_grad: :math:`\\partial u^*_i/\\partial x^*_j = (L/U_\\mathrm{ref})\\partial u_i/\\partial x_j`. Shape (..., d, d).
        :return: Scalar residual. Shape (...).

        Use case: High-speed gas flow where density changes over time or space.
        """
        return rho_t + jnp.sum(u * rho_grad, axis=-1) + rho * jnp.trace(u_grad, axis1=-2, axis2=-1)

    @staticmethod
    @jax.jit
    def momentum_navier_stokes(u_t, u, u_grad, p_grad, u_laplacian, re):
        """
        Navier–Stokes momentum in nondimensional form:
        :math:`\\partial\\mathbf{u}^*/\\partial t^* + \\mathbf{u}^*\\cdot\\nabla^*\\mathbf{u}^* + \\nabla^* p^* - \\mathrm{Re}^{-1}\\nabla^{*2}\\mathbf{u}^* = 0`.

        :param u_t: :math:`\\partial\\mathbf{u}^*/\\partial t^*`. Shape (..., d).
        :param u: :math:`\\mathbf{u}^*`. Shape (..., d).
        :param u_grad: :math:`\\nabla^*\\mathbf{u}^*`. Shape (..., d, d).
        :param p_grad: :math:`\\nabla^* p^*` with :math:`p^* = p/(\\rho_\\mathrm{ref} U^2)`. Shape (..., d).
        :param u_laplacian: :math:`\\nabla^{*2}\\mathbf{u}^*`. Shape (..., d).
        :param re: Reynolds number from :func:`moju.piratio.Groups.re`.
        :return: Vector residual. Shape (..., d).

        Use case: Standard Newtonian fluid flow (water, air, oils).
        """
        advection = u_t + jnp.einsum("...ij,...j->...i", u_grad, u)
        viscous = (1.0 / re) * u_laplacian
        return advection + p_grad - viscous

    @staticmethod
    @jax.jit
    def momentum_incompressible_newtonian_laplacian(u_t, u, u_grad, p_grad, u_laplacian, nu_eff):
        r"""
        Incompressible momentum with explicit effective **kinematic** viscosity :math:`\nu_\mathrm{eff}^*`.

        Residual:
        :math:`\partial\mathbf{u}^*/\partial t^* + \mathbf{u}^*\cdot\nabla^*\mathbf{u}^* + \nabla^* p^* - \nu_\mathrm{eff}^*\nabla^{*2}\mathbf{u}^* = 0`.

        Pair with algebraic :math:`\nu_t` closures by taking :math:`\nu_\mathrm{eff}^* = \nu_m^* + \nu_t^*`
        (nondimensional molecular plus turbulent kinematic viscosity).

        :return: Vector residual, shape (..., d).
        """
        advection = u_t + jnp.einsum("...ij,...j->...i", u_grad, u)
        nu = jnp.asarray(nu_eff)
        if nu.ndim < u_laplacian.ndim:
            nu = nu[..., jnp.newaxis]
        return advection + p_grad - nu * jnp.asarray(u_laplacian)

    @staticmethod
    @jax.jit
    def momentum_compressible_newtonian_laplacian(rho, u_t, u, u_grad, p_grad, u_laplacian, nu_eff):
        r"""
        Compressible momentum in simplified Newtonian form (diagonal viscous term):

        :math:`\rho^*(\partial\mathbf{u}^*/\partial t^* + \mathbf{u}^*\cdot\nabla^*\mathbf{u}^*) + \nabla^* p^* - \rho^*\nu_\mathrm{eff}^*\nabla^{*2}\mathbf{u}^* = 0`.

        Ignores bulk viscosity and :math:`\nabla\nu` contributions; useful as a monitor residual
        when comparing an algebraic :math:`\nu_t` field to a momentum-implied viscous acceleration.

        :return: Vector residual, shape (..., d).
        """
        advection = u_t + jnp.einsum("...ij,...j->...i", u_grad, u)
        rho_b = jnp.asarray(rho)[..., jnp.newaxis]
        nu = jnp.asarray(nu_eff)
        if nu.ndim < u_laplacian.ndim:
            nu = nu[..., jnp.newaxis]
        ul = jnp.asarray(u_laplacian)
        return rho_b * advection + p_grad - rho_b * nu * ul

    @staticmethod
    @jax.jit
    def stokes_flow(p_grad, u_laplacian, re):
        """
        Stokes (creeping) flow, nondimensional: :math:`\\nabla^* p^* - \\mathrm{Re}^{-1}\\nabla^{*2}\\mathbf{u}^* = 0`.

        :param p_grad: :math:`\\nabla^* p^*` (see :meth:`momentum_navier_stokes`). Shape (..., d).
        :param u_laplacian: :math:`\\nabla^{*2}\\mathbf{u}^*`. Shape (..., d).
        :param re: Reynolds number from :func:`moju.piratio.Groups.re`.
        :return: Vector residual. Shape (..., d).

        Use case: Very slow flows where inertia is negligible (Re << 1).
        """
        return p_grad - (1.0 / re) * u_laplacian

    @staticmethod
    @jax.jit
    def euler_momentum(u_t, u, u_grad, p_grad, eu):
        """
        Euler (inviscid) momentum, nondimensional. The ``eu`` argument is kept for API symmetry
        with pressure scaling workflows; the residual is
        :math:`\\partial\\mathbf{u}^*/\\partial t^* + \\mathbf{u}^*\\cdot\\nabla^*\\mathbf{u}^* + \\nabla^* p^*`.

        :param u_t: :math:`\\partial\\mathbf{u}^*/\\partial t^*`. Shape (..., d).
        :param u: :math:`\\mathbf{u}^*`. Shape (..., d).
        :param u_grad: :math:`\\nabla^*\\mathbf{u}^*`. Shape (..., d, d).
        :param p_grad: :math:`\\nabla^* p^*`. Shape (..., d).
        :param eu: Euler number from :func:`moju.piratio.Groups.eu` (for auditing / scaling checks).
        :return: Vector residual. Shape (..., d).

        Use case: High-speed aerodynamics where viscosity is ignored.
        """
        return u_t + jnp.einsum("...ij,...j->...i", u_grad, u) + p_grad

    # --- HEAT & MASS TRANSPORT ---

    @staticmethod
    @jax.jit
    def fourier_conduction(T_t, T_laplacian, fo, t, L):
        """
        Heat equation in Fourier-number form: :math:`T_t - (\\alpha/L^2)\\,L^2\\nabla^2 T^* = 0`
        with :math:`\\alpha = \\mathrm{Fo}\\,L^2/t` from :func:`moju.piratio.Groups.fo`.

        ``T_t`` and ``T_laplacian`` are the **nondimensional** time derivative and Laplacian in the
        chosen temperature and space-time scaling (commonly :math:`T^* = T/\\Delta T`,
        :math:`\\nabla^{*2}` w.r.t. :math:`x^* = x/L`, and Fourier time :math:`t^* = \\alpha t/L^2` so
        :math:`\\partial T^*/\\partial t^* = \\nabla^{*2} T^*`).

        :param T_t: Nondimensional temperature rate term paired with ``fo``, ``t``, ``L``.
        :param T_laplacian: Nondimensional spatial Laplacian term (same convention as ``T_t``).
        :param fo: Fourier number from :func:`moju.piratio.Groups.fo`.
        :param t: Elapsed time [s] (enters :math:`\\alpha = \\mathrm{Fo}\\,L^2/t`).
        :param L: Characteristic length [m].
        :return: Residual; same broadcast shape as ``T_t`` / ``T_laplacian`` / ``alpha``.

        When ``t`` is ``(n_t,)`` but fields are ``(n_t, n_x, ...)``, ``alpha`` broadcasts over
        trailing spatial axes.

        Use case: Pure heat conduction in solids or static fluids.
        """
        alpha = fo * (L**2) / t
        tt = jnp.asarray(T_t)
        lap = jnp.asarray(T_laplacian)
        a = jnp.asarray(alpha)
        target_ndim = max(tt.ndim, lap.ndim, a.ndim)
        if tt.ndim < target_ndim:
            tt = jnp.reshape(tt, tt.shape + (1,) * (target_ndim - tt.ndim))
        if lap.ndim < target_ndim:
            lap = jnp.reshape(lap, lap.shape + (1,) * (target_ndim - lap.ndim))
        if a.ndim < target_ndim:
            a = jnp.reshape(a, a.shape + (1,) * (target_ndim - a.ndim))
        tt, a, lap = jnp.broadcast_arrays(tt, a, lap)
        return tt - a * lap

    @staticmethod
    @jax.jit
    def advection_diffusion(phi_t, u, phi_grad, phi_laplacian, pe):
        """
        Scalar advection–diffusion, nondimensional:
        :math:`\\partial\\phi^*/\\partial t^* + \\mathbf{u}^*\\cdot\\nabla^*\\phi^* - \\mathrm{Pe}^{-1}\\nabla^{*2}\\phi^* = 0`.

        :param phi_t: :math:`\\partial\\phi^*/\\partial t^*`.
        :param u: :math:`\\mathbf{u}^*`. Shape (..., d).
        :param phi_grad: :math:`\\nabla^*\\phi^*`. Shape (..., d).
        :param phi_laplacian: :math:`\\nabla^{*2}\\phi^*`.
        :param pe: Péclet number from :func:`moju.piratio.Groups.pe` or :func:`moju.piratio.Groups.pe_mass`.
        :return: Scalar residual.

        Use case: Temperature or chemical concentration in a moving flow.
        """
        return phi_t + jnp.sum(phi_grad * u, axis=-1) - (1.0 / pe) * phi_laplacian

    @staticmethod
    @jax.jit
    def viscous_dissipation(u_grad, re, ec, U, L):
        """
        Nondimensional viscous dissipation source: :math:`(\\mathrm{Ec}/\\mathrm{Re})\\,2\\|\\mathbf{S}^*\\|^2`
        with :math:`\\mathbf{S}^* = \\frac{L}{U}\\mathbf{S}` and :math:`\\mathbf{S}` the strain rate from ``u_grad``.

        :param u_grad: :math:`\\nabla^*\\mathbf{u}^*` (or equivalent scaled Jacobian). Shape (..., d, d).
        :param re: Reynolds number from :func:`moju.piratio.Groups.re`.
        :param ec: Eckert number from :func:`moju.piratio.Groups.ec`.
        :param U: Characteristic velocity [m/s] (used in :math:`L/U` strain scaling).
        :param L: Characteristic length [m].
        :return: Nondimensional scalar source (dimensional energy equation scaling applies separately).

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
        Nondimensional Hooke residual: :math:`\\boldsymbol{\\sigma}^* - \\mathbf{C}^*:\\boldsymbol{\\varepsilon}^* = 0`.

        :param stress: Nondimensional stress :math:`\\boldsymbol{\\sigma}^* = \\boldsymbol{\\sigma}/E_\\mathrm{ref}`. Shape (..., d).
        :param strain: Strain :math:`\\boldsymbol{\\varepsilon}^*` (already dimensionless). Shape (..., d).
        :param stiffness_tensor: Nondimensional stiffness :math:`\\mathbf{C}^* = \\mathbf{C}/E_\\mathrm{ref}`. Shape (..., d, d).
        :return: Tensor residual. Shape (..., d).

        Use case: Structural AI predicting deformation of metals or polymers.
        """
        return stress - jnp.einsum("...ij,...j->...i", stiffness_tensor, strain)

    @staticmethod
    @jax.jit
    def darcy_flow(u, p_grad, da, L, mu):
        """
        Darcy flow in the same nondimensional **shape** as the dimensional combination
        :math:`\\mathbf{u} + (\\mathrm{Da}\\,L^2/\\mu)\\nabla p = 0`: pass **nondimensional** velocity
        :math:`\\mathbf{u}^*` and form the pressure term so the sum is the ND residual (e.g. absorb
        :math:`\\mathrm{Da}\\,L^2/(\\mu U_\\mathrm{ref})` into an effective :math:`\\nabla^* p^*`).

        :param u: Nondimensional superficial velocity :math:`\\mathbf{u}^*`. Shape (..., d).
        :param p_grad: Pressure-gradient term matching the Darcy–scaled ND convention. Shape (..., d).
        :param da: Darcy number from :func:`moju.piratio.Groups.da`.
        :param L: Characteristic length [m].
        :param mu: Dynamic viscosity [Pa*s].
        :return: Vector residual. Shape (..., d).

        Use case: Ground water flow or oil reservoir modeling.
        """
        return u + (da * (L**2) / mu) * p_grad

    @staticmethod
    @jax.jit
    def brinkman_extension(u, u_laplacian, p_grad, re, da, mu, L):
        """
        Brinkman momentum balance in a **single consistent nondimensional frame** (same formula as
        before): callers must supply :math:`\\mathbf{u}^*`, :math:`\\nabla^{*2}\\mathbf{u}^*`,
        :math:`\\nabla^* p^*`, and material references so ``re``, ``da``, ``mu``, ``L`` combine to a
        ND residual.

        :param u: Nondimensional velocity. Shape (..., d).
        :param u_laplacian: Nondimensional vector Laplacian. Shape (..., d).
        :param p_grad: Nondimensional pressure gradient. Shape (..., d).
        :param re: Reynolds number from :func:`moju.piratio.Groups.re`.
        :param da: Darcy number from :func:`moju.piratio.Groups.da`.
        :param mu: Viscosity [Pa*s] (enters coefficients with ``L``).
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
        Poisson equation in nondimensional form: :math:`\\nabla^{*2}\\phi^* + s^*/\\varepsilon^* = 0`,
        implemented as ``phi_laplacian + source/epsilon`` with **nondimensional** Laplacian and ratio.

        For a dimensional equation :math:`\\nabla^2\\phi + s/\\varepsilon = 0`, a common grouping is
        :func:`moju.piratio.Groups.poisson_rhs_pi`.

        :param phi_laplacian: :math:`\\nabla^{*2}\\phi^*`.
        :param source: Source term in the chosen ND convention.
        :param epsilon: Permittivity or field constant (same convention as ``source``).
        :return: Scalar residual.

        Use case: Electrostatics, gravity, or pressure-Poisson in CFD.
        """
        return phi_laplacian + (source / epsilon)

    @staticmethod
    @jax.jit
    def faraday_law(E_curl, B_t):
        """
        Faraday's law with **nondimensional** :math:`\\nabla^*\\times\\mathbf{E}^*` and
        :math:`\\partial\\mathbf{B}^*/\\partial t^*` chosen so :math:`\\nabla^*\\times\\mathbf{E}^* + \\partial\\mathbf{B}^*/\\partial t^* = 0`
        at zero residual (consistent EM scaling of :math:`\\mathbf{E}`, :math:`\\mathbf{B}`, :math:`x^*`, :math:`t^*`).

        :param E_curl: Nondimensional curl of the electric field. Shape (..., 3).
        :param B_t: Nondimensional :math:`\\partial\\mathbf{B}^*/\\partial t^*`. Shape (..., 3).
        :return: Vector residual. Shape (..., 3).

        Use case: Electromagnetic induction simulations.
        """
        return E_curl + B_t

    # --- VIBRATIONS & WAVES ---

    @staticmethod
    @jax.jit
    def wave_equation(phi_tt, phi_laplacian, st_wave, omega, L):
        """
        Wave equation in nondimensional form:
        :math:`\\partial^2\\phi^*/\\partial {t^*}^2 - c^{*2}\\nabla^{*2}\\phi^* = 0` with
        :math:`c^* = \\omega L / \\mathrm{St}_\\mathrm{wave}` from :func:`moju.piratio.Groups.st_wave`.

        :param phi_tt: :math:`\\partial^2\\phi^*/\\partial {t^*}^2`.
        :param phi_laplacian: :math:`\\nabla^{*2}\\phi^*`.
        :param st_wave: Wave Strouhal from :func:`moju.piratio.Groups.st_wave`.
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
        Helmholtz equation, nondimensional:
        :math:`\\nabla^{*2}\\phi^* + (kL/L)^2\\phi^* = 0` with :math:`kL = kL` from
        :func:`moju.piratio.Groups.wavenumber` (:math:`k` [1/m], :math:`L` [m]).

        :param phi: :math:`\\phi^*`.
        :param phi_laplacian: :math:`\\nabla^{*2}\\phi^*`.
        :param kL: Dimensionless wavenumber :math:`kL`.
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
        Fick’s second law in the same nondimensional pattern as :meth:`fourier_conduction`:
        :math:`\\partial\\phi^*/\\partial t^* - D^*\\nabla^{*2}\\phi^* = 0` with
        :math:`D^* = \\mathrm{Fo}_\\mathrm{mass}\\,L^2/t` from :func:`moju.piratio.Groups.fo_mass`.

        :param phi_t: :math:`\\partial\\phi^*/\\partial t^*`.
        :param phi_laplacian: :math:`\\nabla^{*2}\\phi^*`.
        :param fo_mass: Mass Fourier number from :func:`moju.piratio.Groups.fo_mass`.
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
        Viscous Burgers in a **single nondimensional frame**:
        :math:`\\partial\\mathbf{u}^*/\\partial t^* + \\mathbf{u}^*\\cdot\\nabla^*\\mathbf{u}^* - \\nu^*\\nabla^{*2}\\mathbf{u}^* = 0`
        with :math:`\\nu^* = U L / \\mathrm{Re}` (same coefficient pattern as NS viscous term).

        :param u_t: :math:`\\partial\\mathbf{u}^*/\\partial t^*`. Shape (..., d).
        :param u: :math:`\\mathbf{u}^*`. Shape (..., d).
        :param u_grad: :math:`\\nabla^*\\mathbf{u}^*`. Shape (..., d, d).
        :param u_laplacian: :math:`\\nabla^{*2}\\mathbf{u}^*`. Shape (..., d).
        :param re: Reynolds number from :func:`moju.piratio.Groups.re`.
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
        Laplace equation, nondimensional: :math:`\\nabla^{*2}\\phi^* = 0`.

        :param phi_laplacian: :math:`\\nabla^{*2}\\phi^*`.
        :return: Scalar residual.

        Use case: Steady-state potential flow or steady heat conduction without sources.
        """
        return phi_laplacian

    @staticmethod
    @jax.jit
    def schrodinger_steady(psi_laplacian, V, E, psi, sch_kin_l2):
        """
        Nondimensional steady Schrödinger (scaled TISE):
        :math:`-\\psi_\\mathrm{lap} + K\\,(V-E)\\psi = 0` with
        :math:`\\psi_\\mathrm{lap} = L^2\\nabla^2\\psi` and
        :math:`K = 2mL^2/\\hbar^2` from :func:`moju.piratio.Groups.schrodinger_kinetic_length_squared`.

        Equivalent to the dimensional form :math:`-(\\hbar^2/2m)\\nabla^2\\psi + (V-E)\\psi = 0`
        after multiplying by :math:`2mL^2/\\hbar^2`.

        :param psi_laplacian: :math:`L^2\\nabla^2\\psi` in your chosen :math:`\\psi` scaling.
        :param V: Potential energy [J] (or energy-consistent ND equivalent if you scale :math:`V,E,\\psi` together).
        :param E: Total energy [J] (same convention as ``V``).
        :param psi: Wavefunction amplitude (same scaling as used for ``psi_laplacian``).
        :param sch_kin_l2: :math:`K = 2mL^2/\\hbar^2` from :func:`moju.piratio.Groups.schrodinger_kinetic_length_squared`.
        :return: Scalar residual.

        Use case: AI modeling of quantum states or electron density.
        """
        return -psi_laplacian + sch_kin_l2 * (V - E) * psi

    @staticmethod
    @jax.jit
    def laplace_beltrami(phi_laplacian_g):
        """
        Laplace–Beltrami, nondimensional: residual is the supplied surface Laplacian
        :math:`\\Delta_g \\phi^*` (caller defines metric/scaling).

        :param phi_laplacian_g: :math:`\\Delta_g \\phi^*` on the manifold.
        :return: Scalar residual.

        Use case: Diffusion or geometric deep learning on curved surfaces/manifolds.
        """
        return phi_laplacian_g
