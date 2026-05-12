"""
Dimensional ↔ nondimensional state conversion for Moju's ResidualEngine.

Provides:
- ``NondimScales`` – frozen dataclass holding all reference scales and the time
  convention (convective / Fourier / mass-Fourier / wave).
- ``dimensional_to_nd`` – converts a physical-units state dict into the
  nondimensional form that ``Laws.*`` and ``ResidualEngine`` expect.
- ``nd_to_dimensional`` – exact inverse; useful for interpreting residual fields
  or converting PINN outputs back to physical units.

Scaling convention
------------------
For a field *f* with reference scale *f_ref* and mesh coordinates scaled by
*L_ref*::

    f*        = f / f_ref
    (∂f/∂x)*  = (L_ref / f_ref) · ∂f/∂x
    (∂f/∂t)*  = (t_ref / f_ref) · ∂f/∂t
    (∇²f)*    = (L_ref² / f_ref) · ∇²f

Temperature is affine: T* = (T – T0) / dT_ref.

Time reference depends on ``NondimScales.time_scale``:

+------------------+-------------------------------+-----------------------------------+
| ``time_scale``   | ``t_ref``                     | Intended laws                     |
+==================+===============================+===================================+
| ``"convective"`` | L_ref / U_ref                 | NS, Euler, Burgers, adv-diffusion |
| ``"fourier"``    | L_ref² / alpha_ref            | fourier_conduction                |
| ``"mass_fourier"``| L_ref² / D_ref               | fick_diffusion                    |
| ``"wave"``       | L_ref / c_ref                 | wave_equation                     |
+------------------+-------------------------------+-----------------------------------+

Quick start
-----------
>>> from moju.piratio import NondimScales, dimensional_to_nd
>>> scales = NondimScales(L_ref=0.1, U_ref=1.0, rho_ref=1000.0)
>>> state_nd = dimensional_to_nd(state_phys, scales)
>>> engine = ResidualEngine(laws=[{"name": "momentum_navier_stokes", ...}])
>>> residuals = engine.compute_residuals(state_nd)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Internal type aliases
# ---------------------------------------------------------------------------

# Each rule is a (forward_fn, inverse_fn) pair where
# fn(value: Any, scales: NondimScales) -> Any
_RuleFn = Callable[[Any, "NondimScales"], Any]
_Rule = Tuple[_RuleFn, _RuleFn]


# ---------------------------------------------------------------------------
# NondimScales dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NondimScales:
    """
    Reference scales for dimensional ↔ nondimensional state conversion.

    Parameters
    ----------
    L_ref:
        Characteristic length [m].
    U_ref:
        Characteristic velocity [m/s]. Default 1.0 is suitable for pure
        diffusion problems where there is no convective velocity scale.
    rho_ref:
        Reference density [kg/m³].
    dT_ref:
        Temperature scale ΔT [K]. Temperature is scaled as
        ``T* = (T - T0) / dT_ref``.
    T0:
        Temperature offset [K] for the affine temperature scaling.
    phi_ref:
        Reference scale for the generic scalar field φ (used by
        ``advection_diffusion``, ``poisson_equation``, etc.).
    E_ref:
        Elastic modulus reference [Pa]. Scales stress and stiffness tensors
        in solid-mechanics laws (``hookes_law_residual``).
    p_ref:
        Pressure reference [Pa]. If ``None`` (default), auto-computed as
        ``rho_ref * U_ref²`` (dynamic pressure).
    alpha_ref:
        Thermal diffusivity [m²/s]. **Required** when
        ``time_scale="fourier"``.
    D_ref:
        Mass diffusivity [m²/s]. **Required** when
        ``time_scale="mass_fourier"``.
    c_ref:
        Wave speed [m/s]. **Required** when ``time_scale="wave"``.
    time_scale:
        Convention for nondimensionalising time and time derivatives.
        One of ``"convective"``, ``"fourier"``, ``"mass_fourier"``,
        ``"wave"``.

    Examples
    --------
    Incompressible Navier-Stokes (convective time):

    >>> scales = NondimScales(L_ref=0.1, U_ref=1.0, rho_ref=1000.0)

    Pure heat conduction (Fourier time):

    >>> scales = NondimScales(
    ...     L_ref=0.05, dT_ref=100.0, alpha_ref=1e-7, time_scale="fourier"
    ... )
    """

    L_ref: float
    U_ref: float = 1.0
    rho_ref: float = 1.0
    dT_ref: float = 1.0
    T0: float = 0.0
    phi_ref: float = 1.0
    E_ref: float = 1.0
    p_ref: Optional[float] = None
    alpha_ref: Optional[float] = None
    D_ref: Optional[float] = None
    c_ref: Optional[float] = None
    time_scale: str = "convective"

    def __post_init__(self) -> None:
        valid = ("convective", "fourier", "mass_fourier", "wave")
        if self.time_scale not in valid:
            raise ValueError(
                f"time_scale must be one of {valid!r}; got {self.time_scale!r}"
            )
        if self.time_scale == "fourier" and self.alpha_ref is None:
            raise ValueError(
                "alpha_ref must be provided when time_scale='fourier'"
            )
        if self.time_scale == "mass_fourier" and self.D_ref is None:
            raise ValueError(
                "D_ref must be provided when time_scale='mass_fourier'"
            )
        if self.time_scale == "wave" and self.c_ref is None:
            raise ValueError(
                "c_ref must be provided when time_scale='wave'"
            )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def _p_ref(self) -> float:
        """Effective pressure reference scale [Pa]."""
        if self.p_ref is not None:
            return self.p_ref
        return self.rho_ref * self.U_ref ** 2

    @property
    def t_ref(self) -> float:
        """
        Time reference scale [s], determined by ``time_scale``:

        - ``"convective"``    → ``L_ref / U_ref``
        - ``"fourier"``       → ``L_ref² / alpha_ref``
        - ``"mass_fourier"``  → ``L_ref² / D_ref``
        - ``"wave"``          → ``L_ref / c_ref``
        """
        if self.time_scale == "convective":
            return self.L_ref / self.U_ref
        if self.time_scale == "fourier":
            return self.L_ref ** 2 / self.alpha_ref  # type: ignore[operator]
        if self.time_scale == "mass_fourier":
            return self.L_ref ** 2 / self.D_ref  # type: ignore[operator]
        # wave
        return self.L_ref / self.c_ref  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Internal rule builders
# ---------------------------------------------------------------------------

def _mul(factor_fn: Callable[["NondimScales"], float]) -> _Rule:
    """Return a (forward, inverse) rule pair for a multiplicative scaling."""

    def fwd(v: Any, s: NondimScales) -> Any:
        return jnp.asarray(v) * factor_fn(s)

    def inv(v: Any, s: NondimScales) -> Any:
        return jnp.asarray(v) / factor_fn(s)

    return fwd, inv


def _affine(fwd_fn: _RuleFn, inv_fn: _RuleFn) -> _Rule:
    """Return a (forward, inverse) rule pair for an affine transform."""
    return fwd_fn, inv_fn


# ---------------------------------------------------------------------------
# _FIELD_SCALE_RULES
# ---------------------------------------------------------------------------

_FIELD_SCALE_RULES: Dict[str, _Rule] = {
    # ------------------------------------------------------------------
    # Spatial coordinates
    # x* = x / L_ref
    # ------------------------------------------------------------------
    "x": _mul(lambda s: 1.0 / s.L_ref),
    "y": _mul(lambda s: 1.0 / s.L_ref),
    "z": _mul(lambda s: 1.0 / s.L_ref),

    # ------------------------------------------------------------------
    # Time
    # t* = t / t_ref
    # ------------------------------------------------------------------
    "t": _mul(lambda s: 1.0 / s.t_ref),

    # ------------------------------------------------------------------
    # Velocity components
    # u* = u / U_ref
    # ------------------------------------------------------------------
    "u": _mul(lambda s: 1.0 / s.U_ref),
    "v": _mul(lambda s: 1.0 / s.U_ref),
    "w": _mul(lambda s: 1.0 / s.U_ref),

    # u_t* = ∂u*/∂t* = (t_ref / U_ref) · ∂u/∂t
    "u_t": _mul(lambda s: s.t_ref / s.U_ref),

    # u_grad*_ij = ∂u*_i/∂x*_j = (L_ref / U_ref) · ∂u_i/∂x_j
    "u_grad": _mul(lambda s: s.L_ref / s.U_ref),

    # u_laplacian*_i = (L_ref² / U_ref) · ∇²u_i
    "u_laplacian": _mul(lambda s: s.L_ref ** 2 / s.U_ref),

    # ------------------------------------------------------------------
    # Pressure
    # p* = p / p_ref   (p_ref = rho_ref·U_ref² by default)
    # ------------------------------------------------------------------
    "p": _mul(lambda s: 1.0 / s._p_ref),

    # p_grad* = (L_ref / p_ref) · ∇p
    "p_grad": _mul(lambda s: s.L_ref / s._p_ref),

    # ------------------------------------------------------------------
    # Density (compressible flows)
    # rho* = rho / rho_ref
    # ------------------------------------------------------------------
    "rho": _mul(lambda s: 1.0 / s.rho_ref),

    # rho_t* = ∂ρ*/∂t* = (t_ref / rho_ref) · ∂ρ/∂t
    "rho_t": _mul(lambda s: s.t_ref / s.rho_ref),

    # rho_grad* = (L_ref / rho_ref) · ∇ρ
    "rho_grad": _mul(lambda s: s.L_ref / s.rho_ref),

    # ------------------------------------------------------------------
    # Temperature  (affine: T* = (T − T0) / dT_ref)
    # Time/space derivatives are purely multiplicative (no offset).
    # ------------------------------------------------------------------
    "T": _affine(
        fwd_fn=lambda v, s: (jnp.asarray(v) - s.T0) / s.dT_ref,
        inv_fn=lambda v, s: jnp.asarray(v) * s.dT_ref + s.T0,
    ),

    # T_t* = (t_ref / dT_ref) · ∂T/∂t
    "T_t": _mul(lambda s: s.t_ref / s.dT_ref),

    # T_grad* = (L_ref / dT_ref) · ∇T
    "T_grad": _mul(lambda s: s.L_ref / s.dT_ref),

    # T_laplacian* = (L_ref² / dT_ref) · ∇²T
    "T_laplacian": _mul(lambda s: s.L_ref ** 2 / s.dT_ref),

    # ------------------------------------------------------------------
    # Generic scalar φ  (advection-diffusion, Poisson, Fick diffusion, …)
    # phi* = phi / phi_ref
    # ------------------------------------------------------------------
    "phi": _mul(lambda s: 1.0 / s.phi_ref),

    # phi_t* = (t_ref / phi_ref) · ∂φ/∂t
    "phi_t": _mul(lambda s: s.t_ref / s.phi_ref),

    # phi_grad* = (L_ref / phi_ref) · ∇φ
    "phi_grad": _mul(lambda s: s.L_ref / s.phi_ref),

    # phi_laplacian* = (L_ref² / phi_ref) · ∇²φ
    "phi_laplacian": _mul(lambda s: s.L_ref ** 2 / s.phi_ref),

    # phi_tt* = (t_ref² / phi_ref) · ∂²φ/∂t²  (wave equation)
    "phi_tt": _mul(lambda s: s.t_ref ** 2 / s.phi_ref),

    # ------------------------------------------------------------------
    # Wavefunction (Schrödinger — steady)
    # psi is user-normalised; the law expects L²∇²ψ directly.
    # psi_laplacian = L_ref² · ∇²ψ
    # ------------------------------------------------------------------
    "psi_laplacian": _mul(lambda s: s.L_ref ** 2),

    # ------------------------------------------------------------------
    # Solid mechanics
    # stress*  = σ  / E_ref
    # C*       = C  / E_ref  (stiffness tensor)
    # strain   is dimensionless: ε = ΔL/L  → scale factor 1
    # ------------------------------------------------------------------
    "stress": _mul(lambda s: 1.0 / s.E_ref),
    "stiffness_tensor": _mul(lambda s: 1.0 / s.E_ref),
    "strain": _mul(lambda s: 1.0),

    # ------------------------------------------------------------------
    # Turbulence / kinematic viscosity
    # ν* = ν / (U_ref · L_ref)   (ν has units m²/s = m/s · m)
    # ------------------------------------------------------------------
    "nu_eff": _mul(lambda s: 1.0 / (s.U_ref * s.L_ref)),
    "nu_molecular": _mul(lambda s: 1.0 / (s.U_ref * s.L_ref)),

    # Resolved strain-rate magnitude |S| [1/s]: |S|* = (L_ref / U_ref)|S|
    "strain_rate_magnitude": _mul(lambda s: s.L_ref / s.U_ref),

    # LES filter width Δ [m]: Δ* = Δ / L_ref
    "Delta": _mul(lambda s: 1.0 / s.L_ref),
}

# ---------------------------------------------------------------------------
# _PASSTHROUGH_KEYS
# ---------------------------------------------------------------------------

_PASSTHROUGH_KEYS: frozenset = frozenset({
    # --- Dimensionless groups (outputs of Groups.*) ---
    "re", "pr", "gr", "ma", "we", "pe", "st", "bi", "fo", "sc", "le",
    "kn", "bo", "ca", "eu", "da", "ec", "fo_mass", "kL", "pe_mass",
    "st_wave", "sch_kin_l2",
    # --- Dimensional law constants (passed as-is to Laws.* / Models.* args) ---
    # These are physical constants, not field variables, so the user should
    # supply them in their natural dimensional form.
    "L",        # characteristic length [m]  — used by fourier_conduction, darcy_flow, …
    "U",        # characteristic velocity [m/s]  — used by viscous_dissipation
    "mu",       # dynamic viscosity [Pa·s]
    "k",        # thermal conductivity [W/m·K]
    "cp",       # specific heat [J/kg·K]
    "alpha",    # thermal diffusivity [m²/s]
    "D",        # mass diffusivity [m²/s]
    "c",        # wave speed [m/s]
    "omega",    # angular frequency [rad/s]
    "K",        # permeability [m²]  — Darcy / Brinkman
    "mu_eff",   # effective dynamic viscosity [Pa·s]  — Brinkman
    "P",        # pressure constant (EOS) [Pa]
    "R",        # specific gas constant [J/kg·K]
    "rho0",     # reference density for Boussinesq [kg/m³]
    "beta",     # thermal expansion coefficient [1/K]
    "h_bar",    # reduced Planck constant [J·s]  — Schrödinger
    "m",        # particle mass [kg]  — Schrödinger
    "V",        # potential energy [J]  — Schrödinger
    "E",        # total energy [J]  — Schrödinger
    # Turbulence model constants (dimensionless by construction)
    "Cs",       # Smagorinsky constant
    "C_mu",     # k-ε model constant
    # Sutherland's law constants (dimensional but not field variables)
    "mu0",
    "T0_suth",
    "S_suth",
    # Elastic material constants (used by Models.isotropic_linear_stress)
    "E_young",  # Young's modulus [Pa]
    "nu_p",     # Poisson ratio (dimensionless)
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dimensional_to_nd(
    state: Dict[str, Any],
    scales: NondimScales,
    *,
    extra_rules: Optional[Dict[str, Any]] = None,
    warn_unknown: bool = True,
) -> Dict[str, Any]:
    """
    Convert a physical-units state dictionary to nondimensional form.

    The function applies the scaling rules in ``_FIELD_SCALE_RULES`` (overridden
    by ``extra_rules``) to every key in *state*. Keys in ``_PASSTHROUGH_KEYS``
    (dimensionless groups and known dimensional law constants) are copied
    unchanged without any warning.

    Parameters
    ----------
    state:
        Dictionary mapping field names to values (numpy/JAX arrays or
        Python scalars). Not mutated.
    scales:
        Reference scales for the nondimensionalisation.
    extra_rules:
        Optional per-key overrides applied on top of ``_FIELD_SCALE_RULES``.
        Each value may be:

        - A **float**: ``value_nd = value * extra_rules[key]`` (forward),
          ``value = value_nd / extra_rules[key]`` (inverse via
          :func:`nd_to_dimensional`).
        - A **callable** ``fn(value, scales) → array``: arbitrary forward
          transform.  :func:`nd_to_dimensional` cannot invert callable rules
          and will copy those keys unchanged.
    warn_unknown:
        If ``True`` (default), emit a :class:`UserWarning` for any key that
        is not in ``_FIELD_SCALE_RULES``, ``_PASSTHROUGH_KEYS``, or
        ``extra_rules``. Pass ``False`` to suppress all such warnings.

    Returns
    -------
    Dict[str, Any]
        A new dictionary with nondimensional values.

    Examples
    --------
    >>> from moju.piratio import NondimScales, dimensional_to_nd
    >>> scales = NondimScales(L_ref=0.1, U_ref=1.0, rho_ref=1000.0)
    >>> state_nd = dimensional_to_nd(
    ...     {"x": x_m, "u": u_ms, "p_grad": dpx, "re": 1000.0},
    ...     scales,
    ... )
    """
    effective_rules = _build_effective_rules(extra_rules)

    out: Dict[str, Any] = {}
    for key, value in state.items():
        if key in _PASSTHROUGH_KEYS:
            out[key] = value
        elif key in effective_rules:
            fwd_fn, _ = effective_rules[key]
            out[key] = fwd_fn(value, scales)
        else:
            if warn_unknown:
                warnings.warn(
                    f"dimensional_to_nd: unrecognised key {key!r} — copied "
                    f"unchanged. Supply extra_rules={{'{key}': scale}} to scale "
                    f"it, or pass warn_unknown=False to suppress this warning.",
                    UserWarning,
                    stacklevel=2,
                )
            out[key] = value
    return out


def nd_to_dimensional(
    state_nd: Dict[str, Any],
    scales: NondimScales,
    *,
    extra_rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Inverse of :func:`dimensional_to_nd`: convert a nondimensional state dict
    back to physical (dimensional) units.

    Useful for interpreting residual spatial fields after ``ResidualEngine``
    evaluation, or for converting PINN outputs back to SI units.

    .. note::
       Keys mapped by a **callable** in ``extra_rules`` cannot be automatically
       inverted and are copied unchanged to the output.

    Parameters
    ----------
    state_nd:
        Nondimensional state dictionary (output of :func:`dimensional_to_nd`).
        Not mutated.
    scales:
        The same :class:`NondimScales` used to produce *state_nd*.
    extra_rules:
        Same format as in :func:`dimensional_to_nd`.  Float scalars are
        inverted automatically; callables are copied unchanged.

    Returns
    -------
    Dict[str, Any]
        A new dictionary with dimensional values.
    """
    effective_rules = _build_effective_rules(extra_rules)

    out: Dict[str, Any] = {}
    for key, value in state_nd.items():
        if key in _PASSTHROUGH_KEYS:
            out[key] = value
        elif key in effective_rules:
            _, inv_fn = effective_rules[key]
            if inv_fn is None:
                # Callable-only extra rule — cannot auto-invert
                out[key] = value
            else:
                out[key] = inv_fn(value, scales)
        else:
            out[key] = value  # unknown → pass through (no warning on inverse)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_effective_rules(
    extra_rules: Optional[Dict[str, Any]],
) -> Dict[str, _Rule]:
    """Merge ``_FIELD_SCALE_RULES`` with user-supplied ``extra_rules``."""
    effective: Dict[str, _Rule] = dict(_FIELD_SCALE_RULES)
    if not extra_rules:
        return effective

    for key, rule in extra_rules.items():
        if callable(rule):
            # Forward-only: the user supplies fn(value, scales) → array.
            # The inverse is a no-op (None signals nd_to_dimensional to skip).
            effective[key] = (rule, None)  # type: ignore[assignment]
        else:
            factor = float(rule)
            effective[key] = _mul(lambda s, f=factor: f)
    return effective
