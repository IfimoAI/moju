"""
Law-linked implied constitutive / scaling audits.

For selected :class:`Laws.*` entries, Moju can auto-append ``constitutive_audit`` rows whose
``implied_balance_fn`` evaluates a **governing-equation balance** using ``state_pred`` (and the
law's ``state_map``) and the catalog model prediction ``pred = F(...)``. Residuals use
``implied_delta`` with symmetric normalization of the balance ``raw`` against the magnitudes of
the two sides (see :func:`moju.monitor.closure_registry.compute_implied_delta`).

Some rows (e.g. turbulent viscous acceleration) still use ``implied_fn`` and **subtract** mode:
``pred - implied``.

We avoid pairing **both** a group and a constitutive closure that are algebraically equivalent
given fixed scales (e.g. Fourier: only ``thermal_diffusivity``, not a separate ``fo`` implied row).

See README "Law-linked implied audits" and :func:`merge_law_implied_audit_specs`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import jax.numpy as jnp

from moju.piratio.laws import Laws

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _val(
    state: Dict[str, Any],
    constants: Dict[str, Any],
    law_sm: Dict[str, str],
    arg: str,
) -> Any:
    key = law_sm.get(arg)
    if key is None:
        return None
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _state_or_const(state: Dict[str, Any], constants: Dict[str, Any], key: str) -> Any:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _vec_norm_last(x: Any) -> jnp.ndarray:
    a = jnp.asarray(x)
    if a.ndim == 0:
        return jnp.abs(a)
    return jnp.sqrt(jnp.sum(a**2, axis=-1))


def _re_from_mu(
    u: Any,
    rho: Any,
    L: Any,
    mu_pred: Any,
    *,
    eps: float = 1e-30,
) -> jnp.ndarray:
    u_char = _vec_norm_last(jnp.asarray(u))
    return (jnp.asarray(rho) * u_char * jnp.asarray(L)) / (jnp.asarray(mu_pred) + eps)


# ---------------------------------------------------------------------------
# Balance implied: (law_state_map) -> implied_balance_fn(state, constants, pred) -> (raw, a, b)
# ---------------------------------------------------------------------------


def balance_implied_fourier_conduction(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """``T_t - pred * T_laplacian`` vs :func:`Laws.fourier_conduction` with model ``alpha``."""

    def implied_balance_fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: Any,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        T_t = _val(state, constants, law_sm, "T_t")
        T_lap = _val(state, constants, law_sm, "T_laplacian")
        if T_t is None or T_lap is None:
            return None
        tt = jnp.asarray(T_t)
        lap = jnp.asarray(T_lap)
        a_pred = jnp.asarray(pred)
        diffusive = a_pred * lap
        raw = tt - diffusive
        return raw, tt, diffusive

    return implied_balance_fn


def balance_implied_fick_diffusion(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """``phi_t - pred * phi_laplacian`` vs :func:`Laws.fick_diffusion`."""

    def implied_balance_fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: Any,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        pt = _val(state, constants, law_sm, "phi_t")
        pl = _val(state, constants, law_sm, "phi_laplacian")
        if pt is None or pl is None:
            return None
        pt_a = jnp.asarray(pt)
        pl_a = jnp.asarray(pl)
        d_pred = jnp.asarray(pred)
        diffusive = d_pred * pl_a
        raw = pt_a - diffusive
        return raw, pt_a, diffusive

    return implied_balance_fn


def balance_implied_wave_equation(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """``phi_tt - pred**2 * phi_laplacian`` vs :func:`Laws.wave_equation`."""

    def implied_balance_fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: Any,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        ptt = _val(state, constants, law_sm, "phi_tt")
        pl = _val(state, constants, law_sm, "phi_laplacian")
        if ptt is None or pl is None:
            return None
        ptt_a = jnp.asarray(ptt)
        pl_a = jnp.asarray(pl)
        c = jnp.asarray(pred)
        diffusive = (c**2) * pl_a
        raw = ptt_a - diffusive
        return raw, ptt_a, diffusive

    return implied_balance_fn


def balance_implied_kappa_advection_diffusion(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """
    ``phi_t + u·∇phi - (pred/(|u|L)) phi_laplacian`` vs :func:`Laws.advection_diffusion`
    with ``pred = kappa`` and ``Pe = |u| L / kappa``.
    """

    def implied_balance_fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: Any,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        phi_t = _val(state, constants, law_sm, "phi_t")
        u = _val(state, constants, law_sm, "u")
        g = _val(state, constants, law_sm, "phi_grad")
        lap = _val(state, constants, law_sm, "phi_laplacian")
        L = _state_or_const(state, constants, "L")
        if phi_t is None or u is None or g is None or lap is None or L is None:
            return None
        u = jnp.asarray(u)
        g = jnp.asarray(g)
        if u.shape[-1] != g.shape[-1]:
            return None
        adv = jnp.sum(g * u, axis=-1)
        rhs = jnp.asarray(phi_t) + adv
        U = _vec_norm_last(u)
        denom = jnp.asarray(U) * jnp.asarray(L)
        kappa = jnp.asarray(pred)
        diffusive = (kappa / (denom + 1e-30)) * jnp.asarray(lap)
        raw = rhs - diffusive
        return raw, rhs, diffusive

    return implied_balance_fn


def balance_implied_mu_momentum_navier_stokes(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Vector balance :func:`Laws.momentum_navier_stokes` with ``re`` from model ``mu``."""

    def implied_balance_fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: Any,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        p_grad = _val(state, constants, law_sm, "p_grad")
        u_lap = _val(state, constants, law_sm, "u_laplacian")
        rho = _state_or_const(state, constants, "rho")
        L = _state_or_const(state, constants, "L")
        if u_t is None or u is None or u_grad is None or p_grad is None or u_lap is None:
            return None
        if rho is None or L is None:
            return None
        re_m = _re_from_mu(u, rho, L, pred)
        raw = Laws.momentum_navier_stokes(u_t, u, u_grad, p_grad, u_lap, re_m)
        u = jnp.asarray(u)
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), u)
        inertial = jnp.asarray(u_t) + adv + jnp.asarray(p_grad)
        viscous = (1.0 / re_m)[..., jnp.newaxis] * jnp.asarray(u_lap)
        return raw, inertial, viscous

    return implied_balance_fn


def balance_implied_mu_stokes_flow(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Vector balance :func:`Laws.stokes_flow` with ``re`` from model ``mu``."""

    def implied_balance_fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: Any,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        p_grad = _val(state, constants, law_sm, "p_grad")
        u_lap = _val(state, constants, law_sm, "u_laplacian")
        u = _val(state, constants, law_sm, "u")
        if u is None:
            u = _state_or_const(state, constants, "u")
        rho = _val(state, constants, law_sm, "rho")
        if rho is None:
            rho = _state_or_const(state, constants, "rho")
        L = _val(state, constants, law_sm, "L")
        if L is None:
            L = _state_or_const(state, constants, "L")
        if p_grad is None or u_lap is None:
            return None
        if u is None or rho is None or L is None:
            return None
        re_m = _re_from_mu(u, rho, L, pred)
        raw = Laws.stokes_flow(p_grad, u_lap, re_m)
        pg = jnp.asarray(p_grad)
        viscous = (1.0 / re_m)[..., jnp.newaxis] * jnp.asarray(u_lap)
        return raw, pg, viscous

    return implied_balance_fn


def balance_implied_mu_burgers_equation(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Vector balance :func:`Laws.burgers_equation` with ``re`` from model ``mu``."""

    def implied_balance_fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: Any,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        u_lap = _val(state, constants, law_sm, "u_laplacian")
        U = _val(state, constants, law_sm, "U")
        Llaw = _val(state, constants, law_sm, "L")
        rho = _state_or_const(state, constants, "rho")
        L = _state_or_const(state, constants, "L")
        if u_t is None or u is None or u_grad is None or u_lap is None:
            return None
        if U is None or Llaw is None or rho is None or L is None:
            return None
        re_m = _re_from_mu(u, rho, L, pred)
        raw = Laws.burgers_equation(u_t, u, u_grad, u_lap, re_m, U, Llaw)
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), u)
        inertial = jnp.asarray(u_t) + adv
        nu = (jnp.asarray(U) * jnp.asarray(Llaw)) / re_m
        viscous = nu[..., jnp.newaxis] * jnp.asarray(u_lap)
        return raw, inertial, viscous

    return implied_balance_fn


def implied_viscous_acceleration_incompressible(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Implied Newtonian viscous acceleration u_t + (u·∇)u + ∇p from the momentum balance."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        p_grad = _val(state, constants, law_sm, "p_grad")
        if u_t is None or u is None or u_grad is None or p_grad is None:
            return None
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), jnp.asarray(u))
        return jnp.asarray(u_t) + adv + jnp.asarray(p_grad)

    return implied_fn


def implied_viscous_acceleration_compressible(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Implied ρ(u_t + u·∇u) + ∇p for the simplified compressible momentum form."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        rho = _val(state, constants, law_sm, "rho")
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        p_grad = _val(state, constants, law_sm, "p_grad")
        if rho is None or u_t is None or u is None or u_grad is None or p_grad is None:
            return None
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), jnp.asarray(u))
        rho_b = jnp.asarray(rho)[..., jnp.newaxis]
        return rho_b * (jnp.asarray(u_t) + adv) + jnp.asarray(p_grad)

    return implied_fn


# ---------------------------------------------------------------------------
# Registry: law_name -> list of audit spec dict fragments (merged into engine lists)
# ---------------------------------------------------------------------------

# ``residual_basename`` becomes ``{basename}/implied_delta`` under constitutive/ or scaling/.
# ``include_ref_delta``: when False, skip F(pred)-F(ref) even if state_ref is set.

_LAW_IMPLIED_ROWS: Dict[str, List[Dict[str, Any]]] = {
    "fourier_conduction": [
        {
            "category": "constitutive",
            "name": "thermal_diffusivity",
            "output_key": "alpha",
            "state_map": {"k": "k", "rho": "rho", "cp": "cp"},
            "implied_balance_maker": balance_implied_fourier_conduction,
            "residual_basename": "thermal_diffusivity/law_fourier_conduction",
            "include_ref_delta": True,
        },
    ],
    "fick_diffusion": [
        {
            "category": "constitutive",
            "name": "mass_diffusivity",
            "output_key": "D",
            "state_map": {"fo_mass": "fo_mass", "t": "t", "L": "L"},
            "implied_balance_maker": balance_implied_fick_diffusion,
            "residual_basename": "mass_diffusivity/law_fick_diffusion",
            "include_ref_delta": True,
        },
    ],
    "wave_equation": [
        {
            "category": "constitutive",
            "name": "wave_speed_from_st",
            "output_key": "c",
            "state_map": {"omega": "omega", "L": "L", "st_wave": "st_wave"},
            "implied_balance_maker": balance_implied_wave_equation,
            "residual_basename": "wave_speed_from_st/law_wave_equation",
            "include_ref_delta": True,
        },
    ],
    "advection_diffusion": [
        {
            "category": "constitutive",
            "name": "scalar_diffusivity_from_pe",
            "output_key": "kappa",
            "state_map": {"u": "u", "L": "L", "pe": "pe"},
            "implied_balance_maker": balance_implied_kappa_advection_diffusion,
            "residual_basename": "scalar_diffusivity_from_pe/law_advection_diffusion",
            "include_ref_delta": True,
        },
    ],
    "momentum_navier_stokes": [
        {
            "category": "constitutive",
            "name": "dynamic_viscosity_from_re",
            "output_key": "mu",
            "state_map": {"rho": "rho", "u": "u", "L": "L", "re": "re"},
            "implied_balance_maker": balance_implied_mu_momentum_navier_stokes,
            "residual_basename": "dynamic_viscosity_from_re/law_momentum_navier_stokes",
            "include_ref_delta": True,
        },
    ],
    "stokes_flow": [
        {
            "category": "constitutive",
            "name": "dynamic_viscosity_from_re",
            "output_key": "mu",
            "state_map": {"rho": "rho", "u": "u", "L": "L", "re": "re"},
            "implied_balance_maker": balance_implied_mu_stokes_flow,
            "residual_basename": "dynamic_viscosity_from_re/law_stokes_flow",
            "include_ref_delta": True,
        },
    ],
    "burgers_equation": [
        {
            "category": "constitutive",
            "name": "dynamic_viscosity_from_re",
            "output_key": "mu",
            "state_map": {"rho": "rho", "u": "u", "L": "L", "re": "re"},
            "implied_balance_maker": balance_implied_mu_burgers_equation,
            "residual_basename": "dynamic_viscosity_from_re/law_burgers_equation",
            "include_ref_delta": True,
        },
    ],
    "momentum_incompressible_newtonian_laplacian": [
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_k_omega",
            "output_key": "viscous_accel",
            "state_map": {
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "k": "k",
                "omega": "omega",
                "omega0": "omega0",
            },
            "implied_maker": implied_viscous_acceleration_incompressible,
            "residual_basename": "turbulent_viscous_acceleration_k_omega/law_momentum_incompressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_k_epsilon",
            "output_key": "viscous_accel",
            "state_map": {
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "C_mu": "C_mu",
                "k": "k",
                "epsilon": "epsilon",
                "eps0": "eps0",
            },
            "implied_maker": implied_viscous_acceleration_incompressible,
            "residual_basename": "turbulent_viscous_acceleration_k_epsilon/law_momentum_incompressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_smagorinsky",
            "output_key": "viscous_accel",
            "state_map": {
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "Cs": "Cs",
                "Delta": "Delta",
                "strain_rate_magnitude": "strain_rate_magnitude",
            },
            "implied_maker": implied_viscous_acceleration_incompressible,
            "residual_basename": "turbulent_viscous_acceleration_smagorinsky/law_momentum_incompressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
    ],
    "momentum_compressible_newtonian_laplacian": [
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_compressible_k_omega",
            "output_key": "viscous_accel",
            "state_map": {
                "rho": "rho",
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "k": "k",
                "omega": "omega",
                "omega0": "omega0",
            },
            "implied_maker": implied_viscous_acceleration_compressible,
            "residual_basename": "turbulent_viscous_acceleration_compressible_k_omega/law_momentum_compressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_compressible_k_epsilon",
            "output_key": "viscous_accel",
            "state_map": {
                "rho": "rho",
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "C_mu": "C_mu",
                "k": "k",
                "epsilon": "epsilon",
                "eps0": "eps0",
            },
            "implied_maker": implied_viscous_acceleration_compressible,
            "residual_basename": "turbulent_viscous_acceleration_compressible_k_epsilon/law_momentum_compressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_compressible_smagorinsky",
            "output_key": "viscous_accel",
            "state_map": {
                "rho": "rho",
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "Cs": "Cs",
                "Delta": "Delta",
                "strain_rate_magnitude": "strain_rate_magnitude",
            },
            "implied_maker": implied_viscous_acceleration_compressible,
            "residual_basename": "turbulent_viscous_acceleration_compressible_smagorinsky/law_momentum_compressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
    ],
}


# Best-effort policy: these laws currently have no law-linked implied row because a
# stable constitutive/scaling rearrangement is not encoded in the monitor registry.
_LAW_IMPLIED_UNSUPPORTED_REASONS: Dict[str, str] = {
    "laplace_equation": "no explicit constitutive/scaling term to rearrange",
    "poisson_equation": "requires source-term closure choice not represented by one catalog model",
    "helmholtz_equation": "depends on forcing/wavenumber framing rather than one implied closure target",
    "schrodinger_steady": "complex-valued amplitude/phase closure not encoded as one implied scalar",
    "laplace_beltrami": "metric/geometry operators require chart-specific closure treatment",
    "mass_incompressible": "constraint law (divergence-free) has no direct constitutive target",
    "mass_compressible": "continuity balance has no single implied constitutive quantity",
    "euler_momentum": "inviscid law has no viscosity constitutive closure target",
    "darcy_flow": "permeability/drag closure requires porous-medium model context",
    "brinkman_extension": "mixed porous/viscous closure needs medium-specific constitutive choices",
    "viscous_dissipation": "dissipation source does not map to one implied constitutive/scaling scalar",
    "hookes_law_residual": "elastic constitutive inversion is material-model specific",
    "faraday_law": "curl-based electromagnetic closure requires domain-specific constitutive target choice",
}


def list_laws_with_implied_diagnostics() -> Tuple[str, ...]:
    """Law registry names that contribute auto implied audits."""
    return tuple(sorted(_LAW_IMPLIED_ROWS.keys()))


def law_implied_unsupported_reasons() -> Dict[str, str]:
    """
    Best-effort coverage map for laws without auto implied rows.

    Returns law-name -> short reason. This documents intentional gaps in the
    law-linked implied registry rather than leaving them implicit.
    """
    return dict(_LAW_IMPLIED_UNSUPPORTED_REASONS)


def all_law_names() -> Tuple[str, ...]:
    """All public ``Laws.*`` registry names."""
    names = [
        n
        for n in dir(Laws)
        if not n.startswith("_") and callable(getattr(Laws, n))
    ]
    return tuple(sorted(names))


def classify_laws_for_implied_diagnostics() -> Dict[str, str]:
    """
    Law coverage map for auto implied constitutive diagnostics.

    Values are one of:
    - ``"supported"``: law has entries in ``_LAW_IMPLIED_ROWS``
    - ``"user_specified_only"``: law is intentionally unsupported in
      ``_LAW_IMPLIED_UNSUPPORTED_REASONS`` (user can add manual audits)
    - ``"unclassified"``: law is in ``Laws.*`` but absent from both maps
    """
    supported: Set[str] = set(_LAW_IMPLIED_ROWS.keys())
    unsupported: Set[str] = set(_LAW_IMPLIED_UNSUPPORTED_REASONS.keys())
    out: Dict[str, str] = {}
    for n in all_law_names():
        if n in supported:
            out[n] = "supported"
        elif n in unsupported:
            out[n] = "user_specified_only"
        else:
            out[n] = "unclassified"
    return out


def list_unclassified_laws_for_implied_diagnostics() -> Tuple[str, ...]:
    """``Laws.*`` names missing from both supported and unsupported implied maps."""
    cls = classify_laws_for_implied_diagnostics()
    return tuple(sorted(n for n, c in cls.items() if c == "unclassified"))


def supported_auto_implied_laws_for(
    laws_spec: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    For selected laws, return ``(supported, user_specified_only)`` names.

    ``supported`` are those that auto-prepend implied constitutive rows.
    ``user_specified_only`` are intentionally unsupported laws where users should
    provide explicit constitutive audit specs.
    """
    selected = {
        str(law.get("name") or "")
        for law in laws_spec
        if isinstance(law, dict) and str(law.get("name") or "")
    }
    cls = classify_laws_for_implied_diagnostics()
    supported = sorted(n for n in selected if cls.get(n) == "supported")
    manual = sorted(n for n in selected if cls.get(n) == "user_specified_only")
    return supported, manual


def effective_audit_specs_for_fragment(d: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return ``(constitutive_audit, [])``: law-linked rows (when ``law_implied_audits`` is true)
    merged with the fragment constitutive list. Second element is always empty (legacy
    ``scaling_audit`` removed). Used by Studio dependency planning.
    """
    lic, _lis = merge_law_implied_audit_specs(
        d.get("laws") or [],
        enabled=bool(d.get("law_implied_audits", True)),
    )
    frag_c = list(d.get("constitutive_audit") or [])
    mc, rc = merge_fragment_law_implied_audit_specs(lic, frag_c)
    return mc + rc, []


def merge_law_implied_audit_specs(
    laws_spec: Sequence[Dict[str, Any]],
    *,
    enabled: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build extra ``constitutive_audit`` dict rows for :class:`ResidualEngine` from the selected laws.

    Rows include ``implied_fn`` or ``implied_balance_fn`` (not JSON-serializable) and
    ``residual_basename`` for unique log keys. When ``enabled`` is False, returns ``([], [])``
    (second list is always empty; kept for call-site compatibility).
    """
    if not enabled:
        return [], []
    constitutive: List[Dict[str, Any]] = []
    seen_basenames: set[str] = set()

    for law in laws_spec:
        law_name = str(law.get("name") or "")
        rows = _LAW_IMPLIED_ROWS.get(law_name)
        if not rows:
            continue
        law_sm = dict(law.get("state_map") or {})
        for row in rows:
            basename = str(row["residual_basename"])
            if basename in seen_basenames:
                continue
            seen_basenames.add(basename)
            implied_balance_maker = row.get("implied_balance_maker")
            implied_maker = row.get("implied_maker")
            if implied_balance_maker is not None and implied_maker is not None:
                raise ValueError(
                    f"law_implied row for {law_name!r} has both implied_balance_maker and implied_maker"
                )
            if implied_balance_maker is None and implied_maker is None:
                raise ValueError(
                    f"law_implied row for {law_name!r} missing implied_balance_maker and implied_maker"
                )
            d: Dict[str, Any] = {
                "name": row["name"],
                "output_key": row["output_key"],
                "state_map": dict(row["state_map"]),
                "residual_basename": basename,
                "include_ref_delta": bool(row.get("include_ref_delta", True)),
            }
            if implied_balance_maker is not None:
                d["implied_balance_fn"] = implied_balance_maker(law_sm)
            else:
                d["implied_fn"] = implied_maker(law_sm)
            if row["category"] != "constitutive":
                raise ValueError(
                    f"law_implied row for {law_name!r} has category {row['category']!r}; "
                    "only constitutive is supported."
                )
            constitutive.append(d)

    return constitutive, []


def merge_fragment_law_implied_audit_specs(
    prepended: List[Dict[str, Any]],
    from_config: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return ``(prepended, extras)``. Rows in ``from_config`` that declare the same
    ``residual_basename`` as a prepended law-linked row are dropped (legacy duplicate
    suppression); all other config rows are kept in ``extras``.
    """
    prep_bn = {str(r["residual_basename"]) for r in prepended if r.get("residual_basename")}
    extras: List[Dict[str, Any]] = []
    for d in from_config:
        bn = d.get("residual_basename")
        if bn and str(bn) in prep_bn:
            continue
        extras.append(d)
    return list(prepended), extras
