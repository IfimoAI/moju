"""
Infer :class:`~moju.piratio.nondim.NondimScales` from selected laws and dimensional Path B state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import jax.numpy as jnp

from moju.piratio.nondim import NondimScales

# Law name -> time_scale convention for NondimScales.
LAW_TIME_SCALE: Dict[str, str] = {
    "mass_incompressible": "convective",
    "mass_compressible": "convective",
    "momentum_navier_stokes": "convective",
    "momentum_incompressible_newtonian_laplacian": "convective",
    "momentum_compressible_newtonian_laplacian": "convective",
    "stokes_flow": "convective",
    "euler_momentum": "convective",
    "burgers_equation": "convective",
    "advection_diffusion": "convective",
    "viscous_dissipation": "convective",
    "darcy_flow": "convective",
    "brinkman_extension": "convective",
    "faraday_law": "convective",
    "fourier_conduction": "fourier",
    "fick_diffusion": "mass_fourier",
    "wave_equation": "wave",
    "laplace_equation": "convective",
    "laplace_beltrami": "convective",
    "poisson_equation": "convective",
    "helmholtz_equation": "convective",
    "hookes_law_residual": "convective",
    "schrodinger_steady": "convective",
}

_REQUIRED_BY_TIME_SCALE: Dict[str, Tuple[str, ...]] = {
    "convective": ("L_ref", "U_ref"),
    "fourier": ("L_ref", "dT_ref", "alpha_ref"),
    "mass_fourier": ("L_ref", "phi_ref", "D_ref"),
    "wave": ("L_ref", "phi_ref", "c_ref"),
}


def _lookup(state: Mapping[str, Any], constants: Mapping[str, Any], key: str) -> Any:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _scalar_stat(arr: Any, *, mode: str = "mean") -> Optional[float]:
    if arr is None:
        return None
    a = jnp.ravel(jnp.asarray(arr))
    if a.size == 0:
        return None
    if mode == "span":
        return float(jnp.max(a) - jnp.min(a))
    if mode == "max_abs":
        return float(jnp.max(jnp.abs(a)))
    if mode == "std":
        return float(jnp.std(a))
    return float(jnp.mean(a))


def _rms_velocity(state: Mapping[str, Any], constants: Mapping[str, Any]) -> Optional[float]:
    parts = []
    for k in ("u", "v", "w"):
        v = _lookup(state, constants, k)
        if v is not None:
            parts.append(jnp.ravel(jnp.asarray(v)))
    if not parts:
        return None
    big = jnp.concatenate(parts)
    return float(jnp.sqrt(jnp.mean(big**2)))


def _geom_L_ref(state: Mapping[str, Any]) -> Optional[float]:
    spans = []
    for k in ("x", "y", "z"):
        if k in state:
            s = _scalar_stat(state[k], mode="span")
            if s is not None and s > 0:
                spans.append(s)
    if not spans:
        return None
    return max(spans)


def resolve_time_scale_for_laws(law_names: Sequence[str]) -> str:
    """Pick a single ``time_scale``; raise on incompatible multi-law selection."""
    scales: Set[str] = set()
    unknown: list[str] = []
    for name in law_names:
        ts = LAW_TIME_SCALE.get(name)
        if ts is None:
            unknown.append(name)
        else:
            scales.add(ts)
    if len(scales) > 1:
        raise ValueError(
            "Selected laws require incompatible nondimensional time conventions "
            f"{sorted(scales)}. Set explicit nondim_scales.time_scale and compatible laws, "
            "or run separate audits."
        )
    if scales:
        return next(iter(scales))
    if unknown:
        return "convective"
    return "convective"


def infer_nondim_scales(
    law_names: Sequence[str],
    state: Mapping[str, Any],
    constants: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[NondimScales, Dict[str, str]]:
    """
    Infer reference scales for ``dimensional_to_nd``.

    Returns ``(NondimScales, provenance_dict)`` mapping field names to source labels.
    Raises :class:`ValueError` when required refs cannot be resolved.
    """
    ov = dict(overrides or {})
    prov: Dict[str, str] = {}
    time_scale = str(ov.get("time_scale") or resolve_time_scale_for_laws(law_names))
    if "time_scale" in ov:
        prov["time_scale"] = "override"

    def _pick(
        field: str,
        *,
        keys: Sequence[str] = (),
        stat_mode: str = "mean",
        derived_fn=None,
    ) -> Optional[float]:
        if field in ov and ov[field] is not None:
            prov[field] = "override"
            return float(ov[field])
        for k in keys:
            v = _lookup(state, constants, k)
            if v is not None:
                s = _scalar_stat(v, mode=stat_mode)
                if s is not None and s > 0:
                    prov[field] = f"state.{k}"
                    return s
        if derived_fn is not None:
            d = derived_fn()
            if d is not None and d > 0:
                prov.setdefault(field, "derived")
                return d
        return None

    L_ref = _pick("L_ref", keys=("L",), stat_mode="mean")
    if L_ref is None:
        L_ref = _geom_L_ref(state)
        if L_ref is not None:
            prov["L_ref"] = "geometry.span(x,y,z)"

    U_ref = _pick("U_ref", keys=("U",))
    if U_ref is None:
        U_ref = _rms_velocity(state, constants)
        if U_ref is not None and U_ref > 0:
            prov["U_ref"] = "rms(u,v,w)"

    rho_ref = _pick("rho_ref", keys=("rho",))
    dT_ref = _pick("dT_ref", keys=("dT",), stat_mode="mean")
    if dT_ref is None:
        T = _lookup(state, constants, "T")
        if T is not None:
            span = _scalar_stat(T, mode="span")
            if span is not None and span > 0:
                dT_ref = span
                prov["dT_ref"] = "span(T)"

    T0 = ov.get("T0")
    if T0 is not None:
        prov["T0"] = "override"
        T0_f = float(T0)
    else:
        T0_f = float(_scalar_stat(_lookup(state, constants, "T0"), mode="mean") or 0.0)
        if "T0" in state or "T0" in constants:
            prov["T0"] = "state.T0"

    phi_ref = _pick("phi_ref", keys=("phi",))

    def _alpha_derived() -> Optional[float]:
        k = _lookup(state, constants, "k")
        rho = _lookup(state, constants, "rho")
        cp = _lookup(state, constants, "cp")
        alpha = _lookup(state, constants, "alpha")
        if alpha is not None:
            prov["alpha_ref"] = "state.alpha"
            return _scalar_stat(alpha, mode="mean")
        if k is not None and rho is not None and cp is not None:
            prov["alpha_ref"] = "k/(rho*cp)"
            return float(
                jnp.asarray(k).mean() / (jnp.asarray(rho).mean() * jnp.asarray(cp).mean())
            )
        return None

    alpha_ref = _pick("alpha_ref", keys=("alpha_ref",), derived_fn=_alpha_derived)
    D_ref = _pick("D_ref", keys=("D",))
    c_ref = _pick("c_ref", keys=("c",))

    kwargs: Dict[str, Any] = {
        "L_ref": L_ref,
        "U_ref": U_ref if U_ref is not None else 1.0,
        "rho_ref": rho_ref if rho_ref is not None else 1.0,
        "dT_ref": dT_ref if dT_ref is not None else 1.0,
        "T0": T0_f,
        "phi_ref": phi_ref if phi_ref is not None else 1.0,
        "time_scale": time_scale,
    }
    if alpha_ref is not None:
        kwargs["alpha_ref"] = alpha_ref
    if D_ref is not None:
        kwargs["D_ref"] = D_ref
    if c_ref is not None:
        kwargs["c_ref"] = c_ref

    required = _REQUIRED_BY_TIME_SCALE.get(time_scale, ("L_ref", "U_ref"))
    missing = [f for f in required if kwargs.get(f) is None]
    if missing:
        raise ValueError(
            f"Cannot infer nondim scales for time_scale={time_scale!r}; "
            f"missing: {missing}. Provide nondim_scales overrides or add fields to state "
            f"(e.g. L, U, alpha, k/rho/cp for Fourier)."
        )

    if L_ref is None:
        raise ValueError(
            "Cannot infer L_ref: provide L in state/constants, nondim_scales.L_ref, "
            "or coordinate arrays x/y/z for domain span."
        )

    kwargs["L_ref"] = float(L_ref)
    scales = NondimScales(**kwargs)
    return scales, prov
