"""
Studio-only: build MonitorConfig fragments from allowlisted Laws / Models / Groups.

Uses **identity** ``state_map`` (argument name → same state key). Users should name
``state_pred`` / ``constants`` keys to match ``Models.*`` / ``Groups.*`` / ``Laws.*``
signatures (e.g. ``k_solid`` for :func:`Groups.bi`).

**Implied groups (Studio):** for allowlisted laws, dimensionless arguments that correspond
to a registered ``Groups.*`` function (e.g. ``fo`` for :func:`Groups.fo`) are prepended to
``groups`` so Moju computes them from primitives (``alpha``, ``t``, ``L``) before laws run.
**Law-linked implied audits** (``moju.monitor.law_implied_diagnostics``) prepend e.g.
``thermal_diffusivity`` vs α from ``T_t`` / ``T_laplacian`` when those laws are selected.
Users need not upload ``fo`` / ``re`` / etc. unless they override the matching ``output_key``
via expert JSON.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from apps.moju_studio.config_forms import (
    build_audit_spec_dict,
    build_group_spec,
    build_law_spec,
    group_parameter_names,
    law_parameter_names,
    model_parameter_names,
    scaling_fn_parameter_names,
)
from apps.moju_studio.studio_implied_groups import (
    implied_group_specs_for_laws,
    implied_scaling_audit_specs_for_laws,
    merge_implied_groups_first,
)
from moju.monitor.closure_registry import GROUP_FNS, MODEL_FNS
from moju.monitor.law_implied_diagnostics import merge_law_implied_audit_specs
from moju.monitor.constitutive_closures import list_constitutive_models as closure_model_names
from moju.monitor.law_fd_recipes import list_law_fd_supported_laws

# Coordinate-like keys excluded from "field" prediction lists (grid coords are separate).
_COORD_SPATIAL = frozenset({"x", "y", "z"})
_COORD_ALL = frozenset({"x", "y", "z", "t"})

# Curated Studio allowlists (subset of registries).
STUDIO_LAW_NAMES: tuple[str, ...] = tuple(list_law_fd_supported_laws())

STUDIO_MODEL_NAMES: tuple[str, ...] = tuple(sorted(closure_model_names()))

# Common dimensionless groups for scaling audits + engine ``groups`` specs.
STUDIO_GROUP_NAMES: tuple[str, ...] = (
    "re",
    "pr",
    "pe",
    "gr",
    "bi",
    "fo",
    "eu",
    "we",
    "da",
    "ma",
    "nu",
    "st",
    "sc",
    "le",
    "ec",
    "bo",
    "ca",
    "kn",
    "pe_mass",
)


def _filtered_groups() -> tuple[str, ...]:
    return tuple(n for n in STUDIO_GROUP_NAMES if n in GROUP_FNS)


STUDIO_GROUP_NAMES_EFFECTIVE: tuple[str, ...] = _filtered_groups()

# Default output_key for constitutive audits (pred field vs model).
MODEL_DEFAULT_OUTPUT_KEY: Dict[str, str] = {
    "sutherland_mu": "mu",
    "vft_mu": "mu",
    "ideal_gas_rho": "rho",
    "boussinesq_rho": "rho",
    "thermal_diffusivity": "alpha",
    "kinematic_viscosity": "nu",
    "mass_diffusivity": "D",
    "wave_speed_from_st": "c",
    "dynamic_viscosity_from_re": "mu",
    "scalar_diffusivity_from_pe": "kappa",
    "power_law_mu": "mu_pl",
    "arrhenius_rate": "k_rate",
    "stefan_boltzmann_flux": "q_rad",
    "heat_flux_conduction": "q_flux",
    "speed_of_sound": "a",
    "specific_heat_nasa": "cp",
}


def _group_default_output_key(name: str) -> str:
    """
    Default ``output_key`` for a user-selected scaling ``group`` row.

    Must match **Laws.*** dimensionless argument names (``fo``, ``re``, ``pe``, …) and
    implied-group injection so the planner does not ask for ``Fo`` when the law expects ``fo``.
    Multi-segment registry names keep Pascal-style joining except where ``specials`` override.
    """
    specials = {
        "nu": "Nu",
        "pe_mass": "Pe_m",
        "fo_mass": "Fo_m",
        "st_wave": "St_w",
        "wavenumber": "k_wave",
    }
    if name in specials:
        return specials[name]
    if "_" in name:
        parts = name.split("_")
        return "".join(p[:1].upper() + p[1:] for p in parts if p)
    return name


def build_studio_auto_fragment(
    *,
    law_names: List[str],
    model_names: List[str],
    group_names: List[str],
    pred_keys: Set[str],
    constant_keys: Set[str],
) -> Dict[str, Any]:
    """
    Build ``laws``, ``groups``, ``constitutive_audit``, ``scaling_audit``, ``primary_fields``.

    Raises ``ValueError`` if any name is not allowlisted or missing from registries.
    """
    allow_laws = set(STUDIO_LAW_NAMES)
    allow_models = set(STUDIO_MODEL_NAMES)
    allow_groups = set(STUDIO_GROUP_NAMES_EFFECTIVE)

    for n in law_names:
        if n not in allow_laws:
            raise ValueError(f"Law {n!r} is not in the Studio allowlist")
    for n in model_names:
        if n not in allow_models or n not in MODEL_FNS:
            raise ValueError(f"Model {n!r} is not in the Studio constitutive allowlist")
    for n in group_names:
        if n not in allow_groups or n not in GROUP_FNS:
            raise ValueError(f"Group {n!r} is not in the Studio scaling allowlist")

    laws: List[Dict[str, Any]] = []
    for name in law_names:
        args = law_parameter_names(name)
        sm = {a: a for a in args}
        laws.append(build_law_spec(name, sm))

    constitutive: List[Dict[str, Any]] = []
    for name in model_names:
        out_k = MODEL_DEFAULT_OUTPUT_KEY.get(name)
        if not out_k:
            raise ValueError(f"No default output_key for model {name!r} — extend MODEL_DEFAULT_OUTPUT_KEY")
        args = model_parameter_names(name)
        sm = {a: a for a in args}
        constitutive.append(
            build_audit_spec_dict(
                category="constitutive",
                name=name,
                output_key=out_k,
                state_map=sm,
            )
        )

    user_groups: List[Dict[str, Any]] = []
    scaling: List[Dict[str, Any]] = []
    for name in group_names:
        out_k = _group_default_output_key(name)
        args = group_parameter_names(name)
        sm = {a: a for a in args}
        user_groups.append(build_group_spec(name, out_k, sm))
        scaling.append(
            build_audit_spec_dict(
                category="scaling",
                name=name,
                output_key=out_k,
                state_map=sm,
                invariance_pi_constant=False,
            )
        )

    implied_sa = implied_scaling_audit_specs_for_laws(law_names, pred_keys)
    have_scaling_out = {s.get("output_key") for s in scaling}
    for row in implied_sa:
        ok = row.get("output_key")
        if ok not in have_scaling_out:
            scaling.insert(0, row)
            have_scaling_out.add(ok)

    implied = implied_group_specs_for_laws(law_names)
    groups = merge_implied_groups_first(implied, user_groups)

    li_c, li_s = merge_law_implied_audit_specs(laws, enabled=True)
    constitutive = li_c + constitutive
    scaling = li_s + scaling

    pf = sorted(k for k in pred_keys if k not in _COORD_ALL)[:24]
    if not pf:
        pf = ["T", "u", "rho", "p"]

    return {
        "laws": laws,
        "groups": groups,
        "constitutive_audit": constitutive,
        "scaling_audit": scaling,
        "law_implied_audits": True,
        "primary_fields": pf,
        "derived_state_chain": [],
    }
