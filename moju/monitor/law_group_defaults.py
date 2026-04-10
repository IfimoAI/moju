"""
Default mapping from governing law names to primary dimensionless groups for π-constant audits.

Used only when building an eval :class:`ResidualEngine` with law-linked π-constant defaults
enabled on :class:`MonitorConfig` (opt-in).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from moju.monitor.closure_registry import GROUP_FNS
from moju.monitor.config import AuditSpec, MonitorConfig
from moju.monitor.pi_constant_recipes import GROUP_PI_CONSTANT_RECIPES

# Law name (Laws.* catalog) -> primary Groups.* names for π-constant (non-redundant defaults).
LAW_PRIMARY_PI_GROUPS: Dict[str, List[str]] = {
    "fourier_conduction": ["fo"],
    "fick_diffusion": ["fo_mass"],
    "momentum_navier_stokes": ["re"],
    "stokes_flow": ["re"],
    "burgers_equation": ["re"],
    "advection_diffusion": ["pe"],
    "wave_equation": ["st_wave"],
}


def resolve_pi_groups_for_laws(
    laws_spec: Sequence[Dict[str, Any]],
    *,
    law_group_overrides: Dict[str, List[str]],
    extra_groups: Sequence[str],
) -> List[str]:
    """
    Union of primary π-constant groups for each law, plus extras, de-duplicated in first-seen order.

    Per-law entries in ``law_group_overrides`` **replace** the built-in primary list for that law.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for law in laws_spec:
        name = str(law.get("name") or "")
        if not name:
            continue
        if name in law_group_overrides:
            groups = list(law_group_overrides[name])
        else:
            groups = list(LAW_PRIMARY_PI_GROUPS.get(name, []))
        for g in groups:
            if g not in seen:
                seen.add(g)
                out.append(g)
    for g in extra_groups:
        g = str(g)
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def default_compare_keys_for_pi(config: MonitorConfig) -> List[str]:
    """Fallback ``invariance_compare_keys`` when ``pi_constant_default_compare_keys`` is empty."""
    return list(config.primary_fields)


def build_pi_constant_audit_spec(
    group_name: str,
    *,
    scale_c: float,
    compare_keys: Sequence[str],
) -> AuditSpec:
    """
    Build an :class:`AuditSpec` for ``invariance_pi_constant`` on a registered group with a recipe.

    ``state_map`` uses identity ``arg_name -> state key`` for each group function argument.
    """
    if group_name not in GROUP_PI_CONSTANT_RECIPES:
        raise ValueError(
            f"Group {group_name!r} has no π-constant recipe; supported: "
            f"{sorted(GROUP_PI_CONSTANT_RECIPES.keys())}"
        )
    if group_name not in GROUP_FNS:
        raise ValueError(f"Unknown group {group_name!r}")
    _, arg_names = GROUP_FNS[group_name]
    state_map = {an: an for an in arg_names}
    ck = list(compare_keys)
    if not ck:
        raise ValueError("compare_keys must be non-empty for π-constant AuditSpec")
    if group_name == "fo_mass":
        out_key = "fo_mass"
    elif "_" in group_name:
        out_key = group_name  # e.g. st_wave, pe_mass
    else:
        out_key = group_name[0].upper() + group_name[1:]  # re -> Re, fo -> Fo
    return AuditSpec(
        name=group_name,
        output_key=out_key,
        state_map=state_map,
        invariance_pi_constant=True,
        invariance_compare_keys=ck,
        invariance_scale_c=float(scale_c),
    )


def merge_scaling_audit_with_pi_law_defaults(config: MonitorConfig) -> List[AuditSpec]:
    """
    Return a new scaling_audit list: base specs plus auto π-constant specs for resolved law groups.

    Skips a group if an existing scaling spec already has ``invariance_pi_constant`` for that
    ``name``.
    """
    scaling = list(config.scaling_audit)
    existing = {s.name for s in scaling if s.invariance_pi_constant}
    if not config.pi_constant_law_defaults_enabled:
        return scaling
    groups = resolve_pi_groups_for_laws(
        config.laws,
        law_group_overrides=dict(config.pi_constant_law_group_overrides),
        extra_groups=config.pi_constant_extra_groups,
    )
    ck = (
        list(config.pi_constant_default_compare_keys)
        if config.pi_constant_default_compare_keys
        else default_compare_keys_for_pi(config)
    )
    c = float(config.pi_constant_default_c)
    for g in groups:
        if g in existing:
            continue
        if g not in GROUP_PI_CONSTANT_RECIPES:
            continue
        scaling.append(build_pi_constant_audit_spec(g, scale_c=c, compare_keys=ck))
        existing.add(g)
    return scaling


def build_residual_engine_for_pi_constant_eval(
    base: MonitorConfig,
    *,
    state_builder,
    constants: Dict[str, Any] | None = None,
    **engine_kwargs: Any,
):
    """
    Build a :class:`ResidualEngine` for eval: merged constants, optional appended π-constant specs.

    Does not modify ``base``. π-constant law defaults append only when
    ``base.pi_constant_law_defaults_enabled`` is True.
    """
    from dataclasses import replace

    from moju.monitor.auditor import ResidualEngine

    merged_constants = {**base.constants, **(constants or {})}
    new_scaling = merge_scaling_audit_with_pi_law_defaults(base)
    cfg = replace(
        base,
        constants=merged_constants,
        scaling_audit=new_scaling,
    )
    return ResidualEngine(config=cfg, state_builder=state_builder, **engine_kwargs)
