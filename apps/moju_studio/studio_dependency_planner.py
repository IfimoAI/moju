"""
Studio-only dependency planning: required state keys and law-FD prerequisites.

Does not change ``ResidualEngine`` behavior. Used for preflight UI and checklists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from moju.monitor.derived_state_chain import all_ref_keys_from_chain, keys_produced_by_chain
from moju.monitor.law_fd_recipes import (
    LAW_FD_RECIPES,
    LawFDArgRecipe,
    _resolve_source_state_key,
)
from moju.monitor.law_implied_diagnostics import effective_audit_specs_for_fragment
from moju.monitor.law_implied_diagnostics import supported_auto_implied_laws_for
from moju.monitor.path_b_derivatives import PathBGridConfig


# Alias key (as in NPZ) -> canonical key expected by Moju identity state_map / laws.
BUILTIN_ALIASES_TO_CANONICAL: Dict[str, str] = {
    "temperature": "T",
    "temp": "T",
    "theta": "T",
    "Theta": "T",
    "T_field": "T",
    "pressure": "p",
    "Pressure": "p",
    "density": "rho",
    "rho_field": "rho",
    "velocity": "u",
    "vel": "u",
    "X": "x",
    "coord_x": "x",
    "Y": "y",
    "coord_y": "y",
    "Z": "z",
    "coord_z": "z",
    "time": "t",
    "coords_t": "t",
}


def _law_name_from_spec(spec: Dict[str, Any]) -> Optional[str]:
    n = spec.get("name")
    if n:
        return str(n)
    fn = spec.get("fn")
    if fn is not None:
        return getattr(fn, "__name__", None)
    return None


def collect_group_output_keys_from_fragment(d: Dict[str, Any]) -> Set[str]:
    """State keys written by ``groups`` specs (computed in ``_build_state`` before laws)."""
    out: Set[str] = set()
    for spec in d.get("groups") or []:
        ok = spec.get("output_key")
        if ok is not None:
            out.add(str(ok))
    return out


def collect_required_state_keys_from_fragment(d: Dict[str, Any]) -> Set[str]:
    """Mirror ``ResidualEngine.required_state_keys()`` union logic for a config fragment."""
    keys: Set[str] = set()
    for spec in d.get("laws") or []:
        keys |= set((spec.get("state_map") or {}).values())
    for spec in d.get("groups") or []:
        keys |= set((spec.get("state_map") or {}).values())
        ok = spec.get("output_key")
        if ok:
            keys.add(str(ok))
    ca_eff, sa_eff = effective_audit_specs_for_fragment(d)
    for spec in ca_eff + sa_eff:
        keys |= set((spec.get("state_map") or {}).values())
        ok = spec.get("output_key")
        if ok:
            keys.add(str(ok))
        ivk = spec.get("implied_value_key")
        if ivk:
            keys.add(str(ivk))
    chain = list(d.get("derived_state_chain") or [])
    keys |= all_ref_keys_from_chain(chain)
    keys -= keys_produced_by_chain(chain)
    return keys


def collect_audit_derivative_keys_from_fragment(d: Dict[str, Any]) -> Set[str]:
    """Chain-rule audit derivatives were removed; always empty."""
    _ = d
    return set()


def expand_keys_with_aliases(
    pred_keys: Set[str],
    constant_keys: Set[str],
) -> Tuple[Set[str], List[str]]:
    """
    Return (effective_keys, warnings). ``effective_keys`` includes raw names plus
    canonical keys implied by alias hits (e.g. ``temperature`` -> also treat ``T`` as present).
    """
    raw = set(pred_keys) | set(constant_keys)
    effective = set(raw)
    warnings: List[str] = []
    for k in list(raw):
        canon = BUILTIN_ALIASES_TO_CANONICAL.get(k)
        if canon is not None:
            if canon not in raw:
                warnings.append(
                    f"Alias: NPZ/constants key {k!r} satisfies expected canonical key {canon!r} "
                    f"(prefer renaming to {canon!r} for identity state_map clarity)."
                )
            effective.add(canon)
    return effective, warnings


def _parse_deriv_field(deriv_key: str) -> Optional[str]:
    """``d_T_dx`` -> ``T``."""
    if not deriv_key.startswith("d_"):
        return None
    rest = deriv_key[2:]
    for suf in ("_dx", "_dy", "_dz", "_dt"):
        if rest.endswith(suf):
            return rest[: -len(suf)]
    return None


def coord_keys_for_law_fd_recipe(
    recipe: LawFDArgRecipe,
    grid: PathBGridConfig,
) -> Set[str]:
    """Coordinate *state* keys (respecting ``PathBGridConfig`` key_* names)."""
    out: Set[str] = set()
    kind = recipe.kind
    if kind in ("dt", "dtt"):
        out.add(grid.key_t)
    if kind in ("laplacian", "vector_laplacian", "grad_scalar", "jacobian"):
        out.add(grid.key_x)
        sd = grid.spatial_dimension
        if sd == 1:
            pass
        elif sd == 2:
            out.add(grid.key_y)
        elif sd == 3:
            out.add(grid.key_y)
            out.add(grid.key_z)
        else:
            # auto / unknown at plan time — conservative
            out.add(grid.key_y)
            out.add(grid.key_z)
    return out


def chain_axes_from_audits(d: Dict[str, Any]) -> Set[str]:
    _ = d
    return set()


def coord_keys_for_audit_derivative(
    deriv_key: str,
    grid: PathBGridConfig,
    chain_axes: Set[str],
) -> Set[str]:
    """Minimal coords implied by a ``d_*`` key and configured chain axes."""
    field_name = _parse_deriv_field(deriv_key)
    if field_name is None:
        return set()
    out: Set[str] = set()
    if deriv_key.endswith("_dt"):
        out.add(grid.key_t)
        return out
    suffix_to_coord = {"_dx": "x", "_dy": "y", "_dz": "z"}
    for suf, ax in suffix_to_coord.items():
        if deriv_key.endswith(suf) and ax in chain_axes:
            if ax == "x":
                out.add(grid.key_x)
            elif ax == "y":
                out.add(grid.key_y)
            elif ax == "z":
                out.add(grid.key_z)
    return out


@dataclass
class LawFDRequirement:
    law_name: str
    arg_name: str
    target_state_key: str
    primitive_key: Optional[str]
    recipe_kind: str
    coord_keys: Set[str]


@dataclass
class DependencyPlan:
    """Structured output for Studio preflight / Config preview."""

    required_state_keys: Set[str] = field(default_factory=set)
    required_audit_derivative_keys: Set[str] = field(default_factory=set)
    effective_available_keys: Set[str] = field(default_factory=set)
    alias_warnings: List[str] = field(default_factory=list)
    law_fd_requirements: List[LawFDRequirement] = field(default_factory=list)
    missing_state_direct: List[str] = field(default_factory=list)
    missing_audit_derivatives: List[str] = field(default_factory=list)
    derivable_law_fd_if_enabled: List[str] = field(default_factory=list)
    law_fd_blocked: List[str] = field(default_factory=list)
    derivable_audit_if_auto_fd: List[str] = field(default_factory=list)
    audit_fd_blocked: List[str] = field(default_factory=list)
    unresolved_state_keys: List[str] = field(default_factory=list)
    implied_manual_laws: List[str] = field(default_factory=list)

    def has_blocking_gaps(self) -> bool:
        """True if the user likely needs to change NPZ, constants, or Run FD options."""
        return bool(
            self.missing_state_direct
            or self.missing_audit_derivatives
            or self.law_fd_blocked
            or self.audit_fd_blocked
        )

    def to_markdown(self) -> str:
        lines: List[str] = ["### Dependency plan (Studio)", ""]
        if self.alias_warnings:
            lines.append("**Aliases**")
            for w in self.alias_warnings:
                lines.append(f"- {w}")
            lines.append("")
        if self.missing_state_direct:
            lines.append("**Missing state keys** (not in NPZ/constants and not law-FD-fillable with current options)")
            for k in self.missing_state_direct:
                lines.append(f"- `{k}`")
            lines.append("")
            if "t" in self.missing_state_direct:
                lines.append(
                    "**Note (`t`):** For **`Groups.fo`** / Fourier conduction, **`t`** is usually the **mesh time coordinate** "
                    "in **`state_pred`**, aligned with **Path B — FD grid** **`key_t`** (default `t`). "
                    "Prefer uploading **`t`** (or alias **`time`** / **`coords_t`**) rather than **Constants JSON**, "
                    "unless you intentionally use a scalar broadcast."
                )
                lines.append("")
        if self.derivable_law_fd_if_enabled:
            lines.append(
                "**Law inputs derivable** if **Compute state derivatives (finite difference)** + "
                "**Compute law derivatives (finite difference)** are on (add primitives + coords if listed). "
                "*(API: `auto_path_b_derivatives`, `fill_law_fd`.)*"
            )
            for k in self.derivable_law_fd_if_enabled:
                lines.append(f"- `{k}`")
            lines.append("")
        if self.law_fd_blocked:
            lines.append("**Law FD blocked** (enable FD options or add keys)")
            for k in self.law_fd_blocked:
                lines.append(f"- {k}")
            lines.append("")
        if self.missing_audit_derivatives:
            lines.append("**Missing audit derivative keys** (`d_*`)")
            for k in self.missing_audit_derivatives:
                lines.append(f"- `{k}`")
            lines.append("")
        if self.derivable_audit_if_auto_fd:
            lines.append(
                "**Audit derivatives computable** if **Compute state derivatives (finite difference)** is on "
                "(need base field + coords). *(API: `auto_path_b_derivatives`.)*"
            )
            for k in self.derivable_audit_if_auto_fd:
                lines.append(f"- `{k}`")
            lines.append("")
        if self.audit_fd_blocked:
            lines.append("**Audit FD blocked**")
            for k in self.audit_fd_blocked:
                lines.append(f"- {k}")
            lines.append("")
        if self.unresolved_state_keys:
            lines.append("**Still required after best-effort FD**")
            for k in self.unresolved_state_keys:
                lines.append(f"- `{k}`")
            lines.append("")
        if self.implied_manual_laws:
            lines.append(
                "**Manual constitutive implied specs recommended** (no auto law-linked implied mapping)"
            )
            for n in self.implied_manual_laws:
                lines.append(f"- `{n}`")
            lines.append(
                "Add explicit `constitutive_audit` rows if you want implied constitutive checks for these laws."
            )
            lines.append("")
        if not any(
            [
                self.missing_state_direct,
                self.law_fd_blocked,
                self.missing_audit_derivatives,
                self.audit_fd_blocked,
                self.unresolved_state_keys,
            ]
        ):
            lines.append("No blocking gaps detected for the current fragment and NPZ/constants keys.")
        return "\n".join(lines)


def plan_dependencies(
    frag_d: Dict[str, Any],
    *,
    pred_keys: Set[str],
    constant_keys: Optional[Set[str]] = None,
    auto_path_b_derivatives: bool = False,
    fill_law_fd: bool = False,
    path_b_grid: Optional[PathBGridConfig] = None,
) -> DependencyPlan:
    """
    Pure planner from a merged config fragment (laws/groups/audits) and uploaded key sets.

    ``constant_keys`` should be the set of names present in ``frag_d['constants']`` or the
    constants dict keys the user configured (not values).
    """
    ckeys = set(constant_keys) if constant_keys is not None else set()
    grid = path_b_grid or PathBGridConfig()
    plan = DependencyPlan()
    plan.required_state_keys = collect_required_state_keys_from_fragment(frag_d)
    plan.required_audit_derivative_keys = collect_audit_derivative_keys_from_fragment(frag_d)
    plan.effective_available_keys, plan.alias_warnings = expand_keys_with_aliases(
        set(pred_keys), ckeys
    )
    chain_axes = chain_axes_from_audits(frag_d)
    _li_supported, _li_manual = supported_auto_implied_laws_for(
        frag_d.get("laws") or []
    )
    plan.implied_manual_laws = sorted(_li_manual)

    eff = plan.effective_available_keys
    group_outputs = collect_group_output_keys_from_fragment(frag_d)

    # --- Law FD requirements ---
    law_fd_targets: Dict[str, LawFDRequirement] = {}
    for spec in frag_d.get("laws") or []:
        law_name = _law_name_from_spec(spec)
        if not law_name:
            continue
        sm = spec.get("state_map") or {}
        if not isinstance(sm, dict):
            continue
        recipes = LAW_FD_RECIPES.get(law_name) or {}
        for arg_name, target_sk in sm.items():
            arg_s = str(arg_name)
            tgt = str(target_sk)
            recipe = recipes.get(arg_s)
            if recipe is None:
                continue
            src = _resolve_source_state_key(recipe, arg_s, tgt, sm)
            c_need = coord_keys_for_law_fd_recipe(recipe, grid)
            law_fd_targets[tgt] = LawFDRequirement(
                law_name=law_name,
                arg_name=arg_s,
                target_state_key=tgt,
                primitive_key=src,
                recipe_kind=recipe.kind,
                coord_keys=c_need,
            )
    plan.law_fd_requirements = sorted(law_fd_targets.values(), key=lambda r: (r.law_name, r.target_state_key))

    # Partition required state keys
    missing_direct: List[str] = []
    derivable_law: List[str] = []
    blocked_law: List[str] = []
    for k in sorted(plan.required_state_keys):
        if k in eff:
            continue
        if k in group_outputs:
            # Satisfied by engine ``groups`` before laws (e.g. ``fo`` from ``Groups.fo``).
            continue
        req = law_fd_targets.get(k)
        if req is None:
            missing_direct.append(k)
            continue
        prim_ok = req.primitive_key is not None and req.primitive_key in eff
        coords_ok = all(c in eff for c in req.coord_keys)
        if auto_path_b_derivatives and fill_law_fd and prim_ok and coords_ok:
            derivable_law.append(k)
        elif auto_path_b_derivatives and fill_law_fd:
            parts = []
            if not prim_ok and req.primitive_key:
                parts.append(f"primitive `{req.primitive_key}`")
            if not coords_ok:
                miss_c = sorted(c for c in req.coord_keys if c not in eff)
                parts.append(f"coords {miss_c}")
            blocked_law.append(f"`{k}` ({req.law_name}): need " + ", ".join(parts))
        else:
            blocked_law.append(
                f"`{k}` ({req.law_name}): enable **Compute state derivatives (finite difference)** + "
                f"**Compute law derivatives (finite difference)** (API: `auto_path_b_derivatives` + `fill_law_fd`), "
                f"or provide `{k}` in NPZ/constants"
            )

    plan.missing_state_direct = missing_direct
    plan.derivable_law_fd_if_enabled = derivable_law
    plan.law_fd_blocked = blocked_law

    # --- Audit derivatives ---
    miss_d: List[str] = []
    deriv_audit: List[str] = []
    blocked_audit: List[str] = []
    for dk in sorted(plan.required_audit_derivative_keys):
        if dk in eff:
            continue
        base = _parse_deriv_field(dk)
        c_need = coord_keys_for_audit_derivative(dk, grid, chain_axes)
        base_ok = base is not None and base in eff
        coords_ok = all(c in eff for c in c_need)
        if auto_path_b_derivatives and base_ok and coords_ok:
            deriv_audit.append(dk)
        elif auto_path_b_derivatives:
            parts = []
            if not base_ok and base:
                parts.append(f"field `{base}`")
            if not coords_ok:
                miss_c = sorted(c for c in c_need if c not in eff)
                if miss_c:
                    parts.append(f"coords {miss_c}")
            blocked_audit.append(f"`{dk}`: " + ", ".join(parts) if parts else f"`{dk}`")
        else:
            miss_d.append(dk)

    plan.missing_audit_derivatives = miss_d
    plan.derivable_audit_if_auto_fd = deriv_audit
    plan.audit_fd_blocked = blocked_audit

    # Unresolved: keys still required for laws/groups after FD (conservative)
    unresolved = set(missing_direct)
    plan.unresolved_state_keys = sorted(unresolved)

    return plan


def format_planner_preflight_warning(plan: DependencyPlan) -> str:
    """
    Short summary for Studio ``st.warning`` when :meth:`DependencyPlan.has_blocking_gaps` is true.

    Uses planner partitions (not raw NPZ key lists) so law-FD-derivable keys are not listed as
    missing when primitives and coords are available.
    """
    parts: List[str] = []
    if plan.missing_state_direct:
        parts.append(
            "**Unresolved state** (add to NPZ or Constants): "
            + ", ".join(f"`{k}`" for k in plan.missing_state_direct)
        )
    if plan.law_fd_blocked:
        parts.append("**Law-FD blocked:** " + "; ".join(plan.law_fd_blocked))
    if plan.missing_audit_derivatives:
        parts.append(
            "**Audit derivative keys missing** (NPZ/Constants): "
            + ", ".join(f"`{k}`" for k in plan.missing_audit_derivatives)
        )
    if plan.audit_fd_blocked:
        parts.append("**Audit FD blocked:** " + "; ".join(plan.audit_fd_blocked))
    if not parts:
        return "See **Dependency planner** expander for detail."
    return "\n\n".join(parts) + "\n\n_See **Dependency planner** for full detail._"


def plan_markdown_for_display(
    frag_d: Dict[str, Any],
    *,
    pred_keys: Set[str],
    constant_keys: Optional[Set[str]] = None,
    auto_path_b_derivatives: bool = True,
    fill_law_fd: bool = True,
    path_b_grid: Optional[PathBGridConfig] = None,
) -> str:
    """Convenience wrapper for Streamlit ``st.markdown``."""
    p = plan_dependencies(
        frag_d,
        pred_keys=pred_keys,
        constant_keys=constant_keys,
        auto_path_b_derivatives=auto_path_b_derivatives,
        fill_law_fd=fill_law_fd,
        path_b_grid=path_b_grid,
    )
    return p.to_markdown()
