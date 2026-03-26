"""
Studio-only: inject ``groups`` specs so dimensionless numbers required by laws are
computed by ``Groups.*`` from primitive state/constants (identity ``state_map``).

The core engine already runs ``groups`` before laws via ``_build_state``; this module
only builds the extra MonitorConfig fragment entries when users pick laws in Studio.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Set

from apps.moju_studio.config_forms import (
    build_group_spec,
    group_parameter_names,
    law_parameter_names,
)
from moju.monitor.closure_registry import GROUP_FNS

_MESH_COORD_KEYS = frozenset({"x", "y", "z", "t"})

# Law parameter name -> Groups registry name when it differs from the argument name.
_LAW_ARG_TO_GROUP_NAME: Dict[str, str] = {
    "kL": "wavenumber",
}


def _registry_group_for_law_arg(arg: str) -> str | None:
    if arg in _LAW_ARG_TO_GROUP_NAME:
        gn = _LAW_ARG_TO_GROUP_NAME[arg]
        return gn if gn in GROUP_FNS else None
    if arg in GROUP_FNS:
        return arg
    return None


def _identity_group_state_map(group_name: str) -> Dict[str, str]:
    return {p: p for p in group_parameter_names(group_name)}


def implied_group_specs_for_laws(law_names: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Return ``groups``-style dicts (suitable for ``build_group_spec`` output) so each
    selected law's dimensionless arguments (e.g. ``fo``, ``re``) are computed before laws.

    Uses the same state key as the law argument (e.g. output ``fo`` for ``Laws.fourier_conduction``),
    with identity maps on ``Groups.*`` parameters (user supplies ``alpha``, ``t``, ``L``, not ``fo``).

    For ``pe``, also injects ``re`` and ``pr`` when any selected law needs ``pe``.
    """
    law_names = list(law_names)
    needed_law_args: Set[str] = set()
    for name in law_names:
        needed_law_args.update(law_parameter_names(name))

    by_output: Dict[str, Dict[str, Any]] = {}

    def add_spec(group_registry_name: str, output_key: str) -> None:
        if group_registry_name not in GROUP_FNS:
            return
        sm = _identity_group_state_map(group_registry_name)
        by_output[output_key] = build_group_spec(group_registry_name, output_key, sm)

    for arg in sorted(needed_law_args):
        gn = _registry_group_for_law_arg(arg)
        if gn is None:
            continue
        add_spec(gn, arg)

    if "pe" in needed_law_args:
        add_spec("re", "re")
        add_spec("pr", "pr")
        add_spec("pe", "pe")

    specs = list(by_output.values())
    return _topological_sort_group_specs(specs)


def implied_scaling_audit_specs_for_laws(
    law_names: Iterable[str],
    pred_keys: Set[str],
) -> List[Dict[str, Any]]:
    """Chain-rule scaling audits were removed; no implied scaling rows are injected."""
    _ = (law_names, pred_keys)
    return []


def _topological_sort_group_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run groups whose inputs are other groups' outputs after their producers."""
    if not specs:
        return []
    by_out: Dict[str, Dict[str, Any]] = {}
    for s in specs:
        ok = s.get("output_key")
        if ok is not None:
            by_out[str(ok)] = s

    ids = {id(s): s for s in specs}
    out_edges: Dict[int, Set[int]] = defaultdict(set)
    in_degree: Dict[int, int] = {id(s): 0 for s in specs}

    for s in specs:
        sid = id(s)
        for _arg, in_key in (s.get("state_map") or {}).items():
            if in_key in by_out and by_out[in_key] is not s:
                pid = id(by_out[in_key])
                if sid not in out_edges[pid]:
                    out_edges[pid].add(sid)
                    in_degree[sid] += 1

    q = deque(s for s in specs if in_degree[id(s)] == 0)
    ordered: List[Dict[str, Any]] = []
    while q:
        s = q.popleft()
        ordered.append(s)
        for nid in out_edges[id(s)]:
            in_degree[nid] -= 1
            if in_degree[nid] == 0:
                q.append(ids[nid])

    if len(ordered) != len(specs):
        # Cycle (should not happen); preserve deterministic order
        return sorted(specs, key=lambda x: (str(x.get("name")), str(x.get("output_key"))))
    return ordered


def merge_implied_groups_first(
    implied: List[Dict[str, Any]],
    user_groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Prepend implied specs, then user-selected groups.

    If a user group uses the same ``output_key`` as an implied spec, the **user** entry wins
    (expert override); implied duplicate is dropped.
    """
    user_out = {str(g.get("output_key")) for g in user_groups if g.get("output_key")}
    filtered = [g for g in implied if str(g.get("output_key")) not in user_out]
    return filtered + list(user_groups)
