"""
Core monitor helpers for law-first configuration:

- infer ``groups`` specs from selected ``Laws.*`` arguments
- preserve explicit user overrides by ``output_key``
"""

from __future__ import annotations

import inspect
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Set

from moju.monitor.closure_registry import GROUP_FNS
from moju.piratio.groups import Groups
from moju.piratio.laws import Laws

# Law parameter name -> Groups registry name when it differs from the argument name.
_LAW_ARG_TO_GROUP_NAME: Dict[str, str] = {
    "kL": "wavenumber",
}


def _positional_param_names(fn: Any) -> List[str]:
    sig = inspect.signature(fn)
    names: List[str] = []
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            names.append(str(p.name))
    return names


def law_parameter_names(law_name: str) -> List[str]:
    fn = getattr(Laws, law_name)
    return _positional_param_names(fn)


def group_parameter_names(group_name: str) -> List[str]:
    fn = getattr(Groups, group_name)
    return _positional_param_names(fn)


def build_law_spec_identity(law_name: str) -> Dict[str, Any]:
    args = law_parameter_names(law_name)
    return {"name": law_name, "state_map": {a: a for a in args}}


def _registry_group_for_law_arg(arg: str) -> Optional[str]:
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
    Return ``groups`` specs so selected law dimensionless arguments are computed before laws.
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
        by_output[output_key] = {
            "name": group_registry_name,
            "output_key": output_key,
            "state_map": sm,
        }

    for arg in sorted(needed_law_args):
        gn = _registry_group_for_law_arg(arg)
        if gn is None:
            continue
        add_spec(gn, arg)

    # Pe can depend transitively on Re and Pr in user workflows.
    if "pe" in needed_law_args:
        add_spec("re", "re")
        add_spec("pr", "pr")
        add_spec("pe", "pe")

    specs = list(by_output.values())
    return _topological_sort_group_specs(specs)


def _topological_sort_group_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        return sorted(specs, key=lambda x: (str(x.get("name")), str(x.get("output_key"))))
    return ordered


def merge_implied_groups_first(
    implied: List[Dict[str, Any]],
    user_groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Prepend implied specs, then user-selected groups.
    User spec with same ``output_key`` overrides implied spec.
    """
    user_out = {str(g.get("output_key")) for g in user_groups if g.get("output_key")}
    filtered = [g for g in implied if str(g.get("output_key")) not in user_out]
    return filtered + list(user_groups)
