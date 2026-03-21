"""
π-constant scaling recipes for Groups.* scaling audits.

Each recipe lists (function_arg_name, rule) pairs such that applying the same stretch
c > 1 to selected inputs leaves the dimensionless group value unchanged (for the
canonical monomial form in moju.piratio.groups.Groups).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple, Union

import jax
import jax.numpy as jnp

# Rules: multiply by c, or divide by c (equivalent to multiply by c^{-1}).
ScaleRule = Literal["multiply_c", "divide_c"]

# Group name -> list of (arg_name, rule) in Groups.<name> signature order subset.
GROUP_PI_CONSTANT_RECIPES: Dict[str, List[Tuple[str, ScaleRule]]] = {
    # Re = rho * u * L / mu  ->  L' = c L, mu' = c mu  keeps Re.
    "re": [("L", "multiply_c"), ("mu", "multiply_c")],
    # Pr = cp * mu / k  ->  mu' = c mu, k' = c k  keeps Pr.
    "pr": [("mu", "multiply_c"), ("k", "multiply_c")],
    # Pe = re * pr  ->  re' = c re, pr' = pr / c  keeps Pe.
    "pe": [("re", "multiply_c"), ("pr", "divide_c")],
}


def list_pi_constant_group_names() -> List[str]:
    return sorted(GROUP_PI_CONSTANT_RECIPES.keys())


def apply_pi_constant_recipe(
    constants: Dict[str, Any],
    recipe: List[Tuple[str, ScaleRule]],
    state_map: Dict[str, str],
    c: Union[float, Any],
) -> Dict[str, Any]:
    """
    Copy ``constants`` and scale entries referenced by ``recipe`` via ``state_map``.

    Each recipe ``arg_name`` must appear in ``state_map``; the corresponding state
    key must exist in ``constants`` (Path A: put physical parameters in engine.constants).
    """
    c_arr = jnp.asarray(c)
    c_float = float(jax.device_get(c_arr).item())
    if c_float <= 1.0:
        raise ValueError(f"π-constant scaling requires c > 1, got {c!r}")
    c = c_arr
    out = dict(constants)
    for arg_name, rule in recipe:
        if arg_name not in state_map:
            raise KeyError(f"π-constant: arg {arg_name!r} missing from state_map")
        sk = state_map[arg_name]
        if sk not in out:
            raise KeyError(
                f"π-constant: state key {sk!r} (arg {arg_name!r}) not found in constants; "
                "put scaled parameters in ResidualEngine.constants for Path A."
            )
        v = jnp.asarray(out[sk])
        if rule == "multiply_c":
            out[sk] = v * c
        elif rule == "divide_c":
            out[sk] = v / c
        else:
            raise ValueError(f"Unknown scale rule {rule!r}")
    return out
