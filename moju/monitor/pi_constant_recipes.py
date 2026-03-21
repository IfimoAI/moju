"""
π-constant scaling recipes for Groups.* scaling audits.

Each recipe lists rows ``(arg_name, kind[, power])`` where ``kind`` is ``multiply_c`` or
``divide_c``, and optional integer ``power`` (default 1) means scale by ``c**power`` or
``c**(-power)`` respectively. Combined effect per state key is ``v * c**total_exponent``.

Recipes are **canonical one-parameter orbits** (not unique). For ``gr``, gravity is
fixed inside ``Groups.gr`` (see module docstring in :mod:`moju.piratio.groups`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp

# Row: (arg_name, multiply_c|divide_c) or (arg_name, multiply_c|divide_c, positive_int_power)
RecipeRow = Union[Tuple[str, str], Tuple[str, str, int]]

GROUP_PI_CONSTANT_RECIPES: Dict[str, List[RecipeRow]] = {
    # Re = rho*u*L/mu
    "re": [("L", "multiply_c"), ("mu", "multiply_c")],
    # Pr = cp*mu/k
    "pr": [("mu", "multiply_c"), ("k", "multiply_c")],
    # Nu = h*L/k  ->  h'*k' with L fixed, or scale h,k together
    "nu": [("h", "multiply_c"), ("k", "multiply_c")],
    # Gr = g*beta*dT*L^3/nu^2 (g fixed in Groups.gr). L^2 and nu^3 -> L^6/nu^6 cancels.
    "gr": [("L", "multiply_c", 2), ("nu", "multiply_c", 3)],
    # Ma = u/a
    "ma": [("u", "multiply_c"), ("a", "multiply_c")],
    # We = rho*u^2*L/sigma  ->  u*c, sigma*c^2
    "we": [("u", "multiply_c"), ("sigma", "multiply_c", 2)],
    # Pe = re*pr
    "pe": [("re", "multiply_c"), ("pr", "divide_c")],
    # St = f*L/u  ->  f*c, u*c
    "st": [("f", "multiply_c"), ("u", "multiply_c")],
    # Bi = h*L/k_solid
    "bi": [("h", "multiply_c"), ("k_solid", "multiply_c")],
    # Fo = alpha*t/L^2  ->  alpha*c, t*c, L*c
    "fo": [("alpha", "multiply_c"), ("t", "multiply_c"), ("L", "multiply_c")],
    # Sc = nu/D
    "sc": [("nu", "multiply_c"), ("D", "multiply_c")],
    # Le = sc/pr  ->  scale sc and pr by the same c
    "le": [("sc", "multiply_c"), ("pr", "multiply_c")],
    # Kn = lambda_mfp/L
    "kn": [("lambda_mfp", "multiply_c"), ("L", "multiply_c")],
    # Bo = rho*g*L^2/sigma  ->  L*c, sigma*c^2
    "bo": [("L", "multiply_c"), ("sigma", "multiply_c", 2)],
    # Ca = mu*u/sigma  ->  mu*c, sigma*c
    "ca": [("mu", "multiply_c"), ("sigma", "multiply_c")],
    # Eu = dp/(rho*u^2)  ->  dp*c^2, u*c
    "eu": [("dp", "multiply_c", 2), ("u", "multiply_c")],
    # Da = K/L^2  ->  K*c^2, L*c
    "da": [("K", "multiply_c", 2), ("L", "multiply_c")],
    # Ec = u^2/(cp*dT)
    "ec": [("u", "multiply_c"), ("cp", "multiply_c"), ("dT", "multiply_c")],
    # Fo_mass = D*t/L^2
    "fo_mass": [("D", "multiply_c"), ("t", "multiply_c"), ("L", "multiply_c")],
    # wavenumber = k*L  ->  k/c, L*c
    "wavenumber": [("k", "divide_c"), ("L", "multiply_c")],
    # Pe_mass = re*sc
    "pe_mass": [("re", "multiply_c"), ("sc", "divide_c")],
    # St_wave = omega*L/c_wave
    "st_wave": [("omega", "multiply_c"), ("c", "multiply_c")],
}


def list_pi_constant_group_names() -> List[str]:
    return sorted(GROUP_PI_CONSTANT_RECIPES.keys())


def _row_to_exponent(row: Tuple[Any, ...]) -> Tuple[str, int]:
    if len(row) == 2:
        arg_name, kind = row
        n = 1
    elif len(row) == 3:
        arg_name, kind, n = row
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"recipe row third element must be int >= 1, got {row!r}")
    else:
        raise ValueError(f"recipe row must be length 2 or 3, got {row!r}")
    if kind == "multiply_c":
        return arg_name, n
    if kind == "divide_c":
        return arg_name, -n
    raise ValueError(f"recipe kind must be 'multiply_c' or 'divide_c', got {kind!r}")


def apply_pi_constant_recipe(
    constants: Dict[str, Any],
    recipe: List[RecipeRow],
    state_map: Dict[str, str],
    c: Union[float, Any],
) -> Dict[str, Any]:
    """
    Copy ``constants`` and scale entries referenced by ``recipe`` via ``state_map``.

    Exponents for the same state key (from different args) add. Each state key must
    exist in ``constants`` (Path A: engine.constants).
    """
    c_arr = jnp.asarray(c)
    c_float = float(jax.device_get(c_arr).item())
    if c_float <= 1.0:
        raise ValueError(f"π-constant scaling requires c > 1, got {c!r}")
    c = c_arr

    exp_by_state_key: Dict[str, int] = {}
    for row in recipe:
        arg_name, exp = _row_to_exponent(row)
        if arg_name not in state_map:
            raise KeyError(f"π-constant: arg {arg_name!r} missing from state_map")
        sk = state_map[arg_name]
        exp_by_state_key[sk] = exp_by_state_key.get(sk, 0) + exp

    out = dict(constants)
    for sk, total_exp in exp_by_state_key.items():
        if sk not in out:
            raise KeyError(
                f"π-constant: state key {sk!r} not found in constants; "
                "put scaled parameters in ResidualEngine.constants for Path A."
            )
        v = jnp.asarray(out[sk])
        out[sk] = v * (c ** total_exp)
    return out


def assert_pi_recipes_cover_all_groups() -> None:
    """Dev check: registry keys match recipes (call from tests)."""
    from moju.monitor.closure_registry import GROUP_FNS

    reg = set(GROUP_FNS.keys())
    rec = set(GROUP_PI_CONSTANT_RECIPES.keys())
    if reg != rec:
        missing = sorted(reg - rec)
        extra = sorted(rec - reg)
        raise AssertionError(f"GROUP_FNS vs recipes mismatch missing={missing} extra={extra}")
