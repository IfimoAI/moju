"""π-constant recipe coverage and numeric invariance for all Groups.* entries."""

import pytest
import jax.numpy as jnp

from moju.monitor.closure_registry import GROUP_FNS
from moju.monitor.pi_constant_recipes import (
    GROUP_PI_CONSTANT_RECIPES,
    apply_pi_constant_recipe,
    assert_pi_recipes_cover_all_groups,
)


def test_registry_matches_recipes():
    assert_pi_recipes_cover_all_groups()


@pytest.mark.parametrize("name", sorted(GROUP_FNS.keys()))
def test_recipe_preserves_group_value(name: str):
    fn, arg_names = GROUP_FNS[name]
    recipe = GROUP_PI_CONSTANT_RECIPES[name]
    state_map = {a: a for a in arg_names}
    constants = {a: jnp.array(0.5 + 0.09 * i) for i, a in enumerate(arg_names)}
    scaled = apply_pi_constant_recipe(constants, recipe, state_map, 2.1)
    kw0 = {a: constants[a] for a in arg_names}
    kw1 = {a: scaled[a] for a in arg_names}
    v0 = jnp.asarray(fn(**kw0))
    v1 = jnp.asarray(fn(**kw1))
    assert jnp.allclose(v0, v1, rtol=1e-5, atol=1e-6)


def test_recipe_row_invalid_power():
    with pytest.raises(ValueError, match="third element"):
        apply_pi_constant_recipe(
            {"L": jnp.array(1.0)},
            [("L", "multiply_c", 0)],
            {"L": "L"},
            2.0,
        )


def test_recipe_row_invalid_length():
    with pytest.raises(ValueError, match="length"):
        apply_pi_constant_recipe(
            {"L": jnp.array(1.0)},
            [("L",)],  # type: ignore[list-item]
            {"L": "L"},
            2.0,
        )


def test_accumulated_exponent_same_state_key():
    """Two args mapping to one state key get summed exponents."""
    recipe = [
        ("u", "multiply_c"),
        ("a", "multiply_c"),
    ]
    state_map = {"u": "x", "a": "x"}
    constants = {"x": jnp.array(2.0)}
    out = apply_pi_constant_recipe(constants, recipe, state_map, 3.0)
    assert float(out["x"]) == pytest.approx(2.0 * 3.0 * 3.0)
