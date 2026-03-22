"""Studio π-constant gating (no Streamlit)."""

import jax.numpy as jnp
import pytest

from apps.moju_studio.studio_core import (
    is_studio_npz_shim_state_builder,
    make_session_state_builder,
    validate_studio_pi_gating,
)


def test_npz_shim_detected():
    sb = make_session_state_builder({"T": jnp.array([1.0])})
    assert is_studio_npz_shim_state_builder(sb) is True


def test_validate_no_pi_ok():
    validate_studio_pi_gating(
        use_path_b=True,
        scaling_audit_specs=[{"name": "s", "invariance_pi_constant": False}],
        state_builder=None,
    )


def test_validate_path_b_with_pi_rejected():
    with pytest.raises(ValueError, match="Path A"):
        validate_studio_pi_gating(
            use_path_b=True,
            scaling_audit_specs=[
                {
                    "name": "pi_s",
                    "invariance_pi_constant": True,
                    "invariance_compare_keys": ["u"],
                }
            ],
            state_builder=None,
        )


def test_validate_shim_with_pi_rejected():
    sb = make_session_state_builder({"u": jnp.array([1.0])})
    with pytest.raises(ValueError, match="NPZ Path A shim"):
        validate_studio_pi_gating(
            use_path_b=False,
            scaling_audit_specs=[
                {
                    "name": "pi_s",
                    "invariance_pi_constant": True,
                    "invariance_compare_keys": ["u"],
                }
            ],
            state_builder=sb,
        )


def test_validate_pi_missing_compare_keys():
    sb = lambda model, params, collocation, constants: {"u": jnp.array([1.0])}  # noqa: E731
    with pytest.raises(ValueError, match="invariance_compare_keys"):
        validate_studio_pi_gating(
            use_path_b=False,
            scaling_audit_specs=[
                {"name": "pi_s", "invariance_pi_constant": True, "invariance_compare_keys": []}
            ],
            state_builder=sb,
        )


def test_validate_non_shim_builder_ok():
    def sb(model, params, collocation, constants):  # noqa: ARG001
        c = constants.get("k", 1.0)
        return {"u": jnp.asarray(c) * jnp.ones((2,))}

    validate_studio_pi_gating(
        use_path_b=False,
        scaling_audit_specs=[
            {
                "name": "pi_s",
                "invariance_pi_constant": True,
                "invariance_compare_keys": ["u"],
            }
        ],
        state_builder=sb,
    )
