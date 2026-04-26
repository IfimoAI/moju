"""Legacy π gating hook is a no-op after scaling audit removal."""

from apps.moju_studio.studio_core import validate_studio_pi_gating


def test_validate_studio_pi_gating_is_noop():
    validate_studio_pi_gating(
        use_path_b=True,
        scaling_audit_specs=[],
        state_builder=None,
    )
