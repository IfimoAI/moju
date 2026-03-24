"""Studio implied ``groups`` injection for law dimensionless arguments."""

from apps.moju_studio.studio_auto_config import build_studio_auto_fragment
from apps.moju_studio.studio_implied_groups import implied_group_specs_for_laws, merge_implied_groups_first


def test_implied_pe_includes_re_and_pr_before_pe():
    specs = implied_group_specs_for_laws(["advection_diffusion"])
    outs = [s["output_key"] for s in specs]
    assert outs.index("re") < outs.index("pe")
    assert outs.index("pr") < outs.index("pe")


def test_merge_implied_respects_user_output_key_override():
    implied = [{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}]
    user = [{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha_hat", "t": "t", "L": "L"}}]
    merged = merge_implied_groups_first(implied, user)
    assert len(merged) == 1
    assert merged[0]["state_map"]["alpha"] == "alpha_hat"


def test_build_ns_includes_implied_re_group():
    frag = build_studio_auto_fragment(
        law_names=["momentum_navier_stokes"],
        model_names=[],
        group_names=[],
        pred_keys=set(),
        constant_keys=set(),
    )
    assert any(g.get("name") == "re" and g.get("output_key") == "re" for g in frag["groups"])
