"""Tests for core monitor law->group inference helpers."""

from moju.monitor.law_group_inference import (
    build_law_spec_identity,
    implied_group_specs_for_laws,
    merge_implied_groups_first,
)


def test_build_law_spec_identity_uses_identity_state_map():
    spec = build_law_spec_identity("fourier_conduction")
    assert spec["name"] == "fourier_conduction"
    assert spec["state_map"]["T_t"] == "T_t"
    assert spec["state_map"]["fo"] == "fo"


def test_implied_pe_includes_re_and_pr_before_pe():
    specs = implied_group_specs_for_laws(["advection_diffusion"])
    outs = [s["output_key"] for s in specs]
    assert outs.index("re") < outs.index("pe")
    assert outs.index("pr") < outs.index("pe")


def test_merge_implied_groups_first_respects_user_output_key_override():
    implied = [{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}]
    user = [{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha_hat", "t": "t", "L": "L"}}]
    merged = merge_implied_groups_first(implied, user)
    assert len(merged) == 1
    assert merged[0]["state_map"]["alpha"] == "alpha_hat"
