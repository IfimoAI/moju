"""Tests for moju.monitor.visualize_labels."""

from moju.monitor.auditor import admissibility_level
from moju.monitor.visualize_labels import (
    category_adm_bar_axis_range_percent_full,
    category_adm_bar_x_range,
    format_admissibility_pct,
    pretty_category_name,
    pretty_residual_key,
    truncate_display_label,
)
from moju.monitor.visualize_plotly import format_admissibility_status_label


def test_pretty_residual_key_law_only():
    assert pretty_residual_key("laws/fourier_conduction") == "Fourier Conduction"


def test_pretty_residual_key_scaling_fo_chain_dx():
    assert pretty_residual_key("scaling/fo/chain_dx") == "Fo Spatial Consistency"


def test_pretty_residual_key_chain_dt():
    assert pretty_residual_key("scaling/re/chain_dt") == "Re Temporal Consistency"


def test_pretty_residual_key_constitutive_chain():
    assert pretty_residual_key("constitutive/k_epsilon_nu_t/chain_dy") == "K Epsilon Nu T Spatial Consistency"


def test_pretty_residual_key_ref_implied_pi():
    assert pretty_residual_key("constitutive/foo/ref_delta") == "Foo Reference Consistency"
    assert pretty_residual_key("scaling/bar/implied_delta") == "Bar Implied Consistency"
    assert pretty_residual_key("scaling/re/pi_constant") == "Re Scale Invariance"


def test_pretty_residual_key_data_key():
    assert pretty_residual_key("data/T") == "T"


def test_pretty_category_name():
    assert pretty_category_name("laws") == "Governing Laws"
    assert pretty_category_name("scaling") == "Scaling and Similarity"


def test_category_adm_bar_x_range_tight_when_close():
    x0, x1 = category_adm_bar_x_range([0.98, 0.995])
    assert x1 - x0 < 0.5
    assert x0 <= 0.98 <= x1
    assert x0 <= 0.995 <= x1
    assert x1 <= 1.0


def test_category_adm_bar_x_range_empty_defaults_unit_interval():
    x0, x1 = category_adm_bar_x_range([])
    assert (x0, x1) == (0.0, 1.0)


def test_category_adm_bar_axis_range_percent_full():
    assert category_adm_bar_axis_range_percent_full() == (0.0, 100.0)


def test_category_adm_bar_x_range_high_scores_cap_at_one():
    x0, x1 = category_adm_bar_x_range([1.0, 1.0])
    assert x1 == 1.0
    assert x0 < x1


def test_truncate_display_label():
    assert truncate_display_label("short") == "short"
    long = "a" * 50
    out = truncate_display_label(long, max_len=12)
    assert len(out) <= 13
    assert out.endswith("…")


def test_format_admissibility_pct():
    assert format_admissibility_pct(0.985) == "98.50%"
    assert format_admissibility_pct(0.0) == "0.00%"
    assert format_admissibility_pct(1.0) == "100.00%"
    assert format_admissibility_pct(float("nan")) == "N/A"


def test_format_admissibility_status_label_matches_admissibility_level():
    for s in (0.0, 0.6, 0.8, 0.96, 1.0):
        assert format_admissibility_status_label(s) == admissibility_level(s)
    assert format_admissibility_status_label(float("nan")) == "N/A"


def test_pretty_residual_key_law_linked_implied_constitutive():
    assert (
        pretty_residual_key("constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta")
        == "Thermal diffusivity (implied)"
    )
    assert (
        pretty_residual_key("constitutive/mass_diffusivity/law_fick_diffusion/implied_delta")
        == "Mass diffusivity (implied)"
    )


def test_pretty_residual_key_implied_without_law_slug_constitutive():
    assert pretty_residual_key("constitutive/foo/implied_delta") == "Foo (implied)"
