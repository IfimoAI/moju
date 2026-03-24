"""Tests for moju.monitor.visualize_labels."""

from moju.monitor.visualize_labels import pretty_category_name, pretty_residual_key


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
