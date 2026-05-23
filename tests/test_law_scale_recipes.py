"""Tests for moju.monitor.law_scale_recipes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from moju.monitor.auditor import DEFAULT_NONDIM_R_NORM_SCALE_K, ResidualEngine
from moju.monitor.law_implied_diagnostics import all_law_names
from moju.monitor.law_scale_recipes import (
    characteristic_law_scale_k,
    law_scale_coverage_report,
    list_laws_with_scale_recipes,
)


class TestLawScaleRecipes:
    def test_all_laws_have_recipe_or_generic(self):
        report = law_scale_coverage_report()
        for name in all_law_names():
            assert name in report
            assert report[name] in ("recipe", "generic_only")

    def test_list_recipes_covers_all_laws(self):
        assert set(list_laws_with_scale_recipes()) == set(all_law_names())

    def test_laplace_auto_from_laplacian(self):
        merged = {"phi_laplacian": jnp.array([1.0, 2.0, 3.0])}
        spec = {"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}
        scale, src = characteristic_law_scale_k(
            "laplace_equation",
            merged=merged,
            constants={},
            law_spec=spec,
        )
        assert src == "auto"
        assert scale > DEFAULT_NONDIM_R_NORM_SCALE_K

    def test_missing_keys_fallback(self):
        scale, src = characteristic_law_scale_k(
            "laplace_equation",
            merged={},
            constants={},
            law_spec={"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}},
        )
        assert src == "auto_fallback"
        assert abs(scale - DEFAULT_NONDIM_R_NORM_SCALE_K) < 1e-12

    def test_floor_at_default(self):
        merged = {"phi_laplacian": jnp.array([0.0, 0.0])}
        spec = {"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}
        scale, _ = characteristic_law_scale_k(
            "laplace_equation", merged=merged, constants={}, law_spec=spec
        )
        assert scale >= DEFAULT_NONDIM_R_NORM_SCALE_K - 1e-15

    def test_fourier_term_balance(self):
        merged = {
            "T_t": jnp.array([0.1, 0.2]),
            "T_laplacian": jnp.array([1.0, 2.0]),
            "fo": 0.5,
            "t": 10.0,
            "L": 0.1,
        }
        spec = {
            "name": "fourier_conduction",
            "state_map": {
                "T_t": "T_t",
                "T_laplacian": "T_laplacian",
                "fo": "fo",
                "t": "t",
                "L": "L",
            },
        }
        scale, src = characteristic_law_scale_k(
            "fourier_conduction", merged=merged, constants={}, law_spec=spec
        )
        assert src == "auto"
        assert scale > 0

    def test_engine_auto_mode_logs_scale_source(self):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
        )
        core.compute_residuals({"phi_xx": jnp.array([1.0, 4.0, 9.0])})
        entry = core.log[-1]
        assert entry["scale_source"]["laws/laplace_equation"] == "auto"
        assert entry["scale"]["laws/laplace_equation"] > DEFAULT_NONDIM_R_NORM_SCALE_K

    def test_engine_fixed_mode_uses_gauge(self):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
            law_scale_mode="fixed",
        )
        core.compute_residuals({"phi_xx": jnp.array([100.0])})
        sk = core.log[-1]["scale"]["laws/laplace_equation"]
        assert abs(sk - DEFAULT_NONDIM_R_NORM_SCALE_K) < 1e-9
        assert core.log[-1]["scale_source"]["laws/laplace_equation"] == "fixed"

    def test_closure_scale_stays_fixed_in_auto_mode(self):
        from moju.piratio.models import Models

        P, R, T = jnp.array(1e6), jnp.array(287.0), jnp.array(300.0)
        rho = Models.ideal_gas_rho(P, R, T)
        core = ResidualEngine(
            laws=[],
            constitutive_audit=[
                {
                    "name": "ideal_gas_rho",
                    "output_key": "rho",
                    "state_map": {"P": "P", "R": "R", "T": "T"},
                    "implied_value_key": "rho_implied",
                }
            ],
        )
        state = {"P": P, "R": R, "T": T, "rho": rho, "rho_implied": rho}
        core.compute_residuals(state)
        sk = core.log[-1]["scale"]["constitutive/ideal_gas_rho/implied_delta"]
        assert abs(sk - DEFAULT_NONDIM_R_NORM_SCALE_K) < 1e-9
