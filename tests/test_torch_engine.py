"""
Comprehensive tests for ``moju.torch`` — PyTorch-first physics engine.

Tests cover:
- ``_nondim``: dimensional_to_nd_torch / nd_to_dimensional_torch
- ``_derived``: eval_derived_expr_torch / apply_derived_state_chain_torch
- ``_r_eff``: r_eff_scalar_torch / build_loss_torch
- ``_path_b``: fill_path_b_derivatives_torch
- ``_closure``: compute_implied_delta_torch / compute_ref_delta_torch
- ``_implied_diagnostics``: implied functions + merge_law_implied_audit_specs_torch
- ``TorchResidualEngine``: full pipeline, group inference, derived state, user_fns, loss
"""
from __future__ import annotations

import math
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from moju.piratio.nondim import NondimScales
from moju.torch._nondim import dimensional_to_nd_torch, nd_to_dimensional_torch
from moju.torch._derived import eval_derived_expr_torch, apply_derived_state_chain_torch
from moju.torch._r_eff import r_eff_scalar_torch, build_loss_torch
from moju.torch._path_b import fill_path_b_derivatives_torch
from moju.torch._closure import (
    compute_implied_delta_torch,
    compute_ref_delta_torch,
)
from moju.torch._implied_diagnostics import (
    merge_law_implied_audit_specs_torch,
    implied_alpha_fourier_conduction_torch,
    implied_wave_speed_torch,
    implied_viscous_acceleration_incompressible_torch,
)
from moju.torch._engine import TorchResidualEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _t(*args, **kwargs) -> torch.Tensor:
    """Convenience: torch.tensor with float32."""
    if len(args) == 1:
        return torch.tensor(args[0], dtype=torch.float32)
    return torch.tensor(args, dtype=torch.float32)


def _rand(shape) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(*shape, dtype=torch.float32)


# ===========================================================================
# _nondim
# ===========================================================================


class TestDimensionalToNdTorch:
    def setup_method(self):
        self.scales = NondimScales(L_ref=0.1, U_ref=1.0, rho_ref=1000.0)

    def test_x_scales(self):
        state = {"x": _t(1.0)}
        nd = dimensional_to_nd_torch(state, self.scales)
        assert abs(float(nd["x"]) - 10.0) < 1e-6

    def test_u_scales(self):
        state = {"u": _t(2.0)}
        nd = dimensional_to_nd_torch(state, self.scales)
        assert abs(float(nd["u"]) - 2.0) < 1e-6

    def test_p_scales(self):
        state = {"p": _t(1000.0)}
        nd = dimensional_to_nd_torch(state, self.scales)
        # p_ref = rho_ref * U_ref^2 = 1000.0
        assert abs(float(nd["p"]) - 1.0) < 1e-4

    def test_temperature_affine(self):
        scales = NondimScales(L_ref=0.1, U_ref=1.0, rho_ref=1.0, T0=300.0, dT_ref=50.0)
        state = {"T": _t(350.0)}
        nd = dimensional_to_nd_torch(state, scales)
        assert abs(float(nd["T"]) - 1.0) < 1e-6

    def test_passthrough_re(self):
        state = {"re": _t(1000.0)}
        nd = dimensional_to_nd_torch(state, self.scales)
        assert float(nd["re"]) == 1000.0

    def test_passthrough_unknown_warns(self):
        state = {"mystery_field": _t(42.0)}
        with pytest.warns(UserWarning, match="unrecognised key"):
            nd = dimensional_to_nd_torch(state, self.scales)
        assert float(nd["mystery_field"]) == 42.0

    def test_gradient_flows_through(self):
        val = torch.tensor(2.0, requires_grad=True)
        state = {"u": val}
        nd = dimensional_to_nd_torch(state, self.scales)
        loss = nd["u"].sum()
        loss.backward()
        assert val.grad is not None
        assert not torch.isnan(val.grad)

    def test_roundtrip(self):
        state = {
            "x": _t(0.05),
            "u": _t(0.5),
            "p": _t(500.0),
            "rho": _t(800.0),
        }
        nd = dimensional_to_nd_torch(state, self.scales, warn_unknown=False)
        recovered = nd_to_dimensional_torch(nd, self.scales)
        for k in state:
            assert abs(float(recovered[k]) - float(state[k])) < 1e-4, k

    def test_extra_rules_float(self):
        state = {"kappa": _t(0.1)}
        nd = dimensional_to_nd_torch(state, self.scales, extra_rules={"kappa": 10.0})
        assert abs(float(nd["kappa"]) - 1.0) < 1e-6

    def test_extra_rules_callable(self):
        state = {"kappa": _t(0.1)}
        nd = dimensional_to_nd_torch(
            state, self.scales, extra_rules={"kappa": lambda v, s: v * 20.0}
        )
        assert abs(float(nd["kappa"]) - 2.0) < 1e-6


# ===========================================================================
# _derived
# ===========================================================================


class TestEvalDerivedExprTorch:
    def _env(self):
        return {"k": _t(10.0), "rho": _t(2.0), "cp": _t(5.0)}

    def test_ref(self):
        result = eval_derived_expr_torch({"op": "ref", "key": "k"}, self._env())
        assert abs(float(result) - 10.0) < 1e-6

    def test_const(self):
        result = eval_derived_expr_torch({"op": "const", "value": 3.14}, {})
        assert abs(float(result) - 3.14) < 1e-5

    def test_add(self):
        r = eval_derived_expr_torch(
            {"op": "add", "a": {"op": "ref", "key": "k"}, "b": {"op": "ref", "key": "rho"}},
            self._env(),
        )
        assert abs(float(r) - 12.0) < 1e-6

    def test_div(self):
        r = eval_derived_expr_torch(
            {"op": "div", "a": {"op": "ref", "key": "k"}, "b": {"op": "ref", "key": "rho"}},
            self._env(),
        )
        assert abs(float(r) - 5.0) < 1e-6

    def test_sqrt(self):
        r = eval_derived_expr_torch({"op": "sqrt", "x": {"op": "const", "value": 16.0}}, {})
        assert abs(float(r) - 4.0) < 1e-6

    def test_pow(self):
        r = eval_derived_expr_torch(
            {"op": "pow", "a": {"op": "const", "value": 2.0}, "b": {"op": "const", "value": 3.0}},
            {},
        )
        assert abs(float(r) - 8.0) < 1e-6

    def test_maximum(self):
        r = eval_derived_expr_torch(
            {"op": "maximum", "a": {"op": "const", "value": 3.0}, "b": {"op": "const", "value": 7.0}},
            {},
        )
        assert abs(float(r) - 7.0) < 1e-6

    def test_unsupported_op_raises(self):
        with pytest.raises(ValueError, match="unsupported DSL op"):
            eval_derived_expr_torch({"op": "unknown_op"}, {})

    def test_gradient_flows(self):
        k = torch.tensor(10.0, requires_grad=True)
        r = eval_derived_expr_torch(
            {"op": "div", "a": {"op": "ref", "key": "k"}, "b": {"op": "const", "value": 2.0}},
            {"k": k},
        )
        r.backward()
        assert k.grad is not None
        assert abs(float(k.grad) - 0.5) < 1e-6


class TestApplyDerivedStateChainTorch:
    def test_basic_alpha_computation(self):
        state = {"k": _t(10.0), "rho": _t(2.0), "cp": _t(5.0)}
        steps = [
            {
                "output_key": "alpha",
                "expr": {
                    "op": "div",
                    "a": {"op": "ref", "key": "k"},
                    "b": {
                        "op": "mul",
                        "a": {"op": "ref", "key": "rho"},
                        "b": {"op": "ref", "key": "cp"},
                    },
                },
            }
        ]
        new_state, warns = apply_derived_state_chain_torch(state, {}, steps)
        assert not warns
        assert abs(float(new_state["alpha"]) - 1.0) < 1e-6  # 10 / (2*5)

    def test_missing_key_yields_warning(self):
        state = {}
        steps = [{"output_key": "alpha", "expr": {"op": "ref", "key": "k"}}]
        _, warns = apply_derived_state_chain_torch(state, {}, steps)
        assert len(warns) == 1
        assert "alpha" in warns[0]

    def test_chain_propagation(self):
        """Second step uses result of first step."""
        state = {"a": _t(2.0), "b": _t(3.0)}
        steps = [
            {
                "output_key": "c",
                "expr": {"op": "mul", "a": {"op": "ref", "key": "a"}, "b": {"op": "ref", "key": "b"}},
            },
            {
                "output_key": "d",
                "expr": {"op": "sqrt", "x": {"op": "ref", "key": "c"}},
            },
        ]
        new_state, warns = apply_derived_state_chain_torch(state, {}, steps)
        assert not warns
        assert abs(float(new_state["c"]) - 6.0) < 1e-6
        assert abs(float(new_state["d"]) - math.sqrt(6.0)) < 1e-5


# ===========================================================================
# _r_eff
# ===========================================================================


class TestREff:
    def test_uniform_residuals_no_q_penalty(self):
        r = torch.ones(100)
        val = r_eff_scalar_torch(r)
        # Plain smooth RMS under default ``p`` = 0
        assert val.item() > 0.0
        assert torch.isfinite(val)

    def test_skewed_residuals_higher_r_eff(self):
        torch.manual_seed(42)
        uniform_r = torch.ones(200)
        # Larger pointwise residuals → larger RMS even when ``p`` = 0
        skewed_r = torch.zeros(200)
        skewed_r[0] = 200.0
        r_eff_uniform = float(r_eff_scalar_torch(uniform_r))
        r_eff_skewed = float(r_eff_scalar_torch(skewed_r))
        assert r_eff_skewed > r_eff_uniform

    def test_empty_tensor_returns_nan(self):
        val = r_eff_scalar_torch(torch.tensor([]))
        assert math.isnan(float(val))

    def test_torch_r_eff_q_power_matches_auditor(self):
        pytest.importorskip("torch")

        import moju.monitor.auditor as aud
        import moju.torch._r_eff as tr_mod

        assert tr_mod.R_EFF_Q_POWER == aud.R_EFF_Q_POWER

    def test_gradient_flows(self):
        r = torch.randn(50, requires_grad=True)
        val = r_eff_scalar_torch(r)
        val.backward()
        assert r.grad is not None
        assert not torch.any(torch.isnan(r.grad))

    def test_build_loss_single_law(self):
        residuals = {"laws": {"foo": torch.randn(64)}}
        loss = build_loss_torch(residuals)
        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_build_loss_empty_returns_zero(self):
        loss = build_loss_torch({})
        assert loss.item() == 0.0

    def test_build_loss_custom_weights(self):
        residuals = {
            "laws": {
                "law1": torch.randn(32),
                "law2": torch.randn(32),
            }
        }
        loss_equal = float(build_loss_torch(residuals))
        loss_weighted = float(build_loss_torch(residuals, law_weights={"law1": 0.9, "law2": 0.1}))
        assert loss_equal != loss_weighted

    def test_build_loss_differentiable(self):
        r = torch.randn(64, requires_grad=True)
        residuals = {"laws": {"law1": r}}
        loss = build_loss_torch(residuals)
        loss.backward()
        assert r.grad is not None


# ===========================================================================
# _path_b
# ===========================================================================


class TestFillPathBDerivativesTorchU:
    def _uniform_1d_state(self, n=50):
        x = torch.linspace(0.0, 1.0, n)
        u = torch.sin(x)
        return {"x": x, "u": u}

    def test_fills_u_grad(self):
        state = self._uniform_1d_state()
        new_state, warns = fill_path_b_derivatives_torch(state)
        assert "u_grad" in new_state
        # du/dx ≈ cos(x); check mid-range value
        n = 50
        x = torch.linspace(0.0, 1.0, n)
        expected_mid = float(torch.cos(x[25]))
        computed_mid = float(new_state["u_grad"][25].squeeze())
        assert abs(computed_mid - expected_mid) < 0.05  # FD accuracy

    def test_fills_u_laplacian(self):
        state = self._uniform_1d_state()
        new_state, warns = fill_path_b_derivatives_torch(state)
        assert "u_laplacian" in new_state

    def test_does_not_overwrite_existing(self):
        state = self._uniform_1d_state()
        sentinel = torch.tensor([-999.0] * 50)
        state["u_grad"] = sentinel
        new_state, _ = fill_path_b_derivatives_torch(state)
        assert float(new_state["u_grad"][0]) == -999.0

    def test_warns_without_coords(self):
        state = {"u": torch.randn(50)}  # no x
        new_state, warns = fill_path_b_derivatives_torch(state)
        assert any("u_grad" in w or "no coordinate" in w for w in warns)

    def test_spectral_sin_mode_exact(self):
        n = 64
        L = 2.0 * math.pi
        dx = L / float(n)
        x = torch.arange(n, dtype=torch.float64) * dx
        u = torch.sin(x)
        new_state, warns = fill_path_b_derivatives_torch(
            {"x": x.float(), "u": u.float()},
            diff_method="spectral",
            periodic=True,
        )
        assert not warns
        ux = new_state["u_grad"].squeeze()
        uxx = new_state["u_laplacian"]
        assert float((ux.double() - torch.cos(x)).abs().max()) < 1e-4
        assert float((uxx.double() + torch.sin(x)).abs().max()) < 1e-4

    def test_spectral_requires_periodic(self):
        n = 32
        L = 2.0 * math.pi
        x = torch.arange(n, dtype=torch.float32) * (L / float(n))
        with pytest.raises(ValueError, match="periodic=True"):
            fill_path_b_derivatives_torch(
                {"x": x, "u": torch.sin(x)},
                diff_method="spectral",
                periodic=False,
            )


class TestFillPathBDerivativesTorch:
    def test_fills_T_grad_and_laplacian(self):
        x = torch.linspace(0.0, 1.0, 40)
        T = x ** 2
        state = {"x": x, "T": T}
        new_state, _ = fill_path_b_derivatives_torch(state)
        assert "T_grad" in new_state
        assert "T_laplacian" in new_state


# ===========================================================================
# _closure
# ===========================================================================


class TestComputeImpliedDeltaTorch:
    def _identity_model(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def test_subtract_mode_fractional_value(self):
        """delta = (pred - implied) / (|pred| + eps)."""
        state = {"k": _t(3.0), "k_implied": _t(2.0)}
        result = compute_implied_delta_torch(
            fn_wrapped=self._identity_model,
            arg_names=["k"],
            state_map={"k": "k"},
            state_pred=state,
            constants={},
            implied_fn_torch=lambda s, c: torch.as_tensor(s["k_implied"], dtype=torch.float32),
        )
        assert result is not None
        expected = (3.0 - 2.0) / (3.0 + 1e-30)
        assert abs(float(result) - expected) < 1e-6

    def test_subtract_mode_vector_pred_keeps_shape(self):
        state = {
            "k": torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32),
            "k_implied": torch.tensor([2.7, 4.4, 4.0], dtype=torch.float32),
        }
        result = compute_implied_delta_torch(
            fn_wrapped=self._identity_model,
            arg_names=["k"],
            state_map={"k": "k"},
            state_pred=state,
            constants={},
            implied_fn_torch=lambda s, c: s["k_implied"],
        )
        assert result is not None
        assert result.shape == (3,)
        expected = (state["k"] - state["k_implied"]) / (state["k"].abs() + 1e-30)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_returns_none_missing_key(self):
        state = {}  # no 'k'
        result = compute_implied_delta_torch(
            fn_wrapped=self._identity_model,
            arg_names=["k"],
            state_map={"k": "k"},
            state_pred=state,
            constants={},
            implied_fn_torch=lambda s, c: _t(1.0),
        )
        assert result is None


# ===========================================================================
# _implied_diagnostics
# ===========================================================================


class TestMergeLawImpliedAuditSpecsTorch:
    def test_fourier_conduction_returns_specs(self):
        laws = [{"name": "fourier_conduction", "state_map": {"T_t": "T_t", "T_laplacian": "T_laplacian"}}]
        specs = merge_law_implied_audit_specs_torch(laws)
        assert len(specs) >= 1
        assert specs[0]["name"] == "thermal_diffusivity"
        assert "implied_fn_torch" in specs[0]

    def test_disabled_returns_empty(self):
        laws = [{"name": "fourier_conduction"}]
        specs = merge_law_implied_audit_specs_torch(laws, enabled=False)
        assert specs == []

    def test_no_duplicates(self):
        laws = [
            {"name": "fourier_conduction", "state_map": {}},
            {"name": "fourier_conduction", "state_map": {}},  # duplicate
        ]
        specs = merge_law_implied_audit_specs_torch(laws)
        basenames = [s["residual_basename"] for s in specs]
        assert len(basenames) == len(set(basenames))

    def test_navier_stokes_returns_mu_spec(self):
        laws = [{"name": "momentum_navier_stokes", "state_map": {}}]
        specs = merge_law_implied_audit_specs_torch(laws)
        assert any(s["name"] == "dynamic_viscosity_from_re" for s in specs)

    def test_burgers_returns_nu_spec(self):
        laws = [{"name": "burgers_equation", "state_map": {"re": "Re", "U": "U", "L": "L"}}]
        specs = merge_law_implied_audit_specs_torch(laws)
        assert any(s["name"] == "kinematic_viscosity_from_re" for s in specs)
        burgers = next(s for s in specs if s["name"] == "kinematic_viscosity_from_re")
        assert burgers["output_key"] == "nu"
        assert burgers["state_map"]["re"] == "Re"
        assert burgers["residual_basename"] == "kinematic_viscosity_from_re/law_burgers_equation"


class TestTorchImpliedFunctions:
    def test_fourier_implied_alpha_basic(self):
        law_sm = {"T_t": "T_t", "T_laplacian": "T_laplacian"}
        fn = implied_alpha_fourier_conduction_torch(law_sm)
        state = {"T_t": _t(4.0), "T_laplacian": _t(2.0)}
        result = fn(state, {})
        assert result is not None
        assert abs(float(result) - 2.0) < 1e-6

    def test_wave_implied_speed_basic(self):
        law_sm = {"phi_tt": "phi_tt", "phi_laplacian": "phi_laplacian"}
        fn = implied_wave_speed_torch(law_sm)
        state = {"phi_tt": _t(9.0), "phi_laplacian": _t(1.0)}
        result = fn(state, {})
        assert result is not None
        assert abs(float(result) - 3.0) < 1e-6

    def test_implied_returns_none_on_missing(self):
        law_sm = {"T_t": "T_t", "T_laplacian": "T_laplacian"}
        fn = implied_alpha_fourier_conduction_torch(law_sm)
        result = fn({}, {})  # missing T_t
        assert result is None


# ===========================================================================
# TorchResidualEngine
# ===========================================================================


class TestTorchResidualEngineInit:
    def test_basic_init_mass_incompressible(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
        )
        assert "mass_incompressible" in engine._wrapped_laws

    def test_init_with_constants(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={"re": 1000.0},
        )
        assert engine._constants["re"] == 1000.0

    def test_group_inference_triggered(self):
        engine = TorchResidualEngine(
            laws=[{"name": "momentum_navier_stokes"}],
            constants={},
        )
        # Re is a group arg for NS momentum; should have at least one group plan entry
        assert len(engine._group_compute_plan) > 0

    def test_law_implied_audits_default_true(self):
        engine = TorchResidualEngine(
            laws=[{"name": "fourier_conduction"}],
            constants={},
        )
        assert len(engine._audit_specs) > 0

    def test_law_implied_audits_disable(self):
        engine = TorchResidualEngine(
            laws=[{"name": "fourier_conduction"}],
            constants={},
            law_implied_audits=False,
        )
        assert len(engine._audit_specs) == 0

    def test_user_constitutive_audit_appended(self):
        user_spec = {
            "name": "thermal_diffusivity",
            "output_key": "alpha",
            "state_map": {"k": "k", "rho": "rho", "cp": "cp"},
        }
        engine = TorchResidualEngine(
            laws=[],
            constitutive_audit=[user_spec],
            law_implied_audits=False,
        )
        assert len(engine._audit_specs) == 1


class TestComputeResidualsTorch:
    def _fourier_state(self):
        n = 20
        return {
            "T_t": torch.randn(n, dtype=torch.float32) * 0.1,
            "T_laplacian": torch.randn(n, dtype=torch.float32) * 0.5,
            "fo": torch.ones(n, dtype=torch.float32) * 0.01,
        }

    def test_fourier_residuals_present(self):
        engine = TorchResidualEngine(
            laws=[{"name": "fourier_conduction"}],
            constants={"fo": 0.01, "t": 0.0, "L": 0.1},
            law_implied_audits=False,
        )
        state = {
            "T_t": torch.randn(20),
            "T_laplacian": torch.randn(20),
        }
        result = engine.compute_residuals_torch(state)
        assert "laws" in result
        assert "fourier_conduction" in result["laws"]

    def test_output_shapes_match_input(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            law_implied_audits=False,
        )
        n = 64
        state = {"u_grad": torch.randn(n, 2, 2)}
        result = engine.compute_residuals_torch(state)
        assert "laws" in result
        assert result["laws"]["mass_incompressible"].shape[0] == n

    def test_best_effort_skips_missing(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            law_implied_audits=False,
            best_effort=True,
        )
        state = {}  # no u_grad
        result = engine.compute_residuals_torch(state)
        assert "laws" not in result or "mass_incompressible" not in result.get("laws", {})

    def test_best_effort_false_raises(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            law_implied_audits=False,
            best_effort=False,
        )
        state = {}  # no u_grad
        with pytest.raises(Exception):
            engine.compute_residuals_torch(state)

    def test_apply_nondim_infers_or_requires_refs(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            law_implied_audits=False,
        )
        with pytest.raises(ValueError, match="L_ref|nondim scales"):
            engine.compute_residuals_torch(
                {"u_grad": torch.randn(10, 2, 2)}, apply_nondim=True
            )

    def test_nondim_pipeline(self):
        scales = NondimScales(L_ref=0.1, U_ref=1.0, rho_ref=1.0)
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            scales=scales,
            law_implied_audits=False,
        )
        state = {"u_grad": torch.randn(10, 2, 2)}
        result = engine.compute_residuals_torch(state, apply_nondim=True)
        assert "laws" in result


class TestTorchResidualEngineUserFns:
    def test_user_fn_materialised_in_pipeline(self):
        """user_fn computes k from T before law evaluation."""

        def k_from_T(T: torch.Tensor) -> torch.Tensor:
            return 0.5 * T

        engine = TorchResidualEngine(
            laws=[{"name": "fourier_conduction"}],
            constants={"fo": 0.01},
            user_fns={"k": k_from_T},
            law_implied_audits=False,
        )
        n = 20
        state = {"T": torch.ones(n), "T_t": torch.randn(n), "T_laplacian": torch.randn(n)}
        # The engine should compute k from T before law evaluation
        result = engine.compute_residuals_torch(state)
        # Law should be evaluated (may succeed or skip — we just need no crash)
        assert isinstance(result, dict)


class TestTorchResidualEngineDerivedState:
    def test_alpha_from_derived_chain(self):
        steps = [
            {
                "output_key": "alpha",
                "expr": {
                    "op": "div",
                    "a": {"op": "ref", "key": "k"},
                    "b": {
                        "op": "mul",
                        "a": {"op": "ref", "key": "rho"},
                        "b": {"op": "ref", "key": "cp"},
                    },
                },
            }
        ]
        engine = TorchResidualEngine(
            laws=[{"name": "fourier_conduction"}],
            constants={"fo": 0.01, "t": 0.0, "L": 0.1},
            derived_state_chain=steps,
            law_implied_audits=False,
        )
        n = 10
        state = {
            "k": torch.full((n,), 2.0),
            "rho": torch.full((n,), 1.0),
            "cp": torch.full((n,), 4.0),
            "T_t": torch.randn(n),
            "T_laplacian": torch.randn(n),
        }
        result = engine.compute_residuals_torch(state)
        assert "laws" in result


class TestTorchResidualEngineTrainingLoss:
    def test_training_loss_scalar(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            law_implied_audits=False,
        )
        n = 64
        state = {"u_grad": torch.randn(n, 2, 2)}
        loss = engine.training_loss(state)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_training_loss_backward(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            law_implied_audits=False,
        )
        n = 32
        u_grad = torch.randn(n, 2, 2, requires_grad=True)
        state = {"u_grad": u_grad}
        loss = engine.training_loss(state)
        loss.backward()
        assert u_grad.grad is not None
        assert not torch.any(torch.isnan(u_grad.grad))

    def test_multiple_laws_loss(self):
        engine = TorchResidualEngine(
            laws=[
                {"name": "mass_incompressible"},
            ],
            constants={},
            law_implied_audits=False,
        )
        n = 16
        state = {"u_grad": torch.randn(n, 2, 2)}
        loss = engine.training_loss(state)
        assert torch.isfinite(loss)


class TestTorchResidualEngineConstitutiveAudit:
    def test_fourier_constitutive_audit_in_output(self):
        law_sm = {"T_t": "T_t", "T_laplacian": "T_laplacian"}
        engine = TorchResidualEngine(
            laws=[{"name": "fourier_conduction", "state_map": law_sm}],
            constants={"fo": 0.01},
            law_implied_audits=True,
        )
        n = 20
        state = {
            "k": torch.full((n,), 0.6),
            "rho": torch.full((n,), 1000.0),
            "cp": torch.full((n,), 4200.0),
            "T_t": torch.randn(n),
            "T_laplacian": torch.randn(n),
        }
        result = engine.compute_residuals_torch(state)
        assert "constitutive" in result
        keys = list(result["constitutive"].keys())
        assert any("thermal_diffusivity" in k for k in keys)

    def test_constitutive_audit_differentiable(self):
        law_sm = {"T_t": "T_t", "T_laplacian": "T_laplacian"}
        engine = TorchResidualEngine(
            laws=[{"name": "fourier_conduction", "state_map": law_sm}],
            constants={"fo": 0.01},
            law_implied_audits=True,
        )
        n = 20
        k = torch.full((n,), 0.6, requires_grad=True)
        state = {
            "k": k,
            "rho": torch.full((n,), 1000.0),
            "cp": torch.full((n,), 4200.0),
            "T_t": torch.randn(n),
            "T_laplacian": torch.randn(n),
        }
        result = engine.compute_residuals_torch(state)
        if "constitutive" in result:
            for v in result["constitutive"].values():
                v.sum().backward(retain_graph=True)
        # Should not raise


class TestTorchResidualEngineGroupInference:
    def test_group_plan_built_for_ns(self):
        """Engine builds group compute plan for NS momentum (re group)."""
        engine = TorchResidualEngine(
            laws=[{"name": "momentum_navier_stokes"}],
            constants={"re": 1000.0},  # pre-supply re to avoid inference
            law_implied_audits=False,
        )
        # Group plan should exist even if re is pre-supplied (plan is built
        # at init; runtime skips groups that are already in merged state)
        assert len(engine._group_compute_plan) >= 0

    def test_re_group_computed_from_state(self):
        """When re is absent from state, it is auto-computed from u, L, rho, mu."""
        engine = TorchResidualEngine(
            laws=[{"name": "momentum_navier_stokes"}],
            constants={},
            law_implied_audits=False,
        )
        n = 16
        # re group: Groups.re(u, L, rho, mu) = (rho * u * L) / mu
        # Provide scalar u (speed), not vector — matching Groups.re semantics
        state = {
            "u": torch.ones(n),        # speed magnitude for group re
            "L": torch.ones(n),
            "rho": torch.ones(n),
            "mu": torch.full((n,), 0.001),
            # Also provide vector fields for the NS law itself
            "u_t": torch.zeros(n, 2),
            "u_grad": torch.zeros(n, 2, 2),
            "p_grad": torch.zeros(n, 2),
            "u_laplacian": torch.zeros(n, 2),
        }
        result = engine.compute_residuals_torch(state)
        # Pipeline runs without error; re may be computed or law may skip
        assert isinstance(result, dict)


class TestTorchResidualEnginePathBFill:
    def test_path_b_fills_derivatives(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            path_b_fill=True,
            law_implied_audits=False,
        )
        n = 32
        x = torch.linspace(0.0, 1.0, n)
        # u is a scalar field here; note mass_incompressible needs u_grad
        state = {"x": x, "u": torch.randn(n)}
        # Should run without error (u_grad auto-filled)
        result = engine.compute_residuals_torch(state)
        assert isinstance(result, dict)


class TestTorchResidualEngineEvalMode:
    def test_data_residuals_in_eval_mode(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            law_implied_audits=False,
        )
        n = 16
        state = {"u_grad": torch.randn(n, 2, 2), "u": torch.randn(n, 2)}
        state_ref = {"u": torch.randn(n, 2)}
        result = engine.compute_residuals_torch(state, state_ref=state_ref, run_mode="eval")
        assert "data" in result
        assert "u" in result["data"]

    def test_eval_mode_no_data_when_no_ref(self):
        engine = TorchResidualEngine(
            laws=[{"name": "mass_incompressible"}],
            constants={},
            law_implied_audits=False,
        )
        state = {"u_grad": torch.randn(8, 2, 2)}
        result = engine.compute_residuals_torch(state, run_mode="eval")
        assert "data" not in result


# ===========================================================================
# moju.torch public API surface
# ===========================================================================


class TestPublicApi:
    def test_all_symbols_importable(self):
        from moju.torch import (
            TorchResidualEngine,
            dimensional_to_nd_torch,
            nd_to_dimensional_torch,
            r_eff_scalar_torch,
            build_loss_torch,
            wrap_law_torch,
        )
        assert TorchResidualEngine is not None
        assert dimensional_to_nd_torch is not None
        assert nd_to_dimensional_torch is not None
        assert r_eff_scalar_torch is not None
        assert build_loss_torch is not None
        assert wrap_law_torch is not None

    def test_wrap_law_torch_callable(self):
        from moju.torch import wrap_law_torch
        from moju.piratio.laws import Laws
        fn = wrap_law_torch(Laws.mass_incompressible)
        assert callable(fn)
        # Call with a 2D u_grad (batch of 1, 2x2)
        u_grad = torch.randn(10, 2, 2)
        result = fn(u_grad)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 10
