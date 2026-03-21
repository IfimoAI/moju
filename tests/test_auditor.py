"""Tests for ResidualEngine, build_loss, audit, visualize, constitutive/scaling closures."""

import pytest
import jax
import jax.numpy as jnp
from moju.monitor import (
    AuditSpec,
    MonitorConfig,
    ResidualEngine,
    admissibility_level,
    audit,
    build_loss,
    list_constitutive_models,
    list_pi_constant_group_names,
    list_scaling_closure_ids,
    visualize,
)


class TestAdmissibilityLevel:
    def test_four_levels(self):
        assert admissibility_level(0.0) == "Non-Admissible"
        assert admissibility_level(0.39) == "Non-Admissible"
        assert admissibility_level(0.40) == "Low Admissibility"
        assert admissibility_level(0.69) == "Low Admissibility"
        assert admissibility_level(0.70) == "Moderate Admissibility"
        assert admissibility_level(0.89) == "Moderate Admissibility"
        assert admissibility_level(0.90) == "High Admissibility"
        assert admissibility_level(1.0) == "High Admissibility"


class TestResidualEngineResidualDict:
    def test_laws_only_when_no_audits(self, rtol, atol):
        core = ResidualEngine(
            constants={"L": 0.1},
            laws=[{"name": "mass_incompressible", "state_map": {"u_grad": "u_grad"}}],
        )
        state_pred = {"u_grad": jnp.array([[0.0, 1.0], [-1.0, 0.0]])}
        residuals = core.compute_residuals(state_pred)
        assert "laws" in residuals
        assert jnp.allclose(residuals["laws"]["mass_incompressible"], 0.0, rtol=rtol, atol=atol)
        assert "constitutive" not in residuals
        assert "scaling" not in residuals
        assert "data" not in residuals

    def test_log_appends_rms_per_key(self):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        state_pred = {"phi_laplacian": jnp.array(1.0)}
        core.compute_residuals(state_pred)
        core.compute_residuals(state_pred)
        assert len(core.log) == 2
        assert "laws/laplace_equation" in core.log[0]["rms"]

    def test_constitutive_sutherland_closure(self, rtol, atol):
        core = ResidualEngine(
            constants={"mu0": 1.8e-5, "T0": 273.0, "S": 110.4},
            laws=[],
            constitutive_audit=[
                {
                    "name": "sutherland_mu",
                    "output_key": "mu",
                    "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
                    "predicted_spatial": ["T"],
                }
            ],
        )
        T = 300.0
        mu_true = 1.8e-5 * (T / 273) ** 1.5 * (273 + 110.4) / (T + 110.4)
        state_pred = {
            "mu": mu_true,
            "T": T,
            "d_T_dx": jnp.array(1.0),
            "d_mu_dx": jnp.array(0.0),  # inconsistent on purpose
        }
        residuals = core.compute_residuals(state_pred)
        assert "constitutive" in residuals
        assert "sutherland_mu/chain_dx" in residuals["constitutive"]
        assert abs(float(residuals["constitutive"]["sutherland_mu/chain_dx"])) > 0.0

    def test_scaling_pe_identity_zero(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                    "predicted_spatial": ["Re"],
                }
            ],
        )
        Re, Pr = 100.0, 0.7
        # Provide d_Pe_dx consistent with chain rule for Pe = Re*Pr and dRe/dx = 1, dPr/dx = 0.
        state_pred = {"Pe": Re * Pr, "Re": Re, "Pr": Pr, "d_Re_dx": 1.0, "d_Pe_dx": Pr}
        residuals = core.compute_residuals(state_pred)
        assert "scaling" in residuals
        assert jnp.allclose(residuals["scaling"]["pe/chain_dx"], 0.0, rtol=rtol, atol=atol)

    def test_scaling_pe_weak_chain_dx_weighted_rms(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                    "predicted_spatial": ["Re"],
                    "closure_mode": "weak",
                    "quadrature_weights": {"x": "w_x"},
                }
            ],
        )
        # Pe = Re * Pr, Pr constant, dRe/dx = 1, but we set dPe/dx = 0 -> residual = -Pr everywhere.
        Pr = 5.0
        state_pred = {
            "Pe": jnp.array([10.0, 11.0, 12.0]),
            "Re": jnp.array([2.0, 2.0, 2.0]),
            "Pr": Pr,
            "d_Re_dx": jnp.ones((3,)),
            "d_Pe_dx": jnp.zeros((3,)),
            "w_x": jnp.array([1.0, 2.0, 1.0]),
        }
        residuals = core.compute_residuals(state_pred)
        assert "scaling" in residuals
        # Weighted RMS of constant residual -Pr is |Pr|.
        assert jnp.allclose(residuals["scaling"]["pe/chain_dx"], abs(Pr), rtol=rtol, atol=atol)

    def test_scaling_pe_identity_nonzero(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                    "predicted_spatial": ["Re"],
                }
            ],
        )
        state_pred = {"Pe": 100.0, "Re": 10.0, "Pr": 5.0, "d_Re_dx": 1.0, "d_Pe_dx": 0.0}
        residuals = core.compute_residuals(state_pred)
        # For Pe = Re*Pr, chain expects dPe/dx = Pr * dRe/dx = 5.
        assert jnp.allclose(residuals["scaling"]["pe/chain_dx"], -5.0, rtol=rtol, atol=atol)

    def test_state_ref_adds_data_residual(self, rtol, atol):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        state_pred = {"phi_laplacian": jnp.array(0.5)}
        state_ref = {"phi_laplacian": jnp.array(0.0)}
        residuals = core.compute_residuals(state_pred, state_ref=state_ref)
        assert "data" in residuals
        assert jnp.allclose(residuals["data"]["phi_laplacian"], -0.5, rtol=rtol, atol=atol)

    def test_inferred_predicted_spatial_logged(self):
        core = ResidualEngine(
            constants={"mu0": 1.8e-5, "T0": 273.0, "S": 110.4},
            laws=[],
            constitutive_audit=[
                {
                    "name": "sutherland_mu",
                    "output_key": "mu",
                    "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
                    # predicted_spatial intentionally omitted to trigger inference
                }
            ],
        )
        T = 300.0
        mu_true = 1.8e-5 * (T / 273) ** 1.5 * (273 + 110.4) / (T + 110.4)
        state_pred = {"mu": mu_true, "T": T, "d_T_dx": jnp.array(1.0), "d_mu_dx": jnp.array(0.0)}
        core.compute_residuals(state_pred, collocation={"x": jnp.array([0.0])})
        assert "inferred" in core.log[-1]
        assert any("constitutive:sutherland_mu inferred predicted_spatial=['T']" in s for s in core.log[-1]["inferred"])


class TestBuildLoss:
    def test_cascaded_loss_scalar(self, rtol, atol):
        residual_dict = {
            "laws": {
                "mass_incompressible": jnp.array(0.0),
                "laplace_equation": jnp.array(0.0),
            }
        }
        loss = build_loss(residual_dict)
        assert jnp.allclose(loss, 0.0, rtol=rtol, atol=atol)

    def test_cascaded_loss_nonzero(self, rtol, atol):
        residual_dict = {"laws": {"laplace_equation": jnp.array(3.0)}}
        loss = build_loss(residual_dict)
        assert jnp.allclose(loss, 3.0, rtol=rtol, atol=atol)

    def test_build_loss_law_weights(self, rtol, atol):
        residual_dict = {"laws": {"a": jnp.array(1.0), "b": jnp.array(0.0)}}
        loss = build_loss(residual_dict, law_weights={"a": 1.0, "b": 0.0})
        assert jnp.allclose(loss, 1.0, rtol=rtol, atol=atol)

    def test_build_loss_differentiable(self):
        def loss_fn(phi_laplacian):
            return build_loss({"laws": {"laplace_equation": phi_laplacian}})

        grad = jax.grad(loss_fn)(jnp.array(2.0))
        assert grad is not None


class TestBuildLossBatch:
    def test_batch_law_residuals(self, rtol, atol):
        residual_dict = {"laws": {"mass_incompressible": jnp.zeros((5,))}}
        loss = build_loss(residual_dict)
        assert jnp.allclose(loss, 0.0, rtol=rtol, atol=atol)


class TestAudit:
    def test_audit_writes_back_to_log(self):
        log = [
            {"index": 0, "rms": {"laws/a": 2.0, "constitutive/x": 1.0}},
            {"index": 1, "rms": {"laws/a": 1.0, "constitutive/x": 0.5}},
        ]
        report = audit(log)
        assert "per_key" in report
        assert "r_norm" in log[0]

    def test_audit_r_ref_from_first_entry(self):
        log = [{"index": 0, "rms": {"k": 10.0}}, {"index": 1, "rms": {"k": 5.0}}]
        audit(log)
        assert log[1]["r_norm"]["k"] == 0.5

    def test_audit_uses_r_ref_as_scale_when_supplied(self):
        log = [{"index": 0, "rms": {"k": 100.0}}, {"index": 1, "rms": {"k": 4.0}}]
        audit(log, r_ref={"k": 8.0})
        assert log[1]["r_norm"]["k"] == 0.5
        assert log[0]["r_norm"]["k"] == 100.0 / 8.0

    def test_audit_uses_entry_scale_when_present(self):
        log = [
            {"index": 0, "rms": {"laws/a": 2.0}, "scale": {"laws/a": 4.0}},
            {"index": 1, "rms": {"laws/a": 1.0}, "scale": {"laws/a": 4.0}},
        ]
        audit(log)
        assert log[1]["r_norm"]["laws/a"] == 0.25
        assert log[0]["r_norm"]["laws/a"] == 0.5

    def test_engine_log_has_scale_and_audit_uses_it(self, rtol, atol):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
        )
        state_pred = {"phi_xx": jnp.array(1.0)}
        core.compute_residuals(state_pred)
        assert "scale" in core.log[-1]
        assert "laws/laplace_equation" in core.log[-1]["scale"]
        report = audit(core.log)
        r_norm = core.log[-1]["r_norm"]["laws/laplace_equation"]
        scale_k = core.log[-1]["scale"]["laws/laplace_equation"]
        rms = core.log[-1]["rms"]["laws/laplace_equation"]
        assert abs(r_norm - rms / scale_k) < 1e-6

    def test_audit_export_dir_pdf_with_new_categories(self, tmp_path):
        pytest.importorskip("reportlab")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/chain_dx": 0.5, "scaling/pe/chain_dx": 0.1}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/chain_dx": 0.25, "scaling/pe/chain_dx": 0.05}},
        ]
        report = audit(log, export_dir=str(tmp_path))
        assert "per_key" in report
        dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("audit_")]
        assert len(dirs) == 1
        assert (dirs[0] / "report.pdf").exists()


class TestVisualize:
    def test_visualize_empty_log_returns_none(self):
        assert visualize([], backend="matplotlib") is None

    def test_visualize_backend_none_returns_none(self):
        assert visualize([{"index": 0, "rms": {"k": 1.0}}], backend="none") is None


class TestResidualEngineStateBuilder:
    def test_groups_enrich_state(self, rtol, atol):
        core = ResidualEngine(
            constants={"L": 0.1},
            groups=[{"name": "re", "state_map": {"u": "u", "L": "L", "rho": "rho", "mu": "mu"}, "output_key": "re"}],
        )
        state_pred = {"u": 1.0, "rho": 1000.0, "mu": 1.0}
        state = core._state_builder(state_pred)
        assert "re" in state
        assert jnp.allclose(state["re"], 100.0, rtol=rtol, atol=atol)


class TestCustomFn:
    def test_custom_law_fn(self, rtol, atol):
        def my_residual(x):
            return x - 1.0

        core = ResidualEngine(
            laws=[{"name": "my_law", "state_map": {"x": "x"}, "fn": my_residual}],
        )
        state_pred = {"x": jnp.array(2.0)}
        residuals = core.compute_residuals(state_pred)
        assert jnp.allclose(residuals["laws"]["my_law"], 1.0, rtol=rtol, atol=atol)
        assert jnp.allclose(build_loss(residuals), 1.0, rtol=rtol, atol=atol)

    def test_constitutive_custom_closure(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            constitutive_custom=[{"name": "my_c", "fn": lambda s, c: s["a"] - 2.0 * s["b"]}],
        )
        state_pred = {"a": jnp.array(4.0), "b": jnp.array(2.0)}
        residuals = core.compute_residuals(state_pred)
        assert jnp.allclose(residuals["constitutive"]["custom/my_c"], 0.0, rtol=rtol, atol=atol)

    def test_scaling_custom_closure(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            scaling_custom=[{"name": "diff", "fn": lambda s, c: s["x"] - s["y"]}],
        )
        state_pred = {"x": jnp.array(1.0), "y": jnp.array(3.0)}
        residuals = core.compute_residuals(state_pred)
        assert jnp.allclose(residuals["scaling"]["custom/diff"], -2.0, rtol=rtol, atol=atol)

    def test_custom_group_fn_in_state(self, rtol, atol):
        def my_group(a, b):
            return a * b

        core = ResidualEngine(
            groups=[{"name": "my_ab", "state_map": {"a": "a", "b": "b"}, "output_key": "ab", "fn": my_group}],
        )
        state = core._state_builder({"a": jnp.array(3.0), "b": jnp.array(4.0)})
        assert jnp.allclose(state["ab"], 12.0, rtol=rtol, atol=atol)


class TestRegistryHelpers:
    def test_list_constitutive_models(self):
        names = list_constitutive_models()
        assert "sutherland_mu" in names
        assert "thermal_diffusivity" in names

    def test_list_scaling_closure_ids(self):
        ids = list_scaling_closure_ids()
        assert "pe" in ids
        assert "fo" in ids


class TestMonitorConfig:
    def test_to_from_dict_roundtrip(self):
        cfg = MonitorConfig(
            constants={"cp": 1.0},
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
            constitutive_audit=[
                AuditSpec(
                    name="sutherland_mu",
                    output_key="mu",
                    state_map={"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
                    predicted_temporal=["T"],
                )
            ],
        )
        d = cfg.to_dict()
        cfg2 = MonitorConfig.from_dict(d)
        assert cfg2.to_dict() == d


class TestRequiredKeys:
    def test_required_state_and_derivative_keys(self):
        engine = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                    "predicted_spatial": ["Re", "Pr"],
                    "predicted_temporal": ["Re"],
                }
            ],
        )
        state_keys = engine.required_state_keys()
        assert "phi_xx" in state_keys
        assert "Re" in state_keys and "Pr" in state_keys and "Pe" in state_keys

        deriv_keys = engine.required_derivative_keys()
        assert "d_Pe_dx" in deriv_keys
        assert "d_Re_dx" in deriv_keys
        assert "d_Pr_dx" in deriv_keys
        assert "d_Pe_dt" in deriv_keys
        assert "d_Re_dt" in deriv_keys

    def test_default_inference_uses_primary_fields(self, rtol, atol):
        engine = ResidualEngine(
            laws=[],
            primary_fields=["u", "T"],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "u", "pr": "T"},
                }
            ],
        )
        # No predicted_* provided: with collocation including x and t, engine should pick 'u' first.
        state_pred = {"u": 1.0, "T": 2.0, "Pe": 2.0, "d_u_dx": 0.0, "d_Pe_dx": 0.0}
        residuals = engine.compute_residuals(state_pred, collocation={"x": 0.0, "t": 0.0})
        # With u chosen for predicted_spatial, chain_dx exists only if derivative keys are present; here it is present.
        assert "scaling" in residuals


def _re_pi_spec(*, compare_keys=("out",), scale_c=10.0):
    return {
        "name": "re",
        "output_key": "Re",
        "state_map": {"u": "u", "L": "L", "rho": "rho", "mu": "mu"},
        "invariance_pi_constant": True,
        "invariance_compare_keys": list(compare_keys),
        "invariance_scale_c": scale_c,
    }


class TestPiConstantClosure:
    def test_list_pi_constant_group_names(self):
        names = list_pi_constant_group_names()
        assert "re" in names and "pr" in names and "pe" in names

    def test_apply_pi_constant_recipe_re_preserves_re(self):
        from moju.monitor.pi_constant_recipes import (
            GROUP_PI_CONSTANT_RECIPES,
            apply_pi_constant_recipe,
        )

        const = {"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0}
        sm = {"u": "u", "L": "L", "rho": "rho", "mu": "mu"}
        out = apply_pi_constant_recipe(const, GROUP_PI_CONSTANT_RECIPES["re"], sm, 10.0)
        Re0 = const["rho"] * const["u"] * const["L"] / const["mu"]
        Re1 = float(out["rho"] * out["u"] * out["L"] / out["mu"])
        assert abs(Re0 - Re1) < 1e-9

    def test_apply_pi_constant_recipe_c_must_exceed_one(self):
        from moju.monitor.pi_constant_recipes import (
            GROUP_PI_CONSTANT_RECIPES,
            apply_pi_constant_recipe,
        )

        const = {"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0}
        sm = {"u": "u", "L": "L", "rho": "rho", "mu": "mu"}
        with pytest.raises(ValueError, match="c > 1"):
            apply_pi_constant_recipe(const, GROUP_PI_CONSTANT_RECIPES["re"], sm, 1.0)

    def test_engine_init_requires_recipe_or_compare_keys(self):
        with pytest.raises(ValueError, match="invariance_compare_keys"):
            ResidualEngine(
                constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
                laws=[],
                groups=[],
                scaling_audit=[_re_pi_spec(compare_keys=())],
                state_builder=lambda m, p, col, ct: {"out": jnp.array(1.0)},
            )

    def test_engine_init_unsupported_group_for_pi(self):
        with pytest.raises(ValueError, match="recipe"):
            ResidualEngine(
                constants={"f": 1.0, "u": 1.0, "L": 1.0},
                laws=[],
                groups=[],
                scaling_audit=[
                    {
                        "name": "st",
                        "output_key": "St",
                        "state_map": {"f": "f", "u": "u", "L": "L"},
                        "invariance_pi_constant": True,
                        "invariance_compare_keys": ["x"],
                    }
                ],
                state_builder=lambda m, p, col, ct: {"x": jnp.array(1.0)},
            )

    def test_engine_init_invariance_c_must_exceed_one(self):
        with pytest.raises(ValueError, match="invariance_scale_c"):
            ResidualEngine(
                constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
                laws=[],
                groups=[],
                scaling_audit=[_re_pi_spec(scale_c=1.0)],
                state_builder=lambda m, p, col, ct: {"out": ct["L"] / ct["mu"]},
            )

    def test_path_b_forbidden_when_pi_enabled(self):
        def sb(model, params, collocation, constants):
            return {"out": constants["L"] / constants["mu"]}

        engine = ResidualEngine(
            constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
            laws=[],
            groups=[],
            scaling_audit=[_re_pi_spec()],
            state_builder=sb,
        )
        with pytest.raises(ValueError, match="Path A"):
            engine.compute_residuals({"out": jnp.array(0.5), "Re": jnp.array(0.25), "u": jnp.array(1.0)})

    def test_path_a_pi_residual_zero_when_invariant(self, rtol, atol):
        def sb(model, params, collocation, constants):
            return {"out": constants["L"] / constants["mu"]}

        engine = ResidualEngine(
            constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
            laws=[],
            groups=[],
            scaling_audit=[_re_pi_spec()],
            state_builder=sb,
        )
        residuals = engine.compute_residuals(None, model=0, params=0, collocation={})
        r = residuals["scaling"]["re/pi_constant"]
        assert jnp.allclose(r, 0.0, rtol=rtol, atol=atol)

    def test_pi_constant_scale_uses_mean_abs_scaled_branch(self):
        def sb(model, params, collocation, constants):
            return {"out": constants["L"]}

        engine = ResidualEngine(
            constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
            laws=[],
            groups=[],
            scaling_audit=[_re_pi_spec()],
            state_builder=sb,
        )
        engine.compute_residuals(None, model=0, params=0, collocation={})
        entry = engine.log[-1]
        scale = entry["scale"]["scaling/re/pi_constant"]
        assert scale > 5.0
        rms = entry["rms"]["scaling/re/pi_constant"]
        assert rms > 0.01


class TestAuditSpecPiFieldsRoundtrip:
    def test_monitor_config_scaling_audit_pi_fields(self):
        spec = AuditSpec(
            name="re",
            output_key="Re",
            state_map={"u": "u", "L": "L", "rho": "rho", "mu": "mu"},
            invariance_pi_constant=True,
            invariance_compare_keys=["out"],
            invariance_scale_c=7.0,
        )
        cfg = MonitorConfig(constants={}, scaling_audit=[spec])
        d = cfg.to_dict()
        cfg2 = MonitorConfig.from_dict(d)
        s2 = cfg2.scaling_audit[0]
        assert s2.invariance_pi_constant is True
        assert s2.invariance_compare_keys == ["out"]
        assert s2.invariance_scale_c == 7.0
