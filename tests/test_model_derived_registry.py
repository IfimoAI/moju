"""Tests for ``moju.monitor.model_derived_registry`` and ResidualEngine auto derived-state."""

import jax.numpy as jnp

from moju.monitor import (
    ResidualEngine,
    enrich_derived_state_from_constitutive_audits,
    implied_group_specs_for_laws,
)


def test_enrich_appends_mass_diffusivity_when_D_needed():
    audits = [
        {
            "name": "mass_diffusivity",
            "output_key": "D",
            "state_map": {"fo_mass": "fo_mass", "t": "t", "L": "L"},
        }
    ]
    groups = [{"name": "fo_mass", "output_key": "fo_mass", "state_map": {"D": "D", "t": "t", "L": "L"}}]
    chain = enrich_derived_state_from_constitutive_audits(audits, groups, [])
    assert len(chain) == 1
    assert chain[0]["output_key"] == "D"
    assert chain[0]["expr"]["op"] == "div"


def test_residual_engine_fourier_materializes_alpha_without_npz_alpha():
    laws = [
        {
            "name": "fourier_conduction",
            "state_map": {
                "T_t": "T_t",
                "T_laplacian": "T_laplacian",
                "fo": "fo",
                "t": "t",
                "L": "L",
            },
        }
    ]
    groups = implied_group_specs_for_laws(["fourier_conduction"])
    eng = ResidualEngine(
        laws=laws,
        groups=groups,
        constants={
            "k": jnp.asarray(401.0),
            "rho": jnp.asarray(8960.0),
            "cp": jnp.asarray(385.0),
            "L": jnp.asarray(0.01),
        },
        law_implied_audits=True,
        best_effort_partial=False,
    )
    assert any(s.get("output_key") == "alpha" for s in eng.derived_state_chain)

    x = jnp.linspace(0.0, 1.0, 8)
    T = jnp.sin(jnp.pi * x)
    T_t = jnp.zeros_like(T)
    T_laplacian = -(jnp.pi**2) * T
    t = jnp.asarray(10.0)
    state = {"T": T, "T_t": T_t, "T_laplacian": T_laplacian, "t": t, "x": x}
    res = eng.compute_residuals(state, log_to_python=False)
    assert "fourier_conduction" in res["laws"]
    assert jnp.all(jnp.isfinite(res["laws"]["fourier_conduction"]))


def test_residual_engine_user_fns_k_before_alpha_chain():
    laws = [
        {
            "name": "fourier_conduction",
            "state_map": {
                "T_t": "T_t",
                "T_laplacian": "T_laplacian",
                "fo": "fo",
                "t": "t",
                "L": "L",
            },
        }
    ]
    groups = implied_group_specs_for_laws(["fourier_conduction"])

    def k_of(T):
        return 400.0 + 0.1 * jnp.asarray(T)

    eng = ResidualEngine(
        laws=laws,
        groups=groups,
        constants={
            "rho": jnp.asarray(8960.0),
            "cp": jnp.asarray(385.0),
            "L": jnp.asarray(0.01),
        },
        user_fns={"k": k_of},
        law_implied_audits=True,
        best_effort_partial=False,
    )
    x = jnp.linspace(0.0, 1.0, 8)
    T = jnp.sin(jnp.pi * x)
    T_t = jnp.zeros_like(T)
    T_laplacian = -(jnp.pi**2) * T
    t = jnp.asarray(10.0)
    state = {"T": T, "T_t": T_t, "T_laplacian": T_laplacian, "t": t, "x": x}
    res = eng.compute_residuals(state, log_to_python=False)
    assert "fourier_conduction" in res["laws"]
    assert jnp.all(jnp.isfinite(res["laws"]["fourier_conduction"]))
