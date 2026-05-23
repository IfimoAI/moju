"""Tests for moju.monitor.nondim_inference and Path B dimensional pipeline."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from moju.monitor.auditor import ResidualEngine
from moju.monitor.nondim_inference import infer_nondim_scales, resolve_time_scale_for_laws
from moju.piratio.groups import Groups


class TestNondimInference:
    def test_infer_fourier_from_L_and_alpha(self):
        state = {
            "L": 0.05,
            "alpha": 1e-6,
            "T": jnp.linspace(300.0, 350.0, 5),
            "x": jnp.linspace(0.0, 0.05, 5),
        }
        scales, prov = infer_nondim_scales(
            ["fourier_conduction"], state, {}, {"T0": 300.0}
        )
        assert scales.time_scale == "fourier"
        assert abs(scales.L_ref - 0.05) < 1e-6
        assert abs(scales.alpha_ref - 1e-6) / 1e-6 < 1e-6
        assert "L_ref" in prov

    def test_alpha_derived_from_k_rho_cp(self):
        state = {
            "L": 0.1,
            "k": 200.0,
            "rho": 7800.0,
            "cp": 500.0,
            "T": jnp.array([300.0, 310.0]),
        }
        scales, prov = infer_nondim_scales(["fourier_conduction"], state, {}, None)
        expected = 200.0 / (7800.0 * 500.0)
        assert abs(scales.alpha_ref - expected) / expected < 1e-6
        assert prov.get("alpha_ref") == "k/(rho*cp)"

    def test_missing_L_ref_raises(self):
        with pytest.raises(ValueError, match="L_ref"):
            infer_nondim_scales(["fourier_conduction"], {"T": jnp.array([300.0])}, {}, None)

    def test_time_scale_conflict_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            resolve_time_scale_for_laws(["fourier_conduction", "momentum_navier_stokes"])

    def test_convective_infer_u_from_rms(self):
        state = {"L": 0.2, "u": jnp.array([1.0, 2.0, 3.0])}
        scales, prov = infer_nondim_scales(["momentum_navier_stokes"], state, {}, None)
        assert scales.time_scale == "convective"
        assert prov.get("U_ref") == "rms(u,v,w)"


class TestDimensionalPathBIntegration:
    def test_fourier_dimensional_slab(self):
        L = 0.05
        alpha = 1e-6
        nx = 8
        x = jnp.linspace(0.0, L, nx)
        T = 300.0 + 10.0 * jnp.sin(jnp.pi * x / L)
        T_laplacian = -(jnp.pi / L) ** 2 * 10.0 * jnp.sin(jnp.pi * x / L)
        t = jnp.array(100.0)
        fo = float(Groups.fo(alpha, t, L))

        state = {
            "x": x,
            "T": T,
            "T_laplacian": T_laplacian,
            "T_t": jnp.zeros_like(T),
            "t": t,
            "L": L,
            "alpha": alpha,
            "fo": fo,
        }
        engine = ResidualEngine(
            laws=[
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
            ],
            groups=[
                {
                    "name": "fo",
                    "output_key": "fo",
                    "state_map": {"alpha": "alpha", "t": "t", "L": "L"},
                }
            ],
        )
        engine.compute_residuals(state, state_units="dimensional")
        entry = engine.log[-1]
        assert "nondim_scales" in entry
        assert entry["nondim_scale_source"]
        rms = entry["rms"]["laws/fourier_conduction"]
        assert jnp.isfinite(jnp.asarray(rms))
