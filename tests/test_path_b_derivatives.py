"""Tests for Path B finite-difference derivative fill and derivative key collection."""

import jax.numpy as jnp
import pytest

from moju.monitor.derivative_keys import (
    audit_derivative_keys_for_spec,
    collect_audit_derivative_keys,
    derivative_state_key,
)
from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives


class TestDerivativeKeys:
    def test_derivative_state_key(self):
        assert derivative_state_key("T", "x") == "d_T_dx"
        assert derivative_state_key("T", "y") == "d_T_dy"
        assert derivative_state_key("mu", "t") == "d_mu_dt"

    def test_collect_multi_axis(self):
        spec = {
            "output_key": "mu",
            "predicted_spatial": ["T"],
            "predicted_temporal": [],
            "chain_spatial_axes": ["x", "y"],
        }
        sx, st = audit_derivative_keys_for_spec(spec)
        assert "d_mu_dx" in sx
        assert "d_T_dx" in sx
        assert "d_mu_dy" in sx
        assert "d_T_dy" in sx
        assert not st

    def test_collect_audit_derivative_keys_union(self):
        a = [
            {
                "name": "m",
                "output_key": "mu",
                "predicted_spatial": ["T"],
                "predicted_temporal": ["T"],
                "chain_spatial_axes": ["x"],
            }
        ]
        sx, st = collect_audit_derivative_keys(a, [])
        assert sx == {"d_mu_dx", "d_T_dx"}
        assert st == {"d_mu_dt", "d_T_dt"}


class TestFillPathBDerivatives:
    def test_meshgrid_1d_quadratic_Re(self):
        x = jnp.linspace(0.0, 1.0, 16)
        Re = x**2
        Pr = jnp.array(0.7)
        Pe = Re * Pr
        state = {"Re": Re, "Pe": Pe, "Pr": Pr, "x": x}
        audit = [
            {
                "name": "pe",
                "output_key": "Pe",
                "state_map": {"re": "Re", "pr": "Pr"},
                "predicted_spatial": ["Re"],
            }
        ]
        out, w = fill_path_b_derivatives(
            state,
            constitutive_audit=[],
            scaling_audit=audit,
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert not w
        assert "d_Re_dx" in out
        expect = jnp.gradient(Re, x)
        assert jnp.allclose(out["d_Re_dx"], expect, rtol=1e-5, atol=1e-5)

    def test_skip_existing_non_none(self):
        x = jnp.linspace(0.0, 1.0, 8)
        T = x**2
        sentinel = jnp.full_like(T, 99.0)
        state = {"T": T, "x": x, "d_T_dx": sentinel}
        audit = [
            {
                "name": "sutherland_mu",
                "output_key": "mu",
                "state_map": {"T": "T"},
                "predicted_spatial": ["T"],
            }
        ]
        out, _ = fill_path_b_derivatives(
            state,
            constitutive_audit=audit,
            scaling_audit=[],
            grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
        )
        assert jnp.allclose(out["d_T_dx"], sentinel)

    def test_separable_2d_gradient_y(self):
        nx, ny = 8, 10
        x = jnp.linspace(0.0, 1.0, nx)
        y = jnp.linspace(0.0, 1.0, ny)
        X = x[:, None]
        Y = y[None, :]
        T = X * Y**2
        state = {"T": T, "x": x, "y": y}
        spec = {
            "name": "m",
            "output_key": "T",
            "state_map": {"T": "T"},
            "predicted_spatial": ["T"],
            "chain_spatial_axes": ["y"],
        }
        out, w = fill_path_b_derivatives(
            state,
            constitutive_audit=[spec],
            scaling_audit=[],
            grid=PathBGridConfig(layout="separable", spatial_dimension=2, steady=True),
        )
        assert not w
        assert "d_T_dy" in out
        _, dTdy = jnp.gradient(T, x, y)
        assert jnp.allclose(out["d_T_dy"], dTdy, rtol=1e-4, atol=1e-4)

    def test_unsteady_dt(self):
        t = jnp.linspace(0.0, 1.0, 9)
        nx = 5
        x = jnp.linspace(0.0, 1.0, nx)
        # T(t, x) = t^2 * x
        T = (t[:, None] ** 2) * x[None, :]
        mu = T**2  # placeholder output field for audit derivative keys
        state = {"T": T, "mu": mu, "t": t, "x": x}
        spec = {
            "name": "m",
            "output_key": "mu",
            "state_map": {"T": "T"},
            "predicted_temporal": ["T"],
        }
        out, w = fill_path_b_derivatives(
            state,
            constitutive_audit=[spec],
            scaling_audit=[],
            grid=PathBGridConfig(
                layout="meshgrid",
                spatial_dimension=2,
                steady=False,
            ),
        )
        assert not w
        assert "d_T_dt" in out
        expect = 2.0 * t[:, None] * x[None, :]
        assert jnp.allclose(
            out["d_T_dt"][1:-1], expect[1:-1], rtol=1e-4, atol=1e-4
        )


class TestAuditSpecChainSpatialAxes:
    def test_to_from_dict_roundtrip(self):
        from moju.monitor import AuditSpec

        s = AuditSpec(
            name="pe",
            output_key="Pe",
            state_map={"re": "Re", "pr": "Pr"},
            predicted_spatial=["Re"],
            chain_spatial_axes=["x", "y"],
        )
        d = s.to_dict()
        assert d["chain_spatial_axes"] == ["x", "y"]
        s2 = AuditSpec.from_dict(d)
        assert list(s2.chain_spatial_axes) == ["x", "y"]


class TestResidualEngineMultiAxis:
    def test_invalid_chain_spatial_axes_raises(self):
        from moju.monitor import ResidualEngine

        with pytest.raises(ValueError, match="chain_spatial_axes"):
            ResidualEngine(
                laws=[],
                scaling_audit=[
                    {
                        "name": "pe",
                        "output_key": "Pe",
                        "state_map": {"re": "Re", "pr": "Pr"},
                        "predicted_spatial": ["Re"],
                        "chain_spatial_axes": ["x", "bogus"],
                    }
                ],
            )

    def test_pe_chain_dx_and_chain_dy_zero(self, rtol, atol):
        from moju.monitor import ResidualEngine

        nx, ny = 6, 7
        Re = jnp.broadcast_to(jnp.linspace(0.0, 1.0, nx)[:, None], (nx, ny))
        Pr = jnp.array(0.7)
        Pe = Re * Pr
        d_Re_dx, d_Re_dy = jnp.gradient(Re)
        d_Pe_dx = Pr * d_Re_dx
        d_Pe_dy = Pr * d_Re_dy
        core = ResidualEngine(
            laws=[],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                    "predicted_spatial": ["Re"],
                    "chain_spatial_axes": ["x", "y"],
                }
            ],
        )
        state_pred = {
            "Re": Re,
            "Pe": Pe,
            "Pr": Pr,
            "d_Re_dx": d_Re_dx,
            "d_Pe_dx": d_Pe_dx,
            "d_Re_dy": d_Re_dy,
            "d_Pe_dy": d_Pe_dy,
        }
        r = core.compute_residuals(state_pred)
        assert jnp.allclose(r["scaling"]["pe/chain_dx"], 0.0, rtol=rtol, atol=atol)
        assert jnp.allclose(r["scaling"]["pe/chain_dy"], 0.0, rtol=rtol, atol=atol)

    def test_auto_path_b_derivatives_pe(self, rtol, atol):
        from moju.monitor import ResidualEngine

        x = jnp.linspace(0.0, 1.0, 24)
        Re = x**2
        Pr = jnp.array(0.7)
        Pe = Re * Pr
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
        state_pred = {"Re": Re, "Pe": Pe, "Pr": Pr, "x": x}
        r = core.compute_residuals(state_pred, auto_path_b_derivatives=True)
        assert jnp.allclose(
            r["scaling"]["pe/chain_dx"], 0.0, rtol=rtol, atol=max(atol, 1e-5)
        )

    def test_required_derivative_keys_respects_axes(self):
        from moju.monitor import ResidualEngine

        core = ResidualEngine(
            laws=[],
            constitutive_audit=[
                {
                    "name": "sutherland_mu",
                    "output_key": "mu",
                    "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
                    "predicted_spatial": ["T"],
                    "chain_spatial_axes": ["x", "z"],
                }
            ],
            constants={"mu0": 1.0, "T0": 1.0, "S": 1.0},
        )
        keys = core.required_derivative_keys()
        assert "d_T_dx" in keys
        assert "d_mu_dx" in keys
        assert "d_T_dz" in keys
        assert "d_mu_dz" in keys
        assert "d_T_dy" not in keys
