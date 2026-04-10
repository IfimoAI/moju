#!/usr/bin/env python3
"""Cookbook: constitutive **ref_delta** for ``Models.colebrook_friction`` (Path B)."""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    Re = jnp.linspace(4000.0, 20000.0, 16)
    rr = jnp.array(1e-4)
    f = Models.colebrook_friction(Re, rr)
    state_pred = {"Re": Re, "rr": rr, "f": f}
    cfg = MonitorConfig(
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="colebrook_friction",
                output_key="f",
                state_map={"re": "Re", "epsilon_d": "rr"},
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(
        state_pred, state_ref=dict(state_pred), run_mode="eval"
    )
    report = audit(engine.log)
    flat_key = "constitutive/colebrook_friction/ref_delta"
    rms = engine.log[-1]["rms"][flat_key]
    return {"report": report, "ref_rms": float(rms), "residuals": residuals, "engine": engine, "flat_key": flat_key}


if __name__ == "__main__":
    out = main()
    print("ref_delta RMS:", out["ref_rms"])
