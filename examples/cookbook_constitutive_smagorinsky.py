#!/usr/bin/env python3
"""Cookbook: constitutive **ref_delta** for ``Models.smagorinsky_nu_t`` (Path B)."""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    Cs = jnp.array(0.1)
    Delta = jnp.array(0.02)
    x = jnp.linspace(0.0, 1.0, 32)
    S = 2.0 + 0.5 * x
    nu_t = Models.smagorinsky_nu_t(Cs, Delta, S)
    state_pred = {"Cs": Cs, "Delta": Delta, "S": S, "nu_t": nu_t}
    cfg = MonitorConfig(
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="smagorinsky_nu_t",
                output_key="nu_t",
                state_map={"Cs": "Cs", "Delta": "Delta", "strain_rate_magnitude": "S"},
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(state_pred, state_ref=dict(state_pred))
    report = audit(engine.log)
    flat_key = "constitutive/smagorinsky_nu_t/ref_delta"
    rms = engine.log[-1]["rms"][flat_key]
    return {"report": report, "ref_rms": float(rms), "residuals": residuals, "engine": engine, "flat_key": flat_key}


if __name__ == "__main__":
    out = main()
    print("ref_delta RMS:", out["ref_rms"])
