#!/usr/bin/env python3
"""Cookbook: constitutive **ref_delta** for ``Models.law_of_the_wall`` (Path B)."""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    y_plus = jnp.linspace(30.0, 100.0, 24)
    u_plus = Models.law_of_the_wall(y_plus)
    state_pred = {"y_plus": y_plus, "u_plus": u_plus}
    cfg = MonitorConfig(
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="law_of_the_wall",
                output_key="u_plus",
                state_map={"y_plus": "y_plus"},
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(state_pred, state_ref=dict(state_pred))
    report = audit(engine.log)
    flat_key = "constitutive/law_of_the_wall/ref_delta"
    rms = engine.log[-1]["rms"][flat_key]
    return {"report": report, "ref_rms": float(rms), "residuals": residuals, "engine": engine, "flat_key": flat_key}


if __name__ == "__main__":
    out = main()
    print("ref_delta RMS:", out["ref_rms"])
