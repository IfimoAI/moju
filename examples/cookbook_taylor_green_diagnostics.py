#!/usr/bin/env python3
"""Taylor–Green vortex: exact derivatives, term breakdown, and consistency metadata."""

from __future__ import annotations

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import ns_state_from_analytic, taylor_green_2d


def main() -> None:
    n = 48
    re = 100.0
    t = 0.2
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    state = ns_state_from_analytic(fields, x, y, t=t)

    engine = ResidualEngine(
        constants={},
        laws=[
            {
                "name": "momentum_navier_stokes",
                "state_map": {
                    "u_t": "u_t",
                    "u": "u",
                    "u_grad": "u_grad",
                    "p_grad": "p_grad",
                    "u_laplacian": "u_laplacian",
                    "re": "re",
                },
            }
        ],
        groups=[],
        law_implied_audits=False,
        default_coord_dimension=2,
    )

    meta = {
        "u_t": {
            "source": "solver",
            "evaluation_time": float(t),
            "time_stencil": [float(t)],
        },
        "u_grad": {"source": "solver", "evaluation_time": float(t)},
        "u_laplacian": {"source": "solver", "evaluation_time": float(t)},
        "p_grad": {"source": "solver", "evaluation_time": float(t)},
    }
    residuals = engine.compute_residuals(
        state, run_mode="training", derivative_metadata=meta
    )
    report = audit(engine.log)

    r = residuals["laws"]["momentum_navier_stokes"]
    print("max |NS residual| (exact derivatives):", float(jnp.max(jnp.abs(r))))
    bd = report["term_breakdown"]["momentum_navier_stokes"]
    print("term breakdown keys:", sorted(bd["terms"]))
    print("residual RMS:", bd["residual"]["rms"])
    print("residual / largest term:", bd["residual_over_largest_term"])
    print("scale_k:", bd["scale_k"], bd["scale_source"])
    print("overall admissibility:", report["overall_admissibility_score"])
    print("diagnostic warnings:", report.get("diagnostic_warnings") or [])


if __name__ == "__main__":
    main()
