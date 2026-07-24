#!/usr/bin/env python3
"""
Path B cookbook: periodic Fourier (spectral) fill for Burgers law inputs.

Prefer this over FD on periodic grids (e.g. FNO-style fields). Temporal ``u_t``
is still FD if filled by Moju; here we supply ``u_t`` explicitly.

Canonical docs: ``docs/path_b_derivatives.md``.
Also: ``fill_path_b_spectral`` or
``PathBGridConfig(diff_method=\"spectral\", periodic=True)`` with
``compute_residuals(..., fill_law_fd=True)``. Bare ``auto_path_b_derivatives=True``
stays finite-difference and never enables spectral.

Run::

    python examples/cookbook_path_b_spectral_burgers.py
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from moju.monitor import PathBGridConfig, ResidualEngine, audit, fill_path_b_spectral
from moju.piratio.laws import Laws


def main() -> None:
    n = 64
    L = 2.0 * math.pi
    x = jnp.linspace(0.0, L, n, endpoint=False)
    # Single Fourier mode; spectral ∂ recovers cos / -sin exactly (float32 tol).
    u = jnp.sin(x)[:, None]
    u_t = jnp.zeros_like(u)
    re = 100.0
    U = 1.0

    laws = [
        {
            "name": "burgers_equation",
            "state_map": {
                "u_t": "u_t",
                "u": "u",
                "u_grad": "u_grad",
                "u_laplacian": "u_laplacian",
                "re": "re",
                "U": "U",
                "L": "L",
            },
            "fn": Laws.burgers_equation,
        }
    ]

    # Standalone spectral fill (also available via ResidualEngine + PathBGridConfig).
    filled, warns = fill_path_b_spectral(
        {"u": u, "u_t": u_t, "x": x},
        laws_spec=laws,
        constants={"re": re, "U": U, "L": L},
        grid=PathBGridConfig(spatial_dimension=1, layout="separable", steady=True),
    )
    if warns:
        print("fill warnings:", warns)
    ux = filled["u_grad"][..., 0, 0]
    uxx = filled["u_laplacian"][..., 0]
    print("max |u_x - cos(x)|:", float(jnp.max(jnp.abs(ux - jnp.cos(x)))))
    print("max |u_xx + sin(x)|:", float(jnp.max(jnp.abs(uxx + jnp.sin(x)))))

    engine = ResidualEngine(
        laws=laws,
        constants={"re": re, "U": U, "L": L},
        default_coord_dimension=1,
        law_implied_audits=True,
    )
    grid = PathBGridConfig(
        diff_method="spectral",
        periodic=True,
        spatial_dimension=1,
        layout="separable",
        steady=True,
    )
    residuals = engine.compute_residuals(
        {"u": u, "u_t": u_t, "x": x},
        auto_path_b_derivatives=grid,
        fill_law_fd=True,
        log_to_python=True,
    )
    report = audit(engine.log)
    print("laws keys:", sorted((residuals.get("laws") or {}).keys()))
    print("overall admissibility:", report.get("overall_admissibility"))
    print("OK — spectral Path B Burgers smoke complete.")


if __name__ == "__main__":
    main()
