#!/usr/bin/env python3
"""
Phase A impact: compare legacy Path B FD (fd_order=None) vs proposed fd_order=4.

Reports residual RMS, overall score, and tier for analytic NS, Path B cookbooks,
and a paper-style Fourier Path B slab snapshot. Does not change package defaults.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic, taylor_green_2d
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.tiers import tier_for_score


def _ns_engine() -> ResidualEngine:
    return ResidualEngine(
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


def _run(
    name: str,
    engine: ResidualEngine,
    state: Dict[str, Any],
    grid: PathBGridConfig,
    *,
    fill_law_fd: bool = True,
    exact_residual: Optional[float] = None,
) -> Dict[str, Any]:
    engine.clear_log()
    res = engine.compute_residuals(
        state,
        run_mode="training",
        auto_path_b_derivatives=grid,
        fill_law_fd=fill_law_fd,
    )
    report = audit(engine.log)
    laws = res.get("laws") or {}
    law_rows = {}
    for ln, arr in laws.items():
        a = jnp.asarray(arr)
        rms = float(jnp.sqrt(jnp.mean(a**2))) if a.size else float("nan")
        mx = float(jnp.max(jnp.abs(a))) if a.size else float("nan")
        pk = report.get("per_key", {}).get(f"laws/{ln}", {})
        law_rows[ln] = {
            "residual_rms": rms,
            "residual_max": mx,
            "score": pk.get("admissibility_score"),
            "tier": pk.get("admissibility_level"),
        }
    overall = float(report.get("overall_admissibility_score") or 0.0)
    out = {
        "case": name,
        "fd_order": grid.fd_order,
        "overall_score": overall,
        "overall_tier": tier_for_score(overall) if math.isfinite(overall) else "Unknown",
        "laws": law_rows,
    }
    if exact_residual is not None:
        out["exact_residual_max"] = exact_residual
        # Distance of achieved residual_max to exact (0 for analytic with supplied ∂)
        for ln, row in law_rows.items():
            out.setdefault("closeness", {})[ln] = abs(
                float(row["residual_max"]) - float(exact_residual)
            )
    return out


def compare_pair(
    name: str,
    engine: ResidualEngine,
    state: Dict[str, Any],
    base_grid: PathBGridConfig,
    *,
    exact_max: Optional[float] = None,
) -> Dict[str, Any]:
    legacy = replace(base_grid, fd_order=None, fd_order_time=None)
    prop = replace(base_grid, fd_order=4, fd_order_time=4)
    a = _run(f"{name}/legacy", engine, state, legacy, exact_residual=exact_max)
    b = _run(f"{name}/fd4", engine, state, prop, exact_residual=exact_max)
    tier_changes = []
    for ln in set(a["laws"]) | set(b["laws"]):
        ta = (a["laws"].get(ln) or {}).get("tier")
        tb = (b["laws"].get(ln) or {}).get("tier")
        if ta != tb:
            tier_changes.append(
                {
                    "law": ln,
                    "legacy_tier": ta,
                    "fd4_tier": tb,
                    "legacy_rms": (a["laws"].get(ln) or {}).get("residual_rms"),
                    "fd4_rms": (b["laws"].get(ln) or {}).get("residual_rms"),
                    "legacy_score": (a["laws"].get(ln) or {}).get("score"),
                    "fd4_score": (b["laws"].get(ln) or {}).get("score"),
                }
            )
    if a["overall_tier"] != b["overall_tier"]:
        tier_changes.append(
            {
                "law": "__overall__",
                "legacy_tier": a["overall_tier"],
                "fd4_tier": b["overall_tier"],
                "legacy_score": a["overall_score"],
                "fd4_score": b["overall_score"],
            }
        )
    return {"legacy": a, "fd4": b, "tier_changes": tier_changes}


def case_taylor_green_path_b() -> Dict[str, Any]:
    n, t, re = 32, 0.1, 100.0
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    # Exact residual with supplied derivatives (reference)
    eng = _ns_engine()
    exact_state = ns_state_from_analytic(fields, x, y, t=t)
    eng.compute_residuals(exact_state, run_mode="training")
    exact_r = eng.last_residuals["laws"]["momentum_navier_stokes"]
    exact_max = float(jnp.max(jnp.abs(exact_r)))

    # Path B fill from primitives (+ exact u_t so spatial FD is the variable)
    state = {
        "x": x,
        "y": y,
        "u": fields["u"],
        "u_t": fields["u_t"],
        "p": fields["p"],
        "re": fields["re"],
    }
    grid = PathBGridConfig(layout="meshgrid", spatial_dimension=2, steady=True)
    return compare_pair(
        "taylor_green_path_b_fd",
        _ns_engine(),
        state,
        grid,
        exact_max=exact_max,
    )


def case_kovasznay_path_b() -> Dict[str, Any]:
    nx, ny, re = 40, 40, 40.0
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=re)
    eng = _ns_engine()
    exact_state = ns_state_from_analytic(fields, x, y)
    eng.compute_residuals(exact_state, run_mode="training")
    exact_max = float(
        jnp.max(jnp.abs(eng.last_residuals["laws"]["momentum_navier_stokes"]))
    )
    state = {
        "x": x,
        "y": y,
        "u": fields["u"],
        "u_t": fields["u_t"],
        "p": fields["p"],
        "re": fields["re"],
    }
    grid = PathBGridConfig(layout="meshgrid", spatial_dimension=2, steady=True)
    return compare_pair(
        "kovasznay_path_b_fd", _ns_engine(), state, grid, exact_max=exact_max
    )


def case_laplace_1d_path_b() -> Dict[str, Any]:
    n = 64
    x = jnp.linspace(0.0, 1.0, n)
    phi = jnp.sin(jnp.pi * x)
    state = {"x": x, "phi": phi}
    eng = ResidualEngine(
        constants={},
        laws=[
            {
                "name": "laplace_equation",
                "state_map": {"phi_laplacian": "phi_laplacian"},
            }
        ],
        groups=[],
        law_implied_audits=False,
        default_coord_dimension=1,
    )
    # Exact laplacian = -π² sin(πx); residual of laplace is phi_laplacian
    exact_max = 0.0  # should be near -π²φ + 0 wait: law is just laplacian = 0 for Laplace
    # For phi=sin(πx), lap = -π² sin ≠ 0, so residual is large — still useful for FD compare
    grid = PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True)
    return compare_pair("laplace_1d_sine_path_b", eng, state, grid, exact_max=None)


def case_paper_style_fourier_path_b() -> Dict[str, Any]:
    """1D slab-like Fourier Path B with FD-filled T_laplacian (paper family)."""
    L, k, rho, cp = 0.02, 200.0, 2700.0, 900.0
    alpha = k / (rho * cp)
    n = 64
    x = jnp.linspace(0.0, L, n)
    t = jnp.ones_like(x) * 10.0
    T = 300.0 + 20.0 * (1.0 - x / L) ** 2
    T_xx = jnp.ones_like(x) * (40.0 / (L**2))  # exact
    T_t = alpha * T_xx
    # Primitives only for laplacian fill
    state = {
        "x": x,
        "t": t,
        "T": T,
        "T_t": T_t,
        "L": jnp.ones_like(x) * L,
        "k": jnp.ones_like(x) * k,
        "rho": jnp.ones_like(x) * rho,
        "cp": jnp.ones_like(x) * cp,
        "alpha": jnp.ones_like(x) * alpha,
    }
    from moju.monitor import implied_group_specs_for_laws

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
        constants={},
        laws=laws,
        groups=groups,
        law_implied_audits=False,
        default_coord_dimension=1,
    )
    grid = PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True)
    # Exact with supplied T_laplacian
    eng_ex = ResidualEngine(
        constants={},
        laws=laws,
        groups=groups,
        law_implied_audits=False,
        default_coord_dimension=1,
    )
    st_ex = dict(state)
    st_ex["T_laplacian"] = T_xx
    eng_ex.compute_residuals(st_ex, run_mode="training")
    exact_max = float(
        jnp.max(jnp.abs(eng_ex.last_residuals["laws"]["fourier_conduction"]))
    )
    return compare_pair(
        "paper_style_fourier_path_b", eng, state, grid, exact_max=exact_max
    )


def case_tg_supplied_exact() -> Dict[str, Any]:
    """Control: supplied exact derivatives — FD order must not change scores."""
    n, t, re = 32, 0.1, 100.0
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    state = ns_state_from_analytic(fields, x, y, t=t)
    eng = _ns_engine()
    eng.compute_residuals(state, run_mode="training")
    r1 = audit(eng.log)
    eng2 = _ns_engine()
    # Even if a grid is passed, non-None derivatives are not overwritten
    eng2.compute_residuals(
        state,
        run_mode="training",
        auto_path_b_derivatives=PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order=4
        ),
        fill_law_fd=True,
    )
    r2 = audit(eng2.log)
    return {
        "legacy": {
            "overall_score": r1["overall_admissibility_score"],
            "overall_tier": r1["overall_admissibility_level"],
        },
        "fd4": {
            "overall_score": r2["overall_admissibility_score"],
            "overall_tier": r2["overall_admissibility_level"],
        },
        "tier_changes": [],
        "note": "supplied derivatives; Path B must not overwrite — scores identical",
        "scores_equal": abs(
            float(r1["overall_admissibility_score"])
            - float(r2["overall_admissibility_score"])
        )
        < 1e-12,
    }


def main() -> None:
    cases = [
        ("tg_supplied_exact", case_tg_supplied_exact),
        ("taylor_green_path_b", case_taylor_green_path_b),
        ("kovasznay_path_b", case_kovasznay_path_b),
        ("laplace_1d_path_b", case_laplace_1d_path_b),
        ("paper_style_fourier_path_b", case_paper_style_fourier_path_b),
    ]
    report: Dict[str, Any] = {"cases": {}, "all_tier_changes": []}
    for key, fn in cases:
        print(f"running {key}...")
        result = fn()
        report["cases"][key] = result
        for ch in result.get("tier_changes") or []:
            report["all_tier_changes"].append({"case": key, **ch})
        leg = result.get("legacy") or {}
        fd4 = result.get("fd4") or {}
        print(
            f"  legacy tier={leg.get('overall_tier')} score={leg.get('overall_score')} | "
            f"fd4 tier={fd4.get('overall_tier')} score={fd4.get('overall_score')} | "
            f"tier_changes={len(result.get('tier_changes') or [])}"
        )
    out_path = "docs/phase_a_fd_impact_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {out_path}")
    print("tier changes total:", len(report["all_tier_changes"]))


if __name__ == "__main__":
    main()
