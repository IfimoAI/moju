#!/usr/bin/env python3
"""
Phase A2 impact: legacy vs fd_order=4 (unweighted) vs fd_order=4 (weighted),
plus curvilinear Couette, second-derivative checks, controls, and w2 paper Path B.

Does not change package defaults. Extends docs/phase_a_fd_impact_report.json.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit, implied_group_specs_for_laws
from moju.monitor.analytic_flows import (
    kovasznay_2d,
    ns_state_from_analytic,
    taylor_couette_2d,
    taylor_green_2d,
)
from moju.monitor.curvilinear import (
    fill_vector_grad_lap_curvilinear_2d,
    gradient_curvilinear_2d,
)
from moju.monitor.fd_stencils import diff2_uniform
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.tiers import tier_for_score
from moju.monitor.weighted_norms import structured_point_weights
from moju.piratio import Models


ROOT = Path(__file__).resolve().parents[1]


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


def _summarize(engine: ResidualEngine, res: Dict[str, Any]) -> Dict[str, Any]:
    report = audit(engine.log)
    laws = res.get("laws") or {}
    law_rows = {}
    entry = engine.log[-1] if engine.log else {}
    for ln, arr in laws.items():
        a = jnp.asarray(arr)
        rms = float(jnp.sqrt(jnp.mean(a**2))) if a.size else float("nan")
        mx = float(jnp.max(jnp.abs(a))) if a.size else float("nan")
        pk = report.get("per_key", {}).get(f"laws/{ln}", {})
        row = {
            "residual_rms": rms,
            "residual_max": mx,
            "score": pk.get("admissibility_score"),
            "tier": pk.get("admissibility_level"),
        }
        rw = (entry.get("rms_weighted") or {}).get(f"laws/{ln}")
        if rw is not None:
            row["residual_rms_weighted"] = float(rw)
        ru = (entry.get("rms_unweighted") or {}).get(f"laws/{ln}")
        if ru is not None:
            row["residual_rms_unweighted"] = float(ru)
        law_rows[ln] = row
    overall = float(report.get("overall_admissibility_score") or 0.0)
    return {
        "overall_score": overall,
        "overall_tier": tier_for_score(overall) if math.isfinite(overall) else "Unknown",
        "laws": law_rows,
        "rms_mode": (entry.get("weighted_norms") or {}).get("mode"),
        "scoring_uses_weights": (entry.get("weighted_norms") or {}).get(
            "scoring_uses_weights"
        ),
    }


def _run_triple(
    name: str,
    engine: ResidualEngine,
    state: Dict[str, Any],
    base_grid: PathBGridConfig,
    *,
    fill_law_fd: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"case": name}
    configs = [
        ("legacy", replace(base_grid, fd_order=None, fd_order_time=None), False),
        ("fd4_unweighted", replace(base_grid, fd_order=4, fd_order_time=4), False),
        ("fd4_weighted", replace(base_grid, fd_order=4, fd_order_time=4), True),
    ]
    for label, grid, use_w in configs:
        engine.clear_log()
        res = engine.compute_residuals(
            state,
            run_mode="training",
            auto_path_b_derivatives=grid,
            fill_law_fd=fill_law_fd,
            weighted_norms="score" if use_w else "report",
            resolution_sensor=(label == "fd4_unweighted"),
        )
        summary = _summarize(engine, res)
        summary["fd_order"] = grid.fd_order
        if engine.log and engine.log[-1].get("resolution_sensor"):
            summary["resolution_sensor"] = engine.log[-1]["resolution_sensor"]
        out[label] = summary
    # Tier / score deltas vs legacy
    changes = []
    for label in ("fd4_unweighted", "fd4_weighted"):
        a, b = out["legacy"], out[label]
        if a["overall_tier"] != b["overall_tier"]:
            changes.append(
                {
                    "compare": label,
                    "legacy_tier": a["overall_tier"],
                    "new_tier": b["overall_tier"],
                    "legacy_score": a["overall_score"],
                    "new_score": b["overall_score"],
                }
            )
    out["tier_changes"] = changes
    return out


def case_taylor_green() -> Dict[str, Any]:
    n, t, re = 32, 0.1, 100.0
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    state = {
        "x": x,
        "y": y,
        "u": fields["u"],
        "u_t": fields["u_t"],
        "p": fields["p"],
        "re": fields["re"],
    }
    grid = PathBGridConfig(layout="meshgrid", spatial_dimension=2, steady=True)
    return _run_triple("taylor_green_path_b_fd", _ns_engine(), state, grid)


def case_kovasznay() -> Dict[str, Any]:
    nx, ny, re = 40, 40, 40.0
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=re)
    state = {
        "x": x,
        "y": y,
        "u": fields["u"],
        "u_t": fields["u_t"],
        "p": fields["p"],
        "re": fields["re"],
    }
    grid = PathBGridConfig(layout="meshgrid", spatial_dimension=2, steady=True)
    return _run_triple("kovasznay_path_b_fd", _ns_engine(), state, grid)


def case_laplace() -> Dict[str, Any]:
    n = 64
    x = jnp.linspace(0.0, 1.0, n)
    phi = jnp.sin(jnp.pi * x)
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
    grid = PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True)
    return _run_triple("laplace_1d_sine_path_b", eng, {"x": x, "phi": phi}, grid)


def case_controls() -> Dict[str, Any]:
    """Perturbed TG / Kovasznay with known nonzero residual — must stay below High at fd4."""
    rows = {}
    # TG + p_grad bias
    n, t, re = 32, 0.1, 100.0
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    st = ns_state_from_analytic(fields, x, y, t=t)
    st["p_grad"] = st["p_grad"] + 0.2
    eng = _ns_engine()
    eng.compute_residuals(st, run_mode="training")
    rows["tg_perturbed_p_grad"] = _summarize(eng, eng.last_residuals)

    # Kovasznay * 1.25
    nx, ny = 40, 40
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    st = ns_state_from_analytic(fields, x, y)
    st["u"] = st["u"] * 1.25
    eng = _ns_engine()
    eng.compute_residuals(st, run_mode="training")
    rows["kovasznay_perturbed_u"] = _summarize(eng, eng.last_residuals)
    for k, v in rows.items():
        v["stays_below_high"] = float(v["overall_score"]) < 0.909
    return rows


def case_curvilinear_couette() -> Dict[str, Any]:
    out = {"resolutions": []}
    for nr, nt in ((16, 24), (24, 36), (32, 48)):
        r = jnp.linspace(1.0, 2.0, nr)
        th = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
        fields = taylor_couette_2d(r, th, re=100.0)
        x, y = fields["x"], fields["y"]
        hr = float(r[1] - r[0])
        ht = float(th[1] - th[0])
        filled = fill_vector_grad_lap_curvilinear_2d(
            fields["u"], x, y, d_xi=hr, d_eta=ht, order=4
        )
        p_x, p_y = gradient_curvilinear_2d(
            fields["p"], x, y, d_xi=hr, d_eta=ht, order=4
        )
        pad = 3
        grad_err = float(
            jnp.max(
                jnp.abs(
                    filled["u_grad"][pad:-pad, pad:-pad]
                    - fields["u_grad"][pad:-pad, pad:-pad]
                )
            )
        )
        state = {
            "x": x,
            "y": y,
            "u": fields["u"],
            "u_t": fields["u_t"],
            "u_grad": filled["u_grad"],
            "u_laplacian": filled["u_laplacian"],
            "p": fields["p"],
            "p_grad": jnp.stack([p_x, p_y], axis=-1),
            "re": fields["re"],
        }
        # Polar area weights r dr dθ
        wr = jnp.abs(jnp.concatenate([jnp.array([0.5 * hr]), jnp.full((nr - 2,), hr), jnp.array([0.5 * hr])]))
        wt = jnp.full((nt,), ht)
        # Include Jacobian r
        w = (r[:, None] * wr[:, None]) * wt[None, :]
        eng = _ns_engine()
        eng.compute_residuals(state, run_mode="training", weighted_norms="report", point_weights=w)
        summary = _summarize(eng, eng.last_residuals)
        summary["nr"] = nr
        summary["nt"] = nt
        summary["grad_err_interior"] = grad_err
        out["resolutions"].append(summary)
    # Convergence of grad error
    errs = [r["grad_err_interior"] for r in out["resolutions"]]
    out["grad_error_decreases"] = all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    return out


def case_second_derivative_tg_kov() -> Dict[str, Any]:
    """Convergence of dedicated ∂² stencils on analytic TG/Kovasznay laplacian components."""
    rows = {}
    for order in (2, 4, 6):
        errs = []
        for n in (24, 48):
            xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
            ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
            x, y = jnp.meshgrid(xs, ys, indexing="ij")
            h = float(xs[1] - xs[0])
            fields = taylor_green_2d(x, y, 0.1, re=100.0)
            u = fields["u"][..., 0]
            lap_fd = diff2_uniform(u, h, order=order, axis=0) + diff2_uniform(
                u, h, order=order, axis=1
            )
            exact = fields["u_laplacian"][..., 0]
            pad = order + 1
            errs.append(
                float(jnp.max(jnp.abs(lap_fd[pad:-pad, pad:-pad] - exact[pad:-pad, pad:-pad])))
            )
        rows[f"tg_lap_order_{order}"] = {
            "err_n24": errs[0],
            "err_n48": errs[1],
            "converges": errs[1] < errs[0],
        }
    return rows


def case_w2_paper() -> Dict[str, Any]:
    zpath = (
        ROOT
        / "examples/Notebooks/media/data/wide2_const_prop_1D_cooling_slab_test_state_pred.json.zip"
    )
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json") and "__MACOSX" not in n]
        with zf.open(names[0]) as f:
            raw = json.load(f)
    state = {k: jnp.asarray(v) for k, v in raw.items()}
    L, cp, h, k, rho = 0.1, 900.0, 500.0, 200.0, 2700.0
    engine_kw = {
        "constants": {
            "L": L,
            "cp": cp,
            "h": h,
            "k": k,
            "rho": rho,
            "alpha": Models.thermal_diffusivity(k, rho, cp),
        },
        "laws": [
            {
                "name": "fourier_conduction",
                "state_map": {
                    "T_t": "T_t",
                    "T_laplacian": "T_xx",
                    "fo": "fo",
                    "t": "t",
                    "L": "L",
                },
            }
        ],
        "groups": implied_group_specs_for_laws(["fourier_conduction"]),
        "law_implied_audits": False,
        "default_coord_dimension": 1,
    }
    eng = ResidualEngine(**engine_kw)
    eng.compute_residuals(state, run_mode="eval", weighted_norms="report")
    supplied = _summarize(eng, eng.last_residuals)
    supplied["note"] = "PINN-supplied T_xx (paper Path B); fd_order irrelevant"

    nt, nx = 512, 384
    T2 = jnp.asarray(state["T"]).reshape(nt, nx)
    x1 = jnp.asarray(state["x"]).reshape(nt, nx)[0, :]
    hx = float(x1[1] - x1[0])
    w1 = structured_point_weights([x1])  # 1D spatial weights; broadcast over t later

    def refill(order, *, weighted: bool):
        if order is None:
            Txx = jnp.gradient(jnp.gradient(T2, hx, axis=1), hx, axis=1)
        else:
            Txx = diff2_uniform(T2, hx, order=int(order), axis=1)
        st = dict(state)
        st["T_xx"] = Txx.reshape(-1)
        # Weights: repeat spatial trapezoid across time
        pw = jnp.broadcast_to(w1[None, :], (nt, nx)).reshape(-1) if weighted else None
        e = ResidualEngine(**engine_kw)
        e.compute_residuals(
            st,
            run_mode="eval",
            weighted_norms="score" if weighted else "report",
            point_weights=pw,
        )
        return _summarize(e, e.last_residuals)

    return {
        "supplied_pinn": supplied,
        "fd_legacy_unweighted": refill(None, weighted=False),
        "fd4_unweighted": refill(4, weighted=False),
        "fd4_weighted": refill(4, weighted=True),
    }


def main() -> None:
    report: Dict[str, Any] = {"phase": "A2", "cases": {}, "all_tier_changes": []}
    runners = [
        ("taylor_green_path_b", case_taylor_green),
        ("kovasznay_path_b", case_kovasznay),
        ("laplace_1d_path_b", case_laplace),
        ("controls_perturbed", case_controls),
        ("curvilinear_taylor_couette", case_curvilinear_couette),
        ("second_derivative_convergence", case_second_derivative_tg_kov),
        ("w2_paper_path_b", case_w2_paper),
    ]
    for key, fn in runners:
        print(f"running {key}...")
        result = fn()
        report["cases"][key] = result
        for ch in result.get("tier_changes") or []:
            report["all_tier_changes"].append({"case": key, **ch})
        if key in ("taylor_green_path_b", "kovasznay_path_b", "laplace_1d_path_b"):
            for label in ("legacy", "fd4_unweighted", "fd4_weighted"):
                s = result.get(label) or {}
                print(
                    f"  {label}: tier={s.get('overall_tier')} score={s.get('overall_score')}"
                )
        elif key == "w2_paper_path_b":
            for label in (
                "supplied_pinn",
                "fd_legacy_unweighted",
                "fd4_unweighted",
                "fd4_weighted",
            ):
                s = result.get(label) or {}
                print(
                    f"  {label}: tier={s.get('overall_tier')} score={s.get('overall_score')}"
                )
        elif key == "curvilinear_taylor_couette":
            print(
                f"  grad_error_decreases={result.get('grad_error_decreases')} "
                f"errs={[r.get('grad_err_interior') for r in result.get('resolutions') or []]}"
            )
        elif key == "controls_perturbed":
            for ck, cv in result.items():
                print(
                    f"  {ck}: score={cv.get('overall_score')} tier={cv.get('overall_tier')} "
                    f"below_high={cv.get('stays_below_high')}"
                )

    # Merge with Phase A JSON if present
    phase_a_path = ROOT / "docs/phase_a_fd_impact_report.json"
    if phase_a_path.is_file():
        with open(phase_a_path) as f:
            phase_a = json.load(f)
        report["phase_a_cases"] = phase_a.get("cases")
        report["phase_a_tier_changes"] = phase_a.get("all_tier_changes")

    out_path = ROOT / "docs/phase_a2_fd_impact_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {out_path}")
    print("tier changes:", report["all_tier_changes"])


if __name__ == "__main__":
    main()
