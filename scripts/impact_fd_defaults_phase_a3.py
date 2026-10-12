#!/usr/bin/env python3
"""
Phase A3 impact: clustered grids, localized near-wall violations, noise vs FD order.

Generic analytic flows only. Does not change package defaults.
Writes docs/phase_a3_fd_impact_report.json for the extended markdown report.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import (
    kovasznay_2d,
    ns_state_from_analytic,
    taylor_couette_2d,
    taylor_green_2d,
)
from moju.monitor.curvilinear import (
    fill_vector_grad_lap_polar,
    gradient_polar_physical,
)
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.tiers import tier_for_score
from moju.monitor.weighted_norms import structured_point_weights, trapezoid_weights_1d


ROOT = Path(__file__).resolve().parents[1]
HIGH = 0.909  # ADM_HIGH_THRESHOLD ≈ 1/(1+0.001) style; use same cutoff as report


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


def _law_stats(res: Dict[str, Any], report: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    laws = res.get("laws") or {}
    out = {}
    for ln, arr in laws.items():
        a = jnp.asarray(arr)
        mag = jnp.sqrt(jnp.sum(jnp.square(a), axis=-1)) if a.ndim >= 1 and a.shape[-1] in (2, 3) else jnp.abs(a)
        rms = float(jnp.sqrt(jnp.mean(jnp.square(a)))) if a.size else float("nan")
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
            row["rms_weighted"] = float(rw)
        ru = (entry.get("rms_unweighted") or {}).get(f"laws/{ln}")
        if ru is not None:
            row["rms_unweighted"] = float(ru)
        out[ln] = row
    overall = float(report.get("overall_admissibility_score") or 0.0)
    return {
        "overall_score": overall,
        "overall_tier": tier_for_score(overall) if math.isfinite(overall) else "Unknown",
        "below_high": overall < HIGH,
        "laws": out,
    }


def _run_ns(
    state: Dict[str, Any],
    *,
    grid: Optional[PathBGridConfig] = None,
    fill_law_fd: bool = False,
    weighted_norms: Any = False,
    point_weights: Any = None,
    eval_mask: Any = None,
) -> Dict[str, Any]:
    eng = _ns_engine()
    kw: Dict[str, Any] = {
        "run_mode": "training",
        "weighted_norms": weighted_norms,
        "point_weights": point_weights,
        "eval_mask": eval_mask,
    }
    if grid is not None:
        kw["auto_path_b_derivatives"] = grid
        kw["fill_law_fd"] = fill_law_fd
    res = eng.compute_residuals(state, **kw)
    report = audit(eng.log)
    entry = eng.log[-1]
    summary = _law_stats(res, report, entry)
    # Always attach raw residual for regional post-processing
    summary["_residual"] = res.get("laws", {}).get("momentum_navier_stokes")
    return summary


def _region_breakdown(residual: Any, region_mask: Any) -> Dict[str, float]:
    """region_mask True = near-wall / clustered band."""
    a = jnp.asarray(residual)
    m = jnp.asarray(region_mask).astype(bool)
    if a.ndim == m.ndim + 1 and a.shape[:-1] == m.shape:
        mag = jnp.sqrt(jnp.sum(jnp.square(a), axis=-1) + 1e-30)
    else:
        mag = jnp.abs(a)
    def _stats(sel):
        if not bool(jnp.any(sel)):
            return {"rms": float("nan"), "max": float("nan"), "n": 0}
        v = mag[sel]
        return {
            "rms": float(jnp.sqrt(jnp.mean(jnp.square(v)))),
            "max": float(jnp.max(v)),
            "n": int(jnp.sum(sel)),
        }
    return {
        "cluster_band": _stats(m),
        "exterior": _stats(~m),
        "global": _stats(jnp.ones_like(m)),
    }


def stretched_axis(n: int, lo: float, hi: float, *, power: float = 2.5, toward: str = "lo") -> jnp.ndarray:
    """Monotone stretch clustering points toward ``lo`` or ``hi``."""
    s = jnp.linspace(0.0, 1.0, n)
    if toward == "lo":
        t = s**power
    else:
        t = 1.0 - (1.0 - s) ** power
    return lo + (hi - lo) * t


def polar_area_weights(r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    r1 = jnp.reshape(jnp.asarray(r, dtype=float), (-1,))
    t1 = jnp.reshape(jnp.asarray(theta, dtype=float), (-1,))
    wr = trapezoid_weights_1d(r1)
    wt = trapezoid_weights_1d(t1)
    # Include mapping Jacobian r
    return (r1[:, None] * wr[:, None]) * wt[None, :]


# ---------------------------------------------------------------------------
# 1. Clustered grids (exact / Path B / curvilinear)
# ---------------------------------------------------------------------------


def case_clustered_kovasznay() -> Dict[str, Any]:
    nx, ny, re = 48, 40, 40.0
    xs = stretched_axis(nx, 0.0, 1.0, power=2.5, toward="lo")  # cluster toward x=0
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=re)
    # Path B from primitives (spatial FD is the variable)
    state = {
        "x": x,
        "y": y,
        "u": fields["u"],
        "u_t": fields["u_t"],
        "p": fields["p"],
        "re": fields["re"],
    }
    w = structured_point_weights([xs, ys])
    base = PathBGridConfig(layout="meshgrid", spatial_dimension=2, steady=True)
    rows = {}
    for label, fo, wn in (
        ("legacy", None, "report"),
        ("fd4_unweighted", 4, "report"),
        ("fd4_weighted", 4, "score"),
    ):
        grid = replace(base, fd_order=fo, fd_order_time=fo)
        s = _run_ns(
            state,
            grid=grid,
            fill_law_fd=True,
            weighted_norms=wn,
            point_weights=w if wn == "score" or wn == "report" else None,
        )
        # Drop bulky residual array from JSON later
        rows[label] = {k: v for k, v in s.items() if k != "_residual"}
        rows[label]["fd_order"] = fo
        rows[label]["weight_sum"] = float(jnp.sum(w))
        rows[label]["dx_min"] = float(jnp.min(jnp.diff(xs)))
        rows[label]["dx_max"] = float(jnp.max(jnp.diff(xs)))
    # Exact-derivative control on same grid
    exact = ns_state_from_analytic(fields, x, y)
    s_ex = _run_ns(exact, weighted_norms="report", point_weights=w)
    rows["exact_derivatives"] = {k: v for k, v in s_ex.items() if k != "_residual"}
    rows["meta"] = {
        "nx": nx,
        "ny": ny,
        "stretch_power": 2.5,
        "cluster_toward": "x=0",
        "note": "Path B FD fill on stretched x; y uniform periodic",
    }
    return rows


def case_clustered_couette() -> Dict[str, Any]:
    nr, nt, re = 36, 48, 100.0
    r = stretched_axis(nr, 1.0, 2.0, power=2.5, toward="lo")  # cluster toward inner wall
    th = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
    fields = taylor_couette_2d(r, th, re=re)
    x, y = fields["x"], fields["y"]
    w = polar_area_weights(r, th)

    def _state_at_order(order: int) -> Dict[str, Any]:
        filled = fill_vector_grad_lap_polar(fields["u"], r, th, order=order)
        p_x, p_y = gradient_polar_physical(fields["p"], r, th, order=order)
        return {
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

    state_fd4 = _state_at_order(4)
    state_leg = _state_at_order(2)

    rows = {}
    for label, st, wn in (
        ("legacy", state_leg, "report"),
        ("fd4_unweighted", state_fd4, "report"),
        ("fd4_weighted", state_fd4, "score"),
    ):
        s = _run_ns(st, weighted_norms=wn, point_weights=w)
        rows[label] = {k: v for k, v in s.items() if k != "_residual"}
    exact = {
        "x": x,
        "y": y,
        "u": fields["u"],
        "u_t": fields["u_t"],
        "u_grad": fields["u_grad"],
        "u_laplacian": fields["u_laplacian"],
        "p": fields["p"],
        "p_grad": fields["p_grad"],
        "re": fields["re"],
    }
    s_ex = _run_ns(exact, weighted_norms="report", point_weights=w)
    rows["exact_derivatives"] = {k: v for k, v in s_ex.items() if k != "_residual"}
    rows["meta"] = {
        "nr": nr,
        "nt": nt,
        "stretch_power": 2.5,
        "cluster_toward": "r_inner",
        "dr_min": float(jnp.min(jnp.diff(r))),
        "dr_max": float(jnp.max(jnp.diff(r))),
        "note": (
            "Polar physical (r,θ) Fornberg/central diffs via curvilinear helpers; "
            "legacy column = order 2, fd4 = order 4"
        ),
    }
    return rows


# ---------------------------------------------------------------------------
# 2. Localized violations in the clustered band
# ---------------------------------------------------------------------------


def _detect_table(
    base_state: Dict[str, Any],
    region_mask: Any,
    weights: Any,
    magnitudes: Sequence[float],
    *,
    perturb: str,
    fill_grid: Optional[PathBGridConfig] = None,
) -> Dict[str, Any]:
    """
    Add a localized p_grad (or u) perturbation in region_mask at several amplitudes.

    Reports unweighted vs weighted detection (drop below High), plus regional max/rms
    and eval_mask scores for cluster-only / exterior-only.
    """
    # Baseline (no perturbation)
    baseline = {}
    for wn_label, wn in (("unweighted", "report"), ("weighted", "score")):
        s = _run_ns(
            base_state,
            grid=fill_grid,
            fill_law_fd=fill_grid is not None,
            weighted_norms=wn,
            point_weights=weights,
        )
        baseline[wn_label] = {k: v for k, v in s.items() if k != "_residual"}

    sweeps = []
    for mag in magnitudes:
        st = {k: (jnp.array(v) if hasattr(v, "shape") else v) for k, v in base_state.items()}
        m = jnp.asarray(region_mask)
        if perturb == "p_grad":
            pg = jnp.asarray(st["p_grad"])
            # Add magnitude along x-component in the band only
            delta = jnp.zeros_like(pg)
            delta = delta.at[..., 0].set(jnp.where(m, float(mag), 0.0))
            st["p_grad"] = pg + delta
        elif perturb == "u":
            u = jnp.asarray(st["u"])
            # Relative boost of |u| in the band
            scale = 1.0 + jnp.where(m[..., None], float(mag), 0.0)
            st["u"] = u * scale
            # If derivatives were supplied, leave them inconsistent (violation)
        else:
            raise ValueError(perturb)

        row: Dict[str, Any] = {"magnitude": float(mag), "perturb": perturb}
        for wn_label, wn in (("unweighted", "report"), ("weighted", "score")):
            s = _run_ns(
                st,
                grid=None if "u_grad" in st and fill_grid is None else fill_grid,
                fill_law_fd=fill_grid is not None and "u_grad" not in st,
                weighted_norms=wn,
                point_weights=weights,
            )
            resid = s.pop("_residual")
            regions = _region_breakdown(resid, m)
            # eval_mask: cluster-only and exterior-only
            s_cluster = _run_ns(
                st,
                weighted_norms=wn,
                point_weights=weights,
                eval_mask=m,
                grid=fill_grid if fill_grid is not None and "u_grad" not in st else None,
                fill_law_fd=fill_grid is not None and "u_grad" not in st,
            )
            s_ext = _run_ns(
                st,
                weighted_norms=wn,
                point_weights=weights,
                eval_mask=~m,
                grid=fill_grid if fill_grid is not None and "u_grad" not in st else None,
                fill_law_fd=fill_grid is not None and "u_grad" not in st,
            )
            s_cluster.pop("_residual", None)
            s_ext.pop("_residual", None)
            base_score = float(baseline[wn_label]["overall_score"])
            score = float(s["overall_score"])
            row[wn_label] = {
                **{k: v for k, v in s.items() if k != "_residual"},
                "tier_drop_from_baseline": s["overall_tier"] != baseline[wn_label]["overall_tier"],
                "score_drop": base_score - score,
                "detects_below_high": score < HIGH,
                "region_breakdown": regions,
                "eval_mask_cluster_only": s_cluster,
                "eval_mask_exterior_only": s_ext,
            }
        sweeps.append(row)

    # Detection thresholds: smallest mag that drops below High
    def _first_detect(key: str) -> Optional[float]:
        for row in sweeps:
            if row[key]["detects_below_high"]:
                return float(row["magnitude"])
        return None

    return {
        "baseline": baseline,
        "sweeps": sweeps,
        "first_mag_below_high_unweighted": _first_detect("unweighted"),
        "first_mag_below_high_weighted": _first_detect("weighted"),
    }


def case_localized_kovasznay() -> Dict[str, Any]:
    nx, ny, re = 48, 40, 40.0
    xs = stretched_axis(nx, 0.0, 1.0, power=2.5, toward="lo")
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=re)
    # Use exact derivatives so FD truncation does not confound detection
    state = ns_state_from_analytic(fields, x, y)
    w = structured_point_weights([xs, ys])
    # Cluster band: densest ~20% of x-span near x=0 (use coordinate threshold)
    x_cut = float(xs[max(3, nx // 5)])
    region = x <= x_cut
    mags = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    out = _detect_table(state, region, w, mags, perturb="p_grad", fill_grid=None)
    out["meta"] = {
        "x_cut": x_cut,
        "fraction_points_in_band": float(jnp.mean(region.astype(float))),
        "weight_fraction_in_band": float(jnp.sum(w * region) / jnp.sum(w)),
        "magnitudes": mags,
        "perturb": "p_grad_x += mag in band (exact ∂ elsewhere)",
    }
    return out


def case_localized_couette() -> Dict[str, Any]:
    nr, nt, re = 36, 48, 100.0
    r = stretched_axis(nr, 1.0, 2.0, power=2.5, toward="lo")
    th = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
    fields = taylor_couette_2d(r, th, re=re)
    state = {
        "x": fields["x"],
        "y": fields["y"],
        "u": fields["u"],
        "u_t": fields["u_t"],
        "u_grad": fields["u_grad"],
        "u_laplacian": fields["u_laplacian"],
        "p": fields["p"],
        "p_grad": fields["p_grad"],
        "re": fields["re"],
    }
    w = polar_area_weights(r, th)
    r_cut = float(r[max(3, nr // 5)])
    region = fields["r"] <= r_cut
    mags = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    out = _detect_table(state, region, w, mags, perturb="p_grad", fill_grid=None)
    out["meta"] = {
        "r_cut": r_cut,
        "fraction_points_in_band": float(jnp.mean(region.astype(float))),
        "weight_fraction_in_band": float(jnp.sum(w * region) / jnp.sum(w)),
        "magnitudes": mags,
        "perturb": "p_grad_x += mag in inner-wall band (exact ∂)",
    }
    return out


# ---------------------------------------------------------------------------
# 3. Noise sensitivity vs FD order
# ---------------------------------------------------------------------------


def case_noise_sweep() -> Dict[str, Any]:
    key = jax.random.PRNGKey(0)
    noise_levels = [0.0, 1e-4, 1e-3, 1e-2]
    orders: List[Optional[int]] = [None, 2, 4, 6]
    out: Dict[str, Any] = {"noise_levels": noise_levels, "cases": {}}

    # Taylor–Green
    n, t, re = 32, 0.1, 100.0
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    u_scale = float(jnp.sqrt(jnp.mean(jnp.square(fields["u"]))))
    p_scale = float(jnp.sqrt(jnp.mean(jnp.square(fields["p"]))))

    tg_rows = []
    for eps in noise_levels:
        key, k1, k2 = jax.random.split(key, 3)
        u_n = fields["u"] + eps * u_scale * jax.random.normal(k1, fields["u"].shape)
        p_n = fields["p"] + eps * p_scale * jax.random.normal(k2, fields["p"].shape)
        state = {
            "x": x,
            "y": y,
            "u": u_n,
            "u_t": fields["u_t"],  # keep exact time deriv (noise is spatial)
            "p": p_n,
            "re": fields["re"],
        }
        order_rows = {}
        for fo in orders:
            grid = PathBGridConfig(
                layout="meshgrid",
                spatial_dimension=2,
                steady=True,
                fd_order=fo,
                fd_order_time=fo,
            )
            s = _run_ns(state, grid=grid, fill_law_fd=True, weighted_norms=False)
            order_rows["legacy" if fo is None else f"fd{fo}"] = {
                k: v for k, v in s.items() if k != "_residual"
            }
        tg_rows.append({"noise_rel": eps, "orders": order_rows})
    out["cases"]["taylor_green"] = {
        "u_scale": u_scale,
        "rows": tg_rows,
        "meta": "Gaussian noise on u and p; Path B fills spatial ∂; u_t exact",
    }

    # Kovasznay (uniform grid — isolate order×noise, not clustering)
    nx, ny, re = 40, 40, 40.0
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=re)
    u_scale = float(jnp.sqrt(jnp.mean(jnp.square(fields["u"]))))
    p_scale = float(jnp.sqrt(jnp.mean(jnp.square(fields["p"]))))
    kov_rows = []
    for eps in noise_levels:
        key, k1, k2 = jax.random.split(key, 3)
        u_n = fields["u"] + eps * u_scale * jax.random.normal(k1, fields["u"].shape)
        p_n = fields["p"] + eps * p_scale * jax.random.normal(k2, fields["p"].shape)
        state = {
            "x": x,
            "y": y,
            "u": u_n,
            "u_t": fields["u_t"],
            "p": p_n,
            "re": fields["re"],
        }
        order_rows = {}
        for fo in orders:
            grid = PathBGridConfig(
                layout="meshgrid",
                spatial_dimension=2,
                steady=True,
                fd_order=fo,
                fd_order_time=fo,
            )
            s = _run_ns(state, grid=grid, fill_law_fd=True, weighted_norms=False)
            order_rows["legacy" if fo is None else f"fd{fo}"] = {
                k: v for k, v in s.items() if k != "_residual"
            }
        kov_rows.append({"noise_rel": eps, "orders": order_rows})
    out["cases"]["kovasznay"] = {
        "u_scale": u_scale,
        "rows": kov_rows,
        "meta": "Uniform grid; Gaussian noise on u and p; Path B spatial fill",
    }
    return out


def main() -> None:
    report: Dict[str, Any] = {"phase": "A3", "cases": {}}
    runners = [
        ("clustered_kovasznay", case_clustered_kovasznay),
        ("clustered_couette", case_clustered_couette),
        ("localized_kovasznay", case_localized_kovasznay),
        ("localized_couette", case_localized_couette),
        ("noise_sweep", case_noise_sweep),
    ]
    for key, fn in runners:
        print(f"running {key}...")
        result = fn()
        report["cases"][key] = result
        if key.startswith("clustered"):
            for lab in (
                "legacy",
                "fd4_unweighted",
                "fd4_weighted",
                "exact_derivatives",
            ):
                if lab in result:
                    s = result[lab]
                    print(
                        f"  {lab}: score={s.get('overall_score')} tier={s.get('overall_tier')}"
                    )
        elif key.startswith("localized"):
            print(
                f"  first_below_high unweighted={result.get('first_mag_below_high_unweighted')} "
                f"weighted={result.get('first_mag_below_high_weighted')}"
            )
            print(
                f"  weight_frac_band={result.get('meta', {}).get('weight_fraction_in_band')} "
                f"point_frac={result.get('meta', {}).get('fraction_points_in_band')}"
            )
            for row in result.get("sweeps") or []:
                uw = row["unweighted"]
                w = row["weighted"]
                print(
                    f"  mag={row['magnitude']}: uw score={uw['overall_score']:.4f} "
                    f"({uw['overall_tier']}) max={uw['region_breakdown']['global']['max']:.3g} | "
                    f"w score={w['overall_score']:.4f} ({w['overall_tier']}) "
                    f"max={w['region_breakdown']['global']['max']:.3g}"
                )
        elif key == "noise_sweep":
            for cname, cdata in result.get("cases", {}).items():
                print(f"  --- {cname} ---")
                for row in cdata.get("rows") or []:
                    bits = []
                    for oname, o in row["orders"].items():
                        bits.append(f"{oname}:{o['overall_score']:.4f}/{o['overall_tier'][:4]}")
                    print(f"  eps={row['noise_rel']}: " + " | ".join(bits))

    out_path = ROOT / "docs/phase_a3_fd_impact_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
