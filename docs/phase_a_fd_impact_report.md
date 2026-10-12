# Phase A / A2 impact report: derivative layer (toward Moju 2.0.0)

**Status:** Phase A + A2 approved in principle; **Phase A3 impact complete — defaults still unchanged**.  
**Branch:** `feature/2.0.0-derivative-layer`  
**Machine-readable data:** [`phase_a_fd_impact_report.json`](phase_a_fd_impact_report.json) (A), [`phase_a2_fd_impact_report.json`](phase_a2_fd_impact_report.json) (A2), [`phase_a3_fd_impact_report.json`](phase_a3_fd_impact_report.json) (A3).

This report stops for approval before Phase B (default switch, Torch parity, migration docs, release prep).

---

## What was implemented (defaults still 1.x)

| Capability | How to opt in | Default today |
|------------|---------------|---------------|
| Uniform FD orders 2/4/6 + Fornberg one-sided boundaries | `PathBGridConfig(fd_order=…, fd_order_time=…)` | `None` → legacy `jnp.gradient` |
| Non-uniform FD via Fornberg weights | Same `fd_order` when spacing is non-uniform | n/a (legacy path for `None`) |
| Dedicated second-derivative + mixed stencils | Used automatically when `fd_order` is set (laplacian / `diff2_uniform`) | n/a |
| Dual-spacing derivative-error estimate | `estimate_derivative_error=True` and `derivative_error_method="dual_spacing"` or `"both"` | off |
| Weighted residual norms (trapezoid / user weights) | `weighted_norms="report"` or `True` / `"score"`; optional `point_weights=` | off (unweighted scoring) |
| Curvilinear structured grids | `moju.monitor.curvilinear` (computational FD + metric transform) | not auto Path B |
| Resolution sensor | `resolution_sensor=True` (or options dict) | off; **diagnostic only** |
| Spectral only if periodic + uniform | `ValueError` if not | same hard requirement |

1.4 diagnostics (term breakdown, metadata, masks, flags) remain additive and do not change tier cutoffs. The resolution sensor also never changes tiers.

---

## Suites run

1. **Legacy Path B golden tests** (must stay green with defaults):  
   `test_path_b_derivatives`, `test_law_fd_recipes`, `test_fd_stencils`, `test_analytic_ns_diagnostics`, `test_readme_examples` → **43 passed**.
2. **Phase A2 unit / integration tests:** `tests/test_phase_a2_derivative_layer.py` → **16 passed**.
3. **Phase A impact harness** (`scripts/impact_fd_defaults_phase_a.py`).
4. **Phase A2 impact harness** (`scripts/impact_fd_defaults_phase_a2.py`): legacy vs `fd_order=4` unweighted vs `fd_order=4` weighted, plus curvilinear Couette, second-derivative convergence, perturbed controls, and w2 paper Path B.

---

## Phase A summary (approved)

| Case | Law | Legacy RMS | fd4 RMS | Legacy score / tier | fd4 score / tier | Tier change? |
|------|-----|------------|---------|---------------------|------------------|--------------|
| TG supplied exact ∂ | NS | ~1e-7 | identical | ~1.0 / High | identical | No |
| TG Path B FD | NS | 2.41e-2 | **5.09e-4** | 0.954 / High | 0.999 / High | No |
| Kovasznay Path B FD | NS | 1.24e-1 | **5.62e-2** | 0.836 / **Moderate** | 0.921 / **High** | **Yes** |
| Laplace 1D sine | Laplace | 6.92 | 6.92 | 0.889 / Moderate | 0.889 / Moderate | No |
| Paper-style Fourier Path B | Fourier | 8.14e-1 | **3.52e-2** | 0.988 / High | 0.999 / High | No |

**Approved direction:** default `fd_order=4` / `fd_order_time=4` in Phase B; keep `fd_order="legacy"` / `None` for 1.x bit-exact behaviour.

---

## Phase A2: (a) legacy vs (b) fd4 unweighted vs (c) fd4 weighted

Scores below use the engine’s admissibility path. For (c), primary `rms` is area-weighted (trapezoid from coordinates); unweighted values remain in the `rms_unweighted` sidecar when scoring with weights.

| Case | Law | (a) legacy score / tier | (b) fd4 unweighted | (c) fd4 weighted | Tier change vs legacy? |
|------|-----|-------------------------|--------------------|------------------|------------------------|
| TG Path B FD | NS | 0.954 / High | 0.999 / High | 0.999 / High | No |
| Kovasznay Path B FD | NS | 0.836 / **Moderate** | 0.911 / **High** | 0.909 / **High** | **Yes** (both b and c) |
| Laplace 1D sine | Laplace | 0.889 / Moderate | 0.889 / Moderate | 0.888 / Moderate | No |

Notes:

- On **uniform** grids, weighted ≈ unweighted (TG differs only in the fourth digit of the score).
- **Kovasznay** under Fornberg boundary closures scores ~0.911 (fd4 unweighted), slightly below the Phase A hand-coded-closure number (~0.921), but still crosses High. Weighted is ~0.909 (just above the High cutoff).
- Laplace residual is dominated by the true nonzero Laplacian of `sin(πx)`; FD order and weighting do not hide that.

### Residual magnitudes (Path B FD fill)

| Case | (a) RMS | (b) RMS | (c) scoring RMS (weighted) |
|------|---------|---------|----------------------------|
| TG | 2.41e-2 | 5.09e-4 | ~5.9e-4 (weighted sidecar on b: 5.9e-4) |
| Kovasznay | 1.24e-1 | ~5.6e-2 | ~same order; score uses weighted R_eff |
| Laplace | ~6.92 | ~6.92 | ~6.92 |

---

## Curvilinear: Taylor–Couette on a polar grid

Exact laminar circular Couette between concentric cylinders (`r ∈ [1, 2]`), differentiated in computational `(r, θ)` and transformed with metric terms (`moju.monitor.curvilinear`).

| Resolution `(nr, nt)` | Interior max ‖∇u − ∇u_exact‖ | Notes |
|----------------------|------------------------------|-------|
| 16 × 24 | 3.6e-5 | |
| 24 × 36 | 8.9e-6 | |
| 32 × 48 | 6.3e-6 | **grad error decreases with resolution** |

NS residuals with curvilinear-filled gradients/Laplacians remain small relative to inertial/pressure scales; polar area weights `r·dr·dθ` are supported via `point_weights`.

---

## Second derivatives (dedicated stencils)

Dedicated `diff2_uniform` (not composed first derivatives) plus `mixed_diff_uniform` (successive matching-order first derivatives).

Taylor–Green Laplacian component: errors decrease from `n=24` → `n=48` at orders 2/4/6 (`second_derivative_convergence` in the A2 JSON). Mixed ∂²/∂x∂y on `sin(x)cos(y)` at order 4: interior max error ~2e-5 on a 48² grid.

---

## Boundary closures

- Interior: classic central stencils of order 2/4/6.
- Boundaries: **Fornberg one-sided weights** of matching stencil width (same formal order).
- On wall-normal exponentials (Kovasznay-like), edge error still exceeds interior error; **masking a band of width ~`fd_order` near non-periodic walls** remains recommended when the physics of interest is interior.
- Closure accuracy tests: `tests/test_phase_a2_derivative_layer.py` (`test_boundary_closure_*`).

---

## Resolution sensor (diagnostic only)

`resolution_sensor=True` compares FD order `order_hi` vs `order_lo` on each field and warns when **interior** relative disagreement exceeds a threshold (default 0.5). **Warnings never change tiers.**

On coarse TG (`n=32`), order 2 vs 4 can disagree strongly even when order-4 residuals are already High — that is intentional order-sensitivity signalling, not a tier change.

---

## Control cases (known nonzero residuals)

| Control | Score / tier under supplied ∂ (fd4 irrelevant) | Stays below High? |
|---------|-----------------------------------------------|-------------------|
| TG + `p_grad += 0.2` | 0.741 / Moderate | **Yes** |
| Kovasznay `u *= 1.25` | 0.876 / Moderate | **Yes** |
| Laplace `sin(πx)` (true lap ≠ 0) | 0.889 / Moderate | **Yes** (unchanged by fd4) |

Better derivatives do **not** promote these into High.

---

## Paper reproducibility: w2 Path B bundle

Bundled eval state: `examples/Notebooks/media/data/wide2_const_prop_1D_cooling_slab_test_state_pred.json.zip` (same Path B notebook family as the published paper demo).

| Mode | Score / tier |
|------|--------------|
| **PINN-supplied `T_xx` (paper Path B)** | **1.000 / High** |
| FD refill legacy (`jnp.gradient`) unweighted | 0.9999 / High |
| FD refill `fd_order=4` unweighted | 0.9996 / High |
| FD refill `fd_order=4` weighted | 0.9997 / High |

**Legacy / supplied path reproduces the published High-admissibility paper result.** Changing `fd_order` does not affect audits that already supply `T_xx`. FD refill (strip and recompute) stays High under both legacy and fd4; small score differences are truncation at the slab ends, not tier changes.

---

## Phase A3: clustered grids, localized violations, noise

Harness: `scripts/impact_fd_defaults_phase_a3.py`. Generic analytic flows only.

### 1. Clustered grids (legacy vs fd4 unweighted vs fd4 weighted)

**Kovasznay** on stretched `x` (power 2.5 toward `x=0`; Path B FD fill):

| Config | Residual RMS | Residual max | Score / tier |
|--------|--------------|--------------|--------------|
| legacy | 0.155 | 0.752 | 0.843 / **Moderate** |
| fd4 unweighted | 0.128 | 2.13 | 0.871 / **Moderate** |
| fd4 weighted | (same field) | (same) | **0.934 / High** |
| exact ∂ | ~3e-7 | ~1e-6 | 1.000 / High |

Clustering puts ~21% of points in a thin near-wall band that holds only ~1.8% of trapezoid weight. Unweighted RMS over-weights that dense band (where FD truncation is worst); **area weighting restores High** without changing the residual field.

**Taylor–Couette** on polar `r` clustered toward the inner wall (physical `(r,θ)` Fornberg/central via `curvilinear`; legacy = order 2, fd4 = order 4):

| Config | Residual RMS | Residual max | Score / tier |
|--------|--------------|--------------|--------------|
| legacy (order 2) | 3.9e-3 | 2.7e-2 | 0.994 / High |
| fd4 unweighted | 4.2e-3 | 3.8e-2 | 0.993 / High |
| fd4 weighted | (same field) | (same) | 0.999 / High |
| exact ∂ | ~4e-8 | ~2e-7 | 1.000 / High |

All High; weighting still improves the score slightly.

### 2. Localized violations in the clustered band

Exact derivatives (so FD truncation does not confound detection). Perturbation: `p_grad_x += mag` only inside the near-wall band.

| Case | Point frac in band | Weight frac in band | First mag → below High (unweighted) | First mag → below High (weighted) |
|------|--------------------|---------------------|-------------------------------------|-----------------------------------|
| Clustered Kovasznay | 20.8% | **1.85%** | **0.5** | **1.0** |
| Clustered Couette | 22.2% | **1.44%** | **0.2** | **1.0** |

Weighted scoring needs a **~2–5× larger** near-wall perturbation before the global tier drops. Residual **max** always equals `mag` in the band (global max = band max); exterior max stays ~machine noise.

**Per-region / masks (representative, mag = 0.5, Kovasznay):**

| Mode | Global score / tier | Band RMS / max | Exterior max | `eval_mask` band-only score | Exterior-only score |
|------|---------------------|----------------|--------------|----------------------------|---------------------|
| Unweighted | 0.843 / Moderate | 0.5 / 0.5 | ~1e-6 | 0.711 | 1.000 |
| Weighted | 0.928 / **High** | 0.5 / 0.5 | ~1e-6 | 0.635 | 1.000 |

So: **global weighted RMS can miss a wall-local violation that unweighted catches**, while **max** and **`eval_mask` on the band** still see it clearly. Couette at mag = 0.2: unweighted already Moderate (0.903); weighted still High (0.963) until mag = 1.0.

### 3. Noise sensitivity vs FD order

Gaussian noise on `u` and `p` at relative levels `0`, `1e-4`, `1e-3`, `1e-2` (Path B fills spatial derivatives; `u_t` exact). Uniform grids.

**Taylor–Green** (smooth, periodic) — overall score / tier:

| Noise | legacy | fd2 | fd4 | fd6 |
|------|--------|-----|-----|-----|
| 0 | 0.954 H | 0.985 H | **0.999 H** | 0.9999 H |
| 1e-4 | 0.954 H | 0.985 H | 0.999 H | 0.999 H |
| 1e-3 | 0.954 H | 0.985 H | 0.994 H | 0.989 H |
| 1e-2 | 0.944 H | 0.961 H | 0.945 H | **0.904 M** |

Residual RMS at 1e-2: fd2 0.020, fd4 0.029, **fd6 0.053** (noise amplification).

**Kovasznay** (non-periodic wall-normal structure):

| Noise | legacy | fd2 | fd4 | fd6 |
|------|--------|-----|-----|-----|
| 0 | 0.836 M | 0.915 H | **0.911 H** | 0.934 H |
| 1e-4 | 0.836 M | 0.914 H | **0.909 M** | 0.919 H |
| 1e-3 | 0.832 M | 0.866 M | 0.816 M | **0.673 M** |
| 1e-2 | 0.666 L | 0.581 L | 0.572 L | 0.571 L |

At modest noise, **higher order is not monotonically better**: fd6 degrades fastest; on Kovasznay, fd4 sits on the High/Moderate edge at 1e-4 while fd2 stays High.

---

## Safeguard candidate comparison (post-A3)

Harness: `scripts/compare_weighted_safeguards.py` → [`phase_a3_safeguard_compare.json`](phase_a3_safeguard_compare.json).

On localized near-wall perturbations (exact ∂), first magnitude below High:

| Candidate | Kovasznay clustered | Couette clustered | Unweighted reference |
|-----------|---------------------|-------------------|----------------------|
| (i) residual max | 0.1 | 0.1 | 0.5 / 0.2 |
| (ii) p99 | 0.1 | 0.1 | 0.5 / 0.2 |
| (ii) **p95** (reported) | **0.1** | **0.1** | 0.5 / 0.2 |
| (iii) worst-region wRMS | 0.1 | 0.1 | 0.5 / 0.2 |
| weighted only (no safeguard) | 1.0 | 1.0 | — |

On noise with `fd_order=4`, tier at ε = 1e-3:

| Candidate | Taylor–Green | Kovasznay |
|-----------|--------------|-----------|
| (i) max | High | Non-Admissible |
| (ii) p99 | High | Non-Admissible |
| (ii) **p95** | **High** | Low (unweighted already Moderate) |
| (iii) worst-region | High | Moderate |
| Clean Kov Path B (no noise) | — | **max/p99/region drop High; p95 keeps High** |

**Chosen safeguard: 95th-percentile residual (candidate ii, parameter p=95).**  
Reasons: detects localized violations at mag 0.1 (at least as sensitive as unweighted), stays High on TG noise at 1e-3, and **does not undo** the clean Kovasznay fd4 → High result (unlike max, p99, and worst-region, which engage on wall FD truncation spikes). Effective scoring residual: `max(weighted_R_eff, p95(|r|))`. Defaults: `tiles=(4,4)`, `boundary_frac=0.1`, `min_region_frac=0.05`, `min_region_abs=16` remain available for the `worst_region` method if selected explicitly.

---

## Percentile selection (p95 vs p97.5 vs p99)

Harness update: `scripts/compare_weighted_safeguards.py` → [`phase_a3_safeguard_compare.json`](phase_a3_safeguard_compare.json) (`phase: A3_safeguard_percentile_compare`).

**Prefer rule:** choose **p97.5** only if clean Kovasznay and clean Taylor–Green stay High on **both** coarse and fine grids **and** Taylor–Green stays High at noise `1e-3` (`fd_order=4`); otherwise keep **p95**.

Resolutions: TG coarse/fine = 32² / 64²; Kovasznay coarse/fine = 40² / 80².

### (a) Detection floor (uniform grid, band fraction × first mag below High)

Localized `p_grad` band; values are `(band_frac ≈, first_magnitude)`.

| Flow / res | p95 | p97.5 | p99 | max (ref) |
|------------|-----|-------|-----|-----------|
| TG coarse | 2%→0.5; ≥5%→0.05 | 2%→0.05 | 2%→0.05 | 2%→0.05 |
| TG fine | 2%→1.0; 5%→0.5; ≥10%→0.05 | 2%→1.0; ≥5%→0.05 | 2%→0.05 | 2%→0.05 |
| Kov coarse | 2%→1.0; 5%→0.5; ≥10%→0.1 | 2%→1.0; ≥5%→0.1 | 2%→0.1 | 2%→0.1 |
| Kov fine | 2%→0.5; 5%→0.5; ≥10%→0.1 | 2%→0.5; ≥5%→0.1 | 2%→0.1 | 2%→0.1 |

**Detection-floor takeaway:** violations confined to ≲2–5% of points may not drop the tier under p95 (need larger magnitude or a wider band). p97.5 / p99 detect earlier at mid bands; use `eval_mask` or region scoring for small critical regions.

### (b) Clean Path B (`fd_order=4`, no perturbation)

| Case | p95 | p97.5 | p99 | Boundary-stencil spike frac |
|------|-----|-------|-----|----------------------------|
| TG coarse 32² | High | High | High | 0 |
| TG fine 64² | High | High | High | 0 |
| Kov coarse 40² | **High** | **Low** | Low | 0.050 |
| Kov fine 80² | High | High | High | 0.0005 |

Boundary-stencil spikes: points inside FD boundary layers (`width = fd_order//2`) with `|r|` above the High-tier residual threshold.

### (c) Noise (`fd_order=4`)

| Case | ε | p95 | p97.5 | p99 |
|------|---|-----|-------|-----|
| TG coarse | 1e-3 | High | High | High |
| TG fine | 1e-3 | High | High | High |
| Kov coarse | 0 | High | Low | Low |
| Kov coarse | 1e-3 | Low | Low | Non-Adm |
| Kov fine | 0 | High | High | High |
| Kov fine | 1e-3 | Non-Adm | Non-Adm | Non-Adm |

(TG at 1e-4 High for all three; at 1e-2 Moderate for all three.)

### (d) Clustered A3 (localized near-wall)

First magnitude below High: **p95 = p97.5 = p99 = 0.1** on clustered Kovasznay and Couette (unweighted 0.5 / 0.2).

### Recommendation (step 2 — provisional, superseded by Phase A4)

**Keep default `safeguard_percentile=95`** under plain percentile scoring (no error-aware).

p97.5 fails clean Kovasznay on the coarse 40² grid (Low vs p95 High) because wall FD spikes occupy ~5% of points. See Phase A4 below for unlocking p97.5.

---

## Phase A4: boundary closures + error-aware safeguard

Harnesses: `scripts/phase_a4_boundary_safeguard.py`, `scripts/phase_a4_verification.py` →
[`phase_a4_verification.json`](phase_a4_verification.json).

### (1) Closure operator verification (float64) + fixes

Bugs fixed before re-comparison:

- **bc_aware:** face-mean Dirichlet + odd reflection caused O(1) Kov brow (~6). Replaced with
  per-line high-order Fornberg ghost extrapolation. Kov brow @48 now **9.7e-9** (= matching).
- **order_reduced:** gap row between reduced and matching half-width was all zeros (e₁≈π).
  Fixed; sine boundary order now **~2.03**.

| Closure | d1 bound/int (sine) | Kov brow @48 | Path B spike frac | plain p97.5 Kov |
|---------|--------------------:|-------------:|------------------:|-----------------|
| **matching** | 4.05 / 4.03 | 9.7e-9 | **0.050** | Low |
| order_reduced | 2.03 / 4.03 | 4.1e-5 | 0.068 | worse |
| sbp | 2.03 / ~2.5* | 5.0e-5 | 0.068 | worse |
| bc_aware | 4.05 / 4.03 | 9.7e-9 | **0.050** | Low |

\*SBP interior band includes the 4-row boundary block. No closure unlocks plain p97.5 on clean Kov coarse.

### (2) Spike source (clean Kov 40², matching)

| Field | Boundary RMS | Notes |
|-------|-------------:|-------|
| **u_laplacian** | **7.94** | Dominant; y-edges ≫ x-edges; corners ~14 |
| ∂u/∂y (u0_d1) | 0.326 | y one-sided FD on 2π-periodic field |
| ∂u/∂x (u0_d0) | 2e-8 | Fine |
| p_grad | 4e-7 | Fine |

Spikes are second-derivative / periodic-y boundary truncation, not a broken x-closure.

### (3) Sharp-violation tests

First mag below High (Kov 40², exact ∂):

| Config | gaussian | checker | kink | ≥ plain p95? |
|--------|----------|---------|------|--------------|
| plain p95 | 1.0 | 1.0 | 0.1 | — |
| ea order p95 / p97.5 | 1.0 | 1.0 | 0.1 | **yes** |
| ea dual p97.5 | 1.0 | 1.0 | 0.1 | **yes** |
| ea + smooth p97.5 | 1.0 | 1.0 | 0.1 | sharp yes; **misses** smooth band @0.1 |

### (4) ε variants

ε_order = |r⁽⁴⁾−r⁽²⁾|; ε_dual = |r_h−r_{2h}|/(2⁴−1). On clean suites, order-ε keeps p97.5 and p99 High on Kov coarse; dual-ε drops p99 to Moderate. TG noise 1e-3 with order-ε: p97.5 High on 32²/64².

### (5) Recommendation (defaults **not** switched)

Gate: error-aware defaults only if sharp detection ≥ plain p95 → **passed** (`ea_order_p97_5`).

| Knob | Recommended | Notes |
|------|-------------|-------|
| `boundary_closure` | **matching** | Best accuracy; only matching/bc_aware keep plain p95 High |
| `safeguard_error_aware` | **True** | Sharp ≥ plain p95; unlocks clean p97.5 |
| `safeguard_percentile` | **97.5** | Highest clearing sharp+clean+TG noise 1e-3 |
| ε | **order-difference** | Prefer over dual (clean p99); `smooth_radius=0` |
| Fallback | plain p95 | `error_aware=False` or `weighted_norms=False` |

**Stopped for approval.** Current code defaults remain Phase B (`matching` + plain p95).

## Recommendation for Phase B defaults (final)

1. **`fd_order=4` and `fd_order_time=4`** with docs: noisy inputs → supplied ∂ or `fd_order=2`; avoid 6.
2. **Weighted norms default + p95 safeguard** (not bare max). Unweighted opt-out for 1.x comparisons.
3. Fornberg closures + dedicated second-derivative / mixed stencils.
4. `fd_order=None` / `"legacy"` — 1.x bit-exact.
5. Resolution sensor — opt-in, never changes tiers.
6. Curvilinear — explicit `moju.monitor.curvilinear`; no auto Path B in 2.0.

**Do not** default to order 6.

---

## Deferred to 2.1 (roadmap — not in this PR)

- Element-wise polynomial differentiation on element nodes (e.g. Gauss–Lobatto) with averaging at shared nodes.
- Opt-in weak-form residuals against smooth test functions, with their own scale recipe.

See [`ROADMAP.md`](../ROADMAP.md).

---

## Phase B status (complete; release prep held)

- Defaults: `fd_order=4` / `fd_order_time=4`; `weighted_norms=True` + p95 safeguard; `fd_order="legacy"` / `None` for 1.x bit-exact.
- Torch Path B FD parity (`moju.torch._fd_stencils` / `_path_b`).
- Docs: [`MIGRATION_1x_to_2.0.md`](MIGRATION_1x_to_2.0.md), `getting_derivatives_right.md`, `VERSIONING.md`, CHANGELOG `[2.0.0]`.
- Package version set to `2.0.0` in `pyproject.toml` / `moju.__version__`.

---

## Final approved defaults (A5 → release)

| Knob | Approved 2.0 default | Notes |
|------|----------------------|-------|
| `fd_order` / `fd_order_time` | **4** | `legacy` / `None` for 1.x |
| `boundary_closure` | **`matching`** | Alt closures regress |
| `safeguard_percentile` | **95** | Rule (c) after A5 periodic rerun |
| `safeguard_error_aware` | **False** (opt-in) | A4/A5 evidence documented |
| `periodic` | explicit per axis | Never inferred |
| Resolution-sensor grid-scale | threshold **0.05** | Diagnostic only |

See [`phase_a5_periodic_safeguard.md`](phase_a5_periodic_safeguard.md).
