# Phase A impact report: derivative layer (toward Moju 2.0.0)

**Status:** implementation behind explicit options only — **defaults unchanged**.  
**Branch:** `feature/2.0.0-derivative-layer` (from 1.4.0 diagnostics work).  
**Machine-readable data:** [`phase_a_fd_impact_report.json`](phase_a_fd_impact_report.json).

This report stops for approval before Phase B (default switch, Torch parity, migration docs, release prep).

---

## What was implemented (defaults still 1.x)

| Capability | How to opt in | Default today |
|------------|---------------|---------------|
| Uniform FD orders 2/4/6 + matching one-sided boundaries | `PathBGridConfig(fd_order=…, fd_order_time=…)` | `None` → legacy `jnp.gradient` |
| Non-uniform FD via Fornberg weights (same orders) | Same `fd_order` when spacing is non-uniform | n/a (legacy path for `None`) |
| Dual-spacing derivative-error estimate (every-other-point) | `estimate_derivative_error=True` and `derivative_error_method="dual_spacing"` or `"both"` | off; stencil-order method remains default when estimation is on |
| Stencil-order comparison + `state_pred_alt` | unchanged from 1.4 | opt-in |
| Spectral only if periodic + uniform | `ValueError` if not (no soft fallback) | same hard requirement; soft pre-warn removed |

1.4 diagnostics (term breakdown, metadata, masks, flags) remain additive and do not change tier cutoffs.

---

## Suites run

1. **Legacy Path B golden tests** (must stay green with defaults):  
   `test_path_b_derivatives`, `test_law_fd_recipes`, `test_path_b_spectral`, `test_readme_examples` → **37 passed**.
2. **New unit tests:** Fornberg, dual-spacing, spectral hard error → with existing FD/analytic tests → **33 passed**.
3. **Impact harness** (`scripts/impact_fd_defaults_phase_a.py`): legacy (`fd_order=None`) vs proposed (`fd_order=4`, `fd_order_time=4`) on analytic Path B fills and a paper-style Fourier Path B case.
4. **Order sweep** on Taylor–Green and Kovasznay Path B: `None`, `2`, `4`, `6`.

Paper Path A (PINN autodiff in the Colab notebook) does **not** use Path B FD, so changing `fd_order` defaults will not change those scores. The paper-style case below uses Path B FD fill of `T_laplacian`, which is what a field-export audit would hit.

---

## Impact table (legacy vs `fd_order=4`)

| Case | Law | Legacy residual RMS | fd4 residual RMS | Legacy score / tier | fd4 score / tier | Tier change? |
|------|-----|---------------------|------------------|---------------------|------------------|--------------|
| TG supplied exact ∂ (control) | NS | ~1e-7 (max) | identical | ~1.0 / High | identical | No |
| TG Path B FD fill | NS | 2.41e-2 | **5.09e-4** | 0.954 / High | 0.999 / High | No |
| Kovasznay Path B FD fill | NS | 1.24e-1 | **5.62e-2** | 0.836 / **Moderate** | 0.921 / **High** | **Yes** |
| Laplace 1D sine (lap ≠ 0) | Laplace | 6.92 | 6.92 | 0.889 / Moderate | 0.889 / Moderate | No* |
| Paper-style Fourier Path B | Fourier | 8.14e-1 | **3.52e-2** | 0.988 / High | 0.999 / High | No |

\*Laplace residual is dominated by the true nonzero Laplacian of `sin(πx)`, not FD truncation; score stays on the same `scale_k` floor.

### Closeness to exact residual (analytic Path B)

Exact residual max with supplied derivatives is ~1e-7 for both TG and Kovasznay.

| Case | Legacy max \|R\| | fd4 max \|R\| | Closer to exact? |
|------|------------------|---------------|------------------|
| Taylor–Green Path B | 9.75e-2 | **2.01e-3** | **fd4** (~50× closer) |
| Kovasznay Path B | 7.13e-1 | **5.77e-1** | **fd4** (improved; still limited by boundary one-sided stencils on a wall-normal exponential) |
| Paper-style Fourier | 4.12 (max) | **0.23** | **fd4** |

### Order sweep (Path B fill only)

**Taylor–Green (periodic, smooth):**

| Order | RMS | max \|R\| | Score | Tier |
|-------|-----|-----------|-------|------|
| legacy (`None`) | 2.41e-2 | 9.75e-2 | 0.954 | High |
| 2 | 7.36e-3 | 1.73e-2 | 0.985 | High |
| **4** | **5.09e-4** | **2.01e-3** | **0.999** | High |
| 6 | 5.90e-5 | 2.45e-4 | 0.9999 | High |

**Kovasznay (non-periodic in x):**

| Order | RMS | max \|R\| | Score | Tier |
|-------|-----|-----------|-------|------|
| legacy | 1.24e-1 | 7.13e-1 | 0.836 | **Moderate** |
| 2 | 7.57e-2 | 7.59e-1 | 0.893 | Moderate |
| **4** | **5.62e-2** | **5.77e-1** | **0.921** | **High** |
| 6 | 4.36e-2 | 5.13e-1 | 0.938 | High |

---

## Tier changes (flagged)

Only **Kovasznay Path B FD** changes tier under `fd_order=4`:

- **Moderate → High** for `laws/momentum_navier_stokes` and overall.
- **Cause:** smaller FD truncation in ∇u, ∇²u, ∇p — residual RMS drops ~2× and score crosses the High cutoff (~0.909).  
- **Not** a change in tier cutoffs or R_eff formulas. Exact-derivative Kovasznay remains ~machine precision / High under both settings.

No other case in the harness changed tier. Supplied-derivative audits are bit-identical.

---

## Recommendation

**Propose default `fd_order=4` and `fd_order_time=4` for Moju 2.0.0** (with `fd_order="legacy"` / `None` preserving 1.x `jnp.gradient` exactly).

Reasons:

1. Clear accuracy gain on smooth periodic flow (TG): ~50× lower residual max vs legacy; order 4 captures most of the benefit of order 6 at lower stencil width.
2. On the only tier-changing case (Kovasznay Path B), order 4 is the **first** order that reaches High; order 6 improves further but does not change the decision for a default.
3. Order 2 (explicit) already beats legacy, so documenting `"legacy"` as the 1.x reproduction mode is important; the new default should be a real accuracy step (4), not merely “explicit 2”.
4. Order 6 needs ≥7 points per axis and shows diminishing returns for typical export grids; keep it selectable for highly resolved periodic data.

**Do not recommend order 6 as the default** from this evidence: better on TG, but the practical tier/accuracy jump that matters for the non-periodic case already happens at 4, and 6 is a steeper requirement for short axes / Studio uploads.

---

## Open items deferred to Phase B

- Switch defaults; add `fd_order="legacy"` alias; golden bit-exact tests vs 1.3.1.
- Full Torch parity + JAX/Torch parity test.
- Migration guide, reproducibility note for the published paper (1.x + settings), CHANGELOG `[2.0.0]`, release prep (no upload).

---

## Approval checkpoint

Please confirm:

1. Proceed with Phase B using **default `fd_order=4` / `fd_order_time=4`**, or prefer **6**.
2. Any additional impact cases you want before the default switch (e.g. full Colab Path B notebook with exported w2 states).
