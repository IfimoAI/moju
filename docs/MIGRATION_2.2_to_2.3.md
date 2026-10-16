# Migration: Moju 2.2 → 2.3

**2.3 is additive.** With ``regions=None`` (the default), behaviour is bit-identical
to 2.2.0. Published paper goldens and global scores are unchanged.

## Region-aware implied-law fitting and scoring

Optional named regions select which points enter **per-region** implied-parameter
fits and per-law scores. Derivatives always use the full field. The **global**
audit result is computed exactly as in 2.2 and is unaffected by regions.

### API

Accepted on ``ResidualEngine`` / ``TorchResidualEngine`` constructors,
``compute_residuals``, ``evaluate``, and ``audit`` (``audit`` also accepts the
same kwargs; per-call > constructor / any ``region_results`` already on the log.
``audit(..., regions=...)`` needs ``last_residual_dict``):

| Argument | Default | Meaning |
|----------|---------|---------|
| `regions` | `None` | `dict[str, mask]` or integer label array |
| `region_names` | `None` | label → name when using a label array |
| `region_min_points` | `16` | fit floor; below this → ``Insufficient points`` |

**Precedence:** per-call keyword argument > constructor. Regions are **not**
accepted via ``constants``.

``evaluate(..., regions=...)`` scores the batch in one ``jax.jit`` / ``jax.vmap``
call with region masks as stacked 0/1 weights ``(n_regions, n_points)``. Numeric
per-region outputs (ν̃, fit quality, scores, worst-point index, tail count) are
computed inside the transform; status strings are derived afterward. There is no
silent per-item loop when regions are set — pass ``evaluate_debug_loop=True`` only
for debugging.

``evaluate`` with opt-in implied scoring
(``implied_metric="percentile"`` and/or ``implied_noise_normalization=True``) and
**no** regions also uses the same batched path (jit-safe masked percentile /
noise-normalized δ). Default worst-point settings remain bit-identical to 2.2.0.

Masks are boolean or 0/1 arrays matching the point layout. Regions may overlap;
the report records ``n_unassigned`` for points in no region.

### Batched ``evaluate``: when it raises (no silent fallback)

With ``regions`` set and/or opt-in implied scoring, batching is **required**.
These cases raise ``RuntimeError`` unless ``evaluate_debug_loop=True``:

| Situation | Behaviour |
|-----------|-----------|
| ``best_effort_partial=True`` | Cannot batch — raises |
| ``user_fns`` present | Cannot batch — raises |
| Non-uniform candidate shapes | Cannot batch — raises |
| ``auto_path_b_derivatives`` / ``fill_law_fd`` **with** ``regions`` | Raises (see below) |

Without regions and with default implied scoring, an untraceable batch still
falls back to the per-item loop (same as 2.2).

### Path B / ``fill_law_fd`` (existing limitation)

Path B derivative fill is **loop-only** in ``evaluate``, with or without regions:
the compiled batch path does not run Path B fill. Combining ``regions=...`` with
``auto_path_b_derivatives`` / ``fill_law_fd`` therefore raises unless
``evaluate_debug_loop=True``. Derivatives stay global when you fill outside
``evaluate`` and then pass a complete state; regions only select points for
fits and scores.

### Default ``region_min_points = 16`` (fit floor only)

Chosen so that:

1. A region has at least twice ``implied_min_identifiable_count`` (8) before a
   viscosity gate is even considered.
2. It matches the existing spatial safeguard floor
   (``weighted_safeguard.DEFAULT_MIN_REGION_ABS``).

It is **not** a percentile-adequacy threshold and is **not** tuned to any dataset.
Percentile adequacy is reported separately (see below).

### Percentile adequacy flags

When ``implied_metric="percentile"``, Moju reports expected excluded-tail count
``n_points × (1 − p/100)`` for each region and for the global audit:

| Tail count | ``percentile_status`` |
|------------|------------------------|
| ``tail < 1`` | ``equivalent_to_worst_point`` |
| ``1 ≤ tail < 5`` | ``low_tail_count`` |
| otherwise | ``ok`` |

For **p95**, ``tail < 1`` means ``n < 20`` (percentile cannot exclude any point);
``n ≥ 100`` is recommended for a stable upper-tail estimate (``tail ≥ 5``).
Flags and ``percentile_tail_count`` appear on each region block and on
``percentile_adequacy`` in the audit report / JSON export.

### What Moju does not do

Moju does **not** define regions, choose which regions matter, or aggregate
per-region scores into a gate or overall verdict. Per-region tables are
additional reporting only.

See [`law_implied_audits.md`](law_implied_audits.md#region-aware-fitting-and-scoring-moju-23).
