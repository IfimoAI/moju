# Migration: Moju 2.2 → 2.3

**2.3 is additive.** With ``regions=None`` (the default), behaviour is bit-identical
to 2.2.0. Published paper goldens and global scores are unchanged.

## Region-aware implied-law fitting and scoring

Optional named regions select which points enter **per-region** implied-parameter
fits and per-law scores. Derivatives always use the full field. The **global**
audit result is computed exactly as in 2.2 and is unaffected by regions.

### API

Accepted on ``ResidualEngine`` / ``TorchResidualEngine`` constructors,
``compute_residuals``, ``audit`` (via engine log), and ``evaluate``:

| Argument | Default | Meaning |
|----------|---------|---------|
| `regions` | `None` | `dict[str, mask]` or integer label array |
| `region_names` | `None` | label → name when using a label array |
| `region_min_points` | `16` | below this → status ``Insufficient points`` |

**Precedence:** per-call keyword argument > constructor. Regions are **not**
accepted via ``constants``.

Masks are boolean or 0/1 arrays matching the point layout. Regions may overlap;
the report records ``n_unassigned`` for points in no region.

### Default ``region_min_points = 16``

Chosen so that:

1. A region has at least twice ``implied_min_identifiable_count`` (8) before a
   viscosity gate is even considered.
2. A percentile metric (default p95 when opted in) has enough in-region points
   to be meaningful.
3. It matches the existing spatial safeguard floor
   (``weighted_safeguard.DEFAULT_MIN_REGION_ABS``).

It is **not** tuned to any dataset.

### What Moju does not do

Moju does **not** define regions, choose which regions matter, or aggregate
per-region scores into a gate or overall verdict. Per-region tables are
additional reporting only.

See [`law_implied_audits.md`](law_implied_audits.md#region-aware-fitting-and-scoring-moju-23).
