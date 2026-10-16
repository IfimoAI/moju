# Migration: Moju 2.1 → 2.2

**2.2 is additive.** Default settings match 2.1. Published paper slab-cooling
results (Path A / Path B) reproduce under defaults.

## Opt-in implied constitutive scoring

Configure via **explicit keyword arguments** or engine ``constants``
(``implied_delta`` rows only). Governing-law scoring is unchanged.

| Option | Default | Meaning |
|--------|---------|---------|
| `implied_metric` | `"worst_point"` | `"worst_point"` or `"percentile"` |
| `implied_percentile` | `95` | Used when metric is `"percentile"` |
| `implied_noise_normalization` | `False` | Soften δ by local balance-noise level |

### Per-call API and precedence

Accepted on:

- `ResidualEngine(..., implied_metric=..., implied_percentile=..., implied_noise_normalization=...)`
- `engine.compute_residuals(..., implied_metric=..., ...)`
- `audit(..., implied_metric=..., constants=...)` (re-score needs `last_residual_dict`)
- `evaluate(..., implied_metric=..., ...)` (forces the Python loop when options are non-default)
- `TorchResidualEngine(...)` constructor (forwarded into the JAX audit path)

**Precedence:** per-call keyword argument → constructor keyword argument → `constants` →
module default.

Any conflict between two sources that set the same option to **different** values raises a
`UserWarning` naming both sources and the value used. The `constants` route remains fully
supported (no deprecation in 2.2).

Unknown metrics, percentiles outside `(0, 100]`, and wrong types raise clear
`ValueError` / `TypeError`.

Every `audit` / `evaluate` report (and JSON export summary) includes
`implied_scoring_settings` with `implied_metric`, `implied_percentile`,
`implied_noise_normalization`, and `moju_version`. **Scores and tiers from default and
opt-in settings are not directly comparable** — always record which settings produced a
result before comparing runs.

`MonitorConfig` does **not** yet carry these three fields (follow-up); pass them on the
engine constructor / `constants` / per-call kwargs.

```python
# Kwargs-only (preferred for one-off audits)
engine = ResidualEngine(
    ...,
    implied_metric="percentile",
    implied_percentile=95,
    implied_noise_normalization=True,
)

# Constants-only (unchanged from early 2.2 drafts)
engine = ResidualEngine(
    ...,
    constants={
        "implied_metric": "percentile",
        "implied_percentile": 95,
        "implied_noise_normalization": True,
        # ... other constants ...
    },
)
```

### When to prefer noise normalization

See [`law_implied_audits.md`](law_implied_audits.md#when-to-prefer-noise-normalization).

### Noise-normalization formula

For a property entering the balance linearly as ``lhs ≈ F · basis`` (or LSQ
``rhs ≈ β(F) · basis``):

```text
ε(x) = |r_balance(x)| / (|basis(x)| + ε_m)
δ_nn(x) = (F − implied) / (|F| + ε(x) + ε_norm)
```

- Ratio recovery (Fourier α, …): ``r_balance = lhs − F · basis`` ⇒ ``ε = |F − implied|``.
- LSQ viscosity: ``r_balance`` is the vector residual ``rhs − β_F · ∇²u`` (last-axis
  norm); ``ε`` is converted to μ units when the catalog property is μ.

Aggregation then applies ``worst_point`` or ``percentile`` to ``|δ|`` or ``|δ_nn|``.
Raw ``max |δ|`` is always retained as ``r_max_raw`` in the report.

## Reproducing the paper figures

Seven architecture bundles (`*_moju_data.zip`) are **not** in the git tree. Point
``MOJU_PAPER_DATA`` at a local folder that contains them, then run:

```bash
export MOJU_PAPER_DATA=/path/to/dir_with_seven_zips
pytest tests/test_paper_goldens.py -q
```

Goldens live in ``tests/goldens/paper_slab_cooling/`` (architecture name, SHA-256 of
each input file, expected scores/tiers, Moju version, default settings).

| Reported quantity | Source | Example (`32×32×32`) |
|-------------------|--------|----------------------|
| **Paper constitutive %** | `constitutive_fields.npz` → `nanmax(\|delta_grid\|)` → `1/(1+r_max/1e-2)` | `max\|δ\|=0.00073134` → **0.93185 → 93.19%** |
| Training-log constitutive % | `training_admissibility.npy` last constitutive score | **0.93424 → 93.42%** |

The training-log figure uses the collocation-grid worst-point logged during training;
the paper figure uses the exported eval-grid δ field. Both are valid; cite the
documented quantity. Under **default** settings, Moju 2.2 scores match Moju 2.1
exactly on identical inputs.
