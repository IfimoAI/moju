# Migration: Moju 2.1 → 2.2

**2.2 is additive.** Default settings match 2.1. Published paper slab-cooling
results (Path A / Path B) reproduce under defaults.

## Opt-in implied constitutive scoring

Configure via engine ``constants`` (implied_delta rows only):

| Constant | Default | Meaning |
|----------|---------|---------|
| `implied_metric` | `"worst_point"` | `"worst_point"` or `"percentile"` |
| `implied_percentile` | `95` | Used when metric is `"percentile"` |
| `implied_noise_normalization` | `False` | Soften δ by local balance-noise level |

### When to enable

- Noisy field exports or derivative-limited Path B fills
- Second-derivative-based implied properties (viscosity, diffusivity) where a
  few spikes dominate ``max |δ|``
- You want a percentile view of constitutive closure without changing law scoring

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

```python
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
