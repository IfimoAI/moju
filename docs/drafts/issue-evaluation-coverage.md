# Evaluation coverage on every verdict (non-finite samples)

## Summary

A community report shows that a verdict can claim **High Admissibility** when part of the evaluation is silently missing because non-finite residual samples (NaN) are dropped by NaN-aware reductions. Tolerating NaN is intentional, but a verdict must say what it was computed on. Silent partial coverage is a correctness issue for admissibility claims.

## Reproduction

Minimal case (Moju 2.3.2 / commit `33f107dc`):

```python
from moju.monitor import ResidualEngine, audit, LawSpec, Scoring
import jax.numpy as jnp

eng = ResidualEngine(
    laws=[LawSpec(
        name="unit_law",
        fn=lambda r: r,
        state_map={"r": "r"},
        scoring=Scoring("rms", dimensionless=True),
    )],
    law_implied_audits=False,
)
eng.compute_residuals({"r": jnp.array(values)}, run_mode="eval")
report = audit(eng.log)
```

| Input | Score (approx.) | Tier | Path |
|-------|-----------------|------|------|
| `[0, 0]` | ~0.99999999 | High Admissibility | audit, batched evaluate, loop evaluate |
| `[0, NaN]` | ~0.99999999 | High Admissibility | same (half the evaluation silently missing) |
| `[NaN, NaN]` | 0 | Non-Admissible | same |
| `[0, inf]` / `[0, -inf]` | 0 | Non-Admissible | same (inf propagates through RMS; still no coverage record) |
| p95 / worst-point reduce on `[0, 1, NaN]` or `[0, 1, inf]` | non-finites dropped | — | `reduce_implied_magnitude_arrays` |
| Implied μ with one NaN in a required field | High possible | High | audit |

## Why it matters

Admissibility tiers are evidence claims. A **High** label on `[0, NaN]` is computed only on the finite subset, with no signal that half the samples were unused. Consumers of reports (and automated gates) can treat that as full coverage.

## Proposed design

1. **Coverage record** on every scored item (each law, each implied audit, and the aggregate), JSON-serializable:
   - `n_total`, `n_masked` (explicit `eval_mask` exclusions), `n_selected` (= n_total − n_masked), `n_finite`, `n_nonfinite`
   - `coverage_status`: `"ok"` | `"incomplete"` | `"empty"`
   - Aggregate status = worst of components (`empty` > `incomplete` > `ok`)
2. **`eval_mask` provenance**: masked samples count as `n_masked`, never as non-finite. Optional `eval_mask_label`.
3. **`nonfinite_policy`** on `audit` / scoring configuration:
   - `"ignore"`: current scores; coverage still reported
   - `"flag"` (**default**): scores/tiers unchanged (backward compatible); coverage reported
   - `"strict"`: any item with `coverage_status ≠ "ok"` → `score = null`, `tier = "Incomplete"` (distinct from "Not identifiable"; means not assessable, not failed). Aggregate Incomplete if any required item is.
4. **Empty evaluations**: under `"flag"`, keep today’s score `0` with `coverage_status = "empty"` (docs: “no evidence”, not “evidence of failure”). Under `"strict"`: null / Incomplete.
5. **Batched path parity** with the unbatched path for coverage records (same class of bug as the 2.3.1 batched NI gap).

## Backward compatibility

Default `"flag"` preserves bit-for-bit scores and tiers for existing goldens and paper figures. Callers opt into `"strict"` when they want incomplete coverage to block a High/Moderate/Low verdict.
