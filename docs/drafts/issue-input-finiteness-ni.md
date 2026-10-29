# Undefined conditioning from finite inputs counted as non-finite (2.4.1)

## Summary

In implied-parameter audits, Moju **2.4.1** treated a point as non-finite whenever the viscosity-conditioning value (β / fit quality / term fraction) was non-finite. Conditioning can be non-finite from perfectly finite inputs — for example a 0/0 ratio when viscous and inertial terms both vanish. Under `nonfinite_policy="strict"`, those quiet points inflated `n_nonfinite` and forced **Incomplete**, even though they should be **not identifiable**.

## Reproduction (2.4.1)

Regional NS balance with a quiet band where laplacian and inertial terms are zero (all required state inputs finite). Conditioning is undefined; implied δ is NaN.

| Policy | Implied μ tier (2.4.1) | Coverage (quiet band) |
|--------|------------------------|------------------------|
| `"flag"` | High (selected finite) | quiet points in `n_nonfinite` |
| `"strict"` | **Incomplete** (bug) | same; null score |

## Expected (2.4.2)

- A point is **non-finite** only when a required input (or residual component) at that point is non-finite.
- Otherwise, if conditioning fails **or is undefined**, the point is **not identifiable**.
- Quiet 0/0 regions → `n_not_identifiable`, `n_nonfinite = 0`, strict stays scored when selected samples are finite.
- Non-finite required input at an identifiable point → `n_nonfinite = 1`, strict Incomplete (unchanged).
- Non-finite required input at a would-be NI point → non-finite wins (`n_nonfinite = 1`).
- Default `"flag"` scores/tiers bit-for-bit unchanged (paper goldens).

Implemented in #<PR>
