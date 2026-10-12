# Phase A5: periodic axes + percentile rerun

Defaults are **not** switched in this step.

## Periodic-axis declaration

- `PathBGridConfig.periodic` may be `False`, `True`, or a per-axis sequence such as `(False, True)` for Kovasznay (`x` bounded, `y` periodic).
- Never inferred. Misdeclaring a periodic physical axis as bounded inflates boundary residuals (one-sided closures on a wrap-around field).
- Spectral differentiation still requires **all** spatial axes declared periodic.

## Second-derivative / mixed boundary orders

- poly-exp d2 boundary observed order: 4.015 (pass=True)
- mixed poly-exp boundary observed order: 4.060 (pass=True)
- Couette radial d2 boundary observed order: 3.848 (pass=True)

Matching Fornberg boundary bands for second derivatives now use width `order + 2` so formal order matches the interior.

## Grid-scale resolution-sensor warning

- Warns when Nyquist-bin spectral energy exceeds threshold `0.05` (checkerboard / grid-scale artifacts).
- Diagnostic only; never changes admissibility tiers.

- Checkerboard flag: {'grid_scale_energy': 1.0, 'grid_scale_flag': True, 'n_warnings': 1}
- Narrow Gaussian: {'grid_scale_energy': 8.248595093601346e-07, 'grid_scale_flag': False, 'n_warnings': 1}

## Recommendation (not applied)

- Rule: **c**
- `safeguard_percentile` = 95.0
- `safeguard_error_aware` = False

- Reason: Neither plain nor error-aware p97.5 clears the prefer rule; keep plain p95.

**Why not (a)/(b):** Declaring Kovasznay `periodic=(False, True)` removes the
spurious *y*-edge spikes, but the physical *x*-boundaries still produce ~5% spike
fractions under Path B fd4. Clean Kov therefore stays Moderate/Low for plain and
error-aware p97.5 (and even plain p95) on both resolutions. Taylor–Green with
`periodic=(True, True)` is clean High at all three percentiles. Defaults remain
unchanged until you approve a follow-up.

## Spike fractions (clean Path B)

| case | boundary spike frac | plain p95 | plain p97.5 | plain p99 |
|---|---|---|---|---|
| taylor_green/coarse | 0.0000 | High Admissibility | High Admissibility | High Admissibility |
| taylor_green/fine | 0.0000 | High Admissibility | High Admissibility | High Admissibility |
| kovasznay/coarse | 0.0544 | Moderate Admissibility | Low Admissibility | Non-Admissible |
| kovasznay/fine | 0.0500 | Low Admissibility | Low Admissibility | Non-Admissible |

