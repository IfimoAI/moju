# Versioning policy

moju follows [Semantic Versioning](https://semver.org/) (semver) with one clarification for the 0.x line.

## Version format

**MAJOR.MINOR.PATCH** (e.g. 0.1.3)

- **PATCH** (0.1.x → 0.1.4): Bug fixes and backward-compatible changes only. No API breakage.
- **MINOR** (0.1 → 0.2): New features, optional parameters, or new APIs. Existing public APIs remain supported; breaking changes are avoided but may occur with a deprecation period.
- **MAJOR** (0.x → 1.0, or 1.x → 2.0): Breaking changes allowed. Public APIs may be removed or changed; we document migrations in release notes.

## 0.x vs 1.0

- **0.x:** The library is still evolving. We keep patch releases (0.1.x) backward compatible. Minor releases (0.2, 0.3) may introduce breaking changes after deprecation where feasible.
- **1.0:** We commit to backward compatibility for the 1.x line: no breaking changes in 1.x without a major (2.0) bump.

## Deprecation

Before removing or changing a public API we will:

1. Mark it deprecated (docstring and/or `warnings.warn`) in a release.
2. Remove or change it in a later **major** (or minor, for 0.x) release, with a note in the changelog.

## Summary

| Release type | Backward compatible? |
|--------------|----------------------|
| Patch (x.y.**z**) | Yes |
| Minor (x.**y**.0) | Aim for yes; 0.x may break after deprecation |
| Major (**x**.0.0) | No; breaking changes expected |
