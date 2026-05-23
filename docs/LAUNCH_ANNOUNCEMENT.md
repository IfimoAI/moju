# Project Messaging Snippets

Use these short descriptions for PyPI, GitHub Releases, LinkedIn, X/Twitter, or a short project note.

## One-line Description

Moju provides physics supervision and audit tools for SciML models, with JAX-native residuals and PyTorch support.

## Short Description

Moju turns predicted state fields into governing-law residuals, physics losses, constitutive consistency checks, and audit reports. It separates **average law compliance** (RMS) from **worst-point constitutive integrity** (max |δ|) so PINNs cannot hide closure cheats behind smooth PDE residuals. It is useful for PINNs, CFD surrogates, neural operators, digital twins, and other workflows where model outputs can be represented as state dictionaries.

The core is JAX-native. The `moju.torch` subpackage provides a PyTorch-facing residual engine, R_eff loss helpers, nondimensionalization utilities, and wrappers for using Moju laws with Torch tensors.

Install:

```bash
pip install moju
pip install "moju[viz]"
pip install "moju[torch]"
```

Links:

- PyPI: https://pypi.org/project/moju/
- GitHub: https://github.com/IfimoAI/moju
- Docs: https://ifimoai.github.io/moju/

## Short Social Post

Moju is a lightweight physics supervision layer for SciML and Physics AI: governing-law residuals, constitutive consistency checks, admissibility audits, and Plotly diagnostics from predicted state fields. JAX-native core; PyTorch support via `moju.torch`.

Install: `pip install moju`

Docs: https://ifimoai.github.io/moju/
