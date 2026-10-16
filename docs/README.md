# GitHub Pages Source

This folder is the source for **https://ifimoai.github.io/moju/**.

- **index.html** - concise public landing page with install, links, the Path B slab-cooling quickstart, and `moju.torch` mention.
- **doc/** - static API overview pages for Groups, Models, Laws, Operators, and `moju.monitor` architecture.
- **monitor_training_vs_eval.md** - canonical `run_mode` behavior, split admissibility metrics, **`law_scale_mode`** (auto law scale_k), **`state_units`** (Path B SI), **`r_ref` / `scale_k` calibration**, category rollups, and visualization differences.
- **path_b_derivatives.md** - Path B optional derivative fill: FD (default) vs periodic **spectral** (`diff_method="spectral"`), Torch parity, requirements, and cookbooks.
- **law_implied_audits.md** - law-linked constitutive implied-audit behavior (Fourier → α, Burgers → **ν** via `kinematic_viscosity_from_re`, NS/Stokes → μ), pointwise closure scoring, opt-in robust metrics (2.2), and coverage.
- **MIGRATION_2.1_to_2.2.md** - additive opt-in implied scoring; defaults preserve 2.1 / paper results.
- **MIGRATION_2.2_to_2.3.md** - optional named regions for per-region fits/scores; ``regions=None`` preserves 2.2.
- **LAUNCH_ANNOUNCEMENT.md** - short neutral project descriptions and release messaging snippets.

To enable the site: **GitHub repo -> Settings -> Pages -> Build and deployment -> Source:** choose **Deploy from a branch**. Set **Branch** to `main` and **Folder** to `/docs`, then Save.
