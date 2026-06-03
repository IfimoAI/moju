# Moju notebooks — slab cooling benchmark

Reproducible companion to the [Moju preprint on Zenodo](https://zenodo.org/records/20519331): 1D transient slab cooling with constant material properties and Robin convection at `x = L`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IfimoAI/moju/blob/main/examples/Notebooks/moju_slab_cooling_paper.ipynb)

**Colab badges** open notebooks from GitHub **`main`**. Push merged changes before sharing badge links.

## Path A — paper reproduction (`moju_slab_cooling_paper.ipynb`)

Train the paper's `32×32×32` PINN (`[2, 32, 32, 32, 1]`), run L-BFGS for **14,000** steps, and audit training (`64×48` collocation) and eval (`512×384`) grids with Moju.

**Colab**

1. Click **Open in Colab** above (or open the notebook in Jupyter after cloning).
2. Run **Install** and **Setup (Colab)** — clones the repo to `/content/moju` and `cd`s into this folder for `reference/`.
3. Execute top-to-bottom. The verify cell compares endpoint audit scores to bundled reference outputs.

**Runtime:** full Path A training is on the order of tens of minutes on Colab GPU and longer on CPU (single run, no seed sweep).

**Maintainers:** strip outputs before commit (`nbstripout` or clear all outputs in Jupyter) so notebooks stay small on GitHub.

## Reference data (`reference/32x32x32_opt/`)

Paper reference audit artifacts for the `32×32×32` architecture (Table 1 / §4.2 eval grid). These are **not** MLP checkpoints — they validate agreement of Moju endpoint scores and constitutive fields within tolerance after you re-run training.

| File | Role |
|------|------|
| `training_admissibility.npy` | Training-trajectory governing / constitutive admissibility curves |
| `category_scores.np.npy` | Expected endpoint per-key admissibility scores |
| `constitutive_fields.npz` | Eval-grid constitutive δ fields (worst-point checks) |

## Path B — instant audit / media (`media/`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IfimoAI/moju/blob/main/examples/Notebooks/media/moju_slab_cooling_path_b.ipynb) [`moju_slab_cooling_path_b.ipynb`](media/moju_slab_cooling_path_b.ipynb)

Pre-exported `128×128×128` (w2) state bundles for social posts and quick demos: load `state_pred`, audit, and visualize **without** 14k-step training. Use `media/export_state_zips.py` to export bundles for **any** architecture (`export_state_bundle(..., prefix=...)`). See [`media/README.md`](media/README.md).
