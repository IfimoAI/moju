# Moju notebooks — arXiv slab cooling benchmark

Reproducible companion to the Moju arXiv paper on 1D transient slab cooling with constant material properties and Robin convection at `x = L`.

## Path A — paper reproduction (`moju_slab_cooling_arxiv.ipynb`)

Train the paper's `32×32×32` PINN (`[2, 32, 32, 32, 1]`), run L-BFGS for **14,000** steps, and audit training (`64×48` collocation) and eval (`512×384`) grids with Moju.

**Colab**

1. Clone the repository (reference data is not fetched by `pip install` alone):

   ```bash
   git clone https://github.com/IfimoAI/moju.git
   cd moju/examples/Notebooks
   ```

2. Open `moju_slab_cooling_arxiv.ipynb` in Colab or Jupyter.
3. Run **Install**, then execute top-to-bottom. The final cell compares endpoint audit scores to bundled reference outputs.

**Runtime:** full Path A training is on the order of tens of minutes on Colab GPU and longer on CPU (single run, no seed sweep).

## Reference data (`reference/32x32x32_opt/`)

Paper reference audit artifacts for the `32×32×32` architecture (Table 1 / §4.2 eval grid). These are **not** MLP checkpoints — they validate agreement of Moju endpoint scores and constitutive fields within tolerance after you re-run training.

| File | Role |
|------|------|
| `training_admissibility.npy` | Training-trajectory governing / constitutive admissibility curves |
| `category_scores.np.npy` | Expected endpoint per-key admissibility scores |
| `constitutive_fields.npz` | Eval-grid constitutive δ fields (worst-point checks) |

## Path B — instant audit / media (`media/`)

Pre-exported `128×128×128` (w2) state bundles for social posts and quick demos: load `state_pred`, audit, and visualize **without** 14k-step training. Use `media/export_state_zips.py` to export bundles for **any** architecture (`export_state_bundle(..., prefix=...)`). See [`media/README.md`](media/README.md).
