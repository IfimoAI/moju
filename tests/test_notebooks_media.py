"""Smoke tests for examples/Notebooks Path B zips and Path A reference bundle."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "examples" / "Notebooks"
MEDIA = NOTEBOOKS / "media"
MEDIA_DATA = MEDIA / "data"
REF = NOTEBOOKS / "reference" / "32x32x32_opt"
PATH_A_NB = NOTEBOOKS / "moju_slab_cooling_arxiv.ipynb"
PATH_B_NB = MEDIA / "moju_slab_cooling_path_b.ipynb"
COLAB_BADGE = "colab.research.google.com/assets/colab-badge.svg"
WIDE2_RAW = (
    "https://github.com/IfimoAI/moju/raw/main/examples/Notebooks/media/data/"
    "wide2_const_prop_1D_cooling_slab_test_state_pred.json.zip"
)

sys.path.insert(0, str(ROOT / "examples" / "Notebooks" / "media"))
from export_state_zips import (  # noqa: E402
    DEFAULT_SLAB_PREFIX,
    export_state_bundle,
    load_state_from_json_zip,
    state_bundle_paths,
)

from moju.monitor import ResidualEngine, audit, implied_group_specs_for_laws  # noqa: E402
from moju.piratio import Models  # noqa: E402


def _slab_engine_kw():
    L, cp, h, k, rho = 0.1, 900.0, 500.0, 200.0, 2700.0
    return {
        "constants": {
            "L": L,
            "cp": cp,
            "h": h,
            "k": k,
            "rho": rho,
            "alpha": Models.thermal_diffusivity(k, rho, cp),
        },
        "laws": [
            {
                "name": "fourier_conduction",
                "state_map": {
                    "T_t": "T_t",
                    "T_laplacian": "T_xx",
                    "fo": "fo",
                    "t": "t",
                    "L": "L",
                },
            }
        ],
        "groups": implied_group_specs_for_laws(["fourier_conduction"]),
    }


@pytest.mark.parametrize(
    "zip_name",
    [
        "wide2_const_prop_1D_cooling_slab_final_training_state.json.zip",
        "wide2_const_prop_1D_cooling_slab_test_state_pred.json.zip",
    ],
)
def test_bundled_demo_zip_loads_and_audits(zip_name):
    zip_path = MEDIA_DATA / zip_name
    assert zip_path.exists(), zip_path
    with zipfile.ZipFile(zip_path) as zf:
        assert not any(n.startswith("__MACOSX") for n in zf.namelist())

    state = load_state_from_json_zip(zip_path)
    assert set(state.keys()) >= {"T", "T_t", "T_x", "T_xx", "t", "x"}

    engine = ResidualEngine(**_slab_engine_kw())
    state_jax = {k: jnp.asarray(v) for k, v in state.items()}
    residuals = engine.compute_residuals(state_jax, log_to_python=True)
    report = audit(engine.log, last_residual_dict=residuals)
    assert report["overall_admissibility_score"] > 0.0


def test_path_a_reference_bundle_present():
    assert (REF / "training_admissibility.npy").exists()
    assert (REF / "constitutive_fields.npz").exists()
    cat = REF / "category_scores.np.npy"
    if not cat.exists():
        cat = REF / "category_scores.npy"
    assert cat.exists()


def test_path_a_reference_verify_logic():
    """Mirror the arxiv notebook tolerance check against stored reference only."""
    ref_adm = np.load(REF / "training_admissibility.npy", allow_pickle=True).item()
    cat_path = REF / "category_scores.np.npy"
    if not cat_path.exists():
        cat_path = REF / "category_scores.npy"
    ref_cat = np.load(cat_path, allow_pickle=True).item()

    gov = ref_adm["governing admissibility"][-1]
    const = ref_adm["constitutive admissibility"][-1]
    assert gov > 0.99
    assert 0.90 < const < 0.96
    for key, val in ref_cat.items():
        assert key.startswith(("laws/", "constitutive/"))
        assert 0.0 < float(val) <= 1.0

    arxiv_nb = PATH_A_NB
    assert arxiv_nb.exists()
    assert arxiv_nb.is_file()


def test_colab_notebooks_present_and_lightweight():
    assert PATH_A_NB.exists()
    assert PATH_B_NB.exists()
    assert PATH_A_NB.stat().st_size < 500_000
    assert PATH_B_NB.stat().st_size < 500_000


def test_path_a_notebook_colab_setup():
    nb = json.loads(PATH_A_NB.read_text())
    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert COLAB_BADGE in joined
    assert "git clone" in joined
    assert "optax" in joined
    assert "/content/moju/examples/Notebooks" in joined


def test_path_b_notebook_colab_and_wide2_url():
    nb = json.loads(PATH_B_NB.read_text())
    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert COLAB_BADGE in joined
    assert WIDE2_RAW in joined
    assert "w2_const_prop_1D_cooling_slab" not in joined
    assert "__MACOSX" in joined or "json_members" in joined


def test_readme_colab_badges():
    root_readme = (ROOT / "README.md").read_text()
    notebooks_readme = (NOTEBOOKS / "README.md").read_text()
    media_readme = (MEDIA / "README.md").read_text()
    for text in (root_readme, notebooks_readme, media_readme):
        assert COLAB_BADGE in text
    assert "moju_slab_cooling_arxiv.ipynb" in root_readme
    assert "moju_slab_cooling_path_b.ipynb" in root_readme


def test_export_state_bundle_roundtrip(tmp_path):
    state = {
        "T": [[300.0, 310.0], [320.0, 330.0]],
        "T_t": [[0.0, 0.1], [0.2, 0.3]],
        "T_x": [[1.0, 1.1], [1.2, 1.3]],
        "T_xx": [[0.01, 0.02], [0.03, 0.04]],
        "t": [1.0, 2.0],
        "x": [[0.0], [0.05]],
    }
    prefix = "test_arch_const_prop_1D_cooling_slab"
    train_zip, test_zip = export_state_bundle(state, state, tmp_path, prefix=prefix)
    assert train_zip == state_bundle_paths(tmp_path, prefix)[0]
    assert test_zip == state_bundle_paths(tmp_path, prefix)[1]
    assert load_state_from_json_zip(train_zip)["T"] == state["T"]
    assert load_state_from_json_zip(test_zip)["T"] == state["T"]


def test_export_state_bundle_default_prefix_matches_demo(tmp_path):
    state = {"T": [300.0], "T_t": [0.0], "T_x": [1.0], "T_xx": [0.01], "t": [1.0], "x": [[0.0]]}
    train_zip, test_zip = export_state_bundle(state, state, tmp_path)
    assert train_zip.name == f"{DEFAULT_SLAB_PREFIX}_final_training_state.json.zip"
    assert test_zip.name == f"{DEFAULT_SLAB_PREFIX}_test_state_pred.json.zip"
