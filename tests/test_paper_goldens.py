"""
Paper slab-cooling goldens (seven architectures).

Requires ``MOJU_PAPER_DATA`` pointing at a directory of ``*_moju_data.zip``
bundles (outside the repo or gitignored). Skips with a clear message when unset.

Never commit the zip bundles or extracted arrays — only the small JSON goldens
under ``tests/goldens/paper_slab_cooling/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from moju.monitor.auditor import admissibility_level
from moju.monitor.implied_scoring import (
    score_from_r_max,
    score_implied_delta_array,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "goldens" / "paper_slab_cooling"
ARCHS = [
    "2x2x2_xs",
    "4x4x4_sm",
    "8x8x8_n2",
    "16x16x16_n1",
    "32x32x32_opt",
    "64x64x64_w1",
    "128x128x128_w2",
]
CONST_KEY = "constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta"
LAW_KEY = "laws/fourier_conduction"


def _paper_data_dir() -> Path | None:
    raw = os.environ.get("MOJU_PAPER_DATA", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def paper_data() -> Path:
    d = _paper_data_dir()
    if d is None:
        pytest.skip(
            "MOJU_PAPER_DATA unset — set it to the directory containing the seven "
            "*_moju_data.zip paper bundles to run paper goldens"
        )
    if not d.is_dir():
        pytest.skip(f"MOJU_PAPER_DATA={d} is not a directory")
    missing = [a for a in ARCHS if not (d / f"{a}_moju_data.zip").is_file()]
    if missing:
        pytest.skip(f"MOJU_PAPER_DATA missing bundles for: {', '.join(missing)}")
    return d


def test_paper_goldens_manifest_lists_all_seven():
    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    assert manifest["architectures"] == ARCHS
    assert manifest["settings"]["implied_metric"] == "worst_point"
    assert manifest["settings"]["implied_noise_normalization"] is False


@pytest.mark.parametrize("arch", ARCHS)
def test_paper_figure_and_sha256_match_golden(arch: str, paper_data: Path):
    golden = json.loads((GOLDEN_DIR / f"{arch}.json").read_text())
    zpath = paper_data / f"{arch}_moju_data.zip"
    assert _sha256(zpath) == golden["input_sha256"]["zip"]

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(td)
        root = Path(td) / f"{arch}_moju_data"
        for name, expected_sha in golden["input_sha256"].items():
            if name == "zip":
                continue
            p = root / name
            assert p.is_file(), name
            got = _sha256(p)
            assert got == expected_sha, f"{arch} {name} SHA-256 mismatch"

        npz = np.load(root / "constitutive_fields.npz")
        delta = npz["delta_grid"]
        r_max = float(np.nanmax(np.abs(delta)))
        score = float(score_from_r_max(r_max))
        exp = golden["expected"]["paper_figure"]
        assert r_max == pytest.approx(exp["r_max"], rel=0, abs=0.0)
        assert score == pytest.approx(exp["admissibility_score"], rel=0, abs=0.0)
        assert admissibility_level(score) == exp["admissibility_level"]
        assert round(100.0 * score, 2) == exp["admissibility_percent"]

        # Training-log endpoint (may differ from paper figure; documented in golden).
        adm = np.load(root / "training_admissibility.npy", allow_pickle=True).item()
        train = golden["expected"]["training_log_endpoint"]
        assert float(np.asarray(adm["constitutive admissibility"]).ravel()[-1]) == pytest.approx(
            train["constitutive_admissibility"], rel=0, abs=0.0
        )
        assert float(np.asarray(adm["governing admissibility"]).ravel()[-1]) == pytest.approx(
            train["governing_admissibility"], rel=0, abs=0.0
        )

        cat = np.load(root / "category_scores.np.npy", allow_pickle=True).item()
        for key, val in golden["expected"]["category_scores"].items():
            assert float(cat[key]) == pytest.approx(val, rel=0, abs=0.0)


@pytest.mark.parametrize("arch", ARCHS)
def test_defaults_match_2_1_score_formula_on_fields(arch: str, paper_data: Path):
    """Differential vs Moju 2.1 formula: score = 1/(1 + max|δ|/1e-2) on identical δ."""
    golden = json.loads((GOLDEN_DIR / f"{arch}.json").read_text())
    zpath = paper_data / f"{arch}_moju_data.zip"
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(td)
        root = Path(td) / f"{arch}_moju_data"
        delta = np.load(root / "constitutive_fields.npz")["delta_grid"]
        r_max = float(np.nanmax(np.abs(delta)))
        legacy = 1.0 / (1.0 + r_max / 1.0e-2)
        branch = float(score_from_r_max(r_max))
        assert branch == legacy
        assert branch == pytest.approx(
            golden["expected"]["paper_figure"]["admissibility_score"], rel=0, abs=0.0
        )


@pytest.mark.parametrize("arch", ARCHS)
def test_optin_informational_only_not_golden(arch: str, paper_data: Path):
    """Opt-in percentile + noise-norm runs; results are informational (not asserted to goldens)."""
    zpath = paper_data / f"{arch}_moju_data.zip"
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(td)
        root = Path(td) / f"{arch}_moju_data"
        npz = np.load(root / "constitutive_fields.npz")
        delta = npz["delta_grid"]
        model = npz["model_grid"]
        implied = npz["implied_grid"]
        raw = model - implied
        v_eff, r_raw, meta = score_implied_delta_array(
            delta,
            {"pred": model, "implied": implied, "raw": raw},
            {
                "implied_metric": "percentile",
                "implied_percentile": 95.0,
                "implied_noise_normalization": True,
            },
        )
        assert meta["implied_metric"] == "percentile"
        assert meta["implied_noise_normalization"] is True
        assert float(r_raw) == pytest.approx(float(np.nanmax(np.abs(delta))), rel=0, abs=1e-15)
        # Informational print for suite logs (not a golden assertion).
        score = float(score_from_r_max(float(v_eff)))
        print(
            f"[opt-in informational] {arch}: defaults paper "
            f"{100.0 * float(score_from_r_max(float(r_raw))):.2f}% → "
            f"percentile+noise {100.0 * score:.2f}% ({admissibility_level(score)})"
        )
