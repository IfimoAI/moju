"""Export PINN state dicts as deflated JSON zip bundles for Path B media demos."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

WIDE2_PREFIX = "wide2_const_prop_1D_cooling_slab"


def jax_serializable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def write_state_json_zip(
    state_dict: Mapping[str, Any],
    out_zip_path: str | Path,
    json_name: str,
) -> Path:
    """Write ``state_dict`` to ``json_name`` inside a deflated zip archive."""
    out_zip_path = Path(out_zip_path)
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = out_zip_path.with_suffix(out_zip_path.suffix + ".tmp.json")
    try:
        with tmp_json.open("w", encoding="utf-8") as handle:
            json.dump(state_dict, handle, default=jax_serializable, indent=2)
        with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp_json, arcname=json_name)
    finally:
        if tmp_json.exists():
            tmp_json.unlink()
    return out_zip_path


def wide2_training_zip_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / f"{WIDE2_PREFIX}_final_training_state.json.zip"


def wide2_test_zip_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / f"{WIDE2_PREFIX}_test_state_pred.json.zip"


def export_wide2_states(
    state_final: Mapping[str, Any],
    state_pred: Mapping[str, Any],
    data_dir: str | Path,
) -> tuple[Path, Path]:
    """Write both wide2 Path B zip bundles under ``data_dir``."""
    data_dir = Path(data_dir)
    train_json = f"{WIDE2_PREFIX}_final_training_state.json"
    test_json = f"{WIDE2_PREFIX}_test_state_pred.json"
    train_zip = write_state_json_zip(
        state_final,
        wide2_training_zip_path(data_dir),
        train_json,
    )
    test_zip = write_state_json_zip(
        state_pred,
        wide2_test_zip_path(data_dir),
        test_json,
    )
    return train_zip, test_zip


def load_state_from_json_zip(zip_path: str | Path) -> dict[str, Any]:
    """Load the first ``.json`` member from a state zip archive."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        json_members = [n for n in zf.namelist() if n.endswith(".json") and not n.startswith("__MACOSX")]
        if not json_members:
            raise ValueError(f"No JSON member found in {zip_path}")
        with zf.open(json_members[0]) as handle:
            return json.load(handle)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repack",
        metavar="ZIP",
        nargs="+",
        help="Repack existing zip(s) through the export helper (cleans __MACOSX entries).",
    )
    args = parser.parse_args()
    if args.repack:
        for zip_path in args.repack:
            state = load_state_from_json_zip(zip_path)
            json_name = Path(zip_path).stem.replace(".json", "") + ".json"
            if not json_name.endswith(".json"):
                json_name = Path(zip_path).name.replace(".zip", "")
            write_state_json_zip(state, zip_path, json_name)
            print(f"repacked {zip_path}")
