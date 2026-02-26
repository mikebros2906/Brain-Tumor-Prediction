from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

def main(zip_path: Path, out_dir: Path, variant: str) -> None:
    # Extract ONE dataset variant (BrainTumorYolov8 by default) from the provided zip.
    # The zip contains duplicates for Yolov8/Yolov9/Yolov11. We pick one to avoid bloat.
    assert zip_path.exists(), f"Zip not found: {zip_path}"

    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"BrainTumor/{variant}/"

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix)]
        if not members:
            raise ValueError(f"Could not find {prefix} inside zip.")
        tmp = out_dir.parent / "_tmp_extract"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        zf.extractall(tmp, members)

    extracted_root = tmp / "BrainTumor" / variant
    if not extracted_root.exists():
        raise RuntimeError(f"Unexpected extraction layout; missing: {extracted_root}")

    shutil.copytree(extracted_root, out_dir, dirs_exist_ok=True)
    shutil.rmtree(tmp)

    data_yaml = out_dir / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError(f"Missing data.yaml at: {data_yaml}")

    # Rewrite data.yaml so paths are local and robust
    new_yaml = (
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 3\n"
        "names: ['glioma', 'meningioma', 'pituitary']\n"
    )
    data_yaml.write_text(new_yaml, encoding="utf-8")

    print(f"[OK] Prepared dataset at: {out_dir}")
    print(f"[OK] Rewrote data.yaml with local paths.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True, help="Path to the dataset zip")
    ap.add_argument("--out", type=Path, default=Path("data/brain_tumor_yolov8"), help="Output directory")
    ap.add_argument("--variant", type=str, default="BrainTumorYolov8", help="BrainTumorYolov8/BrainTumorYolov9/BrainTumorYolov11")
    args = ap.parse_args()
    main(args.zip, args.out, args.variant)
