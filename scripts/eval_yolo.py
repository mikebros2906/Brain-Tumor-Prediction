from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO

def main(weights: Path, data_yaml: Path, split: str, device: str) -> None:
    assert weights.exists(), f"Weights not found: {weights}"
    assert data_yaml.exists(), f"data.yaml not found: {data_yaml}"
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, device=device, project="runs/detect", name=f"eval_{split}")
    print("[OK] Evaluation done.")
    try:
        print("mAP50:", float(metrics.box.map50))
        print("mAP50-95:", float(metrics.box.map))
    except Exception:
        pass
    print("See runs/detect/eval_* for plots.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("runs/detect/train/weights/best.pt"))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()
    main(args.weights, args.data, args.split, args.device)
