from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO

def main(data_yaml: Path, epochs: int, imgsz: int, batch: int, device: str, model: str) -> None:
    assert data_yaml.exists(), f"data.yaml not found: {data_yaml}"
    yolo = YOLO(model)
    yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=10,
        project="runs/detect",
        name="train",
        pretrained=True,
        verbose=True,
    )
    print("[OK] Training complete.")
    print("Best weights: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="0", help='GPU id like "0", or "cpu"')
    ap.add_argument("--model", type=str, default="yolov8s.pt", help="yolov8n.pt / yolov8s.pt / yolov8m.pt ...")
    args = ap.parse_args()
    main(args.data, args.epochs, args.img, args.batch, args.device, args.model)
