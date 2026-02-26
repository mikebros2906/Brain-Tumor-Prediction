# Brain Tumor "Presence" Dashboard (Portfolio / Research)

⚠️ **Not medical advice / not for clinical use.**
This project is for ML learning + portfolio demos only.

## What this repo does
- Trains a YOLOv8 object detector to localize tumor regions and classify tumor *type* (glioma / meningioma / pituitary).
- Uses detections to derive a simple "tumor present?" signal: **present = any box above confidence threshold**.
- Runs a Streamlit dashboard where you upload an MRI image and see bounding boxes + confidences.

## Important limitation (read this)
The provided dataset contains only tumor classes (3 types) and **no "no-tumor" / normal** class.
So you cannot honestly train a true present-vs-absent classifier with this dataset alone.
You can still output "No tumor detected" when the model finds no boxes, but it won’t be reliable for truly normal MRIs.

## Quickstart
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare data:
```bash
python scripts/prepare_data.py --zip "/path/to/archive (2).zip"
```

Train:
```bash
python scripts/train_yolo.py --data data/brain_tumor_yolov8/data.yaml --epochs 50 --img 640
```

Run dashboard:
```bash
streamlit run app/app.py
```
