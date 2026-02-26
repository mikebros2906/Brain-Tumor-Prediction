# app/app.py
from __future__ import annotations

import os
import time
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from ultralytics import YOLO


# ----------------------------
# Page config (SaaS-like)
# ----------------------------
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Modern CSS (Streamlit hacking)
# ----------------------------
st.markdown(
    """
<style>
/* App background */
.stApp {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(99,102,241,.18), transparent 55%),
              radial-gradient(1000px 500px at 80% 10%, rgba(34,197,94,.12), transparent 55%),
              #0b1220;
  color: #e5e7eb;
}

/* Remove extra top padding */
.block-container { padding-top: 1.2rem !important; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.01));
  border-right: 1px solid rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] * { color: #e5e7eb; }

/* Headings */
h1, h2, h3 { letter-spacing: -0.02em; }

/* Card component */
.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}
.card h4 { margin: 0 0 6px 0; font-size: 0.9rem; color: rgba(229,231,235,.8); }
.card .big { font-size: 1.35rem; font-weight: 700; color: #ffffff; }
.muted { color: rgba(229,231,235,.7); }

/* Buttons */
.stDownloadButton button, .stButton button {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,.14) !important;
  background: rgba(255,255,255,.06) !important;
}
.stDownloadButton button:hover, .stButton button:hover {
  background: rgba(255,255,255,.10) !important;
}

/* File uploader */
div[data-testid="stFileUploaderDropzone"] {
  border-radius: 16px;
  border: 1px dashed rgba(255,255,255,.18);
  background: rgba(255,255,255,.03);
}

/* Dataframe */
div[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.10);
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers
# ----------------------------
def resolve_default_weights() -> str:
    # Prefer your stable "best.pt" if it exists, else fallback to yolov8n.pt in root
    candidates = [
        Path("models") / "best.pt",
        Path("runs") / "detect" / "train" / "weights" / "best.pt",
        Path("yolov8n.pt"),
        Path("yolov8s.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p.as_posix()
    return ""


@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    if cv2 is None:
        # If OpenCV missing, keep RGB as "fake BGR"
        return rgb[:, :, ::-1].copy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    if cv2 is None:
        rgb = bgr[:, :, ::-1]
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def results_to_table(r) -> pd.DataFrame:
    # Ultralytics Results -> dataframe
    if r.boxes is None or len(r.boxes) == 0:
        return pd.DataFrame(columns=["type", "confidence", "x1", "y1", "x2", "y2"])

    names = r.names
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    xyxy = r.boxes.xyxy.cpu().numpy()

    rows = []
    for i in range(len(cls)):
        rows.append(
            {
                "type": names.get(int(cls[i]), str(int(cls[i]))),
                "confidence": float(conf[i]),
                "x1": float(xyxy[i][0]),
                "y1": float(xyxy[i][1]),
                "x2": float(xyxy[i][2]),
                "y2": float(xyxy[i][3]),
            }
        )
    df = pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)
    return df


def card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
<div class="card">
  <h4>{title}</h4>
  <div class="big">{value}</div>
  <div class="muted">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ----------------------------
# Sidebar: Control Panel
# ----------------------------
st.sidebar.markdown("## Control Panel")
st.sidebar.caption("Tweak thresholds and switch model weights.")

weights_path = st.sidebar.text_input(
    "Weights (.pt) path",
    value=resolve_default_weights(),
    help="Tip: Keep a stable copy at best.pt",
)

conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.65, 0.01)
iou_thres = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.95, 0.80, 0.01)

st.sidebar.divider()
st.sidebar.markdown("### Safety")
st.sidebar.warning("Portfolio demo only. Not medical advice.", icon="⚠️")


# ----------------------------
# Header
# ----------------------------
st.markdown("## 🧠 Brain Tumor Detector")
st.markdown(
    '<span class="muted">Modern dashboard UI • Upload an MRI (jpg/png) • Detection + tumor type + bounding box</span>',
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts


tabs = st.tabs(["🔍 Predict", "📈 Analytics", "ℹ️ Model"])

# ----------------------------
# Tab 1: Predict
# ----------------------------
with tabs[0]:
    colA, colB = st.columns([1.1, 0.9], gap="large")

    with colA:
        st.markdown("### Input")
        upload = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        st.caption("Tip: Start with images from the dataset test folder to sanity check.")

        if upload:
            pil_img = Image.open(upload)
            st.image(pil_img, caption="Uploaded image", use_container_width=True)

    with colB:
        st.markdown("### Output")

        if not weights_path:
            st.error("No weights path provided. Set it in the sidebar (recommended: best.pt).")
            st.stop()

        weights_file = Path(weights_path)
        if not weights_file.exists():
            st.error(f"Weights file not found: {weights_path}")
            st.stop()

        model = load_model(weights_path)

        if upload:
            # Inference
            bgr = pil_to_bgr(pil_img)

            t0 = time.time()
            results = model.predict(
                source=bgr,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False,
            )
            infer_ms = (time.time() - t0) * 1000.0

            r0 = results[0]
            df = results_to_table(r0)

            detected = len(df) > 0
            top_class = df.iloc[0]["type"] if detected else "—"
            top_conf = float(df.iloc[0]["confidence"]) if detected else 0.0

            # Render annotated image
            # Ultralytics plot() returns BGR array
            annotated_bgr = r0.plot()
            annotated_pil = bgr_to_pil(annotated_bgr)

            # KPI cards
            k1, k2 = st.columns(2, gap="medium")
            with k1:
                card("Tumor detected", "✅ Yes" if detected else "❌ No", f"Confidence ≥ {conf_thres:.2f}")
            with k2:
                card("Top prediction", top_class, f"Max conf: {top_conf:.2f}")

            k3, k4 = st.columns(2, gap="medium")
            with k3:
                card("Detections", str(len(df)), "Count after NMS")
            with k4:
                card("Inference time", f"{infer_ms:.0f} ms", "CPU/GPU dependent")

            st.markdown("#### Annotated result")
            st.image(annotated_pil, use_container_width=True)

            # Download annotated
            buf = BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download annotated PNG",
                data=buf.getvalue(),
                file_name="prediction_annotated.png",
                mime="image/png",
                use_container_width=True,
            )

            st.markdown("#### Detections")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Store history
            st.session_state.history.insert(
                0,
                {
                    "filename": upload.name,
                    "detected": detected,
                    "top_class": top_class,
                    "top_conf": top_conf,
                    "detections": len(df),
                    "infer_ms": infer_ms,
                },
            )
            st.session_state.history = st.session_state.history[:25]
        else:
            st.info("Upload an image to see predictions.")


# ----------------------------
# Tab 2: Analytics
# ----------------------------
with tabs[1]:
    st.markdown("### Recent predictions")
    if not st.session_state.history:
        st.caption("No predictions yet.")
    else:
        hdf = pd.DataFrame(st.session_state.history)
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            card("Total runs", str(len(hdf)))
        with c2:
            card("Detected rate", f"{(hdf['detected'].mean()*100):.0f}%", "Over recent runs")
        with c3:
            card("Avg inference", f"{hdf['infer_ms'].mean():.0f} ms")
        with c4:
            card("Most common", hdf["top_class"].value_counts().idxmax())

        st.dataframe(hdf, use_container_width=True, hide_index=True)


# ----------------------------
# Tab 3: Model info
# ----------------------------
with tabs[2]:
    st.markdown("### Model & dataset scope")
    st.markdown(
        """
<div class="card">
  <div class="big">What this demo can do</div>
  <ul class="muted">
    <li>Detect tumor-like regions and classify type: glioma / meningioma / pituitary</li>
    <li>Draw bounding boxes and show confidence scores</li>
  </ul>
  <div class="big" style="margin-top:12px;">Important limitation</div>
  <div class="muted">
    This dataset does not include a true “normal / no tumor” class. So “No detection” ≠ medically “no tumor”.
    This is a portfolio demo, not a diagnostic tool.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Current weights")
    st.code(weights_path or "(none)")

    st.markdown("### Tips to keep it clean")
    st.markdown(
        """
- Put your final model at **`best.pt`** and point the app there.
- You can delete old experiment folders in `runs/` after you’ve saved `best.pt`.
"""
    )
