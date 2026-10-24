"""Contribution tab component for streamlit_app.py.

Import and call render_contribute_tab() inside a st.tab block.
Saves images to dataset/Contributions/<label>/ with metadata CSV.
"""
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import json

ROOT          = Path(__file__).resolve().parents[1]
CONTRIB_DIR   = ROOT / "dataset" / "Contributions"
CONTRIB_CSV   = CONTRIB_DIR / "contributions.csv"
MODEL_PATH    = ROOT / "checkpoints" / "best_model.keras"
NAMES_PATH    = ROOT / "checkpoints" / "class_names.json"

# Known labels — will be extended dynamically
DEFAULT_LABELS = [str(i) for i in range(1, 10)] + \
                 [chr(c) for c in range(ord('A'), ord('Z') + 1)]


def ensure_dirs():
    CONTRIB_DIR.mkdir(parents=True, exist_ok=True)
    if not CONTRIB_CSV.exists():
        with open(CONTRIB_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label", "filename",
                              "is_new_label", "source"])


def save_contribution(img_bgr, label, source="upload"):
    ensure_dirs()
    label_dir = CONTRIB_DIR / label.upper().replace(" ", "_")
    label_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename  = f"{label.upper().replace(' ', '_')}_{timestamp}.jpg"
    path      = label_dir / filename
    cv2.imwrite(str(path), img_bgr)

    # Load existing labels to check if new
    class_names = json.loads(NAMES_PATH.read_text()) if NAMES_PATH.exists() else []
    is_new = label.upper() not in [c.upper() for c in class_names]

    with open(CONTRIB_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, label.upper(), filename, is_new, source])

    return path, is_new


def get_model_prediction(img_bgr, model, class_names):
    """Run the current model on the image and return top 3 predictions."""
    img = cv2.resize(img_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(img.astype(np.float32), axis=0)
    preds = model.predict(inp, verbose=0)[0]
    top3  = np.argsort(preds)[::-1][:3]
    return [(class_names[i], float(preds[i])) for i in top3]


@st.cache_resource
def load_contrib_model():
    if not MODEL_PATH.exists():
        return None, []
    model       = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(NAMES_PATH.read_text())
    return model, class_names


def get_all_known_labels():
    """Combine default labels + any new labels already in Contributions."""
    labels = list(DEFAULT_LABELS)
    if CONTRIB_DIR.exists():
        for d in sorted(CONTRIB_DIR.iterdir()):
            if d.is_dir() and d.name not in labels:
                labels.append(d.name)
    return labels


def render_contribute_tab():
    st.markdown("### Contribute a sign image")
    st.markdown(
        "Help improve the model by submitting labelled sign images. "
        "You can contribute existing signs (A–Z, 1–9) or add a **new label** "
        "such as a word or phrase for future training."
    )

    model, class_names = load_contrib_model()
    known_labels = get_all_known_labels()

    # ── Label selection ───────────────────────────────────────────────────────
    st.markdown("#### 1. Choose a label")
    label_mode = st.radio(
        "Label type",
        ["Existing label (A–Z, 1–9)", "New label (word, phrase, custom)"],
        horizontal=True,
    )

    if label_mode == "Existing label (A–Z, 1–9)":
        label = st.selectbox("Select label", known_labels)
    else:
        label = st.text_input(
            "Enter new label",
            placeholder="e.g. HELLO, WATER, THANK_YOU",
            max_chars=40,
        ).strip().upper().replace(" ", "_")
        if label:
            st.info(
                f"**{label}** is a new label. Images you submit will be saved "
                "and used to train this class when enough are collected."
            )

    if not label:
        st.warning("Please enter or select a label before uploading.")
        return

    # ── Image input ───────────────────────────────────────────────────────────
    st.markdown("#### 2. Provide an image")
    input_mode = st.radio(
        "Source", ["Upload image", "Use webcam"], horizontal=True
    )

    img_bgr = None

    if input_mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload a photo of the sign",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    else:
        img_file = st.camera_input("Take a photo", label_visibility="collapsed")
        if img_file:
            file_bytes = np.frombuffer(img_file.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # ── Preview + model check ─────────────────────────────────────────────────
    if img_bgr is not None:
        st.markdown("#### 3. Preview")

        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Your image — labelled as: {label}",
                use_container_width=True,
            )

        with col_pred:
            st.markdown("**Model's current prediction**")
            if model is not None:
                predictions = get_model_prediction(img_bgr, model, class_names)
                for rank, (pred_label, pred_conf) in enumerate(predictions):
                    bar_color = "🟢" if rank == 0 else "🔵"
                    st.markdown(
                        f"{bar_color} **{pred_label}** — {pred_conf*100:.1f}%"
                    )
                top_label, top_conf = predictions[0]
                if top_label.upper() == label.upper() and top_conf > 0.7:
                    st.success("✓ Model already recognises this sign well.")
                elif label.upper() not in [c.upper() for c in class_names]:
                    st.info("New label — model hasn't seen this sign yet.")
                else:
                    st.warning(
                        f"Model predicted **{top_label}** instead of **{label}**. "
                        "Your contribution will help fix this!"
                    )
            else:
                st.info("No trained model found — prediction unavailable.")

        # ── Submit ────────────────────────────────────────────────────────────
        st.markdown("#### 4. Submit")
        st.caption(
            "By submitting you agree your image may be used to improve "
            "the ISL translator model."
        )

        source = "webcam" if input_mode == "Use webcam" else "upload"

        if st.button("✅ Submit contribution", type="primary",
                     use_container_width=True):
            path, is_new = save_contribution(img_bgr, label, source)
            if is_new:
                st.success(
                    f"Saved! **{label}** is a new label — "
                    "it will be added to training in a future update."
                )
            else:
                st.success(f"Saved! Thank you for contributing to **{label}**.")
            st.caption(f"Saved to: `{path.relative_to(ROOT)}`")

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Contribution stats")
    if CONTRIB_DIR.exists():
        label_dirs = [d for d in CONTRIB_DIR.iterdir() if d.is_dir()]
        if label_dirs:
            total = sum(
                len(list(d.glob("*.jpg"))) for d in label_dirs
            )
            st.metric("Total contributions", total)
            cols = st.columns(4)
            for i, d in enumerate(sorted(label_dirs)):
                n = len(list(d.glob("*.jpg")))
                cols[i % 4].metric(d.name, n)
        else:
            st.info("No contributions yet — be the first!")
    else:
        st.info("No contributions yet — be the first!")