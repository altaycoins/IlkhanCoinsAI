"""
predictv7_1.py
==============
EfficientNetV2 Coin Predictor — Streamlit app.

IMPORTANT: This app does NOT run rembg. Background removal is done
separately by preprocess.py. Upload images from the 'processed/' folder.

Requirements:
    pip install streamlit tensorflow joblib numpy pillow scikit-learn==1.7.2
"""

import os
import io
import joblib
import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import (
    preprocess_input as effnet_preprocess_input,
)

os.environ.setdefault("TF_DISABLE_MKL", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ==============================================================================
# Configuration
# ==============================================================================
MODELS_BASE_DIR = "models"
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
CONFIDENCE_THRESHOLD = 0.1

MODEL_CONFIG = {
    "EfficientNetV2": {
        "Ruler": {
            "model_file": "efficientnetv2_donemi.keras",
            "encoder_file": "coin_donemi_encoder.joblib",
        },
        "Mint": {
            "model_file": "efficientnetv2_darp_yeri.keras",
            "encoder_file": "coin_darpyeri_encoder.joblib",
        },
    }
}

# ==============================================================================
# Helpers
# ==============================================================================
def get_model_versions() -> list:
    p = Path(MODELS_BASE_DIR)
    if not p.exists() or not p.is_dir():
        return []
    return [d for d in os.listdir(p) if os.path.isdir(p / d)]


def format_prediction_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.replace("-", " ").title()


# ==============================================================================
# Model loading (cached — loaded once per session)
# ==============================================================================
@st.cache_resource
def load_effnet_model(version_path: str, feature: str):
    config = MODEL_CONFIG["EfficientNetV2"].get(feature)
    if not config:
        st.error(f"No model config for feature '{feature}'.")
        return None

    base         = Path(MODELS_BASE_DIR) / version_path
    model_path   = base / config["model_file"]
    encoder_path = base / config["encoder_file"]

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None
    if not encoder_path.exists():
        st.error(f"Encoder file not found: {encoder_path}")
        return None

    try:
        model   = load_model(model_path)
        encoder = joblib.load(encoder_path)
        st.success(f"Loaded '{feature}' model from '{version_path}'")
        return {"model": model, "encoder": encoder}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ==============================================================================
# Image processing (cached per file — bytes are hashable, file objects are not)
# No rembg — images must already have background removed by preprocess.py
# ==============================================================================
@st.cache_data
def process_uploaded_image(image_bytes: bytes, filename: str):
    """
    Splits image into left/right halves, resizes, preprocesses for EfficientNetV2.
    Returns (tensor_obverse, tensor_reverse) or (None, None) on error.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        width, height = img.size
        mid       = width // 2
        img_left  = img.crop((0, 0, mid, height))
        img_right = img.crop((mid, 0, width, height))

        tensors = []
        for half in [img_left, img_right]:
            if half.width != IMAGE_WIDTH or half.height != IMAGE_HEIGHT:
                half = ImageOps.fit(
                    half, (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS
                )
            arr = np.asarray(half).astype(np.float32)
            t   = tf.convert_to_tensor(arr)
            t.set_shape([IMAGE_WIDTH, IMAGE_HEIGHT, 3])
            t   = effnet_preprocess_input(t)
            tensors.append(tf.expand_dims(t, axis=0))

        return tensors[0], tensors[1]

    except Exception as e:
        st.error(f"Error processing '{filename}': {e}")
        return None, None


# ==============================================================================
# Prediction
# ==============================================================================
def run_prediction(model_pack: dict, tensor_obverse, tensor_reverse):
    try:
        model   = model_pack["model"]
        encoder = model_pack["encoder"]

        probs_obverse = model.predict(tensor_obverse, verbose=0)[0]
        probs_reverse = model.predict(tensor_reverse, verbose=0)[0]
        probabilities = (probs_obverse + probs_reverse) / 2.0

        top3_idx    = np.argsort(probabilities)[::-1][:3]
        top3_scores = probabilities[top3_idx]
        top3_names  = [format_prediction_text(encoder.classes_[i]) for i in top3_idx]

        prediction  = top3_names[0]
        probability = top3_scores[0]

        if probability < CONFIDENCE_THRESHOLD:
            st.metric(label="Predicted", value="I don't know")
            st.info(f"Top guess: {prediction} ({probability*100:.1f}% conf.)")
        else:
            st.metric(
                label="Predicted",
                value=prediction,
                delta=f"{probability*100:.1f}% Confidence",
            )
            st.write("&nbsp;")

        st.write("**Top 3 Guesses:**")
        for i, (name, score) in enumerate(zip(top3_names, top3_scores)):
            st.write(f"{i+1}. {name} ({score*100:.1f}%)")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ==============================================================================
# Main UI
# ==============================================================================
def main():
    st.set_page_config(page_title="EfficientNetV2 Coin Predictor", layout="wide")
    st.title("🪙 EfficientNetV2 Coin Predictor")
    st.caption(
        "Upload images pre-processed with **preprocess.py** (background already removed)."
    )

    # ── Sidebar: model selection ──────────────────────────────────────────────
    st.sidebar.header("1. Load Model")

    model_versions = get_model_versions()
    if not model_versions:
        st.sidebar.error("No model versions found in 'models/' directory.")
        st.warning("Create a subfolder inside 'models/' to continue.")
        st.stop()

    version_path = st.sidebar.selectbox("Model version", options=model_versions)
    feature      = st.sidebar.selectbox(
        "Feature to predict", options=["Ruler", "Mint"], key="feat"
    )

    if st.sidebar.button("Load Model", type="primary"):
        with st.spinner(f"Loading '{feature}' model..."):
            st.session_state.model_pack = load_effnet_model(version_path, feature)

    # ── Sidebar: file uploader ────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.header("2. Upload Pre-Processed Coin Images")
    st.sidebar.caption("Use images output by preprocess.py (backgrounds already removed).")

    uploaded_files = st.sidebar.file_uploader(
        "Drag and drop JPEG/PNG files here.",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
    )

    # ── Main area ─────────────────────────────────────────────────────────────
    st.divider()

    if "model_pack" not in st.session_state or st.session_state.model_pack is None:
        st.warning("Load a model using the sidebar to begin.")
        st.stop()

    model_pack = st.session_state.model_pack
    st.success(f"**Model loaded:** predicting `{feature}` using `{version_path}`")

    if not uploaded_files:
        st.info("Upload pre-processed coin images in the sidebar to see predictions.")
        st.stop()

    # ── Results grid (3 columns) ──────────────────────────────────────────────
    st.header("3. Prediction Results")

    num_cols = 3
    num_rows = (len(uploaded_files) + num_cols - 1) // num_cols

    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            file_idx = row * num_cols + col_idx
            if file_idx >= len(uploaded_files):
                break

            file = uploaded_files[file_idx]
            with cols[col_idx]:
                with st.container(border=True):
                    st.subheader(file.name)

                    image_bytes = file.read()
                    st.image(image_bytes, use_container_width=True)

                    with st.spinner("Processing..."):
                        tensor_obverse, tensor_reverse = process_uploaded_image(
                            image_bytes, file.name
                        )

                    if tensor_obverse is not None:
                        run_prediction(model_pack, tensor_obverse, tensor_reverse)
                    else:
                        st.error("Could not process this image.")


if __name__ == "__main__":
    main()
