import os
# --- CRITICAL FIX FOR TF 2.20.0 CLOUD DEPLOYMENT ---os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import joblib
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess_input
import rembg

# --- Configuration ---
MODELS_BASE_DIR = "models"
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
CONFIDENCE_THRESHOLD = 0.3 # 30% threshold

# --- Model config just for EfficientNetV2 (Ruler Only) ---
MODEL_CONFIG = {
    "EfficientNetV2": {
        "Ruler": {
            "model_file": "efficientnetv2_donemi.keras", 
            "encoder_file": "coin_donemi_encoder.joblib" 
        }
    }
}

# --- Helper function to scan for models ---
def get_model_versions():
    """Scans the 'models/' directory and returns a list of subfolder names."""
    if not os.path.exists(MODELS_BASE_DIR) or not os.path.isdir(MODELS_BASE_DIR):
        return []
    
    versions = [
        item for item in os.listdir(MODELS_BASE_DIR)
        if os.path.isdir(os.path.join(MODELS_BASE_DIR, item))
    ]
    return versions

# --- Model Loading ---
@st.cache_resource
def load_effnet_model(version_path, feature):
    """ Loads the EfficientNetV2 model and its corresponding encoder. """
    
    config = MODEL_CONFIG["EfficientNetV2"].get(feature, {})
    if not config:
        st.error(f"Configuration error: No model defined for {feature}")
        return None
        
    model_pack = {}
    base_path = Path(MODELS_BASE_DIR) / version_path
    
    try:
        # 1. Load Keras Model
        model_file_path = base_path / config["model_file"]
        if model_file_path.exists():
            model_pack["model"] = load_model(model_file_path)
        else:
            st.error(f"Model file not found: {model_file_path}")
            return None
            
        # 2. Load Encoder
        encoder_file_path = base_path / config["encoder_file"]
        if encoder_file_path.exists():
            model_pack["encoder"] = joblib.load(encoder_file_path)
        else:
            st.error(f"Encoder file not found: {encoder_file_path}")
            return None
        
        return model_pack
        
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None

# --- On-the-fly JPEG Processing ---
@st.cache_data
def process_uploaded_image(image_source):
    """
    Takes an uploaded image, runs rembg, splits it,
    and returns TWO preprocessed tensors exactly matching the local pipeline.
    """
    try:
        # 1. Open image and convert to RGB array (REVERTED EXIF TRANSPOSE)
        img = Image.open(image_source).convert('RGB')
        
        # 2. Run rembg 
        img_no_bg = rembg.remove(img)

        # 3. Convert RGBA to RGB (This flattens transparent pixels to BLACK)
        img_rgb = img_no_bg.convert("RGB")

        # 4. Split into halves
        width, height = img_rgb.size
        midpoint = width // 2
        img_left = img_rgb.crop((0, 0, midpoint, height))
        img_right = img_rgb.crop((midpoint, 0, width, height))

        tensors = []
        for half_img in [img_left, img_right]:
            # 5. Resize if not matching target
            if half_img.width != IMAGE_WIDTH or half_img.height != IMAGE_HEIGHT:
                half_img = ImageOps.fit(half_img, (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
            
            arr = np.asarray(half_img).astype(np.float32)
            
            # 6. Replicate logic from training preprocessing
            image_tensor = tf.convert_to_tensor(arr)
            image_tensor.set_shape([IMAGE_WIDTH, IMAGE_HEIGHT, 3])
            
            # 7. Apply the correct preprocessing
            image_processed = effnet_preprocess_input(image_tensor) 
            
            # 8. Add batch dimension for prediction
            tensors.append(tf.expand_dims(image_processed, axis=0))
        
        return tensors[0], tensors[1]
    
    except Exception as e:
        st.error(f"Error processing {image_source.name}: {e}")
        return None, None

# --- Prediction Function ---
def format_prediction_text(text):
    """Formats a label like 'ghazan-mahmud' to 'Ghazan Mahmud'."""
    if not isinstance(text, str):
        text = str(text)
    return text.replace('-', ' ').title()

def run_prediction(model_pack, tensor_obverse, tensor_reverse):
    """ 
    Runs prediction on BOTH halves and averages the probabilities.
    """
    try:
        model = model_pack["model"]
        encoder = model_pack["encoder"]
        
        # Predict on both halves (verbose=0 hides cloud terminal spam, math remains identical)
        probs_obverse = model.predict(tensor_obverse, verbose=0)[0]
        probs_reverse = model.predict(tensor_reverse, verbose=0)[0]
        
        # Average the probabilities
        probabilities = (probs_obverse + probs_reverse) / 2.0
        
        # Get top 3
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        top_3_scores = probabilities[top_3_indices]
        top_3_names = encoder.classes_[top_3_indices] 
        
        # Format for display
        formatted_names = [format_prediction_text(name) for name in top_3_names]
        prediction = formatted_names[0]
        probability = top_3_scores[0]
        
        if probability < CONFIDENCE_THRESHOLD:
            st.metric(label=f"Predicted", value="I don't know")
            st.info(f"Top guess: {prediction} ({probability*100:.1f}% conf.)")
        else:
            st.metric(label=f"Predicted", value=prediction, delta=f"{probability*100:.1f}% Confidence")
            st.write("&nbsp;")
            
        st.write("**Top 3 Guesses:**")
        for i, (name, score) in enumerate(zip(formatted_names, top_3_scores)):
            st.write(f"{i+1}. {name} ({score*100:.1f}%)")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Main Application UI ---
def main():
    st.set_page_config(page_title="altaycoins Ilkhan Ruler Identification Tool", layout="wide")
    st.title("🪙 altaycoins Ilkhan Ruler Identification Tool")
    
    model_versions = get_model_versions()
    feature = "Ruler" # Hardcoded to only predict the ruler

    if not model_versions:
        st.error("No model versions found in 'models' directory.")
        st.warning("Please create a subfolder in the 'models' directory to continue.")
        st.stop()

    # --- AUTO-LOAD LOGIC ---
    if len(model_versions) == 1:
        version_path = model_versions[0]
        if 'model_pack' not in st.session_state:
            with st.spinner(f"Auto-loading model from '{version_path}'..."):
                st.session_state.model_pack = load_effnet_model(version_path, feature)
                st.toast("✅ Model loaded automatically!")
    else:
        st.sidebar.header("1. Load Model")
        version_path = st.sidebar.selectbox("Model Version", options=model_versions)
        
        if st.sidebar.button("Load Ruler Model", type="primary"):
            with st.spinner(f"Loading {feature} model..."):
                st.session_state.model_pack = load_effnet_model(version_path, feature)
                st.toast("✅ Model loaded successfully!")

    # --- Check if model is loaded ---
    if 'model_pack' not in st.session_state or st.session_state.model_pack is None:
        st.warning("Please wait for the model to load to begin.")
        st.stop()
    
    model_pack = st.session_state.model_pack
    st.success(f"**Ready:** Predicting `{feature}` using `{version_path}`")

    # --- File Uploader (Main Area) ---
    st.divider()
    st.header("Upload Coin Images")
    st.write("Ensure the **obverse** and **reverse** of the coin are side-by-side in a single image.")
    uploaded_files = st.file_uploader(
        "Drag and drop your JPEG/PNG files here.", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True
    )

    # --- Display Results ---
    if uploaded_files:
        st.divider()
        st.header("Prediction Results")
        
        num_files = len(uploaded_files)
        num_cols = 3
        num_rows = (num_files + num_cols - 1) // num_cols 
        
        for i in range(num_rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                file_idx = i * num_cols + j
                
                if file_idx < num_files:
                    file = uploaded_files[file_idx]
                    
                    with cols[j]:
                        with st.container(border=True):
                            st.subheader(f"{file.name}")
                            st.image(file, use_container_width=True) 
                            
                            with st.spinner(f"Processing..."):
                                tensor_obverse, tensor_reverse = process_uploaded_image(file)
                            
                            if tensor_obverse is not None:
                                run_prediction(model_pack, tensor_obverse, tensor_reverse)
                            else:
                                st.error("Could not process this image.")
                                
    elif 'model_pack' in st.session_state:
        st.info("Upload coin images above to see predictions.")

if __name__ == "__main__":
    if "TF_DISABLE_MKL" not in os.environ:
        os.environ["TF_DISABLE_MKL"] = "1"
    main()
