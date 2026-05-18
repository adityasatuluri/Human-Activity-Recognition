import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os



# --- Configuration & Constants ---
IMAGE_SIZE = (300, 300)
CLASS_NAMES = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 
               'fighting', 'hugging', 'laughing', 'listening_to_music', 
               'running', 'sitting', 'sleeping', 'smoking', 'stabbing']
MODEL_NAMES = ["EfficientNetB3", "ResNet50", "MobileNetV2"]

st.set_page_config(page_title="Action Recognition Comparison", layout="wide")

@st.cache_resource
def load_models():
    models = {}
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    for name in MODEL_NAMES:
        # Use r"" strings for Windows paths to avoid backslash issues
        model_path = os.path.join(base_path, "models", f"{name}_final.keras")
        
        if os.path.exists(model_path):
            try:
                # Add safe_mode=False if you used custom layers/optimizers
                models[name] = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e:
                # If it still fails, try this specific Keras 3 fix:
                st.error(f"Error loading {name}. Check if the file is a valid zip (Keras 3 format).")
                st.exception(e) 
        else:
            st.error(f"Path does not exist: {model_path}")
            
    return models

def preprocess_image(image, model_type):
    """Standardize image preprocessing to match training pipeline."""
    img = image.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img).astype(np.float32)
    
    # Apply model-specific preprocessing
    if model_type == "ResNet50":
        img_array = keras.applications.resnet50.preprocess_input(img_array)
    elif model_type == "MobileNetV2":
        img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    # EfficientNetB3 typically expects raw 0-255 or internal scaling
    
    return np.expand_dims(img_array, axis=0)

# --- UI Layout ---
st.title("🏃 Human Action Recognition: 3-Model Comparison")
st.write("Upload a photo to see how different architectures classify the gesture.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.info("Processing with EfficientNetB3, ResNet50, and MobileNetV2...")

    models = load_models()
    
    if models:
        st.divider()
        # Create 3 columns for results
        res_cols = st.columns(3)
        
        for i, name in enumerate(MODEL_NAMES):
            with res_cols[i]:
                if name in models:
                    # Preprocess and Predict
                    processed_img = preprocess_image(image, name)
                    preds = models[name].predict(processed_img, verbose=0)
                    
                    class_idx = np.argmax(preds)
                    label = CLASS_NAMES[class_idx]
                    confidence = np.max(preds)
                    
                    # Display Results
                    st.subheader(f"🤖 {name}")
                    st.metric(label="Predicted Action", value=label.replace('_', ' ').title())
                    st.progress(float(confidence))
                    st.write(f"**Confidence:** {confidence:.2%}")
                else:
                    st.warning(f"{name} not loaded.")

else:
    st.info("Please upload an image to begin inference.")