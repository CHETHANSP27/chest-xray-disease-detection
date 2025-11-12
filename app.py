"""
Streamlit web application for Chest X-ray Disease Detection
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys

# Configure environment
os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'

# Safe import of OpenCV
try:
    import cv2
except ImportError:
    st.error("Failed to import OpenCV. Using PIL for image processing instead.")
    USE_OPENCV = False
else:
    USE_OPENCV = True

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.model import create_model
from src.gradcam import ChestXrayGradCAM, prepare_image_for_cam
from src.data_loader import load_single_image
from src.utils import load_checkpoint

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Disease Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        background-color: #FFEBEE;
        border-left: 4px solid #E53935;
    }
    .negative {
        background-color: #E8F5E9;
        border-left: 4px solid #43A047;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model"""
    # Force CPU for Streamlit deployment
    device = 'cpu'
    
    model = create_model()
    model_path = "models/saved_models/best_model.pth"
    
    if os.path.exists(model_path):
        model, _, _, _ = load_checkpoint(model, None, model_path, device)
        model.eval()
        model.to(device)
        return model
    else:
        st.error(f"Model file not found: {model_path}")
        return None
    
def process_image(image_file):
    """Process uploaded image file"""
    if USE_OPENCV:
        # OpenCV processing path
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # PIL fallback path
        image = Image.open(image_file)
        image = np.array(image)
    return image

def predict_diseases(model, image_tensor):
    """Make predictions"""
    # Force CPU for both model and tensor
    device = 'cpu'
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
        predictions = predictions.cpu().numpy()[0]
    
    return predictions

def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Chest X-Ray Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Explainable Disease Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📋 About")
    st.sidebar.info(
        """
        This application uses a **DenseNet121** deep learning model 
        to detect 14 different thoracic diseases from chest X-ray images.
        
        **Features:**
        - Multi-disease detection
        - Grad-CAM visualization
        - Probability scores
        - Explainable AI
        """
    )
    
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM Visualization", value=True)
    show_all_predictions = st.sidebar.checkbox("Show All Disease Probabilities", value=False)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.header("📤 Upload Chest X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a frontal chest X-ray image"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original X-Ray")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
        
        # Make prediction
        with st.spinner("Analyzing X-ray..."):
            # Preprocess image
            image_tensor = load_single_image(uploaded_file)
            
            # Predict
            predictions = predict_diseases(model, image_tensor)
            
            # Get positive predictions
            positive_diseases = []
            for i, disease in enumerate(Config.DISEASE_LABELS):
                if predictions[i] >= threshold:
                    positive_diseases.append((disease, predictions[i]))
            
            positive_diseases.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        with col2:
            st.subheader("🔍 Analysis Results")
            
            if len(positive_diseases) == 0:
                st.markdown(
                    '<div class="prediction-box negative">'
                    '<h3>✅ No Findings</h3>'
                    '<p>No significant abnormalities detected above the threshold.</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box positive">'
                    f'<h3>⚠️ {len(positive_diseases)} Finding(s) Detected</h3>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                for disease, prob in positive_diseases:
                    st.markdown(
                        f'<div class="prediction-box positive">'
                        f'<b>{disease}</b>: {prob*100:.2f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        # Grad-CAM visualization
        if show_gradcam and len(positive_diseases) > 0:
            st.header("🔥 Grad-CAM Explainability")
            st.info("Heatmaps show which regions of the X-ray influenced the model's decision.")
            
            # Create Grad-CAM
            gradcam = ChestXrayGradCAM(model)
            original_image = prepare_image_for_cam(image_tensor)
            
            # Generate CAMs for top diseases
            top_k = min(3, len(positive_diseases))
            top_disease_indices = [Config.DISEASE_LABELS.index(d[0]) for d in positive_diseases[:top_k]]
            
            cam_cols = st.columns(top_k)
            
            for idx, (col, disease_idx) in enumerate(zip(cam_cols, top_disease_indices)):
                disease = Config.DISEASE_LABELS[disease_idx]
                
                # Generate CAM
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                targets = [ClassifierOutputTarget(disease_idx)]
                
                # Ensure image_tensor is on CPU for GradCAM
                image_tensor_cpu = image_tensor.to('cpu')
                
                grayscale_cam = gradcam.cam(input_tensor=image_tensor_cpu, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                from pytorch_grad_cam.utils.image import show_cam_on_image
                cam_image = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
                
                with col:
                    st.image(cam_image, caption=f"{disease}", use_column_width=True)
        
        # Show all predictions
        if show_all_predictions:
            st.header("📊 All Disease Probabilities")
            
            # Create DataFrame
            import pandas as pd
            results_df = pd.DataFrame({
                'Disease': Config.DISEASE_LABELS,
                'Probability (%)': [f"{p*100:.2f}" for p in predictions],
                'Status': ['Positive' if p >= threshold else 'Negative' for p in predictions]
            })
            
            results_df = results_df.sort_values('Probability (%)', ascending=False, key=lambda x: x.str.replace('%', '').astype(float))
            st.dataframe(results_df, use_container_width=True)
    
    else:
        # Instructions
        st.info(
            """
            ### 📝 Instructions:
            1. Upload a chest X-ray image (PNG, JPG, or JPEG format)
            2. The AI model will analyze the image
            3. View predictions and Grad-CAM visualizations
            4. Adjust threshold in sidebar for sensitivity
            
            ### ⚠️ Disclaimer:
            This is a demonstration tool for educational purposes only. 
            It should **NOT** be used for actual medical diagnosis. 
            Always consult qualified healthcare professionals.
            """
        )
        
        # Sample diseases info
        st.header("🩺 Detectable Diseases")
        diseases_text = ", ".join(Config.DISEASE_LABELS)
        st.write(f"The model can detect: **{diseases_text}**")

if __name__ == "__main__":
    main()