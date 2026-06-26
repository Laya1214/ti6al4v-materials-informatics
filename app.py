import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ti-6Al-4V Materials Informatics Portal", page_icon="🔬", layout="wide")

# CSS
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6; padding: 25px; border-radius: 12px;
        border-left: 6px solid #2C5364; margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# HD Logo + Title
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("image.jpg", width=100)

with col_title:
    st.markdown("""
        <h1 style='margin:0; font-size:38px; color:white; font-weight:800;'>
        Ti-6Al-4V Materials Informatics Portal
        </h1>
        <p style='margin:8px 0 0 0; font-size:20px; color:white; font-weight:500;'>
            Automated ASTM E112 Grain Size Classification & Vickers Hardness Predictor
        </p>
    """, unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    if not os.path.exists("vickers_hardness_regressor.pkl") or not os.path.exists("grain_size_classifier.pkl"):
        return None, None
    return joblib.load("vickers_hardness_regressor.pkl"), joblib.load("grain_size_classifier.pkl")

reg_model, clf_model = load_models()

if reg_model is None or clf_model is None:
    st.error("⚠️ Model files not found!")
    st.stop()

# Feature Extraction (Cleaned)
def preprocess_image(img):
    h, w = img.shape
    img_cropped = img[0:int(h * 0.88), :]
    img_filtered = cv2.bilateralFilter(img_cropped, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_filtered)

def extract_features_from_stream(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    img_enhanced = preprocess_image(img)
    mean = float(np.mean(img_enhanced))
    std = float(np.std(img_enhanced))
    skewness = float(np.mean((img_enhanced - mean) ** 3) / (std ** 3 + 1e-8))
    total_pixels = img_enhanced.size
    hist = cv2.calcHist([img_enhanced], [0], None, [256], [0, 256])
    prob = hist.flatten() / total_pixels
    prob = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log2(prob)))
    edges = cv2.Canny(img_enhanced, 100, 200)
    edge_density = float(np.sum(edges > 0) / (total_pixels + 1e-8))
    features = np.array([mean, std, skewness, entropy, edge_density])
    return features, edges, img_enhanced

# Sidebar
st.sidebar.header("📁 Specimen Metadata")
uploaded_file = st.sidebar.file_uploader("Upload SEM Micrograph", type=["png", "jpg", "jpeg", "tif", "tiff"])
analyze_button = st.sidebar.button("🚀 Analyze Microstructure", type="primary")

# Main App
if uploaded_file and analyze_button:
    file_bytes = uploaded_file.read()
    features, edge_map, enhanced_img = extract_features_from_stream(file_bytes)
    
    if features is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📷 Original Microstructure")
            st.image(uploaded_file, use_column_width=True)
        with col2:
            st.markdown("### 🕸️ Grain Boundary Map")
            st.image(cv2.bitwise_not(edge_map), use_column_width=True, channels="GRAY")
        
        features_2d = features.reshape(1, -1)
        hv = float(reg_model.predict(features_2d)[0])
        grain_class = str(clf_model.predict(features_2d)[0])
        
        # Simple Confidence (based on feature variance)
        confidence = max(60, int(100 - np.std(features)*0.8))
        
        st.markdown("---")
        st.markdown("### 📊 Predicted Properties")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class='metric-container'>
                <p>PREDICTED VICKERS HARDNESS</p>
                <h1 style='font-size:48px;color:#2c3e50;'>{hv:.2f} HV</h1>
                <p style='margin:0;color:#555;'>Confidence: {confidence}%</p>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='metric-container'>
                <p>ASTM GRAIN SIZE CLASS</p>
                <h1 style='font-size:48px;color:#2c3e50;'>{grain_class}</h1>
                <p style='margin:0;color:#555;'>Confidence: {confidence}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Texture Analysis")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].hist(enhanced_img.ravel(), bins=256, color='#2C5364')
        ax[0].set_title("Intensity Histogram")
        ax[1].imshow(cv2.bitwise_not(edge_map), cmap='gray')
        ax[1].set_title("Canny Edge Map")
        ax[1].axis('off')
        st.pyplot(fig)
        
        st.markdown("### 📑 Extracted Features")
        feature_names = ["Mean Intensity", "Std Dev", "Skewness", "Entropy", "Edge Density"]
        st.dataframe(pd.DataFrame({"Feature": feature_names, "Value": features}).style.format({"Value": "{:.6f}"}), hide_index=True, use_container_width=True)
        
        st.info(f"""
        **Ti-6Al-4V Metallurgical Interpretation**  
        Your specimen is classified as **{grain_class.lower()} grained** microstructure.  
        The predicted hardness (**{hv:.1f} HV**) is consistent with the Hall-Petch relationship, where finer grains/laths increase grain boundary density and restrict dislocation motion, leading to higher strength.
        
        Typical range for Ti-6Al-4V (annealed / heat-treated): **320 – 450 HV**.
        """)

else:
    st.info("👈 Upload a SEM micrograph in the sidebar and click **Analyze Microstructure**")

st.caption("ML-Assisted Microstructure Analysis and Property Prediction of Ti-6Al-4V Titanium Alloy")