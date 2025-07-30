import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os

# -----------------------------
# 🔷 App Title and Credit
# -----------------------------
st.set_page_config(page_title="PulseWave EKG Analyzer", layout="wide")
st.markdown("""
    <div style='text-align: center; font-size: 16px; color: gray; margin-bottom: 20px;'>
        <strong>Brahmleen Papneja</strong> — Faculty of Health Sciences, Queen's University
    </div>
""", unsafe_allow_html=True)
st.title("🩺 PulseWave Multi-EKG Abnormality Detector")

# -----------------------------
# 🔶 Load Models
# -----------------------------
@st.cache_resource
def load_models():
    csv_model = joblib.load("ekg_model.pkl")
    cnn_model = torch.load("ekg_cnn_full_model.pth", map_location=torch.device("cpu"))
    cnn_model.eval()
    return csv_model, cnn_model

csv_model, cnn_model = load_models()

# -----------------------------
# 🔷 Mode Selection
# -----------------------------
mode = st.sidebar.selectbox("Select Mode", ["CSV Classifier", "Image Classifier"])

# -----------------------------
# 📊 CSV CLASSIFIER
# -----------------------------
if mode == "CSV Classifier":
    st.subheader("📄 Upload CSV (187 features per row)")
    uploaded_file = st.file_uploader("Choose your .csv file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, header=None)

            if data.shape[1] != 187:
                st.error(f"❌ CSV must have exactly 187 columns (your file has {data.shape[1]}).")
                st.stop()

            st.success(f"✅ Uploaded {data.shape[0]} EKG signals")

            probabilities = csv_model.predict_proba(data)
            predictions = csv_model.predict(data)

            results_df = pd.DataFrame({
                "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in predictions],
                "Prob_Normal (%)": (probabilities[:, 0] * 100).round(2),
                "Prob_Abnormal (%)": (probabilities[:, 1] * 100).round(2),
            })

            st.dataframe(results_df)

            st.markdown("### 📈 Preview of EKG Signals (first 5)")
            for i in range(min(5, data.shape[0])):
                st.line_chart(data.iloc[i])

        except Exception as e:
            st.error(f"Error reading file: {e}")

# -----------------------------
# 🖼️ IMAGE CLASSIFIER
# -----------------------------
elif mode == "Image Classifier":
    st.subheader("🖼️ Upload an EKG image (grayscale PNG)")

    uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("L")
            st.image(image, caption="Uploaded EKG", use_column_width=True)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

            input_tensor = transform(image).unsqueeze(0)
            output = cnn_model(input_tensor)
            _, predicted = torch.max(output, 1)

            class_names = os.listdir("ecg_5class")
            prediction = class_names[predicted.item()]

            st.success(f"📌 Predicted Class: `{prediction}`")

        except Exception as e:
            st.error(f"Error processing image: {e}")
