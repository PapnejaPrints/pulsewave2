import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib
import requests
import os

# Config
st.set_page_config(page_title="PulseWave", layout="centered")
st.title("🫀 PulseWave: EKG Abnormality Classifier")
st.caption("Brahmleen Papneja · Queen's University · Faculty of Health Sciences")
st.markdown("---")

# Constants
EXPECTED_LENGTH = 187
cnn_labels = ["CD", "HYP", "MI", "NORM", "STTC"]
RESNET_MODEL_URL = "https://drive.google.com/file/d/18O3DGE8Y8x466urKTkrSHvOFxAXfBSZs"
RESNET_MODEL_PATH = "ekg_resnet18.pth"

# Download ResNet model from Google Drive
def download_model():
    if not os.path.exists(RESNET_MODEL_PATH):
        st.info("📥 Downloading ResNet18 model from Google Drive...")
        r = requests.get(RESNET_MODEL_URL, stream=True)
        with open(RESNET_MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("✅ Model downloaded.")

# Load models
@st.cache_resource
def load_models():
    download_model()
    rf_model = joblib.load("ekg_model.pkl")

    # Define ResNet18 with 5 output classes
    resnet18 = models.resnet18(pretrained=False)
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # grayscale input
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 5)

    # Load weights (state_dict)
    resnet18.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location="cpu"))
    resnet18.eval()

    return rf_model, resnet18

rf_model, resnet18 = load_models()

# Preprocess CSV signals
def preprocess_signals(df):
    cleaned = []
    for _, row in df.iterrows():
        row_array = row.values.flatten()
        if len(row_array) < EXPECTED_LENGTH:
            row_array = list(row_array) + [0] * (EXPECTED_LENGTH - len(row_array))
        elif len(row_array) > EXPECTED_LENGTH:
            row_array = row_array[:EXPECTED_LENGTH]
        cleaned.append(row_array)
    return pd.DataFrame(cleaned)

# Input mode
mode = st.radio("Select Input Type:", ["📊 CSV EKG Signals", "🖼️ ECG Image"])

# CSV-based classification
if mode == "📊 CSV EKG Signals":
    uploaded_file = st.file_uploader("Upload CSV file (each row = 1 EKG signal)", type="csv")
    if uploaded_file:
        try:
            raw = pd.read_csv(uploaded_file, header=None)
            data = preprocess_signals(raw)

            st.success(f"✅ {data.shape[0]} EKG signals processed.")
            predictions = rf_model.predict(data)
            probs = rf_model.predict_proba(data)

            result_df = pd.DataFrame({
                "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in predictions],
                "Prob_Normal (%)": (probs[:, 0] * 100).round(2),
                "Prob_Abnormal (%)": (probs[:, 1] * 100).round(2),
            })
            result_df.index.name = "Signal #"

            st.markdown("### 🔍 Results")
            filter_choice = st.selectbox("Filter:", ["All", "Normal", "Abnormal"])
            if filter_choice == "Normal":
                st.dataframe(result_df[result_df["Prediction"] == "✅ Normal"])
            elif filter_choice == "Abnormal":
                st.dataframe(result_df[result_df["Prediction"] == "⚠️ Abnormal"])
            else:
                st.dataframe(result_df)

            csv = result_df.to_csv().encode("utf-8")
            st.download_button("📥 Download Results CSV", data=csv, file_name="ekg_predictions.csv", mime="text/csv")

            st.markdown("### 📈 Signal Preview")
            for i in range(min(5, data.shape[0])):
                st.line_chart(data.iloc[i])

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Image-based classification
elif mode == "🖼️ ECG Image":
    uploaded_img = st.file_uploader("Upload grayscale ECG image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        try:
            image = Image.open(uploaded_img).convert("L")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input_tensor = transform(image).unsqueeze(0)  # Add batch dim

            with torch.no_grad():
                outputs = resnet18(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()

            st.image(image, caption="Uploaded ECG", use_column_width=True)
            st.success(f"🧠 ResNet18 Predicted Class: `{cnn_labels[pred_class]}`")

            st.markdown("### 🔬 Class Probabilities")
            st.json({cnn_labels[i]: f"{probs[0][i]*100:.2f}%" for i in range(len(cnn_labels))})

        except Exception as e:
            st.error(f"Image classification error: {e}")
