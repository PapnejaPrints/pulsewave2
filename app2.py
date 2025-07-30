import streamlit as st
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import os
import gdown

# Constants
EXPECTED_LENGTH = 187
RESNET_MODEL_PATH = "ekg_resnet18.pth"
RESNET_FILE_ID = "YOUR_FILE_ID"  # Replace with your actual Google Drive file ID
RF_MODEL_PATH = "ekg_model.pkl"
CNN_LABELS = ["CD", "HYP", "MI", "NORM", "STTC"]

# App config
st.set_page_config(page_title="PulseWave", layout="centered")
st.title("🫀 PulseWave: EKG Abnormality Classifier")
st.caption("Brahmleen Papneja · Queen's University · Faculty of Health Sciences")
st.markdown("---")

# Download ResNet model from Google Drive if not present
@st.cache_resource
def download_and_load_models():
    if not os.path.exists(RESNET_MODEL_PATH):
        gdown.download(f"https://drive.google.com/file/d/18O3DGE8Y8x466urKTkrSHvOFxAXfBSZs", RESNET_MODEL_PATH, quiet=False)
    rf_model = joblib.load(RF_MODEL_PATH)
    resnet_model = torch.load(RESNET_MODEL_PATH, map_location="cpu")
    resnet_model.eval()
    return rf_model, resnet_model

rf_model, resnet_model = download_and_load_models()

# Helper: pad or trim signal
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

# Input selection
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
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = resnet_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()

            st.image(image, caption="Uploaded ECG", use_column_width=True)
            st.success(f"🧠 ResNet18 Predicted Class: `{CNN_LABELS[pred_class]}`")

            st.markdown("### 🔬 Class Probabilities")
            st.json({CNN_LABELS[i]: f"{probs[0][i]*100:.2f}%" for i in range(len(CNN_LABELS))})

        except Exception as e:
            st.error(f"Image classification error: {e}")
