import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import os
import gdown

# Config
st.set_page_config(page_title="PulseWave", layout="centered")
st.title("🫀 PulseWave: EKG Abnormality Classifier")
st.caption("Brahmleen Papneja · Queen's University · Faculty of Health Sciences")
st.markdown("---")

# Constants
EXPECTED_LENGTH = 187
cnn_labels = ["CD", "HYP", "MI", "NORM", "STTC"]
CNN_MODEL_PATH = "ekg_cnn_state_dict.pth"
DRIVE_FILE_ID = "18O3DGE8Y8x466urKTkrSHvOFxAXfBSZs"  # Replace with your actual Drive file ID

# CNN model definition (MUST match training time)
class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (224 // 8) * (224 // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc(x)

# Download from Drive if missing
def ensure_cnn_model_downloaded():
    if not os.path.exists(CNN_MODEL_PATH):
        st.warning("Downloading CNN model from Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, CNN_MODEL_PATH, quiet=False)

# Load models
@st.cache_resource
def load_models():
    rf_model = joblib.load("ekg_model.pkl")

    ensure_cnn_model_downloaded()
    cnn_model = ECGCNN(num_classes=5)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location="cpu"))
    cnn_model.eval()

    return rf_model, cnn_model

rf_model, cnn_model = load_models()

# Input mode
mode = st.radio("Select Input Type:", ["📊 CSV EKG Signals", "🖼️ ECG Image"])

# Preprocess signal data to 187
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

# CSV Signal classification
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

            # Download option
            csv = result_df.to_csv().encode("utf-8")
            st.download_button("📥 Download Results CSV", data=csv, file_name="ekg_predictions.csv", mime="text/csv")

            st.markdown("### 📈 Signal Preview")
            for i in range(min(5, data.shape[0])):
                st.line_chart(data.iloc[i])

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Image-based CNN classification
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
                outputs = cnn_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()

            st.image(image, caption="Uploaded ECG", use_column_width=True)
            st.success(f"🧠 CNN Predicted Class: `{cnn_labels[pred_class]}`")

            st.markdown("### 🔬 Class Probabilities")
            st.json({cnn_labels[i]: f"{probs[0][i]*100:.2f}%" for i in range(len(cnn_labels))})

        except Exception as e:
            st.error(f"Image classification error: {e}")
