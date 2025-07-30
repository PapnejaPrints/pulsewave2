import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib

# Constants
EXPECTED_LENGTH = 187
IMG_SIZE = 128

# ----- CNN Definition -----
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # (B, 1, 128, 128) → (B, 16, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2),            # (B, 16, 64, 64)
            nn.Conv2d(16, 32, 3, 1, 1), # (B, 32, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),            # (B, 32, 32, 32)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification: normal or abnormal
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ----- Model Loading -----
@st.cache_resource
def load_models():
    rf_model = joblib.load("ekg_model.pkl")
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load("ekg_cnn_state_dict.pth", map_location=torch.device("cpu")))
    cnn_model.eval()
    return rf_model, cnn_model

rf_model, cnn_model = load_models()

# ----- Page Setup -----
st.set_page_config(page_title="PulseWave EKG Classifier", layout="centered")
st.title("🩺 PulseWave EKG Classifier")
st.markdown("_Brahmleen Papneja — Queen's University, Faculty of Health Sciences_")

# ----- Input Type Selector -----
input_type = st.radio("📥 Choose input method:", ["📊 CSV EKG signals", "🖼️ ECG image file"])

# ----- CSV Processing -----
def preprocess_signals(df):
    processed = []
    for _, row in df.iterrows():
        arr = row.values.flatten()
        if len(arr) < EXPECTED_LENGTH:
            arr = list(arr) + [0] * (EXPECTED_LENGTH - len(arr))
        elif len(arr) > EXPECTED_LENGTH:
            arr = arr[:EXPECTED_LENGTH]
        processed.append(arr)
    return pd.DataFrame(processed)

# ----- Image Preprocessing -----
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Shape: (1, 1, 128, 128)
    return image

# ----- CSV Upload -----
if input_type == "📊 CSV EKG signals":
    st.markdown("Upload a `.csv` file with one or more EKG signals (rows), each of any length.")
    csv_file = st.file_uploader("📂 Upload CSV", type="csv")

    if csv_file:
        try:
            raw = pd.read_csv(csv_file, header=None)
            st.success(f"✅ Uploaded {raw.shape[0]} signal(s).")
            data = preprocess_signals(raw)
            preds = rf_model.predict(data)
            probs = rf_model.predict_proba(data)

            result_df = pd.DataFrame({
                "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in preds],
                "Prob_Normal (%)": (probs[:, 0] * 100).round(2),
                "Prob_Abnormal (%)": (probs[:, 1] * 100).round(2),
            })
            result_df.index.name = "Signal #"

            st.markdown("### 🔍 Results")
            filt = st.selectbox("Filter:", ["All", "Normal", "Abnormal"])
            if filt == "Normal":
                filtered = result_df[result_df["Prediction"] == "✅ Normal"]
            elif filt == "Abnormal":
                filtered = result_df[result_df["Prediction"] == "⚠️ Abnormal"]
            else:
                filtered = result_df

            st.dataframe(filtered, use_container_width=True)

            # Download
            st.download_button("📥 Download CSV", result_df.to_csv().encode(), "ekg_predictions.csv", "text/csv")

            # Charts
            st.markdown("### 📈 First 5 EKG Signals")
            for i in range(min(5, len(data))):
                st.line_chart(data.iloc[i])

        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")

# ----- Image Upload -----
else:
    st.markdown("Upload a grayscale `.png` or `.jpg` ECG image file (e.g., from PTB-XL images).")
    img_file = st.file_uploader("🖼️ Upload ECG image", type=["png", "jpg", "jpeg"])

    if img_file:
        try:
            image = Image.open(img_file).convert("L")
            st.image(image, caption="Uploaded ECG", width=300)
            img_tensor = preprocess_image(image)
            with torch.no_grad():
                logits = cnn_model(img_tensor)
                prob = torch.softmax(logits, dim=1).squeeze().numpy()
                pred = int(np.argmax(prob))

            label = "✅ Normal" if pred == 0 else "⚠️ Abnormal"
            st.markdown(f"### 🔍 Prediction: **{label}**")
            st.markdown(f"- **Prob Normal:** {prob[0]*100:.2f}%")
            st.markdown(f"- **Prob Abnormal:** {prob[1]*100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
