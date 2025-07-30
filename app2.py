import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import joblib
import os
import gdown

# ------------- CONFIG -------------
st.set_page_config(page_title="PulseWave", layout="centered")
st.title("🫀 PulseWave: EKG Classifier")
st.markdown("_Brahmleen Papneja – Queen's University Faculty of Health Sciences_")
st.markdown("---")

# ------------- CONSTANTS -------------
MODEL_PATH = "ekg_resnet18.pth"
DRIVE_FILE_ID = "18O3DGE8Y8x466urKTkrSHvOFxAXfBSZs"  # Replace with your actual file ID
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]
EXPECTED_SIGNAL_LENGTH = 187

# ------------- DOWNLOAD MODEL FROM DRIVE IF NEEDED -------------
def download_resnet_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

download_resnet_model()

# ------------- LOAD MODELS -------------
@st.cache_resource
def load_models():
    # Load CSV model
    rf_model = joblib.load("ekg_model.pkl")

    # Load ResNet18 image model
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    return rf_model, model

rf_model, cnn_model = load_models()

# ------------- INPUT MODE -------------
mode = st.radio("Choose input type:", ["📊 CSV EKG signals", "🖼️ ECG image"])

# ------------- CSV MODE -------------
if mode == "📊 CSV EKG signals":
    uploaded_csv = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_csv:
        try:
            raw = pd.read_csv(uploaded_csv, header=None)

            def preprocess_signals(df):
                cleaned = []
                for _, row in df.iterrows():
                    row_array = row.values.flatten()
                    if len(row_array) < EXPECTED_SIGNAL_LENGTH:
                        row_array = list(row_array) + [0] * (EXPECTED_SIGNAL_LENGTH - len(row_array))
                    elif len(row_array) > EXPECTED_SIGNAL_LENGTH:
                        row_array = row_array[:EXPECTED_SIGNAL_LENGTH]
                    cleaned.append(row_array)
                return pd.DataFrame(cleaned)

            data = preprocess_signals(raw)
            preds = rf_model.predict(data)
            probs = rf_model.predict_proba(data)

            results = pd.DataFrame({
                "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in preds],
                "Prob_Normal (%)": (probs[:, 0] * 100).round(2),
                "Prob_Abnormal (%)": (probs[:, 1] * 100).round(2)
            })
            st.dataframe(results)

            csv_out = results.to_csv().encode("utf-8")
            st.download_button("📥 Download Results", csv_out, "ekg_predictions.csv", "text/csv")

            st.markdown("### 📈 First 5 Signal Charts")
            for i in range(min(5, data.shape[0])):
                st.line_chart(data.iloc[i])

        except Exception as e:
            st.error(f"Error: {e}")

# ------------- IMAGE MODE -------------
elif mode == "🖼️ ECG image":
    uploaded_img = st.file_uploader("Upload ECG image (grayscale)", type=["png", "jpg", "jpeg"])
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
                output = cnn_model(input_tensor)
                prob = torch.softmax(output, dim=1)[0]
                pred_class = torch.argmax(prob).item()

            st.image(image, caption="Uploaded ECG", use_column_width=True)
            st.success(f"🧠 Predicted Class: `{CLASSES[pred_class]}`")
            st.markdown("### Class Probabilities")
            st.json({CLASSES[i]: f"{prob[i]*100:.2f}%" for i in range(len(CLASSES))})

        except Exception as e:
            st.error(f"Image classification error: {e}")
