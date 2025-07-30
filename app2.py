import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib

# Config
st.set_page_config(page_title="PulseWave", layout="centered")
st.title("🫀 PulseWave: EKG Abnormality Classifier")
st.caption("Brahmleen Papneja · Queen's University · Faculty of Health Sciences")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    # Random Forest
    rf_model = joblib.load("ekg_model.pkl")

    # ResNet18
    resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Grayscale
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 5)
    resnet18.load_state_dict(torch.load("ekg_resnet18.pth", map_location="cpu"))
    resnet18.eval()

    return rf_model, resnet18

rf_model, resnet_model = load_models()

# Constants
EXPECTED_LENGTH = 187
resnet_labels = ["CD", "HYP", "MI", "NORM", "STTC"]

# Input mode
mode = st.radio("Select Input Type:", ["📊 CSV EKG Signals", "🖼️ ECG Image"])

# Preprocessing
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

# CSV classification
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

# Image classification
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
            st.success(f"🧠 ResNet18 Prediction: `{resnet_labels[pred_class]}`")

            st.markdown("### 🔬 Class Probabilities")
            st.json({resnet_labels[i]: f"{probs[0][i]*100:.2f}%" for i in range(len(resnet_labels))})

        except Exception as e:
            st.error(f"⚠️ Image classification error: {e}")
