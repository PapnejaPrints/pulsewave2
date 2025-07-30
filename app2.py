import streamlit as st
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import os

# Title
st.set_page_config(page_title="PulseWave", layout="centered")
st.markdown("### 🫀 PulseWave Multi-EKG Abnormality Detector")
st.markdown("_by Brahmleen Papneja, Faculty of Health Sciences, Queen's University_")
st.markdown("---")

# Load Models
@st.cache_resource
def load_models():
    csv_model = joblib.load("ekg_model.pkl")  # For CSV input
    cnn_model = torch.load("ekg_cnn_full_model.pth", map_location=torch.device("cpu"))  # CNN model (full)
    cnn_model.eval()
    return csv_model, cnn_model

csv_model, cnn_model = load_models()

# Input mode
mode = st.radio("Choose input method:", ["📊 CSV EKG signals", "🖼️ ECG image file"])

if mode == "📊 CSV EKG signals":
    uploaded_file = st.file_uploader("Upload .csv file (187 columns)", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] != 187:
                st.error(f"Expected 187 columns, got {df.shape[1]}")
            else:
                st.success(f"Uploaded {df.shape[0]} EKG signal(s)")

                preds = csv_model.predict(df)
                probs = csv_model.predict_proba(df)

                result_df = pd.DataFrame({
                    "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in preds],
                    "Prob_Normal (%)": (probs[:, 0] * 100).round(2),
                    "Prob_Abnormal (%)": (probs[:, 1] * 100).round(2),
                })
                st.dataframe(result_df)

                st.markdown("### 📈 Signal Preview (first 5):")
                for i in range(min(5, df.shape[0])):
                    st.line_chart(df.iloc[i])

        except Exception as e:
            st.error(f"Error processing file: {e}")

elif mode == "🖼️ ECG image file":
    uploaded_img = st.file_uploader("Upload grayscale ECG image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        try:
            image = Image.open(uploaded_img).convert("L")  # Convert to grayscale
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                outputs = cnn_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                classes = ["CD", "HYP", "MI", "NORM", "STTC"]

                st.image(image, caption="Uploaded ECG", use_column_width=True)
                st.success(f"🧠 Predicted class: `{classes[pred_class]}`")

                st.markdown("### Class Probabilities:")
                prob_dict = {classes[i]: f"{probs[0][i]*100:.2f}%" for i in range(len(classes))}
                st.json(prob_dict)

        except Exception as e:
            st.error(f"Image classification error: {e}")
