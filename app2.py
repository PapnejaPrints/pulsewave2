import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import joblib

# ========================
# 📌 Author & App Header
# ========================
st.set_page_config(page_title="PulseWave EKG Classifier", layout="centered")
st.title("🩺 PulseWave Multi-EKG Abnormality Detector")
st.markdown("**Brahmleen Papneja · Faculty of Health Sciences · Queen’s University**")

# ========================
# 📦 Load Models
# ========================
@st.cache_resource
def load_models():
    csv_model = joblib.load("ekg_model.pkl")
    
    class ECGCNN(nn.Module):
        def __init__(self, num_classes=5):
            super(ECGCNN, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 28 * 28, 128),  # 224/8 = 28
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            return self.fc(x)

    cnn_model = ECGCNN(num_classes=5)
    cnn_model.load_state_dict(torch.load("ekg_cnn_state_dict.pth", map_location=torch.device("cpu")))
    cnn_model.eval()
    return csv_model, cnn_model

csv_model, cnn_model = load_models()

# ========================
# 📁 Tabs: CSV vs CNN
# ========================
tab1, tab2 = st.tabs(["📈 CSV-Based Classification", "🖼️ Image-Based CNN Classification"])

# ========================
# 🧪 CSV-BASED (RandomForest)
# ========================
with tab1:
    st.subheader("Upload a CSV file (187 features per EKG signal)")
    uploaded_file = st.file_uploader("Choose your .csv file", type="csv", key="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] != 187:
                st.error("⚠️ CSV must have exactly 187 columns.")
            else:
                preds = csv_model.predict(df)
                probs = csv_model.predict_proba(df)

                result_df = pd.DataFrame({
                    "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in preds],
                    "Prob_Normal (%)": (probs[:, 0] * 100).round(2),
                    "Prob_Abnormal (%)": (probs[:, 1] * 100).round(2),
                })
                st.success(f"{len(df)} EKG signals processed.")
                st.dataframe(result_df)

                st.markdown("### Preview of first 5 EKG signals")
                for i in range(min(5, len(df))):
                    st.line_chart(df.iloc[i])
        except Exception as e:
            st.error(f"Error: {e}")

# ========================
# 🧠 CNN-BASED (Image Classifier)
# ========================
with tab2:
    st.subheader("Upload a grayscale EKG image (224x224 recommended)")
    uploaded_image = st.file_uploader("Upload a .png or .jpg image", type=["png", "jpg", "jpeg"], key="image")

    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert("L")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = cnn_model(input_tensor)
                _, pred_class = torch.max(output, 1)

            class_names = ["CD", "HYP", "MI", "NORM", "STTC"]
            pred_label = class_names[pred_class.item()]
            st.image(image, caption=f"Predicted: {pred_label}", width=300)
            st.success(f"✅ CNN Prediction: {pred_label}")
        except Exception as e:
            st.error(f"Error processing image: {e}")
