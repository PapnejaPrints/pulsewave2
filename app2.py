import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib

# --------- Page Config & Styling ---------
st.set_page_config(page_title="PulseWave", layout="centered", page_icon="🫀")
st.markdown("""
    <style>
    body {
        color: white;
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stRadio > div {
        background-color: #1c1e26;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stFileUploader {
        background-color: #1c1e26;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: white;
    }
    .stButton > button {
        background-color: #1b4332;
        color: white;
        border: None;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

# --------- Title ---------
st.markdown("### 🫀 PulseWave Multi-EKG Abnormality Detector")
st.markdown("_by Brahmleen Papneja, Faculty of Health Sciences, Queen's University_")
st.markdown("---")

# --------- CNN Model ---------
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
            nn.Linear(64 * (224 // 8) * (224 // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

# --------- Load Models ---------
@st.cache_resource
def load_models():
    csv_model = joblib.load("ekg_model.pkl")
    cnn_model = ECGCNN()
    cnn_model.load_state_dict(torch.load("ekg_cnn_state_dict.pth", map_location=torch.device("cpu")))
    cnn_model.eval()
    return csv_model, cnn_model

csv_model, cnn_model = load_models()
classes = ["CD", "HYP", "MI", "NORM", "STTC"]

# --------- Input Selector ---------
mode = st.radio("📤 Choose input method:", ["📊 CSV EKG signals", "🖼️ ECG image file"])

# --------- CSV Input Handling ---------
if mode == "📊 CSV EKG signals":
    uploaded_file = st.file_uploader("Upload a .csv file (187 columns)", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] != 187:
                st.error(f"❌ Expected 187 columns, got {df.shape[1]}")
            else:
                st.success(f"✅ Uploaded {df.shape[0]} EKG signal(s)")
                preds = csv_model.predict(df)
                probs = csv_model.predict_proba(df)

                result_df = pd.DataFrame({
                    "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in preds],
                    "Prob_Normal (%)": (probs[:, 0] * 100).round(2),
                    "Prob_Abnormal (%)": (probs[:, 1] * 100).round(2),
                })
                st.dataframe(result_df)

                st.markdown("### 📈 Signal Preview (first 5 rows):")
                for i in range(min(5, df.shape[0])):
                    st.line_chart(df.iloc[i])

        except Exception as e:
            st.error(f"Error: {e}")

# --------- Image Input Handling ---------
elif mode == "🖼️ ECG image file":
    uploaded_img = st.file_uploader("Upload grayscale ECG image (e.g. 224x224)", type=["png", "jpg", "jpeg"])
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

            st.image(image, caption="📷 Uploaded ECG", use_column_width=True)
            st.success(f"🧠 Predicted class: `{classes[pred_class]}`")

            st.markdown("### 🔍 Class Probabilities")
            st.json({classes[i]: f"{probs[0][i]*100:.2f}%" for i in range(5)})

        except Exception as e:
            st.error(f"Image classification error: {e}")
