import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib

# ----- Page Config -----
st.set_page_config(page_title="PulseWave", layout="centered", page_icon="🫀")

# ----- Custom CSS Styling -----
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #f4f6f9;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4 {
            color: #1b4332;
        }
        .stButton>button {
            background-color: #1b4332;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            border: none;
            font-weight: bold;
        }
        .stFileUploader, .stRadio, .stSelectbox {
            background-color: #ffffff;
            padding: 1em;
            border-radius: 10px;
        }
        .stMarkdown code {
            background-color: #d8f3dc;
            color: #1b4332;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ----- App Header -----
st.markdown("""
    <div style="text-align: center;">
        <h1>🫀 PulseWave</h1>
        <h4 style="color:gray;">Multi-EKG Abnormality Detector</h4>
        <p>by <strong>Brahmleen Papneja</strong>, Faculty of Health Sciences, Queen's University</p>
        <hr style="width: 60%; margin: auto;">
    </div>
""", unsafe_allow_html=True)

# ----- CNN Model Definition -----
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
            nn.Linear(64 * (224//8) * (224//8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

# ----- Load Models -----
@st.cache_resource
def load_models():
    csv_model = joblib.load("ekg_model.pkl")
    cnn_model = ECGCNN()
    cnn_model.load_state_dict(torch.load("ekg_cnn_state_dict.pth", map_location=torch.device("cpu")))
    cnn_model.eval()
    return csv_model, cnn_model

csv_model, cnn_model = load_models()

# ----- Input Mode -----
mode = st.radio("📥 Choose input method:", ["📊 CSV EKG signals", "🖼️ ECG image file"])

# ----- CSV Upload and Prediction -----
if mode == "📊 CSV EKG signals":
    uploaded_file = st.file_uploader("Upload a .csv file with 187 signal values per row", type="csv")
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
                st.dataframe(result_df, use_container_width=True)

                st.markdown("### 📈 Signal Preview (first 5 rows):")
                for i in range(min(5, df.shape[0])):
                    st.line_chart(df.iloc[i])

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ----- Image Upload and Prediction -----
elif mode == "🖼️ ECG image file":
    uploaded_img = st.file_uploader("Upload grayscale ECG image (100x100 or larger)", type=["png", "jpg", "jpeg"])
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
                classes = ["CD", "HYP", "MI", "NORM", "STTC"]

                st.image(image, caption="📷 Uploaded ECG", use_column_width=True)

                st.markdown(f"""
                <div style='background-color:#1b4332; padding: 1rem; border-radius: 10px; text-align: center;'>
                    <h3 style='color:white;'>🧠 Predicted class: <code>{classes[pred_class]}</code></h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 🔍 Class Probabilities")
                st.json({classes[i]: f"{probs[0][i]*100:.2f}%" for i in range(5)})

        except Exception as e:
            st.error(f"❌ Image classification error: {e}")
