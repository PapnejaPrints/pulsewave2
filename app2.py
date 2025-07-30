import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import pandas as pd

# Define your CNN model architecture here
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 25 * 25, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 5)  # 5 output classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Streamlit page setup
st.set_page_config(page_title="PulseWave", layout="centered", page_icon="🫀")

# Page title and description
st.markdown("""
    <h1 style='text-align: center;'>🫀 PulseWave</h1>
    <h4 style='text-align: center; color: gray;'>Multi-Class ECG Classifier</h4>
    <p style='text-align: center;'>by <b>Brahmleen Papneja</b>, Queen's University</p>
    <hr>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    rf_model = joblib.load("ekg_model.pkl")
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load("ekg_cnn_state_dict.pth", map_location="cpu"))
    cnn_model.eval()
    return rf_model, cnn_model

rf_model, cnn_model = load_models()
classes = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# Sidebar
st.sidebar.title("⚙️ Input Mode")
mode = st.sidebar.radio("Choose input type:", ["CSV (EKG Features)", "Image (Grayscale ECG)"])

# CSV mode
if mode == "CSV (EKG Features)":
    st.subheader("📥 Upload CSV File")
    uploaded_csv = st.file_uploader("Upload CSV with 187 numeric features", type="csv")

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv, header=None)
        if df.shape[1] != 187:
            st.error("❌ CSV must contain exactly 187 features.")
        else:
            pred = rf_model.predict(df)[0]
            st.success(f"🧠 Predicted Class: {classes[pred]}")

# Image mode
else:
    st.subheader("🖼️ Upload ECG Image")
    uploaded_img = st.file_uploader("Upload a 100x100 grayscale ECG image", type=["png", "jpg", "jpeg"])

    if uploaded_img:
        image = Image.open(uploaded_img).convert("L")
        st.image(image, caption="Uploaded ECG", use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()
        ])
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = cnn_model(img_tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]
            pred_idx = np.argmax(probs)

        st.markdown(f"""
        <div style='background-color: #1b4332; color: white; padding: 1em; border-radius: 8px; text-align: center;'>
            <h4>🧠 Predicted Class: <code>{classes[pred_idx]}</code></h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Class Probabilities")
        st.json({cls: f"{probs[i]*100:.2f}%" for i, cls in enumerate(classes)})
