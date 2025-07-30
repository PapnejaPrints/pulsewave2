import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
from cnn_model import CNN  # Update to match your model file name

# Streamlit page settings
st.set_page_config(page_title="PulseWave", layout="centered", page_icon="🫀")

# ---------- Custom Styling ----------
st.markdown("""
    <style>
    .reportview-container .main {{
        background-color: #f9f9f9;
    }}
    .stButton>button {{
        background-color: #1b4332;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        border: none;
    }}
    .stFileUploader {{
        margin-top: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
    <h1 style='text-align: center;'>🫀 PulseWave</h1>
    <h4 style='text-align: center; color: gray;'>Multi-Label ECG Abnormality Classifier</h4>
    <p style='text-align: center;'>by <b>Brahmleen Papneja</b>, Faculty of Health Sciences, Queen's University</p>
    <hr style='border-top: 1px solid #bbb;'>
""", unsafe_allow_html=True)

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    csv_model = joblib.load("ekg_rf_model.pkl")
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load("ekg_cnn_state_dict.pth", map_location=torch.device("cpu")))
    cnn_model.eval()
    return csv_model, cnn_model

csv_model, cnn_model = load_models()
classes = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# ---------- Sidebar Settings ----------
st.sidebar.markdown("## ⚙️ Choose Input Mode")
mode = st.sidebar.radio("Select input type:", ["📊 CSV EKG Signals", "🖼️ ECG Image File"])

# ---------- CSV Input Mode ----------
if mode == "📊 CSV EKG Signals":
    st.subheader("📥 Upload CSV File")
    uploaded_csv = st.file_uploader("Upload a CSV with 187 features", type="csv")

    if uploaded_csv is not None:
        import pandas as pd
        data = pd.read_csv(uploaded_csv, header=None)

        if data.shape[1] != 187:
            st.error("CSV must contain exactly 187 values per row.")
        else:
            prediction = csv_model.predict(data)[0]
            st.success(f"🧠 Predicted class: {classes[prediction]}")

# ---------- Image Input Mode ----------
else:
    st.subheader("🖼️ Upload ECG Image")
    uploaded_img = st.file_uploader("Upload a grayscale ECG image (100x100)", type=["png", "jpg", "jpeg"])

    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert("L")
        st.image(image, caption="🩺 Uploaded ECG", use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()
        ])

        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = cnn_model(img_tensor)
            probs = torch.softmax(output, dim=1).numpy()
            pred_class = np.argmax(probs)

        st.markdown(f"""
        <div style='background-color:#1b4332; padding: 1em; border-radius: 10px; text-align: center;'>
            <h3 style='color:white;'>🧠 Predicted class: <code>{classes[pred_class]}</code></h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Class Probabilities")
        st.json({cls: f"{probs[0][i]*100:.2f}%" for i, cls in enumerate(classes)})
