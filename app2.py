import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
from model import CNN

# Set Streamlit page config
st.set_page_config(page_title="PulseWave", layout="centered")

# ---------- UI HEADER ----------
st.markdown("""
    <h1 style='text-align: center;'>🫀 PulseWave</h1>
    <h4 style='text-align: center; color: gray;'>Multi-Label ECG Abnormality Classifier</h4>
    <p style='text-align: center;'>by <b>Brahmleen Papneja</b>, Faculty of Health Sciences, Queen's University</p>
    <hr style='border-top: 1px solid #bbb;'>
""", unsafe_allow_html=True)

# ---------- Model Loader ----------
@st.cache_resource
def load_models():
    # Load CSV model
    csv_model = joblib.load("ekg_rf_model.pkl")

    # Load CNN model
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load("ekg_cnn_state_dict.pth", map_location=torch.device("cpu")))
    cnn_model.eval()

    return csv_model, cnn_model

csv_model, cnn_model = load_models()

# ---------- Input Options ----------
st.sidebar.markdown("## ⚙️ Settings")
mode = st.sidebar.radio("Choose input method:", ["📊 CSV EKG signals", "🖼️ ECG image file"])

classes = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# ---------- CSV Mode ----------
if mode == "📊 CSV EKG signals":
    st.subheader("📥 Upload CSV file")
    uploaded_csv = st.file_uploader("Upload a CSV file with 187 features", type="csv")

    if uploaded_csv is not None:
        import pandas as pd
        data = pd.read_csv(uploaded_csv, header=None)
        if data.shape[1] != 187:
            st.error("CSV must contain exactly 187 values per row.")
        else:
            prediction = csv_model.predict(data)[0]
            st.markdown(f"""
            <div style='background-color:#1b4332; padding: 1em; border-radius: 10px; text-align: center;'>
                <h3 style='color:white;'>🧠 Predicted class: <code>{classes[prediction]}</code></h3>
            </div>
            """, unsafe_allow_html=True)

# ---------- Image Mode ----------
else:
    st.subheader("🖼️ Upload ECG Image")
    uploaded_img = st.file_uploader("Upload an ECG image (grayscale, 100x100)", type=["png", "jpg", "jpeg"])

    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert("L")
        st.image(image, caption="Uploaded ECG", use_container_width=True)

        # Transform and predict
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()
        ])

        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = cnn_model(img_tensor)
            probs = torch.softmax(output, dim=1).numpy()
            pred_class = np.argmax(probs)

        # Display result
        st.markdown(f"""
        <div style='background-color:#1b4332; padding: 1em; border-radius: 10px; text-align: center;'>
            <h3 style='color:white;'>🧠 Predicted class: <code>{classes[pred_class]}</code></h3>
        </div>
        """, unsafe_allow_html=True)

        # Show class probabilities
        st.markdown("### 🔍 Class Probabilities:")
        for i in range(5):
            st.markdown(f"- **{classes[i]}**: `{probs[0][i]*100:.2f}%`")
