import streamlit as st
import pandas as pd
import joblib

# Constants
EXPECTED_LENGTH = 187

# Load model
@st.cache_resource
def load_model():
    return joblib.load("ekg_model.pkl")

model = load_model()

# Page config
st.set_page_config(page_title="PulseWave CSV Classifier", layout="centered")
st.title("🩺 PulseWave EKG Classifier (Flexible CSV)")
st.markdown("_Brahmleen Papneja — Queen's University, Faculty of Health Sciences_")
st.markdown("Upload a CSV file with one or more EKG signals. Each row = 1 signal. Number of columns can vary — signals will be padded or trimmed to 187 samples.")

# File upload
uploaded_file = st.file_uploader("📂 Upload EKG CSV file", type="csv")

# Preprocessing
def preprocess_signals(df):
    processed = []
    for _, row in df.iterrows():
        arr = row.values.flatten()
        # Pad or trim
        if len(arr) < EXPECTED_LENGTH:
            arr = list(arr) + [0] * (EXPECTED_LENGTH - len(arr))
        elif len(arr) > EXPECTED_LENGTH:
            arr = arr[:EXPECTED_LENGTH]
        processed.append(arr)
    return pd.DataFrame(processed)

# Handle file
if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file, header=None)
        st.success(f"✅ Uploaded {raw.shape[0]} signal(s).")

        X = preprocess_signals(raw)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        # Build output table
        results_df = pd.DataFrame({
            "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in y_pred],
            "Prob_Normal (%)": (y_proba[:, 0] * 100).round(2),
            "Prob_Abnormal (%)": (y_proba[:, 1] * 100).round(2),
        })
        results_df.index.name = "Signal #"

        st.markdown("### 🔍 Results")
        filter_choice = st.selectbox("Filter results:", ["All", "Normal", "Abnormal"])
        if filter_choice == "Normal":
            filtered = results_df[results_df["Prediction"] == "✅ Normal"]
        elif filter_choice == "Abnormal":
            filtered = results_df[results_df["Prediction"] == "⚠️ Abnormal"]
        else:
            filtered = results_df

        st.dataframe(filtered, use_container_width=True)

        # Download button
        csv_bytes = results_df.to_csv().encode("utf-8")
        st.download_button("📥 Download results as CSV", data=csv_bytes, file_name="ekg_predictions.csv", mime="text/csv")

        # Plot signals
        st.markdown("### 📈 First 5 EKG Signals")
        for i in range(min(5, len(X))):
            st.line_chart(X.iloc[i])

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
