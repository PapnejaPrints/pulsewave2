import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("ekg_model.pkl")

# Set page config and header
st.set_page_config(page_title="EKG Abnormality Detector", layout="centered")
st.title("🩺 EKG Abnormality Detector (Batch Mode)")
st.markdown("**By Brahmleen Papneja – Queen's University, Faculty of Health Sciences**")
st.markdown("Upload a CSV where each row contains an EKG signal (any length). The app will automatically handle the formatting.")

EXPECTED_LENGTH = 187

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

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file, header=None)
        st.success(f"✅ Uploaded {raw.shape[0]} EKG signal(s)")

        data = preprocess_signals(raw)

        probs = model.predict_proba(data)
        preds = model.predict(data)

        results_df = pd.DataFrame({
            "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in preds],
            "Prob_Normal (%)": (probs[:, 0] * 100).round(2),
            "Prob_Abnormal (%)": (probs[:, 1] * 100).round(2),
        })
        results_df.index.name = "Signal #"

        st.markdown("### 🔍 Results")
        filter_opt = st.selectbox("Filter results:", ["All", "Normal", "Abnormal"])

        if filter_opt == "Normal":
            filtered = results_df[results_df["Prediction"] == "✅ Normal"]
        elif filter_opt == "Abnormal":
            filtered = results_df[results_df["Prediction"] == "⚠️ Abnormal"]
        else:
            filtered = results_df

        st.dataframe(filtered, use_container_width=True)

        # CSV download
        csv_data = results_df.to_csv().encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv_data, "ekg_results.csv", "text/csv")

        st.markdown("### 📈 Preview First 5 Signals")
        for i in range(min(5, data.shape[0])):
            st.line_chart(data.iloc[i])

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
