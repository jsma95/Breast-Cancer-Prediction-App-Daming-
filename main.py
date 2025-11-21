import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ============================
# Load Model & Scaler
# ============================
model_rf = joblib.load("model_rf.joblib")
model_svm = joblib.load("model_svm.joblib")
model_vote = joblib.load("model_vote.joblib")
scaler = joblib.load("scaler.joblib")

# ============================
# Konfigurasi Halaman
# ============================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)

st.title("🔬 Breast Cancer Prediction App")
st.write("""
Aplikasi ini memprediksi apakah sampel termasuk **Malignant (Kanker Berbahaya)**  
atau **Benign (Tidak berbahaya)** berdasarkan 30 fitur medis.

Gunakan model yang kamu inginkan di sidebar.
""")

# ============================
# Sidebar: Model Selector
# ============================
st.sidebar.header("⚙️ Pengaturan Model")

model_choice = st.sidebar.selectbox(
    "Pilih Model:",
    ("Random Forest", "SVM", "Voting Ensemble")
)

model_dict = {
    "Random Forest": model_rf,
    "SVM": model_svm,
    "Voting Ensemble": model_vote
}

selected_model = model_dict[model_choice]

# ============================
# Form Input User
# ============================
st.header("🧪 Input Data Diagnosis")

st.write("Masukkan nilai 30 fitur berikut:")

# Untuk memudahkan user → tampilkan angka default (rata-rata dataset)
default_value = 10.0

feature_values = []

# 30 fitur aslinya
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(feature, value=float(default_value))
        feature_values.append(val)

# ============================
# Prediksi
# ============================
if st.button("🔍 Prediksi"):
    # Format input ke numpy array
    input_array = np.array(feature_values).reshape(1, -1)

    # Scaling
    input_scaled = scaler.transform(input_array)

    # Predict
    pred = selected_model.predict(input_scaled)[0]
    proba = selected_model.predict_proba(input_scaled)[0]

    # ============================
    # Tampilkan Hasil
    # ============================
    st.subheader("📊 Hasil Prediksi")

    if pred == 1:
        st.error("**Hasil: Malignant (Kanker Berbahaya)**")
        st.write(f"Probabilitas: {proba[1]*100:.2f}%")
    else:
        st.success("**Hasil: Benign (Tidak Berbahaya)**")
        st.write(f"Probabilitas: {proba[0]*100:.2f}%")

    st.info(f"Model yang digunakan: **{model_choice}**")

