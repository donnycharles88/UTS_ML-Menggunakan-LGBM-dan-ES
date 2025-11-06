import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1️⃣ Load Model, Scaler, dan Feature Columns
# ============================================================
xgb_model = joblib.load("model/xgb_model.pkl")
rf_model = joblib.load("model/rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

st.title("🚗 Prediksi Harga Mobil Bekas")

st.sidebar.header("Masukkan Data Mobil")

# ============================================================
# 2️⃣ Input User
# ============================================================
year = st.sidebar.number_input("Tahun", 2000, 2025, 2018)
mileage = st.sidebar.number_input("Kilometer (km)", 0, 500000, 50000)
instalment = st.sidebar.number_input("Cicilan per bulan (Rp)", 0, 20000000, 5000000)
transmission = st.sidebar.selectbox("Transmisi", ["Automatic", "Manual"])
brand = st.sidebar.text_input("Merek Mobil", "Toyota")
location = st.sidebar.text_input("Lokasi", "Jakarta")

# ============================================================
# 3️⃣ Siapkan DataFrame dengan semua kolom fitur
# ============================================================
input_data = pd.DataFrame(columns=feature_columns)
input_data.loc[0] = 0  # isi default semua 0

# Isi kolom yang tersedia dari input user
input_data['year'] = year
input_data['mileage (km)'] = mileage
input_data['instalment (Rp|Monthly)'] = instalment
input_data['transmission'] = 0 if transmission == "Manual" else 1
input_data['brand'] = 0  # sementara dummy encoder
input_data['location'] = 0

# ============================================================
# 4️⃣ Scale kolom numerik
# ============================================================
num_cols = ['year', 'mileage (km)', 'instalment (Rp|Monthly)']
input_data[num_cols] = scaler.transform(input_data[num_cols])

# ============================================================
# 5️⃣ Pilih Model dan Prediksi
# ============================================================
model_choice = st.selectbox("Pilih Model", ["XGBoost", "Random Forest"])

if st.button("🔍 Prediksi Harga"):
    if model_choice == "XGBoost":
        pred = xgb_model.predict(input_data)[0]
    else:
        pred = rf_model.predict(input_data)[0]
    
    st.success(f"💰 Estimasi Harga Mobil: Rp {pred:,.0f}")