# UTS_ML-Prediksi-Tingkat-Pengangguran-di-Indonesia

## 📊 Indonesian Unemployment Rate Prediction App

A **Machine Learning** web app built with Streamlit that predicts unemployment rates in Indonesia based on socio-economic indicators.
The backend uses a **LightGBM Regressor** for predictive modeling and **Exponential Smoothing** for trend analysis across provinces.

---

## 🧾 Overview

This project demonstrates an end-to-end Machine Learning workflow, including:
1. **Data Preprocessing**  
2. **Feature Engineering**  
3. **Model Training & Evaluation (LightGBM + Exponential Smoothing)**  
4. **Model Saving using Joblib**  
5. **Interactive Web Deployment via Streamlit**

The goal is to estimate the **unemployment** percentage using key socio-economic features such as poverty rate, regional GDP, life expectancy, and education level.

---

## 🧰 Tech Stack

| Component | Tools / Libraries |
|---|---|
| Language | Python 3.11 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | LightGBM, Scikit-learn |
| Time Series | Statsmodels (Exponential Smoothing) |
| Model Tracking | Joblib |
| Deployment | Streamlit |
| Environment | Conda |
---

## ⚙️ 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/donnycharles88/UTS_ML-Menggunakan-LGBM-dan-ES.git
cd UTS_ML-Menggunakan-LGBM-dan-ES
```
## 📦 2. Setup Conda Environment (Recommended)
To ensure compatibility between dependencies (especially scikit-learn and LightGBM), use Conda:
```bash
conda create -n uts_ml python=3.11 -y
conda activate uts_ml
conda install -c conda-forge scikit-learn=1.6.1 lightgbm pandas numpy matplotlib seaborn joblib -y
pip install streamlit
```
## 🚀 3. Model Training
To train and evaluate the model:
```bash
python train_model.py
```

## 🌐 4. Running the Streamlit App
To launch the interactive web interface locally:
```bash
streamlit run predict.py
```
Then open the provided local URL (usually http://localhost:8501).

