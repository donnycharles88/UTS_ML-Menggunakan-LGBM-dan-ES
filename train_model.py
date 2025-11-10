import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD DATA
# =====================================================
df = pd.read_csv("2021socio_economic_indonesia.csv")
print("Kolom dataset:", df.columns.tolist())

# Tambahkan kolom unemployment jika belum ada
if "unemployment" not in df.columns:
    np.random.seed(42)
    df["unemployment"] = (
        0.5 * df["poorpeople_percentage"]
        - 0.000005 * df["reg_gdp"]
        + np.random.uniform(3, 10, size=len(df))
    )
    print("⚠️ Kolom 'unemployment' tidak ditemukan, kolom dummy otomatis ditambahkan.")

# =====================================================
# 2. DEFINE FEATURES & TARGET
# =====================================================
target = "unemployment"
numeric_features = [
    "poorpeople_percentage", "reg_gdp", "life_exp", "avg_schooltime", "exp_percap"
]
categorical_features = ["province", "cities_reg"]

df = df.dropna(subset=numeric_features + [target])
X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================
# 3. PIPELINE LIGHTGBM
# =====================================================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model_lgbm = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

pipeline_lgbm = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model_lgbm)
])

# =====================================================
# 4. TRAIN MODEL LIGHTGBM
# =====================================================
pipeline_lgbm.fit(X_train, y_train)
y_pred = pipeline_lgbm.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n📊 Evaluasi Model LightGBM")
print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"R²   : {r2:.3f}")
print(f"MAPE : {mape:.3f}")

# =====================================================
# 5. EXPONENTIAL SMOOTHING TRAINING & SAVE
# =====================================================
print("\n📈 Latih Model Exponential Smoothing berdasarkan rata-rata provinsi...")

prov_mean = df.groupby("province")[target].mean().reset_index().sort_values(target)
series = prov_mean[target].values

alphas = np.linspace(0.1, 0.9, 9)
best_alpha, best_mse, best_fit = None, float("inf"), None

for alpha in alphas:
    model_es = SimpleExpSmoothing(series)
    fit = model_es.fit(smoothing_level=alpha, optimized=False)
    mse_es = mean_squared_error(series, fit.fittedvalues)
    if mse_es < best_mse:
        best_alpha, best_mse, best_fit = alpha, mse_es, fit

print(f"✅ Alpha terbaik: {best_alpha:.2f} | MSE: {best_mse:.3f}")

# Simpan model
joblib.dump(pipeline_lgbm, "model/model_lightgbm.pkl")
joblib.dump(best_fit, "model/model_expsmooth.pkl")
print("\n💾 Model disimpan sebagai:")
print(" - model_lightgbm.pkl")
print(" - model_expsmooth.pkl")

# Visualisasi
plt.figure(figsize=(10, 5))
plt.plot(series, "o-", label="Aktual", color="blue")
plt.plot(best_fit.fittedvalues, "r--", label=f"Exponential Smoothing α={best_alpha:.2f}")
plt.legend()
plt.title("Aktual vs Exponential Smoothing")
plt.show()
