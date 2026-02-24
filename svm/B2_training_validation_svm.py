# =========================================================
# B2 - TRAINING & VALIDASI SVM DENGAN GRID SEARCH (FINAL)
# =========================================================

import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# =========================================================
# PATH FILE
# =========================================================
TRAIN_CSV = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_split\fitur_train.csv"
VAL_CSV   = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_split\fitur_val.csv"

MODEL_PATH  = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\model\model_svm.pkl"
SCALER_PATH = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\model\scaler.pkl"
VAL_OUTPUT_CSV = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\hasil\hasil_validasi.csv"

# =========================================================
# LOAD DATA
# =========================================================
print("📥 Memuat data training & validasi...")
df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)

# =========================================================
# PEMISAHAN FITUR & LABEL
# =========================================================
X_train = df_train.drop(columns=["Nama_Citra", "Label"])
y_train = df_train["Label"]

X_val = df_val.drop(columns=["Nama_Citra", "Label"])
y_val = df_val["Label"]

# =========================================================
# NORMALISASI DATA (WAJIB DISIMPAN)
# =========================================================
print("⚙️ Normalisasi data menggunakan StandardScaler...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# =========================================================
# GRID SEARCH SVM
# =========================================================
print("🔍 Melakukan Grid Search untuk SVM...")

param_grid = {
    "C": [1, 10, 50, 100, 500],
    "gamma": [0.001, 0.01, 0.1, 1],
    "kernel": ["rbf"]
}

svm = SVC(probability=True)

grid = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1,     # FIX error multiprocessing Windows
    verbose=2
)

grid.fit(X_train_scaled, y_train)

# =========================================================
# MODEL TERBAIK
# =========================================================
svm_model = grid.best_estimator_

print("\n🔥 HASIL GRID SEARCH")
print("Parameter terbaik :", grid.best_params_)
print("Akurasi CV terbaik:", round(grid.best_score_, 4))

# =========================================================
# EVALUASI TRAINING & VALIDASI
# =========================================================
train_pred = svm_model.predict(X_train_scaled)
val_pred   = svm_model.predict(X_val_scaled)

train_acc = accuracy_score(y_train, train_pred)
val_acc   = accuracy_score(y_val, val_pred)

print("\n🎯 HASIL AKHIR")
print(f"✅ Akurasi Training : {train_acc:.4f}")
print(f"✅ Akurasi Validasi : {val_acc:.4f}")

# =========================================================
# SIMPAN MODEL & SCALER (PISAH → PALING AMAN)
# =========================================================
joblib.dump(svm_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n💾 Model SVM disimpan di:")
print(MODEL_PATH)

print("💾 Scaler disimpan di:")
print(SCALER_PATH)

# =========================================================
# SIMPAN HASIL VALIDASI KE CSV
# =========================================================
df_val_result = df_val.copy()
df_val_result["Prediksi"] = val_pred
df_val_result.to_csv(VAL_OUTPUT_CSV, index=False)

print("\n📄 Hasil validasi disimpan di:")
print(VAL_OUTPUT_CSV)

print("\n✅ B2 TRAINING & VALIDASI SELESAI")