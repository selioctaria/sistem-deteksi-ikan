# =========================================================
# SVM - HYBRID (GLCM + CANNY)
# =========================================================

import pandas as pd
import joblib
import os

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# PATH CONFIG
# =========================================================
BASE_PATH = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN"

TRAIN_CSV = os.path.join(BASE_PATH, "fitur/fitur_split/fitur_train.csv")
VAL_CSV   = os.path.join(BASE_PATH, "fitur/fitur_split/fitur_val.csv")
TEST_CSV  = os.path.join(BASE_PATH, "fitur/fitur_split/fitur_test.csv")

MODEL_DIR = os.path.join(BASE_PATH, "support_vector_machine/hybrid/model")
HASIL_DIR = os.path.join(BASE_PATH, "support_vector_machine/hybrid/hasil")

MODEL_PATH  = os.path.join(MODEL_DIR, "svm_hybrid.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_hybrid.pkl")

TRAIN_OUTPUT = os.path.join(HASIL_DIR, "hasil_training.csv")
VAL_OUTPUT   = os.path.join(HASIL_DIR, "hasil_validasi.csv")
TEST_OUTPUT  = os.path.join(HASIL_DIR, "hasil_testing.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HASIL_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
print("📥 Memuat dataset...")
df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)

# =========================================================
# VALIDASI KOLOM
# =========================================================
required_cols = [
    "Contrast", "Correlation", "Energy", "Homogeneity",
    "Edge_Pixels", "Edge_Ratio",
    "Label"
]

for col in required_cols:
    if col not in df_train.columns:
        raise ValueError(f"❌ Kolom '{col}' tidak ditemukan!")

# =========================================================
# FITUR HYBRID
# =========================================================
features = [
    "Contrast", "Correlation", "Energy", "Homogeneity",
    "Edge_Pixels", "Edge_Ratio"
]

X_train = df_train[features]
y_train = df_train["Label"]

X_val = df_val[features]
y_val = df_val["Label"]

X_test = df_test[features]
y_test = df_test["Label"]

print("\n📊 Jumlah Data:")
print("Train:", len(X_train))
print("Val  :", len(X_val))
print("Test :", len(X_test))

# =========================================================
# NORMALISASI
# =========================================================
print("\n⚙️ Normalisasi (StandardScaler)...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# =========================================================
# TRAINING SVM
# =========================================================
print("\n🚀 Training SVM (HYBRID)...")

svm_model = SVC(
    kernel='rbf',
    C=500,
    gamma=0.1,
    probability=True
)

svm_model.fit(X_train_scaled, y_train)

# =========================================================
# PREDIKSI
# =========================================================
train_pred = svm_model.predict(X_train_scaled)
val_pred   = svm_model.predict(X_val_scaled)
test_pred  = svm_model.predict(X_test_scaled)

# =========================================================
# AKURASI
# =========================================================
train_acc = accuracy_score(y_train, train_pred)
val_acc   = accuracy_score(y_val, val_pred)
test_acc  = accuracy_score(y_test, test_pred)

print("\n🎯 HASIL AKURASI")
print(f"✅ Training : {train_acc:.4f}")
print(f"✅ Validasi : {val_acc:.4f}")
print(f"✅ Testing  : {test_acc:.4f}")

# =========================================================
# CONFUSION MATRIX
# =========================================================
print("\n📊 Confusion Matrix (TEST):")
print(confusion_matrix(y_test, test_pred))

print("\n📄 Classification Report (TEST):")
print(classification_report(y_test, test_pred))

# =========================================================
# SIMPAN HASIL TRAINING
# =========================================================
df_train_result = df_train.copy()
df_train_result["Prediksi"] = train_pred
df_train_result.to_csv(TRAIN_OUTPUT, index=False)

# =========================================================
# SIMPAN HASIL VALIDASI
# =========================================================
df_val_result = df_val.copy()
df_val_result["Prediksi"] = val_pred
df_val_result.to_csv(VAL_OUTPUT, index=False)

# =========================================================
# SIMPAN HASIL TESTING
# =========================================================
df_test_result = df_test.copy()
df_test_result["Prediksi"] = test_pred
df_test_result.to_csv(TEST_OUTPUT, index=False)

print("\n📄 File hasil disimpan:")
print(TRAIN_OUTPUT)
print(VAL_OUTPUT)
print(TEST_OUTPUT)

# =========================================================
# SIMPAN MODEL & SCALER
# =========================================================
joblib.dump(svm_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n💾 Model disimpan:", MODEL_PATH)
print("💾 Scaler disimpan:", SCALER_PATH)

print("\n🎉 SELESAI - HYBRID MODEL")