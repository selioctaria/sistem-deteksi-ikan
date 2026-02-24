# =========================================================
# B3 - TESTING AKHIR SISTEM SVM
# =========================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================================
# PATH FILE
# =========================================================
TEST_CSV = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_split\fitur_test.csv"
MODEL_PATH = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\model\model_svm.pkl"

OUTPUT_CSV = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\hasil\hasil_testing.csv"
CONF_MATRIX_IMG = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\confusion_matrix.png"

# =========================================================
# LOAD DATA TEST
# =========================================================
print("📥 Memuat data testing...")
df_test = pd.read_csv(TEST_CSV)

X_test = df_test.drop(columns=["Nama_Citra", "Label"])
y_test = df_test["Label"]

# =========================================================
# LOAD MODEL & SCALER
# =========================================================

print("📦 Memuat model & scaler...")
svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(
    r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\model\scaler.pkl"
)


# =========================================================
# NORMALISASI DATA TEST
# =========================================================
X_test_scaled = scaler.transform(X_test)

# =========================================================
# PREDIKSI
# =========================================================
y_pred = svm_model.predict(X_test_scaled)

# =========================================================
# EVALUASI
# =========================================================
test_acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=["tinggi", "sedang", "rendah"])
report = classification_report(y_test, y_pred)

# =========================================================
# SIMPAN CSV HASIL TESTING
# =========================================================
df_result = df_test.copy()
df_result["Prediksi"] = y_pred
df_result.to_csv(OUTPUT_CSV, index=False)

# =========================================================
# VISUALISASI CONFUSION MATRIX
# =========================================================
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["tinggi", "sedang", "rendah"],
    yticklabels=["tinggi", "sedang", "rendah"]
)
plt.xlabel("Prediksi")
plt.ylabel("Label Aktual")
plt.title("Confusion Matrix Klasifikasi Kadar Garam Ikan Benggol")
plt.tight_layout()
plt.savefig(CONF_MATRIX_IMG)
plt.show()

# ===============================
# DATAFRAME DETAIL HASIL TESTING
# ===============================
df_testing = pd.DataFrame({
    "Label_Asli": y_test,
    "Label_Prediksi": y_pred
})

# Status benar / salah
df_testing["Status"] = np.where(
    df_testing["Label_Asli"] == df_testing["Label_Prediksi"],
    "Benar",
    "Salah"
)

# Simpan CSV detail
csv_detail_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\hasil\hasil_testing_detail.csv"
df_testing.to_csv(csv_detail_path, index=False)

print("📄 CSV detail testing disimpan di:", csv_detail_path)

# ===============================
# REKAP BENAR & SALAH PER KELAS
# ===============================
rekap = (
    df_testing
    .groupby(["Label_Asli", "Status"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Hitung total & akurasi per kelas
rekap["Total"] = rekap["Benar"] + rekap["Salah"]
rekap["Akurasi_Kelas (%)"] = (rekap["Benar"] / rekap["Total"]) * 100

# Simpan CSV rekap
csv_rekap_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\hasil\rekap_testing_per_kelas.csv"
rekap.to_csv(csv_rekap_path, index=False)

print("📊 CSV rekap per kelas disimpan di:", csv_rekap_path)


# ===============================
# DATAFRAME DASAR
# ===============================
df_eval = pd.DataFrame({
    "Label_Asli": y_test,
    "Label_Prediksi": y_pred
})

kelas_list = sorted(df_eval["Label_Asli"].unique())

precision_data = []

# ===============================
# HITUNG TP, FP, PRECISION
# ===============================
for kelas in kelas_list:
    TP = np.sum(
        (df_eval["Label_Asli"] == kelas) &
        (df_eval["Label_Prediksi"] == kelas)
    )

    FP = np.sum(
        (df_eval["Label_Asli"] != kelas) &
        (df_eval["Label_Prediksi"] == kelas)
    )

    precision = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0

    precision_data.append([
        kelas,
        TP,
        FP,
        round(precision, 2)
    ])

# ===============================
# DATAFRAME PRECISION
# ===============================
df_precision = pd.DataFrame(
    precision_data,
    columns=["Kelas", "TP", "FP", "Precision (%)"]
)

# ===============================
# SIMPAN CSV
# ===============================
csv_precision_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\svm\hasil\precision_per_kelas.csv"
df_precision.to_csv(csv_precision_path, index=False)

print("\n📊 HASIL PRECISION PER KELAS:")
print(df_precision)
print("\n💾 CSV precision disimpan di:", csv_precision_path)

# ===============================
# TAMPILKAN KE TERMINAL
# ===============================
print("\n📌 RINGKASAN HASIL TESTING PER KELAS:")
print(rekap)

# =========================================================
# OUTPUT TERMINAL
# =========================================================
print("\n🎯 HASIL PENGUJIAN AKHIR (TESTING)")
print(f"✅ Akurasi Testing : {test_acc:.4f}")
print("\n📊 Classification Report:")
print(report)

print("\n💾 File output:")
print("CSV :", OUTPUT_CSV)
print("Gambar CM :", CONF_MATRIX_IMG)
