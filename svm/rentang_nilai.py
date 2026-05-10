import os
import pandas as pd

# =========================
# PATH CSV
# =========================
CSV_DIR = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_split"

train_csv = os.path.join(CSV_DIR, "fitur_train.csv")
val_csv = os.path.join(CSV_DIR, "fitur_val.csv")
test_csv = os.path.join(CSV_DIR, "fitur_test.csv")

# =========================
# BACA CSV
# =========================
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)

# =========================
# GABUNGKAN DATA
# =========================
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

# =========================
# PILIH KOLOM FITUR
# =========================
features = [
    "Contrast",
    "Correlation",
    "Energy",
    "Homogeneity",
    "Edge_Pixels",
    "Edge_Ratio"
]

# =========================
# HITUNG STATISTIK
# =========================
stats = pd.DataFrame({
    "Min": df_all[features].min(),
    "Max": df_all[features].max(),
    "Mean": df_all[features].mean()
})

print("\nSTATISTIK FITUR SELURUH DATASET\n")
print(stats)