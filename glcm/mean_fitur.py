import pandas as pd

# =========================
# PATH FILE CSV
# =========================
TRAIN_CSV = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_split\fitur_train.csv"
VAL_CSV   = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_split\fitur_val.csv"
TEST_CSV  = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_split\fitur_test.csv"

# =========================
# LOAD DATA
# =========================
print("📥 Membaca file CSV...")

df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)

# =========================
# GABUNG SEMUA DATA
# =========================
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

print("Total data gabungan:", len(df_all))

# =========================
# HITUNG MIN MAX MEAN
# =========================
result = df_all.groupby("Label")[["Edge_Pixels", "Edge_Ratio"]].agg(
    ["min", "max", "mean"]
)

# =========================
# TAMPILKAN HASIL
# =========================
print("\n📊 STATISTIK EDGE FEATURES")
print(result)

# =========================
# SIMPAN KE CSV
# =========================
output_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\statistik_edge.csv"
result.to_csv(output_path)

print("\n✅ Hasil disimpan di:")
print(output_path)