import pandas as pd

# =========================
# PATH FILE
# =========================
input_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_hybrid_normalized.csv"
output_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_statistik_semua_kelas.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(input_csv)

print("Jumlah data:", len(df))
print("Kolom:", df.columns.tolist())

# =========================
# HAPUS KOLOM LABEL
# =========================
df_features = df.drop(columns=["label"])

# =========================
# HITUNG MIN, MAX, MEAN
# =========================
stats = pd.DataFrame({
    "Fitur": df_features.columns,
    "Min": df_features.min().values,
    "Max": df_features.max().values,
    "Mean": df_features.mean().values
})

# =========================
# SIMPAN CSV
# =========================
stats.to_csv(output_csv, index=False)

print("\n✅ Statistik fitur berhasil dibuat")
print("📁 File:", output_csv)

print("\nIsi statistik fitur:")
print(stats)