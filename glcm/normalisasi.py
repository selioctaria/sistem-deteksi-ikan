import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =========================
# PATH FILE
# =========================
input_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_hybrid_final.csv"
output_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_hybrid_normalized.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(input_csv)

print("Jumlah data:", len(df))
print("Kolom data:", df.columns.tolist())

# =========================
# PISAH FITUR & LABEL
# =========================
X = df.drop(columns=["label"])
y = df["label"]

# =========================
# NORMALISASI MIN-MAX
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# =========================
# GABUNG KEMBALI
# =========================
df_normalized = pd.DataFrame(X_scaled, columns=X.columns)
df_normalized["label"] = y.values

# =========================
# SIMPAN CSV
# =========================
df_normalized.to_csv(output_csv, index=False)

print("\n✅ Normalisasi selesai")
print("📁 File output:", output_csv)
print("\nContoh 5 data pertama:")
print(df_normalized.head())
