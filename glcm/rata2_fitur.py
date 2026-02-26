import pandas as pd

# =========================
# PATH FILE
# =========================
input_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_hybrid_normalized.csv"
output_avg_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_hybrid_rata_rata_per_label.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(input_csv)

# =========================
# HITUNG RATA-RATA PER LABEL
# =========================
# group by label
df_avg_per_label = df.groupby("label").mean().reset_index()

# =========================
# SIMPAN KE CSV
# =========================
df_avg_per_label.to_csv(output_avg_csv, index=False)

print(f"✅ File rata-rata per label berhasil dibuat: {output_avg_csv}")
print("\nContoh isi file rata-rata per label:")
print(df_avg_per_label)