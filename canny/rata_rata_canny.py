import pandas as pd

# =========================
# PATH FILE
# =========================
input_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\canny\hasil_piksel_tepi.csv"
output_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\canny\rata_rata_perkelas.csv"

# =========================
# BACA DATA CSV
# =========================
data = pd.read_csv(input_csv)

# =========================
# HITUNG RATA-RATA PER KELAS
# =========================
result = data.groupby("Kelas").agg({
    "Edge_Pixels": ["mean", "min", "max"],
    "Edge_Ratio": ["mean", "min", "max"]
})

# rapikan nama kolom
result.columns = [
    "Mean Edge Pixel",
    "Min Edge Pixel",
    "Max Edge Pixel",
    "Mean Edge Ratio",
    "Min Edge Ratio",
    "Max Edge Ratio"
]

# reset index supaya kolom kelas muncul
result = result.reset_index()

# =========================
# SIMPAN KE CSV BARU
# =========================
result.to_csv(output_csv, index=False)

print("✅ File rata-rata per kelas berhasil dibuat")
print(result)