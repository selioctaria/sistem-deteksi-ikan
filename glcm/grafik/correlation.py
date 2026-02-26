import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# PATH FILE CSV
# =========================
input_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_hybrid_rata_rata_per_label.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(input_csv)

print("Kolom dalam file:", df.columns.tolist())

# =========================
# CARI KOLOM Correlation (AMAN HURUF BESAR/KECIL)
# =========================
correlation_column = None
for col in df.columns:
    if col.lower() == "correlation":
        correlation_column = col
        break

if correlation_column is None:
    raise ValueError("Kolom correlation tidak ditemukan di CSV!")

# =========================
# DATA UNTUK GRAFIK
# =========================
kelas = df["label"].tolist()
correlation_values = df[correlation_column].tolist()

# =========================
# BUAT GRAFIK MODERN
# =========================
plt.figure(figsize=(9,6))

bars = plt.bar(kelas, correlation_values)

# Tambahkan nilai di atas batang
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + (max(correlation_values)*0.02),
        f"{yval:.4f}",
        ha='center',
        va='bottom',
        fontsize=11
    )

plt.title("Perbandingan Rata-rata Correlation per Kadar Garam Ikan", fontsize=14, fontweight='bold')
plt.xlabel("Kategori Kadar Garam", fontsize=12)
plt.ylabel("Rata-rata Correlation", fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# =========================
# SIMPAN FILE
# =========================
output_folder = os.path.dirname(input_csv)
output_path = os.path.join(output_folder, "grafik_correlation_per_label.png")

plt.savefig(output_path, dpi=300)
print(f"\n✅ Grafik berhasil disimpan di:\n{output_path}")

# =========================
# TAMPILKAN
# =========================
plt.show()