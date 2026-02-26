import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

# =========================
# PATH CSV RATA-RATA
# =========================
csv_avg_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\canny\hasil_piksel_tepi_rata_rata.csv"

# =========================
# FOLDER OUTPUT UNTUK GRAFIK
# =========================
output_folder = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\canny"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "grafik_rata_rata_piksel_3D.png")

# =========================
# BACA DATA RATA-RATA
# =========================
kelas = []
avg_edge_pixels = []

with open(csv_avg_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row['Nama File'] == 'RATA-RATA':
            kelas.append(row['Kelas'])
            avg_edge_pixels.append(int(row['Jumlah Piksel Tepi']))

# =========================
# PERSIAPAN 3D BAR
# =========================
x_pos = np.arange(len(kelas))
y_pos = np.zeros(len(kelas))  # hanya satu lapisan di sumbu Y
z_pos = np.zeros(len(kelas))

dx = 0.6  # lebar batang di X
dy = 0.6  # lebar batang di Y
dz = avg_edge_pixels  # tinggi batang

# Warna gradien berdasarkan tinggi
min_val = min(dz)
max_val = max(dz)
if max_val - min_val == 0:
    norm = [0.5 for _ in dz]
else:
    norm = [(val - min_val) / (max_val - min_val) for val in dz]
colors = plt.cm.viridis(norm)  # palet modern viridis

# =========================
# BUAT FIGURE 3D
# =========================
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

# Tambahkan label angka di atas batang
for i in range(len(dz)):
    ax.text(x_pos[i] + dx/2, y_pos[i] + dy/2, dz[i] + max(dz)*0.02,
            f"{dz[i]}", ha='center', va='bottom', color='black', fontsize=10)

# =========================
# SET LABEL DAN TITLE
# =========================
ax.set_xticks(x_pos + dx/2)
ax.set_xticklabels(kelas, fontsize=12)
ax.set_zlabel('Rata-rata Jumlah Piksel Tepi', fontsize=12)
ax.set_title('Perbandingan Rata-rata Piksel Tepi per Kadar Garam', fontsize=14)

# Putar view supaya lebih menarik
ax.view_init(elev=30, azim=-60)

# =========================
# SIMPAN DAN TAMPILKAN
# =========================
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"✅ Grafik 3D berhasil disimpan di: {output_path}")
plt.show()