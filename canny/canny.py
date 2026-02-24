import cv2
import os
import numpy as np
import csv

# =========================
# KONFIGURASI PARAMETER
# =========================
input_root = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\augmented"
output_root = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\canny\citra_canny"
csv_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\canny\hasil_piksel_tepi.csv"

# Gaussian Blur
gaussian_kernel = (5, 5)
gaussian_sigma = 1.4

# Canny Threshold
low_threshold = 50
high_threshold = 150

os.makedirs(output_root, exist_ok=True)

# =========================
# SIAPKAN FILE CSV
# =========================
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Kelas",
        "Nama File",
        "Jumlah Piksel Tepi",
        "Total Piksel",
        "Rasio Piksel Tepi"
    ])

# =========================
# PROSES DATASET
# =========================
for class_name in os.listdir(input_root):
    class_input_path = os.path.join(input_root, class_name)

    if not os.path.isdir(class_input_path):
        continue

    class_output_path = os.path.join(output_root, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    print(f"\n📁 Memproses kelas: {class_name}")

    for file in os.listdir(class_input_path):
        if file.lower().endswith(".png"):
            img_path = os.path.join(class_input_path, file)

            # Baca citra grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ Gagal membaca {file}")
                continue

            # 1. Gaussian Smoothing
            blurred = cv2.GaussianBlur(
                img,
                gaussian_kernel,
                gaussian_sigma
            )

            # 2. Canny Edge Detection
            edges = cv2.Canny(
                blurred,
                low_threshold,
                high_threshold
            )

            # 3. Hitung jumlah piksel tepi
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            edge_ratio = edge_pixels / total_pixels

            # 4. Simpan hasil citra
            output_path = os.path.join(class_output_path, file)
            cv2.imwrite(output_path, edges)

            # 5. Simpan ke CSV
            with open(csv_path, mode='a', newline='') as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow([
                    class_name,
                    file,
                    edge_pixels,
                    total_pixels,
                    round(edge_ratio, 6)
                ])

            print(f"✅ {class_name}/{file} | Edge: {edge_pixels} | Rasio: {edge_ratio:.4f}")
