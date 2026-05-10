import cv2
import os
import numpy as np
import csv 

# =========================
# KONFIGURASI PATH
# =========================
input_root = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\augmented"
output_root = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\canny\citra_canny"
csv_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\canny\hasil_piksel_tepii.csv"

os.makedirs(output_root, exist_ok=True)

# =========================
# PARAMETER CANNY
# =========================
gaussian_kernel = (5, 5)
gaussian_sigma = 1.4
low_threshold = 50
high_threshold = 150

# =========================
# SIAPKAN CSV
# =========================
with open(csv_path, mode='w', newline='') as file_csv:

    writer = csv.writer(file_csv)
    writer.writerow([
        "Kelas",
        "Nama_File",
        "Edge_Pixels",
        "Object_Pixels",
        "Edge_Ratio"
    ])

    # =========================
    # PROSES SETIAP KELAS
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

                # =========================
                # BACA CITRA GRAYSCALE
                # =========================
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"❌ Gagal membaca {file}")
                    continue

                # =========================
                # GAUSSIAN SMOOTHING
                # =========================
                blurred = cv2.GaussianBlur(
                    img,
                    gaussian_kernel,
                    gaussian_sigma
                )

                # =========================
                # CANNY EDGE DETECTION
                # =========================
                edges = cv2.Canny(
                    blurred,
                    low_threshold,
                    high_threshold
                )

                # =========================
                # HITUNG EDGE PIXEL
                # =========================
                edge_pixels = np.sum(edges > 0)

                # HITUNG OBJECT PIXEl
                object_pixels = np.sum(img > 0)

                # EDGE RATIO
                if object_pixels == 0:
                    edge_ratio = 0
                else:
                    edge_ratio = edge_pixels / object_pixels

                # =========================
                # SIMPAN CITRA EDGE
                # =========================
                output_path = os.path.join(class_output_path, file)
                cv2.imwrite(output_path, edges)

                # =========================
                # SIMPAN KE CSV
                # =========================
                writer.writerow([
                    class_name,
                    file,
                    edge_pixels,
                    object_pixels,
                    round(edge_ratio, 6)
                ])

                print(
                    f"✅ {class_name}/{file} | Edge: {edge_pixels} | Ratio: {edge_ratio:.5f}"
                )

print("\n🎉 Proses Canny selesai dan data disimpan ke CSV.")
