import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# =========================
# KONFIGURASI PATH
# =========================
DATASET_ROOT = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset_split"
OUTPUT_DIR = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = ["train", "val", "test"]
classes = ["tinggi", "sedang", "rendah"]

# =========================
# FUNGSI EKSTRAKSI FITUR
# =========================
def extract_features(img):
    # ---- GLCM ----
    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # ---- CANNY ----
    blur = cv2.GaussianBlur(img, (5, 5), 1.4)
    edges = cv2.Canny(blur, 50, 150)

    edge_pixels = np.sum(edges > 0)
    edge_ratio = edge_pixels / edges.size

    return [
        contrast,
        correlation,
        energy,
        homogeneity,
        edge_pixels,
        edge_ratio
    ]

# =========================
# PROSES PER SPLIT
# =========================
for split in splits:
    data_rows = []

    print(f"\n📂 Memproses data {split.upper()}")

    for label in classes:
        class_dir = os.path.join(DATASET_ROOT, split, label)

        for file in os.listdir(class_dir):
            if file.lower().endswith(".png"):
                img_path = os.path.join(class_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                features = extract_features(img)

                data_rows.append([
                    file,
                    label
                ] + features)

    # Buat DataFrame
    df = pd.DataFrame(data_rows, columns=[
        "Nama_Citra",
        "Label",
        "Contrast",
        "Correlation",
        "Energy",
        "Homogeneity",
        "Edge_Pixels",
        "Edge_Ratio"
    ])

    # Simpan CSV
    output_csv = os.path.join(OUTPUT_DIR, f"fitur_{split}.csv")
    df.to_csv(output_csv, index=False)

    print(f"✅ CSV {split} berhasil dibuat: {output_csv}")
    print(f"   Total data: {len(df)}")

print("\n🎉 LANGKAH B1 SELESAI")
