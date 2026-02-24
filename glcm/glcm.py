import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Path dataset
DATASET_DIR = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\augmented"
OUTPUT_CSV_ALL = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_glcm.csv"

# Kelas kadar garam
classes = ["tinggi", "sedang", "rendah"]

# List penampung data
data_all = []

for label in classes:
    folder_path = os.path.join(DATASET_DIR, label)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            
            # Baca citra grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Pastikan tipe uint8
            img = img.astype(np.uint8)
            
            # Hitung GLCM
            glcm = graycomatrix(
                img,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # Ekstraksi fitur
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            
            # Simpan ke list
            data_all.append([
                filename,
                label,
                contrast,
                correlation,
                energy,
                homogeneity
            ])

# Buat DataFrame
df_all = pd.DataFrame(data_all, columns=[
    "Nama_Citra",
    "Kelas",
    "Contrast",
    "Correlation",
    "Energy",
    "Homogeneity"
])

# Simpan CSV
df_all.to_csv(OUTPUT_CSV_ALL, index=False)

print("CSV seluruh citra berhasil dibuat:", OUTPUT_CSV_ALL)
