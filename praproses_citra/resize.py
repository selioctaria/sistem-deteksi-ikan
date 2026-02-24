import cv2
import os

input_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\remove_bg\tinggi\tinggi_39.png"
output_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\resize\tinggi\tinggi_resize_39.png"

img = cv2.imread(input_path)

if img is None:
    print("❌ Gambar gagal dibaca")
else:
    print("✅ Gambar terbaca")
    print("Ukuran awal:", img.shape)

    resized_img = cv2.resize(
        img,
        (225, 225),
        interpolation=cv2.INTER_AREA
    )

    print("Ukuran setelah resize:", resized_img.shape)

    cv2.imwrite(output_path, resized_img)
    print("✅ Resize selesai dan gambar disimpan")