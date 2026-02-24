import cv2

# Path input dan output
input_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\grayscale\tinggi\tinggi_grayscale_39.png"
output_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\clahe\tinggi\tinggi_clahe_39.png"

# Baca citra grayscale
gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if gray is None:
    print("❌ Gambar grayscale tidak terbaca")
else:
    print("✅ Gambar grayscale terbaca")
    print("Ukuran:", gray.shape)

    # Inisialisasi CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=5.0,
        tileGridSize=(8, 8)
    )

    # Terapkan CLAHE
    clahe_img = clahe.apply(gray)

    # Simpan hasil
    cv2.imwrite(output_path, clahe_img)
    print("✅ CLAHE berhasil diterapkan")