import cv2

# Path input dan output
input_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\resize\tinggi\tinggi_resize_39.png"
output_path = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\grayscale\tinggi\tinggi_grayscale_39.png"

# Baca gambar
img = cv2.imread(input_path)

if img is None:
    print("❌ Gambar tidak terbaca")
else:
    print("✅ Gambar terbaca")
    print("Ukuran sebelum grayscale:", img.shape)

    # Konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Ukuran setelah grayscale:", gray.shape)

    # Simpan hasil
    cv2.imwrite(output_path, gray)
    print("✅ Konversi grayscale selesai")