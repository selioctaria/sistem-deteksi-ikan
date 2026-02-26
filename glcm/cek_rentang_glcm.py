import pandas as pd

input_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_hybrid_normalized.csv"

df = pd.read_csv(input_csv)

print("=== DAFTAR KOLOM DALAM FILE ===")
for col in df.columns:
    print(f"'{col}'")
print("="*50)

fitur_list = ["Contrast", "Homogeneity", "Energy", "Correlation"]

for fitur in fitur_list:
    
    kolom_asli = None
    for col in df.columns:
        if col.strip().lower() == fitur.lower():
            kolom_asli = col
            break

    if kolom_asli is None:
        print(f"❌ Kolom {fitur} TIDAK ditemukan dalam dataset!\n")
        continue

    print(f"📊 Mengecek: {kolom_asli}")

    if df[kolom_asli].isna().all():
        print("   ⚠ Semua nilai adalah NaN (kosong)\n")
        continue

    nilai_min = df[kolom_asli].min()
    nilai_max = df[kolom_asli].max()
    nilai_mean = df[kolom_asli].mean()

    print(f"   Min  : {nilai_min}")
    print(f"   Max  : {nilai_max}")
    print(f"   Mean : {nilai_mean}")
    print("-"*50)