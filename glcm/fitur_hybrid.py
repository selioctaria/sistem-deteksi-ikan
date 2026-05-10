import pandas as pd

# =========================
# PATH FILE
# =========================
glcm_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_glcm.csv"
canny_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\canny\hasil_piksel_tepi.csv"
output_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_hybrid_final.csv"

# =========================
# LOAD DATA
# =========================
df_glcm = pd.read_csv(glcm_csv)
df_canny = pd.read_csv(canny_csv)

print("Jumlah data GLCM :", len(df_glcm))
print("Jumlah data Canny:", len(df_canny))

# =========================
# CEK NAMA KOLOM AWAL
# =========================
print("\nKolom awal GLCM:")
print(df_glcm.columns)

print("\nKolom awal Canny:")
print(df_canny.columns)

# =========================
# BERSIHKAN NAMA KOLOM
# =========================
df_glcm.columns = df_glcm.columns.str.strip()
df_canny.columns = df_canny.columns.str.strip()

# =========================
# STANDARISASI NAMA KOLOM
# =========================
df_glcm.rename(columns={
    "Nama_Citra": "filename",
    "Kelas": "label"
}, inplace=True)

df_canny.rename(columns={
    "Nama_File": "filename",
    "Nama File": "filename",
    "Kelas": "label",
    "Edge_Pixels": "edge_pixels",
    "Object_Pixels": "total_pixels",
    "Edge_Ratio": "edge_ratio"
}, inplace=True)

# =========================
# CEK KOLOM SETELAH RENAME
# =========================
print("\nKolom setelah rename GLCM:")
print(df_glcm.columns)

print("\nKolom setelah rename Canny:")
print(df_canny.columns)

# =========================
# VALIDASI DATA SEBELUM MERGE
# =========================
missing_glcm = set(df_canny["filename"]) - set(df_glcm["filename"])
missing_canny = set(df_glcm["filename"]) - set(df_canny["filename"])

print("\nFile di Canny tapi tidak ada di GLCM:", len(missing_glcm))
print("File di GLCM tapi tidak ada di Canny:", len(missing_canny))

if len(missing_glcm) > 0:
    print("Contoh missing (Canny → GLCM):", list(missing_glcm)[:5])

if len(missing_canny) > 0:
    print("Contoh missing (GLCM → Canny):", list(missing_canny)[:5])

# =========================
# MERGE DATA
# =========================
df_merge = pd.merge(
    df_glcm,
    df_canny[["filename", "edge_pixels", "edge_ratio"]],
    on="filename",
    how="inner"
)

print("\nJumlah data setelah merge:", len(df_merge))

# =========================
# SELEKSI FITUR FINAL
# =========================
df_final = df_merge[[
    "Contrast",
    "Correlation",
    "Energy",
    "Homogeneity",
    "edge_pixels",
    "edge_ratio",
    "label"
]]

# =========================
# SIMPAN CSV FINAL
# =========================
df_final.to_csv(output_csv, index=False)

print("\n✅ Penggabungan fitur BERHASIL")
print("📁 File output:", output_csv)

print("\nContoh 5 baris data:")
print(df_final.head())