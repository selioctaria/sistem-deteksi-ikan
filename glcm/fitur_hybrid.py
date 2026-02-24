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
# STANDARISASI NAMA KOLOM
# =========================
df_glcm.rename(columns={
    "Nama_Citra": "filename",
    "Kelas": "label"
}, inplace=True)

df_canny.rename(columns={
    "Nama File": "filename",
    "Kelas": "label",
    "Jumlah Piksel Tepi": "edge_pixels",
    "Total Piksel": "total_pixels",
    "Rasio Piksel Tepi": "edge_ratio"
}, inplace=True)

# =========================
# VALIDASI DATA SEBELUM MERGE
# =========================
missing_glcm = set(df_canny["filename"]) - set(df_glcm["filename"])
missing_canny = set(df_glcm["filename"]) - set(df_canny["filename"])

print(f"File di Canny tapi tidak ada di GLCM: {len(missing_glcm)}")
print(f"File di GLCM tapi tidak ada di Canny: {len(missing_canny)}")

if len(missing_glcm) > 0:
    print("⚠ Contoh missing (Canny → GLCM):", list(missing_glcm)[:5])

if len(missing_canny) > 0:
    print("⚠ Contoh missing (GLCM → Canny):", list(missing_canny)[:5])

# =========================
# MERGE DATA (INNER JOIN)
# =========================
df_merge = pd.merge(
    df_glcm,
    df_canny[["filename", "edge_pixels", "edge_ratio"]],
    on="filename",
    how="inner"
)

print("Jumlah data setelah merge:", len(df_merge))

# =========================
# SELEKSI KOLOM FINAL
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
