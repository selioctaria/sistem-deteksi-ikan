import pandas as pd
from sklearn.model_selection import train_test_split
import os

# =========================
# PATH FILE
# =========================
input_csv = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_hybrid_normalized.csv"

output_dir = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\fitur\fitur_split"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(input_csv)

print("Total data:", len(df))

# =========================
# SPLIT TEST (165 DATA)
# =========================
train_val_df, test_df = train_test_split(
    df,
    test_size=165,
    stratify=df["label"],
    random_state=42
)

# =========================
# SPLIT VALIDASI (162 DATA)
# =========================
train_df, val_df = train_test_split(
    train_val_df,
    test_size=162,
    stratify=train_val_df["label"],
    random_state=42
)

# =========================
# SIMPAN CSV
# =========================
train_path = os.path.join(output_dir, "train.csv")
val_path = os.path.join(output_dir, "val.csv")
test_path = os.path.join(output_dir, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# =========================
# INFO HASIL SPLIT
# =========================
print("\nJumlah dataset setelah split:")
print("Train :", len(train_df))
print("Val   :", len(val_df))
print("Test  :", len(test_df))

print("\nDistribusi kelas TRAIN:")
print(train_df["label"].value_counts())

print("\nDistribusi kelas VALIDASI:")
print(val_df["label"].value_counts())

print("\nDistribusi kelas TEST:")
print(test_df["label"].value_counts())

print("\n✅ Dataset berhasil dibagi sesuai jumlah yang diinginkan")