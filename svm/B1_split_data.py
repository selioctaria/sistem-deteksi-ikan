import os
import shutil
import random

# =========================
# KONFIGURASI PATH
# =========================
SOURCE_DIR = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset\augmented"
TARGET_DIR = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset_split"

# Proporsi split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # supaya konsisten

classes = ["tinggi", "sedang", "rendah"]

# =========================
# PROSES SPLIT DATASET
# =========================
for cls in classes:
    src_cls_path = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(src_cls_path) if f.endswith(".png")]
    
    random.shuffle(images)

    total = len(images)
    train_end = int(TRAIN_RATIO * total)
    val_end = train_end + int(VAL_RATIO * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split, file_list in splits.items():
        target_cls_dir = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(target_cls_dir, exist_ok=True)

        for file in file_list:
            src_path = os.path.join(src_cls_path, file)
            dst_path = os.path.join(target_cls_dir, file)
            shutil.copy(src_path, dst_path)

    print(f"\nKelas: {cls}")
    print(f"  Train: {len(train_imgs)}")
    print(f"  Val  : {len(val_imgs)}")
    print(f"  Test : {len(test_imgs)}")

print("\n✅ Dataset berhasil dipisahkan ke folder train/val/test")
