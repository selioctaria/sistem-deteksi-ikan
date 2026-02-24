import cv2
import os

# Folder input (split asli) dan output (hasil augmentasi)
input_root = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset_split"
output_root = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\dataset_split_augmented"

# Sudut rotasi
angles = {
    "r0": None,
    "r90": cv2.ROTATE_90_CLOCKWISE,
    "r270": cv2.ROTATE_90_COUNTERCLOCKWISE
}

splits = ["train", "val", "test"]  # augmentasi per split

for split in splits:
    split_input_path = os.path.join(input_root, split)
    split_output_path = os.path.join(output_root, split)
    os.makedirs(split_output_path, exist_ok=True)

    for class_name in os.listdir(split_input_path):
        class_input_path = os.path.join(split_input_path, class_name)
        if not os.path.isdir(class_input_path):
            continue

        class_output_path = os.path.join(split_output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for file in os.listdir(class_input_path):
            if file.lower().endswith(".png"):
                img_path = os.path.join(class_input_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"❌ Gagal membaca {file}")
                    continue

                filename = os.path.splitext(file)[0]

                # Variasi mirroring
                variants = {
                    "orig": img,
                    "flipH": cv2.flip(img, 1),  # Horizontal
                    "flipV": cv2.flip(img, 0)   # Vertikal
                }

                # Rotasi
                for var_name, var_img in variants.items():
                    for angle_name, angle_code in angles.items():
                        if angle_code is None:
                            rotated = var_img
                        else:
                            rotated = cv2.rotate(var_img, angle_code)

                        output_name = f"{filename}_{var_name}_{angle_name}.png"
                        output_path = os.path.join(class_output_path, output_name)
                        cv2.imwrite(output_path, rotated)

                print(f"✅ {split}/{class_name}/{file} selesai diaugmentasi")

print("\n✅ Semua citra di split train/val/test berhasil diaugmentasi")
