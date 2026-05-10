import os
import joblib
import numpy as np

# ===============================
# SET PATH OTOMATIS (ANTI ERROR)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "svm_hybrid.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_hybrid.pkl")

print("===== CEK PATH =====")
print("Folder aktif:", BASE_DIR)
print("Model path:", MODEL_PATH)
print("Scaler path:", SCALER_PATH)

# ===============================
# CEK FILE ADA ATAU TIDAK
# ===============================
print("\n===== CEK FILE =====")
print("Isi folder:")
print(os.listdir(BASE_DIR))

if not os.path.exists(MODEL_PATH):
    print("\n❌ ERROR: svm_hybrid.pkl TIDAK DITEMUKAN!")
    exit()

if not os.path.exists(SCALER_PATH):
    print("\n❌ ERROR: scaler.pkl TIDAK DITEMUKAN!")
    exit()

# ===============================
# LOAD MODEL & SCALER
# ===============================
print("\n===== LOAD MODEL =====")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ Model berhasil diload")
print("Tipe model:", type(model))

# ===============================
# CEK KELAS MODEL
# ===============================
print("\n===== INFO KELAS =====")
try:
    print("Kelas:", model.classes_)
except:
    print("Model tidak punya atribut classes_")

# ===============================
# CEK JUMLAH FITUR
# ===============================
print("\n===== INFO FITUR =====")
try:
    jumlah_fitur = scaler.n_features_in_
    print("Jumlah fitur (scaler):", jumlah_fitur)
except:
    print("Tidak bisa membaca jumlah fitur")

# ===============================
# CEK PARAMETER MODEL
# ===============================
print("\n===== PARAMETER MODEL =====")
try:
    print(model.get_params())
except:
    print("Tidak bisa membaca parameter model")

# ===============================
# TEST PREDIKSI DUMMY
# ===============================
print("\n===== TEST DUMMY =====")
try:
    dummy = np.zeros((1, jumlah_fitur))
    dummy_scaled = scaler.transform(dummy)

    pred = model.predict(dummy_scaled)
    prob = model.predict_proba(dummy_scaled)

    print("Prediksi dummy:", pred)
    print("Probabilitas:", prob)

except Exception as e:
    print("❌ Error test dummy:", str(e))

# ===============================
# TEST FITUR MANUAL (SIMULASI)
# ===============================
print("\n===== TEST FITUR MANUAL =====")
try:
    # GANTI sesuai jumlah fitur kamu
    fitur_manual = [0] * jumlah_fitur

    fitur_manual = np.array(fitur_manual).reshape(1, -1)
    fitur_scaled = scaler.transform(fitur_manual)

    pred = model.predict(fitur_scaled)
    prob = model.predict_proba(fitur_scaled)

    print("Prediksi manual:", pred)
    print("Prob manual:", prob)

except Exception as e:
    print("❌ Error fitur manual:", str(e))