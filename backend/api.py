from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops

import base64

def encode_image(image_np):
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")

# ===============================
# INISIALISASI FLASK
# ===============================
app = Flask(__name__)

# ===============================
# PATH MODEL & SCALER
# ===============================
MODEL_PATH = "model_svm.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

kelas_map = {
    "rendah": "Rendah",
    "sedang": "Sedang",
    "tinggi": "Tinggi"
}

# ===============================
# FUNGSI EKSTRAKSI GLCM
# ===============================
def extract_glcm(img_gray):
    glcm = graycomatrix(
        img_gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    return [
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0]
    ]


# ===============================
# ENDPOINT PREDIKSI
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        # ===============================
        # PREPROCESSING (SAMA PERSIS)
        # ===============================
        img = np.array(image)
        img = cv2.resize(img, (256, 256))

        # Remove background
        img_no_bg = remove(img)

        if img_no_bg.shape[2] == 4:
            img_no_bg = cv2.cvtColor(img_no_bg, cv2.COLOR_RGBA2RGB)

        img_no_bg = cv2.resize(img_no_bg, (225, 225))

        # Grayscale
        gray = cv2.cvtColor(img_no_bg, cv2.COLOR_RGB2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        # Canny
        edges = cv2.Canny(gray_clahe, 50, 150)

        # ===============================
        # EKSTRAKSI FITUR
        # ===============================
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_ratio = edge_pixels / total_pixels

        glcm_features = extract_glcm(gray_clahe)

        fitur = [
            glcm_features[0],
            glcm_features[1],
            glcm_features[2],
            glcm_features[3],
            edge_pixels,
            edge_ratio
        ]

        fitur_scaled = scaler.transform([fitur])

        # ===============================
        # PREDIKSI SVM
        # ===============================
        pred = model.predict(fitur_scaled)[0]
        prob = model.predict_proba(fitur_scaled)[0]
        confidence = float(np.max(prob) * 100)

        # ===============================
        # RESPONSE JSON
        # ===============================
        return jsonify({
            "status": "success",
            "kelas": kelas_map[pred],
            "confidence": round(confidence, 2),
            "fitur": {
                "contrast": round(fitur[0], 4),
                "correlation": round(fitur[1], 4),
                "energy": round(fitur[2], 4),
                "homogeneity": round(fitur[3], 4),
                "edge_pixel": int(fitur[4]),
                "edge_ratio": round(fitur[5], 6)
            },
            "gambar": {
                "remove_bg": encode_image(img_no_bg),
                "grayscale": encode_image(gray),
                "clahe": encode_image(gray_clahe),
                "canny": encode_image(edges)
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)