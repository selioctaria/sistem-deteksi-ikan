from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import base64
import os

# ===============================
# INIT FLASK
# ===============================
app = Flask(__name__)

# ===============================
# PATH MODEL & SCALER
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "svm_hybrid.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_hybrid.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# LABEL MAP
# ===============================
kelas_map = {
    "rendah": "Rendah",
    "sedang": "Sedang",
    "tinggi": "Tinggi"
}

# ===============================
# ENCODE IMAGE
# ===============================
def encode_image(image_np):
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")

# ===============================
# GLCM FEATURE EXTRACTION
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
# PREDICTION ENDPOINT
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        # ===============================
        # PREPROCESSING
        # ===============================
        img = np.array(image)
        img = cv2.resize(img, (256, 256))

        img_no_bg = img

        # safety check (kalau ada alpha channel)
        if img_no_bg.shape[2] == 4:
            img_no_bg = cv2.cvtColor(img_no_bg, cv2.COLOR_RGBA2RGB)

        img_no_bg = cv2.resize(img_no_bg, (256, 256))

        gray = cv2.cvtColor(img_no_bg, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        edges = cv2.Canny(gray_clahe, 50, 150)

        # ===============================
        # FEATURE EXTRACTION
        # ===============================
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_ratio = edge_pixels / total_pixels

        glcm_features = extract_glcm(gray_clahe)

        print("Contrast     :", glcm_features[0])
        print("Correlation  :", glcm_features[1])
        print("Energy       :", glcm_features[2])
        print("Homogeneity  :", glcm_features[3])
        print("Edge Pixels  :", edge_pixels)
        print("Edge Ratio   :", edge_ratio)

        fitur = [
            glcm_features[0],
            glcm_features[1],
            glcm_features[2],
            glcm_features[3],
            edge_pixels,
            edge_ratio
        ]

        print("FITUR ARRAY:", fitur)
        if edge_ratio < 0.04:
            pred = "rendah"
        elif edge_ratio < 0.05:
            pred = "sedang"
        else:
            pred = "tinggi"

        # ===============================
        # RESPONSE JSON
        # ===============================
        return jsonify({
            "status": "success",
            "kelas": kelas_map[pred],

            "confidence": None,

            "fitur": {
                "contrast": round(glcm_features[0], 4),
                "correlation": round(glcm_features[1], 4),
                "energy": round(glcm_features[2], 4),
                "homogeneity": round(glcm_features[3], 4),
                "edge_pixel": int(edge_pixels),
                "edge_ratio": round(edge_ratio, 6)
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