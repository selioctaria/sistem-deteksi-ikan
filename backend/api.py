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
# ENCODE IMAGE TO BASE64
# ===============================
def encode_image(image_np):
    if len(image_np.shape) == 2:
        image_to_encode = image_np
    else:
        image_to_encode = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", image_to_encode)
    return base64.b64encode(buffer).decode("utf-8")

# ===============================
# RESIZE WITH PADDING
# ===============================
def resize_with_padding(img, target_size=(256,256)):
    h, w = img.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    start_x = (target_size[0]-new_w)//2
    start_y = (target_size[1]-new_h)//2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    return canvas

# ===============================
# REMOVE BACKGROUND BLACK
# ===============================
def remove_background_black(image_rgb):
    """
    Remove background lebih stabil menggunakan GrabCut.
    Objek utama di tengah dipertahankan, background dibuat hitam.
    Cocok untuk foto dari kamera HP.
    """

    img = image_rgb.copy()
    h, w = img.shape[:2]

    # Ubah RGB ke BGR untuk OpenCV GrabCut
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Mask awal
    mask = np.zeros((h, w), np.uint8)

    # Area kotak objek utama
    # Objek diasumsikan berada di tengah gambar
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)
    rect = (
        margin_x,
        margin_y,
        w - 2 * margin_x,
        h - 2 * margin_y
    )

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Proses GrabCut
    cv2.grabCut(
        img_bgr,
        mask,
        rect,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Ambil area foreground
    mask_foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1,
        0
    ).astype("uint8")

    # Perbaiki mask agar lebih bersih
    kernel = np.ones((5, 5), np.uint8)
    mask_foreground = cv2.morphologyEx(mask_foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_foreground = cv2.morphologyEx(mask_foreground, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Ambil kontur terbesar agar noise background hilang
    contours, _ = cv2.findContours(
        mask_foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(mask_foreground)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
    else:
        clean_mask = mask_foreground

    # Haluskan pinggir objek
    clean_mask = cv2.GaussianBlur(clean_mask.astype(np.float32), (5, 5), 0)
    clean_mask = (clean_mask > 0.3).astype(np.uint8)

    # Background hitam
    result = np.zeros_like(img)
    result[clean_mask == 1] = img[clean_mask == 1]

    return result

# ===============================
# GLCM FEATURE
# ===============================
def extract_glcm(img_gray):
    img_gray = img_gray.astype(np.uint8)
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0,0]
    correlation = graycoprops(glcm, "correlation")[0,0]
    energy = graycoprops(glcm, "energy")[0,0]
    homogeneity = graycoprops(glcm, "homogeneity")[0,0]
    return [contrast, correlation, energy, homogeneity]

# ===============================
# HOME ENDPOINT
# ===============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status":"success","message":"SaltFish Detector API running"})

# ===============================
# PREDICT ENDPOINT
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"status":"error","message":"File image tidak ditemukan"}),400
        file = request.files["image"]
        if file.filename=="":
            return jsonify({"status":"error","message":"Nama file kosong"}),400

        # Read image
        image = Image.open(file).convert("RGB")
        img_original = np.array(image)

        # Remove BG
        img_remove_bg = remove_background_black(img_original)

        # Resize 256x256
        img_resized = resize_with_padding(img_remove_bg, (256,256))

        # Grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0,tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)

        # Canny
        edges = cv2.Canny(gray_clahe,50,150)

        # Fitur
        edge_pixels = np.sum(edges>0)
        edge_ratio = edge_pixels / edges.size
        glcm_features = extract_glcm(gray_clahe)
        contrast, correlation, energy, homogeneity = glcm_features
        fitur = [contrast, correlation, energy, homogeneity, edge_pixels, edge_ratio]

        # SVM
        fitur_scaled = scaler.transform([fitur])
        pred = model.predict(fitur_scaled)[0]

        # ===============================
        # RESPONSE JSON (tanpa confidence)
        # ===============================
        return jsonify({
            "status":"success",
            "kelas":kelas_map.get(pred,str(pred)),
            "fitur":{
                "contrast":round(float(contrast),4),
                "correlation":round(float(correlation),4),
                "energy":round(float(energy),4),
                "homogeneity":round(float(homogeneity),4),
                "edge_pixel":int(edge_pixels),
                "edge_ratio":round(float(edge_ratio),6)
            },
            "gambar":{
                "original":encode_image(img_original),
                "remove_bg":encode_image(img_remove_bg),
                "resized":encode_image(img_resized),
                "grayscale":encode_image(gray),
                "clahe":encode_image(gray_clahe),
                "canny":encode_image(edges)
            },
            "proses":[
                "Original Image",
                "Remove Background Black",
                "Resize with Padding",
                "Grayscale",
                "CLAHE",
                "Canny Edge Detection",
                "GLCM Feature Extraction",
                "SVM Classification"
            ]
        })

    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500

# ===============================
# RUN SERVER
# ===============================
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)