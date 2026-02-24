import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Deteksi Kadar Garam Ikan Benggol",
    layout="centered"
)

# ============================================================================
# PATH MODEL & SCALER
# ============================================================================
MODEL_PATH = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\streamlit_app\model\model_svm.pkl"
SCALER_PATH = r"D:\Documents\SKRIPSI\PROGRAM SISTEM DETEKSI IKAN\streamlit_app\model\scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

kelas_map = {
    "rendah": "Rendah",
    "sedang": "Sedang",
    "tinggi": "Tinggi"
}

# ============================================================================
# FUNGSI BANTUAN UI
# ============================================================================
def show_image(img, caption, width=140):
    st.image(img, caption=caption, width=width)

def arrow():
    st.markdown(
        "<div style='text-align:center; font-size:26px;'>➡️</div>",
        unsafe_allow_html=True
    )

# ============================================================================
# FUNGSI EKSTRAKSI GLCM
# ============================================================================
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

# ============================================================================
# HEADER UI
# ============================================================================
st.markdown(
    """
    <h1 style="
        font-size:48px;
        font-weight:800;
        text-align:center;
        margin-bottom:20px;
    ">
        🐟 Deteksi Kadar Garam Ikan Benggol
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
        text-align:center;
        font-size:16px;
        color:#aaa;
        margin-bottom:40px;
    ">
        Sistem ini mendeteksi <b>kadar garam ikan benggol</b> menggunakan alur:  
        <br>
        <b>Remove Background → CLAHE → Canny → GLCM → SVM</b>
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# UPLOAD GAMBAR
# ============================================================================
uploaded_file = st.file_uploader(
    "📤 Upload citra ikan (.jpg / .png)",
    type=["jpg", "png"]
)

# ============================================================================
# PROSES DETEKSI
# ============================================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Citra Asli", width=300)

    if st.button("🔍 Deteksi Kadar Garam"):
        with st.spinner("⏳ Memproses citra..."):
            # ============================================================================
            # PREPROCESSING
            # ============================================================================
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

            # ============================================================================
            # TAMPILAN PROSES
            # ============================================================================
            st.markdown("## 🧪 Tahapan Pengolahan Citra")
            
            import base64
            from io import BytesIO

            def to_base64(img):
                pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()
            
            def image_card(img, caption):
                st.markdown(
                    f"""
                    <div style="
                        background:#000;
                        padding:12px;
                        border-radius:12px;
                        text-align:center;
                        width:180px;
                        margin:auto;
                    ">
                        <img src="data:image/png;base64,{img}" style="width:150px;border-radius:8px"/>
                        <p style="color:#aaa;font-size:13px;margin-top:6px">{caption}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # ========== GRID HORIZONTAL =================
            cols = st.columns(4)
            
            with cols[0]:
                image_card(to_base64(img_no_bg), "Remove Background")

            with cols[1]:
                image_card(to_base64(gray), "Grayscale")

            with cols[2]:
                image_card(to_base64(gray_clahe), "CLAHE")

            with cols[3]:
                image_card(to_base64(edges), "Canny")

            # ============================================================================
            # SPASI JARAK KE HASIL PREDIKSI
            # ============================================================================
            st.markdown(
                """
                <div style="margin-top:30px;"></div>
                """,
                unsafe_allow_html=True
            )

            # ============================================================================
            # EKSTRAKSI FITUR
            # ============================================================================
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

            # ============================================================================
            # PREDIKSI
            # ============================================================================
            pred = model.predict(fitur_scaled)[0]
            prob = model.predict_proba(fitur_scaled)[0]
            confidence = np.max(prob) * 100

        # ============================================================================
        # HASIL
        # ============================================================================
        st.success(f"🎯 **Kadar Garam: {kelas_map[pred]}**")
        st.info(f"📈 Confidence: **{confidence:.2f}%**")

        # ============================================================================
        # TABEL FITUR
        # ============================================================================
        st.subheader("📊 Nilai Fitur Ekstraksi")
        st.table({
            "Fitur": [
                "Contrast",
                "Correlation",
                "Energy",
                "Homogeneity",
                "Edge Pixel",
                "Edge Ratio"
            ],
            "Nilai": [
                round(fitur[0], 4),
                round(fitur[1], 4),
                round(fitur[2], 4),
                round(fitur[3], 4),
                int(fitur[4]),
                round(fitur[5], 6)
            ]
        })
