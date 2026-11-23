import sys
import shutil
import importlib
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

import ultralytics


# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Deteksi Tumor Otak",
    layout="wide",
)

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Hilangkan tulisan bawaan:
       200MB per file • JPG, PNG */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

    [data-testid="stFileUploaderDropzone"] small {
        display: none !important;
    }

    [data-testid="stFileUploader"] small {
        display: none !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# JUDUL
# ============================================================

st.title("Deteksi Tumor Otak pada Citra MRI")

st.caption(
    "Perbandingan YOLOv8s Baseline dan YOLOv8s + Triplet Attention"
)


# ============================================================
# PATH
# ============================================================

APP_DIR = Path(__file__).resolve().parent

BASELINE_PATH = APP_DIR / "baseline_best.pt"
TRIPLET_PATH = APP_DIR / "triplet_best.pt"
TRIPLET_MODULE = APP_DIR / "triplet_attention.py"


# ============================================================
# TRIPLET ATTENTION
# ============================================================

def prepare_triplet_module():

    if not TRIPLET_MODULE.exists():
        raise FileNotFoundError(
            f"triplet_attention.py tidak ditemukan:\n{TRIPLET_MODULE}"
        )

    ultralytics_root = Path(
        ultralytics.__file__
    ).resolve().parent

    target = (
        ultralytics_root
        / "nn"
        / "modules"
        / "triplet_attention.py"
    )

    shutil.copy2(
        TRIPLET_MODULE,
        target
    )

    sys.modules.pop(
        "ultralytics.nn.modules.triplet_attention",
        None
    )

    importlib.invalidate_caches()

    importlib.import_module(
        "ultralytics.nn.modules.triplet_attention"
    )


try:
    prepare_triplet_module()

except Exception as e:
    st.error("Triplet Attention gagal dimuat.")
    st.code(str(e))
    st.stop()


from ultralytics import YOLO


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"baseline_best.pt tidak ditemukan:\n{BASELINE_PATH}"
        )

    if not TRIPLET_PATH.exists():
        raise FileNotFoundError(
            f"triplet_best.pt tidak ditemukan:\n{TRIPLET_PATH}"
        )

    baseline = YOLO(
        str(BASELINE_PATH)
    )

    triplet = YOLO(
        str(TRIPLET_PATH)
    )

    return baseline, triplet


try:
    baseline_model, triplet_model = load_models()

except Exception as e:
    st.error("Model gagal dimuat.")
    st.code(str(e))
    st.stop()


# ============================================================
# GAMBAR 640 x 640
# ============================================================

def make_square(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    return ImageOps.pad(
        image.convert("RGB"),
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )


# ============================================================
# DETEKSI
# ============================================================

def run_detection(model, image):

    results = model.predict(
        source=np.array(image),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    result = results[0]

    inference_ms = float(
        result.speed["inference"]
    )

    plotted = result.plot()

    plotted = cv2.cvtColor(
        plotted,
        cv2.COLOR_BGR2RGB
    )

    plotted = make_square(
        plotted
    )

    return result, plotted, inference_ms


# ============================================================
# PREDIKSI UTAMA
# ============================================================

def get_prediction(result):

    if result.boxes is None or len(result.boxes) == 0:

        return (
            "Tidak terdeteksi tumor",
            "-"
        )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidences)
    )

    best_box = result.boxes[
        best_index
    ]

    class_id = int(
        best_box.cls[0].item()
    )

    confidence = float(
        best_box.conf[0].item()
    )

    class_name = (
        result.names[class_id]
        .replace("_", " ")
        .title()
    )

    return (
        class_name,
        f"{confidence * 100:.2f}%"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

if "image_name" not in st.session_state:
    st.session_state.image_name = None


# ============================================================
# UPLOAD
# ============================================================

st.markdown("### Upload Citra MRI Otak")

st.caption(
    "Format: JPG, JPEG, PNG • Maksimal 200 MB"
)


if st.session_state.image_bytes is None:

    uploaded_file = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_file is not None:

        st.session_state.image_bytes = (
            uploaded_file.getvalue()
        )

        st.session_state.image_name = (
            uploaded_file.name
        )

        st.rerun()


else:

    col_info, col_button = st.columns(
        [5, 1]
    )

    with col_info:
        st.success(
            f"Gambar dipilih: "
            f"{st.session_state.image_name}"
        )

    with col_button:

        if st.button(
            "Upload Gambar",
            use_container_width=True,
        ):

            st.session_state.image_bytes = None
            st.session_state.image_name = None

            st.rerun()


# ============================================================
# HASIL
# ============================================================

if st.session_state.image_bytes is not None:

    image = Image.open(
        BytesIO(
            st.session_state.image_bytes
        )
    ).convert("RGB")

    display_image = make_square(
        image
    )


    # --------------------------------------------------------
    # CITRA INPUT
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("Citra Input")

    _, center, _ = st.columns(
        [1, 2, 1]
    )

    with center:

        st.image(
            display_image,
            width=640
        )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with st.spinner(
        "Melakukan deteksi..."
    ):

        (
            baseline_result,
            baseline_image,
            baseline_time,
        ) = run_detection(
            baseline_model,
            image
        )

        (
            triplet_result,
            triplet_image,
            triplet_time,
        ) = run_detection(
            triplet_model,
            image
        )


    # --------------------------------------------------------
    # GAMBAR HASIL
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("Hasil Deteksi")

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### YOLOv8s Baseline"
        )

        st.image(
            baseline_image,
            width=640
        )


    with col2:

        st.markdown(
            "### YOLOv8s + Triplet Attention"
        )

        st.image(
            triplet_image,
            width=640
        )


    # --------------------------------------------------------
    # AMBIL HASIL
    # --------------------------------------------------------

    baseline_pred, baseline_conf = (
        get_prediction(
            baseline_result
        )
    )

    triplet_pred, triplet_conf = (
        get_prediction(
            triplet_result
        )
    )


    # --------------------------------------------------------
    # TABEL
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    hasil = pd.DataFrame(
        [
            {
                "Model": "YOLOv8s Baseline",
                "Prediksi": baseline_pred,
                "Confidence Score": baseline_conf,
                "Waktu Inferensi": f"{baseline_time:.2f} ms",
            },
            {
                "Model": "YOLOv8s + Triplet Attention",
                "Prediksi": triplet_pred,
                "Confidence Score": triplet_conf,
                "Waktu Inferensi": f"{triplet_time:.2f} ms",
            },
        ]
    )

    st.dataframe(
        hasil,
        use_container_width=True,
        hide_index=True,
    )