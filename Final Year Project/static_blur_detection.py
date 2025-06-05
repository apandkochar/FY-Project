import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Blur Detector", layout="wide")

# Sidebar: choose mode
mode = st.sidebar.radio(
    "Select mode",
    ("1️⃣ Threshold setup", "2️⃣ Blur detection")
)

# Initialize threshold in session state
if "threshold" not in st.session_state:
    st.session_state.threshold = 300.0

# 1️⃣ Threshold setup
if mode == "1️⃣ Threshold setup":
    st.header("Step 1: Set Your Threshold")
    st.write(
        "Upload one clean (sharp) image and one blurry image. "
        "We'll compute their Laplacian variances and let you pick a threshold."
    )

    col1, col2 = st.columns(2)
    with col1:
        clean_file = st.file_uploader(
            "Upload a clean image",
            type=["png", "jpg", "jpeg", "bmp"],
            key="clean_uploader"
        )
    with col2:
        blur_file = st.file_uploader(
            "Upload a blurry image",
            type=["png", "jpg", "jpeg", "bmp"],
            key="blur_uploader"
        )

    if clean_file and blur_file:
        # Read files into OpenCV
        def to_gray(uploaded_file):
            arr = np.frombuffer(uploaded_file.read(), np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        img_clean = to_gray(clean_file)
        img_blur  = to_gray(blur_file)

        # Compute variances
        var_clean = cv2.Laplacian(img_clean, cv2.CV_64F).var()
        var_blur  = cv2.Laplacian(img_blur,  cv2.CV_64F).var()

        st.write(f"**Clean image variance:** {var_clean:.2f}")
        st.write(f"**Blurry image variance:** {var_blur:.2f}")

        # Slider to choose threshold
        new_thr = st.slider(
            "Select threshold",
            min_value=float(min(var_clean, var_blur)),
            max_value=float(max(var_clean, var_blur)),
            value=float(st.session_state.threshold),
            help="Images with variance below this will be labeled “Blurry.”"
        )
        st.session_state.threshold = new_thr

        st.success(f"Threshold set to **{st.session_state.threshold:.2f}**")

# 2️⃣ Blur detection
else:
    st.header("Step 2: Detect Blur in a Batch")
    st.write(
        "Upload one or more images. "
        f"Images with Laplacian variance below **{st.session_state.threshold:.2f}** "
        "will be labeled **Blurry**."
    )

    uploaded_files = st.file_uploader(
        "Upload images for blur detection",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True
    )  # :contentReference[oaicite:0]{index=0}

    if uploaded_files:
        results = []
        for uf in uploaded_files:
            # Convert to grayscale OpenCV image
            data = np.frombuffer(uf.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                st.warning(f"Could not process {uf.name}")
                continue

            lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
            label   = "Blurry" if lap_var < st.session_state.threshold else "Sharp"
            results.append((uf.name, lap_var, label))

        # Show results in a table
        df = pd.DataFrame(results, columns=["Filename", "Laplacian Variance", "Label"])
        st.table(df)

    else:
        st.info("Please upload one or more images for detection.")
