import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.detector import PlateDetector
from src.ocr_engine import OCREngine
from src.utils import draw_detection_box

st.set_page_config(
    page_title="Number Plate Detection System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Number Plate Detection System (Computer Vision)")
st.markdown("Automated License Plate Recognition (ALPR / ANPR) using OpenCV, Haar Cascades, Contour Segmentation, and OCR Engines.")

st.sidebar.header("⚙️ Detection Controls")
detection_method = st.sidebar.selectbox("Detection Algorithm", ["hybrid", "cascade", "contour"], index=0)
ocr_choice = st.sidebar.selectbox("OCR Engine", ["auto", "easyocr", "tesseract"], index=0)
min_aspect = st.sidebar.slider("Min Aspect Ratio", 1.5, 3.0, 2.0, 0.1)
max_aspect = st.sidebar.slider("Max Aspect Ratio", 4.0, 8.0, 6.0, 0.1)

detector = PlateDetector()
ocr = OCREngine(engine_type=ocr_choice)

uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Detect
    with st.spinner("Processing image & recognizing license plates..."):
        detections = detector.detect_plates(image, method=detection_method)

    annotated = image.copy()
    detection_data = []

    with col2:
        st.subheader("Detection Results")
        if not detections:
            st.warning("No license plates detected in the image.")
        else:
            for idx, (x, y, w, h, roi, enhanced_roi) in enumerate(detections, 1):
                text, conf = ocr.extract_text(enhanced_roi)
                annotated = draw_detection_box(annotated, (x, y, w, h), label=text if text else f"Plate #{idx}", confidence=conf)
                detection_data.append({
                    "Plate ID": f"Plate #{idx}",
                    "Extracted Text": text if text else "N/A",
                    "Confidence": f"{conf * 100:.1f}%",
                    "Bounding Box (X,Y,W,H)": f"({x}, {y}, {w}, {h})"
                })

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    if detections:
        st.markdown("---")
        st.subheader("🔍 Cropped License Plate ROIs & Data")
        
        cols = st.columns(len(detections))
        for idx, (col, (x, y, w, h, roi, enhanced_roi)) in enumerate(zip(cols, detections), 1):
            with col:
                st.caption(f"Plate Candidate #{idx}")
                st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="Cropped ROI", use_container_width=True)
                st.image(enhanced_roi, caption="Preprocessed ROI", use_container_width=True)

        st.subheader("📋 Detection Logs")
        df = pd.DataFrame(detection_data)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Detection Report (CSV)",
            data=csv,
            file_name="license_plate_detections.csv",
            mime="text/csv"
        )
else:
    st.info("👈 Please upload an image using the file uploader above to begin plate detection.")
