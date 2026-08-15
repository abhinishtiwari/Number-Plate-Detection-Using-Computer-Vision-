import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io
import re

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract

app = FastAPI(title="Number Plate Detection API")

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local & Vercel/GitHub Pages deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "cascades" / "haarcascade_russian_plate_number.xml"

# Initialize Haar Cascade Classifier
plate_cascade = None
if CASCADE_PATH.exists():
    plate_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

# Lazy-loaded EasyOCR reader instance
easyocr_reader = None

def get_easyocr_reader():
    """Initializes EasyOCR reader on demand if available."""
    global easyocr_reader
    if easyocr_reader is None:
        try:
            import easyocr
            easyocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception:
            pass
    return easyocr_reader


@app.get("/health")
def health_check():
    """Lightweight health check endpoint for cold-start wakeups and monitoring."""
    return {"status": "ok"}


def clean_text(raw_text: str) -> str:
    """Sanitizes OCR text to keep only uppercase alphanumeric characters."""
    if not raw_text:
        return ""
    return re.sub(r'[^A-Z0-9]', '', raw_text.upper())


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """Preprocesses cropped license plate region for optimal OCR extraction."""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    # Resize ROI for better OCR resolution
    height, width = gray.shape
    if height < 80 or width < 160:
        gray = cv2.resize(gray, (max(width * 2, 200), max(height * 2, 80)), interpolation=cv2.INTER_CUBIC)

    # Bilateral filter to smooth noise while preserving character edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Otsu thresholding
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def run_ocr(roi: np.ndarray) -> str:
    """
    Runs multi-engine OCR (EasyOCR -> PyTesseract -> Heuristic Fallback)
    to extract license plate alphanumeric text.
    """
    # 1. EasyOCR (Deep Learning Engine - Works without system Tesseract binary)
    reader = get_easyocr_reader()
    if reader is not None:
        try:
            ocr_results = reader.readtext(roi)
            extracted_parts = []
            for (bbox, text, prob) in ocr_results:
                cleaned = clean_text(text)
                if len(cleaned) >= 2:
                    extracted_parts.append(cleaned)
            if extracted_parts:
                return "".join(extracted_parts)
        except Exception:
            pass

    # 2. PyTesseract OCR Engine
    thresh = preprocess_roi(roi)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    try:
        raw_text = pytesseract.image_to_string(thresh, config=config)
        cleaned = clean_text(raw_text)
        if cleaned:
            return cleaned
    except Exception:
        pass

    # 3. Secondary PyTesseract on raw ROI
    try:
        raw_text = pytesseract.image_to_string(roi, config=config)
        cleaned = clean_text(raw_text)
        if cleaned:
            return cleaned
    except Exception:
        pass

    return "DETECTED"


@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, detects license plate bounding boxes,
    runs OCR on detected ROIs, and returns JSON formatted predictions.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image format.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected_boxes = []

    # Method 1: Haar Cascade Detection
    if plate_cascade and not plate_cascade.empty():
        cascade_plates = plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 15)
        )
        for (x, y, w, h) in cascade_plates:
            detected_boxes.append((int(x), int(y), int(w), int(h)))

    # Method 2: Contour Aspect-Ratio Detection (Fallback / Hybrid)
    if not detected_boxes:
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        for c in contours:
            area = cv2.contourArea(c)
            if area < 400 or area > 60000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 2.0 <= aspect_ratio <= 6.5:
                detected_boxes.append((int(x), int(y), int(w), int(h)))
                break

    results = []
    for (x, y, w, h) in detected_boxes:
        # Extract ROI with safety margin
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        roi = image[y1:y2, x1:x2]
        recognized_text = run_ocr(roi)

        results.append({
            "box": [x, y, w, h],
            "text": recognized_text
        })

    return {"plates": results}
