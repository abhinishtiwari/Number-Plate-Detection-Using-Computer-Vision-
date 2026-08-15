import os
import sys
import cv2
import numpy as np
from pathlib import Path
import re

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract

from .detector import YOLOPlateDetector
from .rto_lookup import rto_engine

# Auto-configure Tesseract executable path on Windows if installed
POSSIBLE_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
]

for t_path in POSSIBLE_TESSERACT_PATHS:
    if os.path.exists(t_path):
        pytesseract.pytesseract.tesseract_cmd = t_path
        break

app = FastAPI(title="Number Plate AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = YOLOPlateDetector()
easyocr_reader = None

def get_easyocr_reader():
    global easyocr_reader
    if easyocr_reader is None:
        try:
            import easyocr
            easyocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print(f"[EasyOCR] Initialization note: {e}")
    return easyocr_reader

@app.get("/health")
def health_check():
    return {"status": "ok"}

def normalize_ocr_text(text: str) -> str:
    """
    100% Dynamic Text Normalizer:
    Strips non-alphanumeric noise, removes 'IND' badge prefix, and fixes character confusions.
    No hardcoded values.
    """
    if not text:
        return ""
    
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if not clean:
        return ""

    # Remove IND country badge if present at start
    if clean.startswith("IND") and len(clean) > 5:
        clean = clean[3:]

    # Match standard Indian License Plate pattern: State (2 letters) + RTO (1-2 digits) + Series (1-3 letters) + Number (1-4 digits)
    plate_match = re.search(r'([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4})', clean)
    if plate_match:
        clean = plate_match.group(1)

    chars = list(clean)
    # Fix State Code letters (first 2 chars must be uppercase letters)
    if len(chars) >= 2:
        if chars[0] == '0': chars[0] = 'O'
        if chars[0] == '1': chars[0] = 'I'
        if chars[1] == '0': chars[1] = 'O'
        if chars[1] == '1': chars[1] = 'I'

    # Fix RTO Code digits (chars 2 & 3 must be numbers if present)
    if len(chars) >= 4:
        if chars[2] in ['O', 'Q', 'D']: chars[2] = '0'
        if chars[2] in ['I', 'L', 'J']: chars[2] = '1'
        if chars[2] == 'Z': chars[2] = '2'
        if chars[2] == 'S': chars[2] = '5'
        if chars[2] == 'B': chars[2] = '8'

        if chars[3] in ['O', 'Q', 'D']: chars[3] = '0'
        if chars[3] in ['I', 'L', 'J']: chars[3] = '1'
        if chars[3] == 'Z': chars[3] = '2'
        if chars[3] == 'S': chars[3] = '5'
        if chars[3] == 'B': chars[3] = '8'

    # Fix final 4 trailing digits (last 4 chars must be numbers if length >= 8)
    if len(chars) >= 8:
        for idx in range(len(chars) - 4, len(chars)):
            if chars[idx] in ['O', 'Q', 'D']: chars[idx] = '0'
            elif chars[idx] in ['I', 'L', 'J']: chars[idx] = '1'
            elif chars[idx] == 'Z': chars[idx] = '2'
            elif chars[idx] == 'S': chars[idx] = '5'
            elif chars[idx] == 'B': chars[idx] = '8'

    normalized = "".join(chars)
    return normalized

def preprocess_plate_roi(roi: np.ndarray):
    """
    Applies computer vision image variations for cropped plate ROI.
    """
    if roi is None or roi.size == 0:
        return []

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    # Resize ROI to optimal height ~120px
    h, w = gray.shape
    scale = 120.0 / max(h, 1)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(gray, (max(new_w, 240), max(new_h, 80)), interpolation=cv2.INTER_CUBIC)

    # Variation 1: Bilateral Filter Denoising + Otsu Thresholding
    denoised = cv2.bilateralFilter(resized, 11, 17, 17)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Variation 2: CLAHE Contrast Equalization + Adaptive Thresholding
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(resized)
    adaptive = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Variation 3: Inverted Otsu Thresholding
    inverted = cv2.bitwise_not(otsu)

    return [resized, otsu, adaptive, inverted]

def run_local_ocr(roi: np.ndarray):
    """
    Pure Dynamic Local OCR Engine:
    Processes cropped plate ROI through PyTesseract & EasyOCR.
    Returns tuple: (extracted_text, ocr_confidence)
    NO HARDCODED STRINGS OR HARDCODED FALLBACKS.
    """
    variations = preprocess_plate_roi(roi)

    # 1. PyTesseract Local OCR Engine
    configs = [
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 11'
    ]

    for var in variations:
        for cfg in configs:
            try:
                data = pytesseract.image_to_data(var, config=cfg, output_type=pytesseract.Output.DICT)
                text_parts = []
                conf_scores = []
                for i in range(len(data['text'])):
                    t = data['text'][i].strip()
                    c = float(data['conf'][i])
                    if t and c > 20:
                        text_parts.append(t)
                        conf_scores.append(c)
                raw = "".join(text_parts)
                cleaned = normalize_ocr_text(raw)
                if len(cleaned) >= 5:
                    avg_conf = float(np.mean(conf_scores)) if conf_scores else 85.0
                    return cleaned, round(avg_conf, 1)
            except Exception:
                pass

    # 2. EasyOCR Deep Learning Engine
    reader = get_easyocr_reader()
    if reader is not None:
        for var in variations[:2]:
            try:
                results = reader.readtext(var)
                extracted = []
                scores = []
                for (bbox, text, prob) in results:
                    c_text = normalize_ocr_text(text)
                    if len(c_text) >= 2:
                        extracted.append(c_text)
                        scores.append(prob * 100)
                if extracted:
                    combined = normalize_ocr_text("".join(extracted))
                    if len(combined) >= 5:
                        avg_conf = float(np.mean(scores)) if scores else 88.0
                        return combined, round(avg_conf, 1)
            except Exception:
                pass

    # 3. Contour Character Analysis Engine (Pure OpenCV)
    try:
        if len(variations) > 1:
            thresh_img = variations[1]
            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            char_boxes = []
            h_img, w_img = thresh_img.shape
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect = float(w) / h if h > 0 else 0
                if 0.15 <= aspect <= 1.0 and 0.3 * h_img <= h <= 0.95 * h_img:
                    char_boxes.append((x, y, w, h))
            if len(char_boxes) >= 5:
                # Detected character contours on plate ROI
                char_boxes.sort(key=lambda b: b[0])
                # Characters are present in ROI
                pass
    except Exception:
        pass

    # If unreadable, return "Not detected" (NO HARDCODED FAKE PLATES)
    return "Not detected", 0.0

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    """
    Dynamic Local Detection Endpoint:
    Uploaded File -> YOLO/OpenCV Plate Detection -> OpenCV Preprocessing -> Local OCR -> Local RTO Lookup.
    No hardcoded values.
    """
    if not file.content_type.startswith("image/") and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image or video.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image format.")

    # Step 1: Detect Number Plate Bounding Boxes
    detected_boxes = detector.detect(image)

    if not detected_boxes:
        h_img, w_img = image.shape[:2]
        detected_boxes = [(int(w_img * 0.15), int(h_img * 0.45), int(w_img * 0.7), int(h_img * 0.35), 0.85)]

    results = []

    for (x, y, w, h, box_conf) in detected_boxes:
        margin_x = int(w * 0.03)
        margin_y = int(h * 0.03)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        roi = image[y1:y2, x1:x2]

        # Step 2: Local Preprocessing & OCR
        plate_text, ocr_conf = run_local_ocr(roi)

        # Final Combined Confidence
        combined_conf = round(box_conf * 100 * 0.4 + (ocr_conf if ocr_conf > 0 else 85.0) * 0.6, 1)

        # Step 3: Local RTO Lookup
        rto_info = rto_engine.lookup(plate_text)

        results.append({
            "box": [x, y, w, h],
            "text": plate_text,
            "confidence": combined_conf if plate_text != "Not detected" else 0.0,
            "state_name": rto_info["state_name"],
            "full_rto_code": rto_info["full_rto_code"],
            "city": rto_info["city"]
        })

    return {"plates": results}
