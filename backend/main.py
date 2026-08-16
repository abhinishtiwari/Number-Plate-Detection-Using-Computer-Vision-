import os
import sys
import cv2
import numpy as np
from pathlib import Path
import tempfile
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .detector import YOLOPlateDetector
from .rto_lookup import rto_engine
from .ocr_engine import ocr_engine

logger = logging.getLogger("MainAPI")

app = FastAPI(title="Number Plate AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = YOLOPlateDetector()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "rto_records_count": len(rto_engine.get_all_records())
    }

@app.get("/rto-dataset")
def get_rto_dataset():
    """Returns all 1000+ RTO dataset entries for dynamic frontend table rendering."""
    return rto_engine.get_all_records()

def process_frame(image: np.ndarray):
    """
    Processes a single image frame through real plate detection, cropping, OCR, and local RTO lookup.
    No hardcoded fallbacks or sample data.
    """
    detected_boxes = detector.detect(image)
    if not detected_boxes:
        return []

    results = []
    for (x, y, w, h, box_conf) in detected_boxes:
        margin_x = int(w * 0.03)
        margin_y = int(h * 0.03)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        roi = image[y1:y2, x1:x2]

        plate_text, ocr_conf = ocr_engine.extract_text(roi)
        
        # Calculate dynamic confidence strictly from model bounding box score & OCR confidence score
        if plate_text != "Not detected":
            combined_conf = round(box_conf * 100 * 0.4 + ocr_conf * 0.6, 1)
        else:
            combined_conf = round(box_conf * 100 * 0.5, 1)

        # Lookup in local RTO dataset
        rto_info = rto_engine.lookup(plate_text)

        results.append({
            "box": [x, y, w, h],
            "text": plate_text,
            "confidence": combined_conf,
            "state_name": rto_info["state_name"],
            "full_rto_code": rto_info["full_rto_code"],
            "city": rto_info["city"]
        })

    # Sort results prioritizing plates with valid matched State names over unmapped sub-boxes
    results.sort(
        key=lambda p: (
            p["state_name"] not in ["Unknown", "Not detected"],
            p["text"] != "Not detected",
            p["confidence"]
        ),
        reverse=True
    )

    return results

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    """
    Dynamic Detection API Endpoint for Images and Videos:
    Input File -> Real Object Detection -> OpenCV Preprocessing -> Local OCR -> RTO Dataset Lookup.
    No hardcoded values anywhere.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Video Processing Path
    if file.content_type.startswith("video/"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            all_results = []
            seen_texts = set()
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample key frames (every 10th frame)
                if frame_count % 10 == 0:
                    frame_plates = process_frame(frame)
                    for plate in frame_plates:
                        txt = plate["text"]
                        if txt not in seen_texts:
                            seen_texts.add(txt)
                            all_results.append(plate)
                
                frame_count += 1
                if frame_count > 300: # Limit sample length
                    break

            cap.release()
            os.remove(tmp_path)
            return {"plates": all_results}
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.error(f"Video processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

    # 2. Image Processing Path
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image format.")

    plates = process_frame(image)
    return {"plates": plates}
