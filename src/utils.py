import cv2
import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path

def preprocess_image(image):
    """
    Converts image to grayscale, applies bilateral filter for noise reduction,
    and returns edge-detected image via Canny algorithm.
    """
    if image is None:
        raise ValueError("Input image is None")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Noise reduction while preserving edges
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection
    edged = cv2.Canny(bfilter, 30, 200)
    return gray, edged

def enhance_plate_roi(plate_roi):
    """
    Applies adaptive thresholding and contrast enhancement to cropped license plate ROI.
    """
    if plate_roi is None or plate_roi.size == 0:
        return plate_roi
        
    if len(plate_roi.shape) == 3:
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_roi
        
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Otsu thresholding
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def draw_detection_box(image, bbox, label="Number Plate", confidence=None):
    """
    Draws a stylized bounding box and label banner over detected plate region.
    """
    output = image.copy()
    x, y, w, h = bbox
    
    # Draw bounding rectangle
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Text string
    text = label if confidence is None else f"{label} ({confidence:.2f})"
    
    # Text background box
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    cv2.rectangle(
        output,
        (x, y - text_height - 10),
        (x + text_width + 10, y),
        (0, 255, 0),
        -1
    )
    cv2.putText(
        output,
        text,
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    return output

def log_detection_to_csv(plate_text, confidence=1.0, csv_path="output/detections.csv"):
    """
    Logs timestamped license plate detections into CSV file.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Plate_Text", "Confidence"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            plate_text,
            f"{confidence:.2f}"
        ])
