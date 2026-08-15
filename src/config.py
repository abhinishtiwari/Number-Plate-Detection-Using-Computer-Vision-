import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CASCADE_PATH = BASE_DIR / "cascades" / "haarcascade_russian_plate_number.xml"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detection Parameters
MIN_PLATE_AREA = 500
MAX_PLATE_AREA = 50000
MIN_ASPECT_RATIO = 2.0
MAX_ASPECT_RATIO = 6.0

# Preprocessing Parameters
BILATERAL_D = 11
BILATERAL_SIGMA_COLOR = 17
BILATERAL_SIGMA_SPACE = 17
CANNY_THRESHOLD1 = 30
CANNY_THRESHOLD2 = 200

# OCR Options
DEFAULT_OCR_ENGINE = "tesseract"  # Options: "easyocr", "tesseract", "contour"
