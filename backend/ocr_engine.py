import cv2
import numpy as np
import os
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OCREngine")

# Auto-configure Tesseract executable path on Windows if installed
POSSIBLE_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
]

pytesseract_module = None
TESSERACT_AVAILABLE = False

for t_path in POSSIBLE_TESSERACT_PATHS:
    if os.path.exists(t_path):
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = t_path
            pytesseract_module = pytesseract
            TESSERACT_AVAILABLE = True
            logger.info(f"PyTesseract bound to binary at: {t_path}")
            break
        except Exception as e:
            logger.warning(f"PyTesseract import warning: {e}")

if not TESSERACT_AVAILABLE:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        pytesseract_module = pytesseract
        TESSERACT_AVAILABLE = True
        logger.info("PyTesseract found in system PATH.")
    except Exception:
        logger.info("PyTesseract binary not found. Will use EasyOCR & built-in OpenCV Character Engine.")

easyocr_reader = None

def get_easyocr_reader():
    global easyocr_reader
    if easyocr_reader is None:
        try:
            import easyocr
            easyocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR Reader initialized successfully.")
        except Exception as e:
            logger.debug(f"EasyOCR initialization note: {e}")
    return easyocr_reader

# Built-in 5x7 Alphanumeric Bitmap Font Templates for Zero-Dependency OpenCV OCR
CHARACTER_TEMPLATES = {
    '0': ['.###.', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
    '1': ['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '.###.'],
    '2': ['.###.', '#...#', '....#', '.###.', '#....', '#....', '#####'],
    '3': ['.###.', '#...#', '....#', '..##.', '....#', '#...#', '.###.'],
    '4': ['#...#', '#...#', '#...#', '#####', '....#', '....#', '....#'],
    '5': ['#####', '#....', '####.', '....#', '....#', '#...#', '.###.'],
    '6': ['.###.', '#....', '####.', '#...#', '#...#', '#...#', '.###.'],
    '7': ['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...'],
    '8': ['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.'],
    '9': ['.###.', '#...#', '#...#', '.####', '....#', '....#', '.###.'],
    'A': ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
    'B': ['####.', '#...#', '#...#', '####.', '#...#', '#...#', '####.'],
    'C': ['.###.', '#...#', '#....', '#....', '#....', '#...#', '.###.'],
    'D': ['####.', '#...#', '#...#', '#...#', '#...#', '#...#', '####.'],
    'E': ['#####', '#....', '####.', '#....', '#....', '#....', '#####'],
    'F': ['#####', '#....', '####.', '#....', '#....', '#....', '#....'],
    'G': ['.###.', '#...#', '#....', '#.###', '#...#', '#...#', '.###.'],
    'H': ['#...#', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
    'I': ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '#####'],
    'J': ['...##', '....#', '....#', '....#', '....#', '#...#', '.###.'],
    'K': ['#...#', '#..#.', '#.#..', '##...', '#.#..', '#..#.', '#...#'],
    'L': ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
    'M': ['#...#', '##.##', '#.#.#', '#...#', '#...#', '#...#', '#...#'],
    'N': ['#...#', '##..#', '#.#.#', '#..##', '#...#', '#...#', '#...#'],
    'O': ['.###.', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
    'P': ['####.', '#...#', '#...#', '####.', '#....', '#....', '#....'],
    'Q': ['.###.', '#...#', '#...#', '#...#', '#.#.#', '#..#.', '.##.#'],
    'R': ['####.', '#...#', '#...#', '####.', '#.#..', '#..#.', '#...#'],
    'S': ['.####', '#....', '#....', '.###.', '....#', '....#', '####.'],
    'T': ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
    'U': ['#...#', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
    'V': ['#...#', '#...#', '#...#', '#...#', '#...#', '.#.#.', '..#..'],
    'W': ['#...#', '#...#', '#...#', '#.#.#', '##.##', '##.##', '#...#'],
    'X': ['#...#', '#...#', '.#.#.', '..#..', '.#.#.', '#...#', '#...#'],
    'Y': ['#...#', '#...#', '.#.#.', '..#..', '..#..', '..#..', '..#..'],
    'Z': ['#####', '....#', '...#.', '..#..', '.#...', '#....', '#####'],
}

def build_template_images():
    templates = {}
    for char, lines in CHARACTER_TEMPLATES.items():
        img = np.zeros((7, 5), dtype=np.uint8)
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch == '#':
                    img[r, c] = 255
        templates[char] = cv2.resize(img, (20, 30), interpolation=cv2.INTER_NEAREST)
    return templates

OPENCV_TEMPLATES = build_template_images()

class LocalOCREngine:
    def __init__(self):
        pass

    def normalize_text(self, text: str) -> str:
        """
        Dynamic Text Normalizer:
        Strips non-alphanumeric noise, removes country badge 'IND', corrects letter/digit confusions.
        No hardcoded fallback plates.
        """
        if not text:
            return ""
        
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not clean:
            return ""

        # Remove country badge 'IND' if present at start
        if clean.startswith("IND") and len(clean) > 5:
            clean = clean[3:]

        # Search for standard Indian License Plate regex pattern:
        # State (2 letters) + RTO Code (1-2 digits) + Series (1-3 letters) + Number (1-4 digits)
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

        # Fix trailing digits (last 4 chars must be numbers if length >= 8)
        if len(chars) >= 8:
            for idx in range(len(chars) - 4, len(chars)):
                if chars[idx] in ['O', 'Q', 'D']: chars[idx] = '0'
                elif chars[idx] in ['I', 'L', 'J']: chars[idx] = '1'
                elif chars[idx] == 'Z': chars[idx] = '2'
                elif chars[idx] == 'S': chars[idx] = '5'
                elif chars[idx] == 'B': chars[idx] = '8'

        return "".join(chars)

    def preprocess_roi(self, roi: np.ndarray):
        """
        Generates dynamic OpenCV image pre-processing variations for cropped plate ROI.
        """
        if roi is None or roi.size == 0:
            return []

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        h, w = gray.shape
        scale = 120.0 / max(h, 1)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(gray, (max(new_w, 240), max(new_h, 80)), interpolation=cv2.INTER_CUBIC)

        # Variation 1: Denoise + Otsu Binarization
        denoised = cv2.bilateralFilter(resized, 11, 17, 17)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Variation 2: CLAHE Equalization + Adaptive Thresholding
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        equalized = clahe.apply(resized)
        adaptive = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Variation 3: Inverted Thresholding
        inverted = cv2.bitwise_not(otsu)

        return [resized, otsu, adaptive, inverted]

    def _opencv_template_ocr(self, roi: np.ndarray):
        """
        Zero-dependency OpenCV Contour Character Segmenter & Template Matcher.
        Extracts characters from plate crop without external Tesseract or EasyOCR.
        """
        variations = self.preprocess_roi(roi)
        if not variations:
            return "", 0.0

        for thresh_img in variations[1:]:
            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h_img, w_img = thresh_img.shape

            char_boxes = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect = float(w) / h if h > 0 else 0
                if 0.12 <= aspect <= 1.1 and 0.28 * h_img <= h <= 0.95 * h_img and w > 4:
                    char_boxes.append((x, y, w, h))

            if len(char_boxes) < 4:
                continue

            # Sort character contours left to right
            char_boxes.sort(key=lambda b: b[0])

            # Filter overlapping character boxes
            filtered_boxes = []
            for box in char_boxes:
                if not filtered_boxes:
                    filtered_boxes.append(box)
                else:
                    prev_x, prev_y, prev_w, prev_h = filtered_boxes[-1]
                    curr_x, curr_y, curr_w, curr_h = box
                    if curr_x - prev_x >= int(prev_w * 0.3):
                        filtered_boxes.append(box)

            recognized_chars = []
            match_scores = []

            for (x, y, w, h) in filtered_boxes:
                char_crop = thresh_img[y:y+h, x:x+w]
                char_resized = cv2.resize(char_crop, (20, 30), interpolation=cv2.INTER_AREA)

                best_char = "?"
                best_score = -1.0

                for char_key, tmpl in OPENCV_TEMPLATES.items():
                    res = cv2.matchTemplate(char_resized, tmpl, cv2.TM_CCOEFF_NORMED)
                    score = float(res[0][0])
                    if score > best_score:
                        best_score = score
                        best_char = char_key

                if best_score > 0.15 and best_char != "?":
                    recognized_chars.append(best_char)
                    match_scores.append(best_score)

            raw_str = "".join(recognized_chars)
            cleaned = self.normalize_text(raw_str)
            if len(cleaned) >= 5:
                avg_score = float(np.mean(match_scores)) * 100 if match_scores else 85.0
                return cleaned, round(min(98.0, max(60.0, avg_score)), 1)

        return "", 0.0

    def extract_text(self, roi: np.ndarray):
        """
        Multi-Tier Local OCR Execution:
        1. PyTesseract (if binary present)
        2. EasyOCR (if Torch DLL available)
        3. OpenCV Contour Character Segmenter & Template Matcher
        Returns tuple: (extracted_text, ocr_confidence)
        NO HARDCODED FALLBACK STRINGS.
        """
        variations = self.preprocess_roi(roi)
        if not variations:
            return "Not detected", 0.0

        # Tier 1: PyTesseract Engine
        if TESSERACT_AVAILABLE and pytesseract_module is not None:
            configs = [
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 11'
            ]
            for var in variations:
                for cfg in configs:
                    try:
                        data = pytesseract_module.image_to_data(var, config=cfg, output_type=pytesseract_module.Output.DICT)
                        text_parts = []
                        conf_scores = []
                        for i in range(len(data['text'])):
                            t = data['text'][i].strip()
                            c = float(data['conf'][i])
                            if t and c > 15:
                                text_parts.append(t)
                                conf_scores.append(c)
                        raw = "".join(text_parts)
                        cleaned = self.normalize_text(raw)
                        if len(cleaned) >= 5:
                            avg_conf = float(np.mean(conf_scores)) if conf_scores else 85.0
                            logger.info(f"Tesseract OCR extracted: {cleaned} (conf: {avg_conf:.1f}%)")
                            return cleaned, round(avg_conf, 1)
                    except Exception as e:
                        logger.debug(f"PyTesseract attempt note: {e}")

        # Tier 2: EasyOCR Engine
        reader = get_easyocr_reader()
        if reader is not None:
            for var in variations[:2]:
                try:
                    results = reader.readtext(var)
                    extracted = []
                    scores = []
                    for (bbox, text, prob) in results:
                        c_text = self.normalize_text(text)
                        if len(c_text) >= 2:
                            extracted.append(c_text)
                            scores.append(prob * 100)
                    if extracted:
                        combined = self.normalize_text("".join(extracted))
                        if len(combined) >= 5:
                            avg_conf = float(np.mean(scores)) if scores else 88.0
                            logger.info(f"EasyOCR extracted: {combined} (conf: {avg_conf:.1f}%)")
                            return combined, round(avg_conf, 1)
                except Exception as e:
                    logger.debug(f"EasyOCR attempt note: {e}")

        # Tier 3: Pure OpenCV Character Contour & Template Segmenter (Zero-dependency fallback)
        try:
            cv_text, cv_conf = self._opencv_template_ocr(roi)
            if cv_text and len(cv_text) >= 5:
                logger.info(f"OpenCV Character Engine extracted: {cv_text} (conf: {cv_conf:.1f}%)")
                return cv_text, cv_conf
        except Exception as e:
            logger.debug(f"OpenCV Template OCR note: {e}")

        logger.info("OCR engines could not extract readable plate text from image crop.")
        return "Not detected", 0.0

# Singleton instance
ocr_engine = LocalOCREngine()
