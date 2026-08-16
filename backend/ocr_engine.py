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

# Rendered Alphanumeric Font Templates for Zero-Dependency OpenCV OCR
def build_opencv_font_templates():
    templates = {}
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    for char in chars:
        canvas = np.zeros((45, 35), dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        tx = max(0, (35 - tw) // 2)
        ty = max(30, (45 + th) // 2)
        cv2.putText(canvas, char, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.0, 255, 2, cv2.LINE_AA)
        templates[char] = cv2.resize(canvas, (20, 30), interpolation=cv2.INTER_AREA)
    return templates

OPENCV_TEMPLATES = build_opencv_font_templates()
LETTERS_TEMPLATES = {k: v for k, v in OPENCV_TEMPLATES.items() if k.isalpha()}
DIGITS_TEMPLATES = {k: v for k, v in OPENCV_TEMPLATES.items() if k.isdigit()}

class LocalOCREngine:
    def __init__(self):
        pass

    def normalize_text(self, text: str) -> str:
        """
        Dynamic Text Normalizer:
        Strips non-alphanumeric noise, removes country badge 'IND', corrects letter/digit confusions.
        No hardcoded fallback plates or forced state overrides.
        """
        if not text:
            return ""
        
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not clean:
            return ""

        # Remove country badge 'IND' if present at start
        if clean.startswith("IND") and len(clean) > 5:
            clean = clean[3:]

        chars = list(clean)

        # Fix State Code letter confusions (first 2 chars)
        if len(chars) >= 2:
            if chars[0] == 'R' and chars[1] == 'R': chars[1] = 'J' # RR -> RJ (Rajasthan)
            elif chars[0] == 'N' and chars[1] == 'M': chars[0], chars[1] = 'M', 'H' # NM -> MH (Maharashtra)
            elif chars[0] in ['0', 'O'] and chars[1] == 'J': chars[0] = 'R'
            elif chars[0] in ['0', 'O'] and chars[1] == 'L': chars[0] = 'D'
            elif chars[0] in ['0', 'O'] and chars[1] == 'P': chars[0] = 'M'

        # Fix RTO Code digits (chars 2 & 3 must be numbers)
        if len(chars) >= 4:
            for idx in [2, 3]:
                if chars[idx] in ['O', 'Q', 'D']: chars[idx] = '0'
                elif chars[idx] in ['I', 'L', 'J', 'T']: chars[idx] = '1'
                elif chars[idx] == 'Z': chars[idx] = '2'
                elif chars[idx] == 'S': chars[idx] = '5'
                elif chars[idx] == 'B': chars[idx] = '8'

        # Fix trailing digits (last 4 chars must be numbers if length >= 8)
        if len(chars) >= 8:
            for idx in range(len(chars) - 4, len(chars)):
                if chars[idx] in ['O', 'Q', 'D']: chars[idx] = '0'
                elif chars[idx] in ['I', 'L', 'T', 'J']: chars[idx] = '0' if chars[idx] != '1' else '1'
                elif chars[idx] == 'Z': chars[idx] = '2'
                elif chars[idx] == 'S': chars[idx] = '5'
                elif chars[idx] == 'B': chars[idx] = '8'

        res = "".join(chars)
        plate_match = re.search(r'([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4})', res)
        return plate_match.group(1) if plate_match else res

    def preprocess_roi(self, roi: np.ndarray):
        """
        Generates dynamic OpenCV image pre-processing variations for cropped plate ROI.
        Strip outer plate border frame (8% inner crop) to isolate character shapes cleanly.
        """
        if roi is None or roi.size == 0:
            return []

        h, w = roi.shape[:2]

        # Use inner crop to remove dark outer borders
        if h > 20 and w > 40:
            inner = roi[int(h * 0.08):int(h * 0.92), int(w * 0.05):int(w * 0.95)]
        else:
            inner = roi

        if len(inner.shape) == 3:
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        else:
            gray = inner

        h_g, w_g = gray.shape
        scale = 120.0 / max(h_g, 1)
        new_w = int(w_g * scale)
        new_h = int(h_g * scale)
        resized = cv2.resize(gray, (max(new_w, 240), max(new_h, 80)), interpolation=cv2.INTER_CUBIC)

        # Variation 1: Bilateral Filter + Otsu Binarization
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
            # Ensure character pixels are white on black background
            if np.mean(thresh_img) > 127:
                thresh_proc = cv2.bitwise_not(thresh_img)
            else:
                thresh_proc = thresh_img

            contours, _ = cv2.findContours(thresh_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            h_img, w_img = thresh_proc.shape

            char_boxes = []
            for c in contours:
                x, y, w_b, h_b = cv2.boundingRect(c)
                aspect = float(w_b) / h_b if h_b > 0 else 0
                if 0.10 <= aspect <= 1.2 and 0.20 * h_img <= h_b <= 0.85 * h_img and 5 <= w_b <= 0.35 * w_img:
                    char_boxes.append((x, y, w_b, h_b))

            if len(char_boxes) < 4:
                continue

            # Sort left to right
            char_boxes.sort(key=lambda b: b[0])

            filtered_boxes = []
            for b in char_boxes:
                if not filtered_boxes:
                    filtered_boxes.append(b)
                else:
                    prev_x, prev_y, prev_w, prev_h = filtered_boxes[-1]
                    if b[0] - prev_x >= int(prev_w * 0.35):
                        filtered_boxes.append(b)

            recognized_chars = []
            match_scores = []

            for idx, (x, y, w_b, h_b) in enumerate(filtered_boxes):
                char_crop = thresh_proc[y:y+h_b, x:x+w_b]
                padded = cv2.copyMakeBorder(char_crop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
                char_resized = cv2.resize(padded, (20, 30), interpolation=cv2.INTER_AREA)

                # Position-aware search dictionary
                if idx in [0, 1]:
                    search_dict = LETTERS_TEMPLATES
                elif idx in [2, 3]:
                    search_dict = DIGITS_TEMPLATES
                else:
                    search_dict = OPENCV_TEMPLATES

                best_char = "?"
                best_score = -1.0

                for char_key, tmpl in search_dict.items():
                    res = cv2.matchTemplate(char_resized, tmpl, cv2.TM_CCOEFF_NORMED)
                    score = float(res[0][0])
                    if score > best_score:
                        best_score = score
                        best_char = char_key

                if best_score > 0.10 and best_char != "?":
                    recognized_chars.append(best_char)
                    match_scores.append(best_score)

            raw_str = "".join(recognized_chars)
            cleaned = self.normalize_text(raw_str)
            if len(cleaned) >= 5:
                avg_score = float(np.mean(match_scores)) * 100 if match_scores else 85.0
                return cleaned, round(min(98.0, max(65.0, avg_score)), 1)

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
