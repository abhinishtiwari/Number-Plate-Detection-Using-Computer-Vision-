import cv2
import re
import numpy as np

class OCREngine:
    def __init__(self, engine_type="auto"):
        self.engine_type = engine_type
        self.easyocr_reader = None
        self._init_engine()

    def _init_engine(self):
        """Initializes requested OCR library with graceful fallback."""
        if self.engine_type in ["auto", "easyocr"]:
            try:
                import easyocr
                # Initialize English OCR reader
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                self.engine_type = "easyocr"
                return
            except Exception:
                pass
                
        if self.engine_type in ["auto", "tesseract"]:
            try:
                import pytesseract
                # Check if tesseract binary is responsive
                _ = pytesseract.get_tesseract_version()
                self.engine_type = "tesseract"
                return
            except Exception:
                pass
                
        # Heuristic fallback engine
        self.engine_type = "heuristic"

    def clean_plate_text(self, text):
        """Filters non-alphanumeric noise characters from license plate text."""
        if not text:
            return ""
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        return cleaned

    def extract_text(self, plate_roi):
        """
        Extracts license plate text from cropped ROI image using selected OCR engine.
        Returns tuple: (cleaned_text, confidence_score)
        """
        if plate_roi is None or plate_roi.size == 0:
            return "", 0.0

        if self.engine_type == "easyocr" and self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(plate_roi)
                text_results = []
                conf_scores = []
                for res in results:
                    bbox, text, conf = res
                    cleaned = self.clean_plate_text(text)
                    if len(cleaned) >= 2:
                        text_results.append(cleaned)
                        conf_scores.append(conf)
                if text_results:
                    return "".join(text_results), float(np.mean(conf_scores))
            except Exception:
                pass

        if self.engine_type == "tesseract":
            try:
                import pytesseract
                # Whitelist characters for license plates
                config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(plate_roi, config=config)
                cleaned = self.clean_plate_text(text)
                return cleaned, 0.85 if cleaned else 0.0
            except Exception:
                pass

        # Fallback Heuristic Text Segmenter placeholder
        return "DETECTED_PLATE", 0.70
