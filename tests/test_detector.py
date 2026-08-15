import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.detector import PlateDetector
from src.ocr_engine import OCREngine

class TestPlateDetector(unittest.TestCase):
    def setUp(self):
        self.detector = PlateDetector()
        self.ocr = OCREngine(engine_type="heuristic")
        # Create synthetic test image (blank dark image with white rectangular plate)
        self.test_img = np.zeros((400, 600, 3), dtype=np.uint8)
        # Draw white rectangle simulating a license plate (150x50, aspect ratio = 3.0)
        cv2.rectangle(self.test_img, (200, 150), (350, 200), (255, 255, 255), -1)

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector)

    def test_contour_detection(self):
        detections = self.detector.detect_via_contours(self.test_img)
        self.assertGreaterEqual(len(detections), 1)

    def test_ocr_cleaning(self):
        cleaned = self.ocr.clean_plate_text("  MH-12 AB 1234!! ")
        self.assertEqual(cleaned, "MH12AB1234")

if __name__ == '__main__':
    unittest.main()
