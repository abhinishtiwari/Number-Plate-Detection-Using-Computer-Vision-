import cv2
import numpy as np
import os
from pathlib import Path
from .config import CASCADE_PATH, MIN_PLATE_AREA, MAX_PLATE_AREA, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO
from .utils import preprocess_image, enhance_plate_roi

class PlateDetector:
    def __init__(self, cascade_path=None):
        self.cascade_path = str(cascade_path) if cascade_path else str(CASCADE_PATH)
        self.cascade = None
        self._load_cascade()

    def _load_cascade(self):
        """Loads Haar Cascade classifier if XML file exists."""
        if os.path.exists(self.cascade_path):
            self.cascade = cv2.CascadeClassifier(self.cascade_path)

    def detect_via_cascade(self, gray_image):
        """Detects candidate plate regions using Haar Cascade Classifier."""
        if self.cascade is None or self.cascade.empty():
            return []
        
        plates = self.cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 20)
        )
        return list(plates)

    def detect_via_contours(self, image):
        """
        Detects candidate plate regions using image segmentation, Canny edge detection,
        and contour aspect-ratio filtering.
        """
        gray, edged = preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_PLATE_AREA or area > MAX_PLATE_AREA:
                continue
                
            # Polygon approximation
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # Rectangular bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Check aspect ratio typical for license plates
            if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
                candidates.append((x, y, w, h))
                
        return candidates

    def detect_plates(self, image, method="hybrid"):
        """
        Main entry point for plate detection.
        Methods: 'cascade', 'contour', 'hybrid'
        Returns list of (x, y, w, h, cropped_roi_enhanced)
        """
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        boxes = []

        if method in ["cascade", "hybrid"]:
            cascade_boxes = self.detect_via_cascade(gray)
            boxes.extend(cascade_boxes)

        if method in ["contour", "hybrid"] or len(boxes) == 0:
            contour_boxes = self.detect_via_contours(image)
            boxes.extend(contour_boxes)

        # Remove duplicate bounding boxes via Non-Maximum Suppression (NMS) or simple IOUs
        unique_boxes = self._remove_overlapping_boxes(boxes)
        
        results = []
        for (x, y, w, h) in unique_boxes:
            # Crop ROI with small margin
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.05)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)
            
            roi = image[y1:y2, x1:x2]
            enhanced_roi = enhance_plate_roi(roi)
            results.append((x, y, w, h, roi, enhanced_roi))

        return results

    def _remove_overlapping_boxes(self, boxes, overlap_threshold=0.3):
        """Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes."""
        if len(boxes) == 0:
            return []

        boxes_arr = np.array(boxes)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        w  = boxes_arr[:, 2]
        h  = boxes_arr[:, 3]
        x2 = x1 + w
        y2 = y1 + h

        areas = w * h
        order = np.argsort(y2)

        keep = []
        while order.size > 0:
            i = order[-1]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[:-1]])
            yy1 = np.maximum(y1[i], y1[order[:-1]])
            xx2 = np.minimum(x2[i], x2[order[:-1]])
            yy2 = np.minimum(y2[i], y2[order[:-1]])

            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            intersection = w_inter * h_inter

            iou = intersection / (areas[i] + areas[order[:-1]] - intersection)
            inds = np.where(iou <= overlap_threshold)[0]
            order = order[inds]

        return [tuple(boxes[k]) for k in keep]
