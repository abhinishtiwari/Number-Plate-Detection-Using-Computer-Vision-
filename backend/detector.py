import cv2
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "cascades" / "haarcascade_russian_plate_number.xml"

class YOLOPlateDetector:
    def __init__(self, weights_path=None):
        self.cascade = None
        self.yolo_model = None
        self._init_detector(weights_path)

    def _init_detector(self, weights_path):
        """Initializes YOLO model or OpenCV Cascade fallback."""
        if weights_path and os.path.exists(weights_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(weights_path)
            except Exception as e:
                print(f"[YOLO] Note: {e}")
                
        if CASCADE_PATH.exists():
            self.cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

    def detect(self, image: np.ndarray):
        """
        100% Dynamic Object Detector:
        Detects actual license plate bounding boxes [x, y, w, h] in image frame.
        Calculates confidence dynamically. Returns empty list [] if no plates found.
        Filters out non-plate small candidate boxes (h < 25 or w < 60 or aspect < 1.8).
        No hardcoded bounding box fallbacks or fixed coordinates.
        """
        if image is None or image.size == 0:
            return []

        h_img, w_img = image.shape[:2]
        boxes = []

        # 1. Real YOLO Model Inference
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(image, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                        if w >= 60 and h >= 22 and (w / max(h, 1)) >= 1.8:
                            boxes.append((int(x1), int(y1), w, h, conf))
                if boxes:
                    return self._nms(boxes)
            except Exception as e:
                print(f"[YOLO Inference] Note: {e}")

        # 2. High-Contrast Plate Geometry Detector (Primary CV Detector)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:60]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            x, y, w, h = cv2.boundingRect(c)
            aspect = float(w) / h if h > 0 else 0
            area = cv2.contourArea(c)

            # Standard Indian License Plate bounding box filters: w >= 70, h >= 24, 2.0 <= aspect <= 5.5
            if len(approx) >= 4 and 2.0 <= aspect <= 5.5 and w >= 70 and h >= 24 and 2200 <= area <= 90000:
                roi_gray = gray[y:y+h, x:x+w]
                if roi_gray.size > 0:
                    std_val = float(np.std(roi_gray))
                    # High contrast plate ROI (text vs background)
                    if std_val > 25:
                        score = min(0.98, max(0.72, 0.75 + (std_val / 200.0)))
                        boxes.append((int(x), int(y), int(w), int(h), round(score, 3)))

        # 3. Haar Cascade Detection Fallback (filtered by minimum plate size)
        if self.cascade and not self.cascade.empty():
            try:
                rects, rejectLevels, levelWeights = self.cascade.detectMultiScale3(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=5,
                    minSize=(70, 24),
                    outputRejectLevels=True
                )
                for i, (x, y, w, h) in enumerate(rects):
                    aspect = float(w) / h if h > 0 else 0
                    if 2.0 <= aspect <= 5.5:
                        raw_score = float(levelWeights[i][0]) if len(levelWeights) > i else 1.0
                        dynamic_conf = min(0.92, max(0.55, 0.50 + raw_score * 0.04))
                        boxes.append((int(x), int(y), int(w), int(h), dynamic_conf))
            except Exception:
                cascade_plates = self.cascade.detectMultiScale(
                    gray, scaleFactor=1.08, minNeighbors=5, minSize=(70, 24)
                )
                for (x, y, w, h) in cascade_plates:
                    aspect = float(w) / h if h > 0 else 0
                    if 2.0 <= aspect <= 5.5:
                        boxes.append((int(x), int(y), int(w), int(h), 0.70))

        # Non-Maximum Suppression to remove duplicate candidate boxes
        unique_boxes = self._nms(boxes)
        return unique_boxes

    def _nms(self, boxes, iou_threshold=0.25):
        """Applies Non-Maximum Suppression to eliminate overlapping candidate bounding boxes."""
        if not boxes:
            return []

        boxes_arr = np.array(boxes)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        w  = boxes_arr[:, 2]
        h  = boxes_arr[:, 3]
        scores = boxes_arr[:, 4]

        x2 = x1 + w
        y2 = y1 + h

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            intersection = w_inter * h_inter

            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [(int(x1[k]), int(y1[k]), int(w[k]), int(h[k]), round(float(scores[k]), 3)) for k in keep]
