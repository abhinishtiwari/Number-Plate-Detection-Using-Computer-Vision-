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
            except Exception:
                pass
                
        if CASCADE_PATH.exists():
            self.cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

    def detect(self, image: np.ndarray):
        """
        Detects license plate bounding boxes [x, y, w, h] in image.
        Supports single or multiple plates.
        """
        if image is None:
            return []

        h_img, w_img = image.shape[:2]

        # 1. Try YOLO model if loaded
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(image, verbose=False)
                boxes = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                        boxes.append((int(x1), int(y1), w, h, conf))
                if boxes:
                    return boxes
            except Exception:
                pass

        # 2. Haar Cascade & Contour Aspect Ratio Detection (OpenCV Pipeline)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        boxes = []

        if self.cascade and not self.cascade.empty():
            cascade_plates = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 15)
            )
            for (x, y, w, h) in cascade_plates:
                boxes.append((int(x), int(y), int(w), int(h), 0.92))

        # Contour Edge Detection
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        for c in contours:
            area = cv2.contourArea(c)
            if area < 300 or area > 70000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 2.0 <= aspect_ratio <= 6.5:
                boxes.append((int(x), int(y), int(w), int(h), 0.88))

        # Perform Non-Maximum Suppression to remove duplicate bounding boxes
        unique_boxes = self._nms(boxes)
        return unique_boxes

    def _nms(self, boxes, iou_threshold=0.3):
        """Applies Non-Maximum Suppression to eliminate overlapping bounding boxes."""
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

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [(int(x1[k]), int(y1[k]), int(w[k]), int(h[k]), float(scores[k])) for k in keep]
