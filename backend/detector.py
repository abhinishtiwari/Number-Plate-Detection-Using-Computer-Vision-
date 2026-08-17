"""Number plate localisation.

Three proposal sources, merged with non-maximum suppression:

1. **YOLO** - used only when real weights are present on disk. The class no
   longer pretends to be a YOLO detector when it is not; ``active_sources``
   reports exactly what ran.
2. **Text-anchored proposals** - a full-frame OCR pass returns text boxes, and
   the plate-shaped ones become candidates. Because the box comes from the text
   itself, the box drawn on screen always corresponds to the text that was read.
3. **Geometric CV** - contour + Haar cascade candidates, filtered by ratios of
   the image size rather than absolute pixel counts, so the same thresholds
   work on a 480p frame and a 12 MP photo.

Confidence semantics: YOLO boxes carry the model probability. Geometric boxes
carry ``None``, because edge-contrast has no probabilistic meaning; the API
reports OCR confidence in that case instead of inventing a detector score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import (
    CASCADE_PATH,
    DETECTION_MAX_EDGE,
    MAX_PLATE_AREA_RATIO,
    MAX_PLATE_ASPECT,
    MIN_PLATE_AREA_RATIO,
    MIN_PLATE_ASPECT,
    MIN_PLATE_HEIGHT_PX,
    MIN_PLATE_WIDTH_PX,
    NMS_IOU_THRESHOLD,
    YOLO_WEIGHTS_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class PlateCandidate:
    """A proposed plate region in original-image coordinates."""

    x: int
    y: int
    w: int
    h: int
    source: str
    detector_confidence: float | None = None
    """Model probability (0-1). ``None`` for geometric proposals, which have no
    calibrated score - the API must not present a fabricated number."""

    text_hint: str = ""
    """Text already read from this region by the full-frame OCR pass, if any."""

    text_hint_confidence: float | None = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def area(self) -> int:
        return self.w * self.h


class PlateDetector:
    """Locates candidate number plate regions in a frame."""

    def __init__(self, weights_path: Path | str | None = None, ocr=None):
        self._ocr = ocr
        self._yolo = None
        self._cascade = None
        self._sources: list[str] = []
        self._load_yolo(Path(weights_path) if weights_path else YOLO_WEIGHTS_PATH)
        self._load_cascade()

    # -------------------------------------------------------------- loading

    def _load_yolo(self, weights: Path) -> None:
        if not weights.is_file():
            logger.info(
                "No YOLO weights at %s - using text-anchored and geometric detection. "
                "Train weights with backend/dataset/data.yaml to enable YOLO.", weights,
            )
            return
        try:
            from ultralytics import YOLO  # noqa: PLC0415 - optional dep

            self._yolo = YOLO(str(weights))
            self._sources.append("yolo")
            logger.info("YOLO plate detector loaded from %s", weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO weights found at %s but could not be loaded: %s: %s",
                           weights, type(exc).__name__, exc)

    def _load_cascade(self) -> None:
        if not hasattr(cv2, "CascadeClassifier"):
            # Some OpenCV builds ship without the legacy cascade API.
            logger.info("This OpenCV build has no CascadeClassifier; skipping Haar proposals.")
            return
        if not CASCADE_PATH.is_file():
            logger.warning("Haar cascade missing at %s; geometric detection will use "
                           "contours only.", CASCADE_PATH)
            return
        cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
        if cascade.empty():
            logger.warning("Haar cascade at %s failed to load.", CASCADE_PATH)
            return
        self._cascade = cascade

    @property
    def active_sources(self) -> list[str]:
        sources = list(self._sources)
        if self._ocr is not None and self._ocr.full_frame_engine_name:
            sources.append(f"text:{self._ocr.full_frame_engine_name}")
        sources.append("contour")
        if self._cascade is not None:
            sources.append("haar")
        return sources

    # ------------------------------------------------------------ filtering

    @staticmethod
    def _plausible(w: int, h: int, image_area: int) -> bool:
        if w < MIN_PLATE_WIDTH_PX or h < MIN_PLATE_HEIGHT_PX:
            return False
        aspect = w / h
        if not MIN_PLATE_ASPECT <= aspect <= MAX_PLATE_ASPECT:
            return False
        area_ratio = (w * h) / image_area if image_area else 0.0
        return MIN_PLATE_AREA_RATIO <= area_ratio <= MAX_PLATE_AREA_RATIO

    # ------------------------------------------------------------- detection

    def detect(self, image: np.ndarray) -> list[PlateCandidate]:
        """Return de-duplicated plate candidates for ``image``."""
        if image is None or image.size == 0:
            return []

        working, scale = self._downscale(image)
        h_img, w_img = working.shape[:2]
        image_area = h_img * w_img

        candidates: list[PlateCandidate] = []
        candidates.extend(self._from_yolo(working, image_area))
        candidates.extend(self._from_text(working, image_area))
        candidates.extend(self._from_contours(working, image_area))
        candidates.extend(self._from_cascade(working, image_area))

        merged = self._non_max_suppression(candidates)
        if scale != 1.0:
            merged = [self._rescale(c, 1.0 / scale) for c in merged]
        return [self._clamp(c, image.shape) for c in merged]

    @staticmethod
    def _downscale(image: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= DETECTION_MAX_EDGE:
            return image, 1.0
        scale = DETECTION_MAX_EDGE / longest
        resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        return resized, scale

    @staticmethod
    def _rescale(candidate: PlateCandidate, factor: float) -> PlateCandidate:
        return PlateCandidate(
            x=int(round(candidate.x * factor)),
            y=int(round(candidate.y * factor)),
            w=int(round(candidate.w * factor)),
            h=int(round(candidate.h * factor)),
            source=candidate.source,
            detector_confidence=candidate.detector_confidence,
            text_hint=candidate.text_hint,
            text_hint_confidence=candidate.text_hint_confidence,
        )

    @staticmethod
    def _clamp(candidate: PlateCandidate, shape) -> PlateCandidate:
        h_img, w_img = shape[:2]
        x = max(0, min(candidate.x, w_img - 1))
        y = max(0, min(candidate.y, h_img - 1))
        candidate.x, candidate.y = x, y
        candidate.w = max(1, min(candidate.w, w_img - x))
        candidate.h = max(1, min(candidate.h, h_img - y))
        return candidate

    def _from_yolo(self, image: np.ndarray, image_area: int) -> list[PlateCandidate]:
        if self._yolo is None:
            return []
        try:
            results = self._yolo(image, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO inference failed: %s: %s", type(exc).__name__, exc)
            return []

        candidates = []
        for result in results:
            for box in getattr(result, "boxes", []) or []:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                w, h = int(round(x2 - x1)), int(round(y2 - y1))
                if self._plausible(w, h, image_area):
                    candidates.append(PlateCandidate(
                        x=int(round(x1)), y=int(round(y1)), w=w, h=h,
                        source="yolo", detector_confidence=float(box.conf[0]),
                    ))
        return candidates

    def _from_text(self, image: np.ndarray, image_area: int) -> list[PlateCandidate]:
        """Turn plate-shaped text boxes into candidates."""
        if self._ocr is None:
            return []

        candidates = []
        for region in self._ocr.find_text_regions(image):
            x, y, w, h = region.box
            # Pad, because a text box hugs the glyphs and clips the plate edge.
            pad_x, pad_y = int(w * 0.08), int(h * 0.30)
            x, y = max(0, x - pad_x), max(0, y - pad_y)
            w, h = w + 2 * pad_x, h + 2 * pad_y
            if not self._plausible(w, h, image_area):
                continue
            candidates.append(PlateCandidate(
                x=x, y=y, w=w, h=h, source="text",
                text_hint=region.text, text_hint_confidence=region.confidence,
            ))
        return candidates

    def _from_contours(self, image: np.ndarray, image_area: int) -> list[PlateCandidate]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)
        # Close gaps so a plate border becomes one contour.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        candidates = []
        for source in (edges, closed):
            contours, _ = cv2.findContours(source, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not self._plausible(w, h, image_area):
                    continue
                roi = gray[y:y + h, x:x + w]
                # A plate carries high-contrast glyphs; flat regions are not plates.
                if roi.size == 0 or float(np.std(roi)) < 22.0:
                    continue
                candidates.append(PlateCandidate(x=x, y=y, w=w, h=h, source="contour"))
        return candidates

    def _from_cascade(self, image: np.ndarray, image_area: int) -> list[PlateCandidate]:
        if self._cascade is None:
            return []
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            rects = self._cascade.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=4,
                minSize=(MIN_PLATE_WIDTH_PX, MIN_PLATE_HEIGHT_PX),
            )
        except cv2.error as exc:
            logger.warning("Haar cascade detection failed: %s", exc)
            return []

        return [
            PlateCandidate(x=int(x), y=int(y), w=int(w), h=int(h), source="haar")
            for (x, y, w, h) in rects
            if self._plausible(int(w), int(h), image_area)
        ]

    # --------------------------------------------------------------- merging

    #: A smaller box sitting mostly inside a larger one is the same plate, even
    #: when IoU is low. Without this, a tight text box and the plate border box
    #: were both reported and the same plate appeared twice.
    _CONTAINMENT_THRESHOLD = 0.65

    def _non_max_suppression(self, candidates: list[PlateCandidate]) -> list[PlateCandidate]:
        """Keep one candidate per plate region.

        A trusted YOLO box wins outright. Otherwise the larger box wins, because
        it gives OCR more of the plate to work with. When a text-anchored box is
        absorbed, its already-read text is carried over to the surviving box so
        no information is lost.
        """
        if not candidates:
            return []

        ordered = sorted(
            candidates,
            key=lambda c: (
                1 if c.source == "yolo" else 0,
                c.detector_confidence if c.detector_confidence is not None else 0.0,
                c.area,
            ),
            reverse=True,
        )

        kept: list[PlateCandidate] = []
        for candidate in ordered:
            duplicate_of = next(
                (k for k in kept
                 if self._iou(candidate, k) > NMS_IOU_THRESHOLD
                 or self._containment(candidate, k) > self._CONTAINMENT_THRESHOLD),
                None,
            )
            if duplicate_of is None:
                kept.append(candidate)
            elif candidate.text_hint and not duplicate_of.text_hint:
                duplicate_of.text_hint = candidate.text_hint
                duplicate_of.text_hint_confidence = candidate.text_hint_confidence
        return kept

    @staticmethod
    def _intersection(a: PlateCandidate, b: PlateCandidate) -> int:
        inter_w = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        inter_h = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        return inter_w * inter_h

    @classmethod
    def _iou(cls, a: PlateCandidate, b: PlateCandidate) -> float:
        intersection = cls._intersection(a, b)
        if intersection == 0:
            return 0.0
        union = a.area + b.area - intersection
        return intersection / union if union else 0.0

    @classmethod
    def _containment(cls, a: PlateCandidate, b: PlateCandidate) -> float:
        """Fraction of the smaller box that lies inside the larger one."""
        intersection = cls._intersection(a, b)
        smaller = min(a.area, b.area)
        return intersection / smaller if smaller else 0.0
