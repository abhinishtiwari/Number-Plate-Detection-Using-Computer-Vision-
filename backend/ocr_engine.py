"""Local OCR for cropped number plates.

Engine chain, best first:

1. **RapidOCR** (ONNX Runtime, ships its own models) - pip installable, CPU only,
   no torch. This is the engine that actually reads plates offline.
2. **EasyOCR** - used when a working torch install is present.
3. **Tesseract** - used when the ``tesseract`` binary is installed.
4. **Template matcher** - dependency-free last resort. Its output is only
   accepted when it validates as a real Indian registration, so it can no
   longer emit noise like "MM72M0" as if it were a plate.

Confidences are whatever the engine reported, rescaled to 0-100. There are no
floors, no defaults and no invented numbers: if an engine gives no score the
result is reported with the score it earned.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from . import plate_text
from .config import OCR_ENGINE_ORDER, OCR_MIN_CONFIDENCE, TESSERACT_CANDIDATE_PATHS
from .plate_text import PlateReading

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A block of text located by an OCR engine, in image coordinates."""

    box: tuple[int, int, int, int]
    text: str
    confidence: float


@dataclass
class PlateResult:
    """Outcome of reading one plate crop."""

    reading: PlateReading
    confidence: float = 0.0
    engine: str = "none"
    fragments: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.reading.text

    @property
    def is_valid(self) -> bool:
        return self.reading.is_valid


def _quad_to_box(quad: Sequence[Sequence[float]]) -> tuple[int, int, int, int]:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    x1, y1 = int(round(min(xs))), int(round(min(ys)))
    x2, y2 = int(round(max(xs))), int(round(max(ys)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _reading_order(regions: list[TextRegion]) -> list[TextRegion]:
    """Order OCR boxes by visual row, then left-to-right within each row."""
    rows: list[list[TextRegion]] = []
    for region in sorted(regions, key=lambda r: r.box[1] + r.box[3] / 2):
        _, y, _, h = region.box
        center = y + h / 2
        matching_row = None
        for row in rows:
            centers = [r.box[1] + r.box[3] / 2 for r in row]
            heights = [r.box[3] for r in row]
            if abs(center - float(np.mean(centers))) <= 0.55 * max(h, float(np.mean(heights))):
                matching_row = row
                break
        if matching_row is None:
            rows.append([region])
        else:
            matching_row.append(region)

    rows.sort(key=lambda row: min(r.box[1] for r in row))
    ordered: list[TextRegion] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda r: r.box[0]))
    return ordered


# --------------------------------------------------------------------- engines


class _RapidOCR:
    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415 - optional dep

        self._reader = RapidOCR()

    def detect_and_read(self, image: np.ndarray) -> list[TextRegion]:
        result, _ = self._reader(image)
        regions = []
        for entry in result or []:
            quad, text, score = entry[0], entry[1], entry[2]
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                continue
            regions.append(TextRegion(_quad_to_box(quad), str(text), confidence))
        return regions


class _EasyOCR:
    name = "easyocr"

    def __init__(self):
        import easyocr  # noqa: PLC0415 - optional dep

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def detect_and_read(self, image: np.ndarray) -> list[TextRegion]:
        regions = []
        for quad, text, score in self._reader.readtext(image):
            regions.append(TextRegion(_quad_to_box(quad), str(text), float(score)))
        return regions


class _Tesseract:
    name = "tesseract"
    _WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    _CONFIGS = (
        f"--oem 3 --psm 7 -c tessedit_char_whitelist={_WHITELIST}",
        f"--oem 3 --psm 6 -c tessedit_char_whitelist={_WHITELIST}",
        f"--oem 3 --psm 11 -c tessedit_char_whitelist={_WHITELIST}",
    )

    def __init__(self):
        import pytesseract  # noqa: PLC0415 - optional dep

        binary = next((p for p in TESSERACT_CANDIDATE_PATHS if os.path.exists(p)), None)
        if binary:
            pytesseract.pytesseract.tesseract_cmd = binary
        # Raises if the binary is missing, which is how availability is decided.
        pytesseract.get_tesseract_version()
        self._pytesseract = pytesseract

    def detect_and_read(self, image: np.ndarray) -> list[TextRegion]:
        regions: list[TextRegion] = []
        for config in self._CONFIGS:
            data = self._pytesseract.image_to_data(
                image, config=config, output_type=self._pytesseract.Output.DICT
            )
            found = False
            for i, raw_text in enumerate(data["text"]):
                text = (raw_text or "").strip()
                if not text:
                    continue
                try:
                    conf = float(data["conf"][i])
                except (TypeError, ValueError):
                    continue
                if conf < 0:
                    continue
                box = (int(data["left"][i]), int(data["top"][i]),
                       int(data["width"][i]), int(data["height"][i]))
                regions.append(TextRegion(box, text, conf / 100.0))
                found = True
            if found:
                break
        return regions


class _TemplateMatcher:
    """Dependency-free segment-and-match reader.

    Kept as a genuine last resort for air-gapped installs. It is deliberately
    conservative: the caller only accepts its output when the reading validates
    as a real Indian registration.
    """

    name = "template"
    _CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    _SIZE = (24, 36)  # (w, h) used for both templates and character crops

    def __init__(self):
        self._templates = self._build_templates()

    def _build_templates(self) -> dict[str, list[np.ndarray]]:
        fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX)
        templates: dict[str, list[np.ndarray]] = {}
        for char in self._CHARS:
            variants = []
            for font in fonts:
                canvas = np.zeros((72, 56), dtype=np.uint8)
                (tw, th), _ = cv2.getTextSize(char, font, 2.0, 3)
                org = ((56 - tw) // 2, (72 + th) // 2)
                cv2.putText(canvas, char, org, font, 2.0, 255, 3, cv2.LINE_AA)
                variants.append(self._normalise(canvas))
            templates[char] = variants
        return templates

    def _normalise(self, glyph: np.ndarray) -> np.ndarray:
        """Tight-crop to the ink and rescale, so stroke size stops mattering."""
        coords = cv2.findNonZero(glyph)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            glyph = glyph[y:y + h, x:x + w]
        if glyph.size == 0:
            glyph = np.zeros(self._SIZE[::-1], dtype=np.uint8)
        resized = cv2.resize(glyph, self._SIZE, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    @staticmethod
    def _binarise(gray: np.ndarray) -> list[np.ndarray]:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 9
        )
        variants = []
        for image in (otsu, adaptive):
            # Characters must end up white on black.
            variants.append(cv2.bitwise_not(image) if np.mean(image) > 127 else image)
        return variants

    def _segment(self, binary: np.ndarray) -> list[tuple[int, int, int, int]]:
        h_img, w_img = binary.shape
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 0.35 * h_img or h > 0.95 * h_img:
                continue
            if w < 0.02 * w_img or w > 0.25 * w_img:
                continue
            if not 0.12 <= w / h <= 1.15:
                continue
            boxes.append((x, y, w, h))
        boxes.sort(key=lambda b: b[0])

        # Drop boxes that overlap a kept box horizontally (nested contours).
        kept: list[tuple[int, int, int, int]] = []
        for box in boxes:
            if kept:
                px, _, pw, _ = kept[-1]
                if box[0] < px + pw * 0.6:
                    continue
            kept.append(box)
        return kept

    def _classify(self, crop: np.ndarray) -> tuple[str, float]:
        probe = self._normalise(crop)
        best_char, best_score = "", -1.0
        for char, variants in self._templates.items():
            for template in variants:
                score = float(cv2.matchTemplate(probe, template, cv2.TM_CCOEFF_NORMED)[0][0])
                if score > best_score:
                    best_score, best_char = score, char
        return best_char, best_score

    def detect_and_read(self, image: np.ndarray) -> list[TextRegion]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 40:
            scale = 40.0 / max(gray.shape[0], 1)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        best: list[TextRegion] = []
        best_score = 0.0
        for binary in self._binarise(gray):
            boxes = self._segment(binary)
            if len(boxes) < 6:
                continue
            chars, scores = [], []
            for x, y, w, h in boxes:
                char, score = self._classify(binary[y:y + h, x:x + w])
                if not char or score <= 0.25:
                    continue
                chars.append(char)
                scores.append(score)
            if len(chars) < 6:
                continue
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                h_img, w_img = binary.shape
                best = [TextRegion((0, 0, w_img, h_img), "".join(chars), mean_score)]
        return best


_ENGINE_FACTORIES = {
    _RapidOCR.name: _RapidOCR,
    _EasyOCR.name: _EasyOCR,
    _Tesseract.name: _Tesseract,
    _TemplateMatcher.name: _TemplateMatcher,
}

#: Engines that scan a whole frame reliably enough to be used as plate locators.
_FULL_FRAME_CAPABLE = {_RapidOCR.name, _EasyOCR.name}


class OCREngine:
    """Lazily initialised chain of local OCR engines."""

    def __init__(self, order: Sequence[str] = OCR_ENGINE_ORDER):
        self._order = [name for name in order if name in _ENGINE_FACTORIES]
        unknown = [name for name in order if name not in _ENGINE_FACTORIES]
        if unknown:
            logger.warning("Ignoring unknown OCR engine(s) in OCR_ENGINE_ORDER: %s",
                           ", ".join(unknown))
        self._engines: dict[str, object | None] = {}
        self._load_errors: dict[str, str] = {}

    # ------------------------------------------------------------ lifecycle

    def _engine(self, name: str):
        if name not in self._engines:
            try:
                self._engines[name] = _ENGINE_FACTORIES[name]()
                logger.info("OCR engine ready: %s", name)
            except Exception as exc:  # noqa: BLE001 - report, then fall through
                self._engines[name] = None
                self._load_errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("OCR engine '%s' unavailable - %s: %s", name, type(exc).__name__, exc)
        return self._engines[name]

    def warm_up(self) -> None:
        """Initialise engines eagerly so the first request is not slow."""
        for name in self._order:
            self._engine(name)

    def status(self) -> dict:
        """Report which engines loaded, for /health. No guessing in the UI."""
        self.warm_up()
        available = [n for n in self._order if self._engines.get(n) is not None]
        return {
            "configured_order": list(self._order),
            "available": available,
            "active": available[0] if available else None,
            "unavailable": dict(self._load_errors),
            "neural_engine_available": any(n in _FULL_FRAME_CAPABLE for n in available),
        }

    @property
    def full_frame_engine_name(self) -> str | None:
        for name in self._order:
            if name in _FULL_FRAME_CAPABLE and self._engine(name) is not None:
                return name
        return None

    # -------------------------------------------------------------- reading

    def find_text_regions(self, image: np.ndarray) -> list[TextRegion]:
        """Locate every text block in a full frame.

        Used to propose plate regions directly from text, which guarantees that
        a reported box is the box the text came from.
        """
        name = self.full_frame_engine_name
        if name is None or image is None or image.size == 0:
            return []
        engine = self._engine(name)
        try:
            return engine.detect_and_read(image)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Full-frame OCR pass failed (%s): %s", name, exc)
            return []

    @staticmethod
    def _preprocessed_variants(roi: np.ndarray):
        """Yield OCR crops lazily, cheapest/most faithful first.

        RapidOCR commonly succeeds on the original colour crop. Grayscale,
        upscaling, and CLAHE are therefore computed only when a previous attempt
        failed, while preserving every fallback used for difficult plates.
        """
        if roi is None or roi.size == 0:
            return

        if roi.ndim == 3:
            yield roi
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return

        target_h = 96
        if h < target_h:
            scale = target_h / h
            gray = cv2.resize(gray, (max(1, int(round(w * scale))), target_h),
                              interpolation=cv2.INTER_CUBIC)
        yield gray

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yield clahe.apply(gray)

    @staticmethod
    def preprocess(roi: np.ndarray) -> list[np.ndarray]:
        """Return all preprocessing variants for callers that need inspection."""
        return list(OCREngine._preprocessed_variants(roi))

    def read_plate(self, roi: np.ndarray) -> PlateResult:
        """Read one plate crop, trying each engine and image variant in turn."""
        if roi is None or roi.size == 0:
            return PlateResult(reading=plate_text.parse(""))

        # Variants are created on demand and cached only if another OCR engine
        # needs the same fallback image.
        variant_source = iter(self._preprocessed_variants(roi))
        variants: list[np.ndarray] = []
        variants_exhausted = False
        best_valid: PlateResult | None = None
        best_invalid: PlateResult | None = None

        for name in self._order:
            engine = self._engine(name)
            if engine is None:
                continue
            variant_index = 0
            while True:
                if variant_index == len(variants):
                    if variants_exhausted:
                        break
                    try:
                        variants.append(next(variant_source))
                    except StopIteration:
                        variants_exhausted = True
                        break
                variant = variants[variant_index]
                variant_index += 1
                try:
                    regions = engine.detect_and_read(variant)
                except Exception as exc:  # noqa: BLE001 - visible, then continue
                    logger.warning("OCR engine '%s' failed on a crop: %s: %s",
                                   name, type(exc).__name__, exc)
                    break

                usable = [r for r in regions if r.confidence >= OCR_MIN_CONFIDENCE]
                if not usable:
                    continue

                usable = _reading_order(usable)
                fragments = [r.text for r in usable]
                reading = plate_text.extract_best(fragments)
                confidence = round(float(np.mean([r.confidence for r in usable])) * 100.0, 1)

                if reading.is_valid and reading.plate_format in ("standard", "bharat"):
                    candidate_result = PlateResult(
                        reading=reading, confidence=confidence,
                        engine=name, fragments=fragments,
                    )
                    complete = (
                        reading.plate_format == "bharat"
                        or len(reading.number) == 4
                    )
                    if complete:
                        logger.info("%s read plate %s (%.1f%% confidence)",
                                    name, reading.text, confidence)
                        return candidate_result

                    # A 1-3 digit serial can be legal, but it can also be one
                    # row of a multi-line plate whose lower digits were missed.
                    # Try the remaining lazy variants before accepting it.
                    if (best_valid is None
                            or (len(reading.number), -reading.corrections, confidence)
                            > (len(best_valid.reading.number),
                               -best_valid.reading.corrections,
                               best_valid.confidence)):
                        best_valid = candidate_result
                    continue

                # Remember the best unvalidated reading, but never from the
                # template matcher - unvalidated template output is noise.
                if name != _TemplateMatcher.name and reading.text and best_invalid is None:
                    best_invalid = PlateResult(reading=reading, confidence=confidence,
                                              engine=name, fragments=fragments)

        if best_valid is not None:
            logger.info("%s read short-serial plate %s (%.1f%% confidence)",
                        best_valid.engine, best_valid.text, best_valid.confidence)
            return best_valid

        if best_invalid is not None:
            logger.info("OCR produced %r but it is not a valid Indian registration.",
                        best_invalid.text)
            return best_invalid

        logger.info("No OCR engine could read text from this crop.")
        return PlateResult(reading=plate_text.parse(""))


#: Process-wide instance.
ocr_engine = OCREngine()
