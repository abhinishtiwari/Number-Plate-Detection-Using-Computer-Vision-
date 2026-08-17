"""FastAPI service for the Number Plate AI pipeline.

Flow: upload -> plate localisation -> crop -> OpenCV preprocessing -> local OCR
-> plate grammar validation -> RTO dataset lookup -> JSON response.

Every field returned is derived from the uploaded media or the RTO CSV. Fields
that could not be resolved are ``null`` rather than a placeholder string, so the
frontend can say "not in dataset" instead of showing an invented city.
"""
from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .config import (
    ALLOWED_IMAGE_SUFFIXES,
    ALLOWED_VIDEO_SUFFIXES,
    CORS_ALLOWED_ORIGINS,
    FRONTEND_DIR,
    LOG_LEVEL,
    MAX_IMAGE_PIXELS,
    MAX_OCR_CANDIDATES,
    MAX_OCR_CORRECTIONS,
    MAX_UPLOAD_BYTES,
    OCR_MIN_CONFIDENCE,
    ONLY_VALID_PLATES,
    PROCESSING_TIMEOUT_SECONDS,
    REQUIRE_KNOWN_RTO,
    ROI_PAD_RATIO,
    SERVE_FRONTEND,
    VIDEO_FRAME_STRIDE,
    VIDEO_MAX_FRAMES_SCANNED,
)
from .detector import PlateCandidate, PlateDetector
from .ocr_engine import PlateResult, ocr_engine
from .plate_text import PlateReading, parse
from .rto_lookup import rto_engine

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

#: The detector shares the OCR engine so it can use the full-frame text pass to
#: anchor plate proposals on real text.
detector = PlateDetector(ocr=ocr_engine)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialise constrained native libraries and report capabilities."""
    cv2.setNumThreads(1)
    status = ocr_engine.status()
    logger.info("OCR engines available: %s", status["available"] or "none")
    if not status["neural_engine_available"]:
        logger.warning(
            "No neural OCR engine is installed, so plate text accuracy will be poor. "
            "Install one with: pip install rapidocr-onnxruntime"
        )
    logger.info("Plate detection sources: %s", ", ".join(detector.active_sources))
    logger.info("RTO dataset: %s (%d records, %d states/UTs)",
                rto_engine.dataset_path.name, rto_engine.record_count, rto_engine.state_count)
    yield


app = FastAPI(
    title="Number Plate AI API",
    version="2.0.0",
    description="Local ANPR pipeline: plate detection, OpenCV preprocessing, "
                "offline OCR and RTO dataset lookup.",
    lifespan=lifespan,
)

# A wildcard origin cannot be combined with credentials; browsers reject it and
# it would also let any site make credentialed calls. Credentials are only
# enabled when an explicit origin list is configured.
_allow_credentials = "*" not in CORS_ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ALLOWED_ORIGINS),
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ endpoints


@app.get("/health")
def health_check() -> dict:
    """Service, dataset and OCR engine status. Used by the frontend on load."""
    ocr_status = ocr_engine.status()
    return {
        "status": "ok",
        "rto_dataset": {
            "file": rto_engine.dataset_path.name,
            "records": rto_engine.record_count,
            "states": rto_engine.state_count,
        },
        "ocr": ocr_status,
        "detection_sources": detector.active_sources,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "pipeline_ready": bool(ocr_status["available"]),
    }


@app.get("/rto-dataset")
def get_rto_dataset() -> dict:
    """The full RTO dataset, for the reference table in the UI."""
    records = rto_engine.get_all_records()
    return {"count": len(records), "records": records}


@app.post("/detect")
def detect_plate(file: UploadFile = File(...)) -> dict:
    """Detect plates without blocking the ASGI event loop.

    FastAPI runs this synchronous endpoint in its thread pool, so health checks
    and Gunicorn heartbeats remain responsive while OpenCV/ONNX does CPU work.
    The bounded read prevents an oversized request from becoming a second large
    in-memory copy.
    """
    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()
    is_video = content_type.startswith("video/") or suffix in ALLOWED_VIDEO_SUFFIXES
    is_image = content_type.startswith("image/") or suffix in ALLOWED_IMAGE_SUFFIXES

    try:
        contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        file.file.close()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {limit_mb:.0f} MB limit.",
        )

    if is_video:
        return _process_video(contents, suffix or ".mp4")
    if not is_image:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload an image "
                   f"({', '.join(sorted(ALLOWED_IMAGE_SUFFIXES))}) or a video "
                   f"({', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}).",
        )

    try:
        with Image.open(BytesIO(contents)) as probe:
            width, height = probe.size
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.") from None

    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        max_megapixels = MAX_IMAGE_PIXELS / 1_000_000
        raise HTTPException(
            status_code=413,
            detail=f"Decoded image is too large; maximum is {max_megapixels:g} megapixels.",
        )

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

    try:
        plates = process_frame(image)
        return {
            "media_type": "image",
            "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "plates": plates,
            "plate_count": len(plates),
        }
    finally:
        del image, contents
        gc.collect()


# ------------------------------------------------------------------- pipeline


def process_frame(image: np.ndarray) -> list[dict]:
    """Run detection + OCR + RTO lookup over one frame.

    Only real number plates come back. Text that is not a valid Indian
    registration (badges, dealer stickers, watermarks) and regions where nothing
    could be read are dropped, unless ONLY_VALID_PLATES is disabled.
    """
    results: list[dict] = []

    for candidate in _rank_candidates(detector.detect(image)):
        # A valid, confident full-frame OCR result already came from this exact
        # region. Re-running OCR over several crop variants is duplicate work
        # and was the largest source of request time and native allocations.
        trusted = _trusted_text_hint(candidate)
        if trusted is not None:
            result, rto = trusted
        else:
            roi = _crop(image, candidate)
            if roi.size == 0:
                continue
            result = ocr_engine.read_plate(roi)
            result = _prefer_better_reading(result, candidate)
            rto = rto_engine.lookup(result.reading if result.text else None)

        if ONLY_VALID_PLATES:
            rejection = _rejection_reason(result.reading, rto)
            if rejection:
                logger.debug("Discarding region %s (%r): %s",
                             candidate.box, result.text, rejection)
                continue

        results.append(_build_plate_payload(candidate, result.reading, result, rto))

    results = _deduplicate(results)
    # Readable plates first, then higher confidence.
    results.sort(key=lambda p: (p["is_valid_format"], p["confidence"] or 0.0), reverse=True)
    return results


def _trusted_text_hint(candidate: PlateCandidate) -> tuple[PlateResult, dict] | None:
    """Reuse a strong full-frame OCR reading instead of invoking OCR again."""
    confidence = candidate.text_hint_confidence or 0.0
    if not candidate.text_hint or confidence < OCR_MIN_CONFIDENCE:
        return None

    reading = parse(candidate.text_hint)
    rto = rto_engine.lookup(reading if reading.text else None)
    if _rejection_reason(reading, rto) is not None:
        return None

    result = PlateResult(
        reading=reading,
        confidence=round(confidence * 100.0, 1),
        engine="rapidocr:text-box",
        fragments=[candidate.text_hint],
    )
    return result, rto


def _rejection_reason(reading: PlateReading, rto: dict) -> str | None:
    """Why this reading is not a number plate, or None when it is one."""
    if not reading.text:
        return "nothing readable"
    if not reading.is_valid:
        return "not an Indian registration format"
    if reading.corrections > MAX_OCR_CORRECTIONS:
        return f"needed {reading.corrections} character repairs"
    if REQUIRE_KNOWN_RTO and rto.get("match_level") not in ("exact", "national"):
        return f"RTO prefix is not in the dataset (match_level={rto.get('match_level')})"
    return None


def _rank_candidates(candidates: list[PlateCandidate]) -> list[PlateCandidate]:
    """Order candidates by how likely they are to be a plate, then cap the list."""
    ordered = sorted(
        candidates,
        key=lambda c: (
            c.source == "yolo",
            bool(c.text_hint),                 # a region that already produced text
            c.detector_confidence or 0.0,
            c.area,
        ),
        reverse=True,
    )
    if len(ordered) > MAX_OCR_CANDIDATES:
        logger.debug("Capping %d candidate regions to %d.", len(ordered), MAX_OCR_CANDIDATES)
    return ordered[:MAX_OCR_CANDIDATES]


def _prefer_better_reading(result, candidate: PlateCandidate):
    """Choose between the crop reading and the full-frame text-detection reading.

    Both describe the same region. The crop is usually better, but when the crop
    includes chrome trim above the plate the engine can return two overlapping
    text boxes that merge into a doubled reading. The tighter text-anchored read
    wins when it needs fewer repairs.
    """
    if not candidate.text_hint:
        return result

    from . import plate_text
    from .ocr_engine import PlateResult

    hint = plate_text.parse(candidate.text_hint)
    if not hint.is_valid:
        return result

    hint_confidence = round((candidate.text_hint_confidence or 0.0) * 100.0, 1)
    if result.is_valid:
        if result.reading.corrections < hint.corrections:
            return result
        # Equal repair counts: trust whichever read scored higher.
        if result.reading.corrections == hint.corrections and result.confidence >= hint_confidence:
            return result

    return PlateResult(
        reading=hint,
        confidence=hint_confidence,
        engine=f"{result.engine or 'ocr'}:text-box",
        fragments=[candidate.text_hint],
    )


def _deduplicate(plates: list[dict]) -> list[dict]:
    """Collapse repeated readings of the same plate, keeping the best one.

    Two candidate boxes over one plate can both produce a reading; without this
    the same vehicle is listed twice with slightly different text.
    """
    best: dict[str, dict] = {}
    unreadable: list[dict] = []
    for plate in plates:
        text = plate["text"]
        if not text:
            unreadable.append(plate)
            continue
        current = best.get(text)
        if current is None or _quality(plate) > _quality(current):
            best[text] = plate

    kept = list(best.values())
    # An unreadable box that overlaps a readable one is the same plate.
    for plate in unreadable:
        if not any(_boxes_overlap(plate["box"], other["box"]) for other in kept):
            kept.append(plate)
    return kept


def _quality(plate: dict) -> tuple:
    """Rank readings: valid format, then fewer corrections, then confidence."""
    return (
        plate["is_valid_format"],
        -plate["ocr_corrections"],
        plate["confidence"] or 0.0,
    )


def _boxes_overlap(a: list[int], b: list[int], threshold: float = 0.5) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = inter_w * inter_h
    smaller = min(aw * ah, bw * bh)
    return bool(smaller) and intersection / smaller > threshold


def _crop(image: np.ndarray, candidate: PlateCandidate) -> np.ndarray:
    pad_x = int(candidate.w * ROI_PAD_RATIO)
    pad_y = int(candidate.h * ROI_PAD_RATIO)
    x1 = max(0, candidate.x - pad_x)
    y1 = max(0, candidate.y - pad_y)
    x2 = min(image.shape[1], candidate.x + candidate.w + pad_x)
    y2 = min(image.shape[0], candidate.y + candidate.h + pad_y)
    return image[y1:y2, x1:x2]


def _build_plate_payload(candidate: PlateCandidate, reading: PlateReading,
                         result, rto: dict) -> dict:
    """Assemble the JSON for one plate.

    ``confidence`` is the OCR confidence when text was read, otherwise the
    detector's own probability when it has one, otherwise ``null``. It is never
    a synthesised value.
    """
    if result.text and result.confidence > 0:
        confidence = result.confidence
        confidence_basis = f"ocr:{result.engine}"
    elif candidate.detector_confidence is not None:
        confidence = round(candidate.detector_confidence * 100.0, 1)
        confidence_basis = f"detector:{candidate.source}"
    else:
        confidence = None
        confidence_basis = "unscored"

    return {
        "box": list(candidate.box),
        "text": result.text or None,
        "is_valid_format": bool(reading.is_valid),
        "plate_format": reading.plate_format,
        "ocr_corrections": reading.corrections,
        "raw_ocr_text": reading.raw or None,
        "confidence": confidence,
        "confidence_basis": confidence_basis,
        "ocr_engine": result.engine,
        "detection_source": candidate.source,
        "detector_confidence": (
            round(candidate.detector_confidence * 100.0, 1)
            if candidate.detector_confidence is not None else None
        ),
        "state_name": rto.get("state_name"),
        "state_code": rto.get("state_code") or None,
        "full_rto_code": rto.get("full_rto_code"),
        "city": rto.get("city") or None,
        "rto_match_level": rto.get("match_level"),
    }


def _process_video(contents: bytes, suffix: str) -> dict:
    """Sample a bounded number of frames within the request time budget."""
    tmp_path = None
    capture = None
    deadline = time.monotonic() + PROCESSING_TIMEOUT_SECONDS
    timed_out = False
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Drop the upload copy before native video decoding starts.
        del contents
        capture = cv2.VideoCapture(tmp_path)
        if not capture.isOpened():
            raise HTTPException(status_code=400,
                                detail="Could not decode the video file. Try MP4 (H.264).")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        best_by_plate: dict[str, dict] = {}
        unreadable: list[dict] = []
        frame_index = 0
        scanned = 0

        while scanned < VIDEO_MAX_FRAMES_SCANNED:
            if time.monotonic() >= deadline:
                timed_out = True
                break

            # Seek directly to sampled frames instead of decoding every frame in
            # between. This keeps free-tier video requests bounded.
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break

            scanned += 1
            for plate in process_frame(frame):
                plate["frame_index"] = frame_index
                plate["timestamp_seconds"] = round(frame_index / fps, 2) if fps > 0 else None
                key = plate["text"]
                if not key:
                    if len(unreadable) < 5:
                        unreadable.append(plate)
                    continue
                previous = best_by_plate.get(key)
                if previous is None:
                    plate["times_seen"] = 1
                    best_by_plate[key] = plate
                else:
                    previous["times_seen"] += 1
                    if (plate["confidence"] or 0) > (previous["confidence"] or 0):
                        plate["times_seen"] = previous["times_seen"]
                        best_by_plate[key] = plate
            del frame
            frame_index += VIDEO_FRAME_STRIDE

        plates = sorted(best_by_plate.values(),
                        key=lambda p: (p["is_valid_format"], p["confidence"] or 0.0),
                        reverse=True)
        if not plates:
            plates = unreadable[:5]

        return {
            "media_type": "video",
            "frames_scanned": scanned,
            "frame_stride": VIDEO_FRAME_STRIDE,
            "processing_limited": timed_out or scanned >= VIDEO_MAX_FRAMES_SCANNED,
            "fps": round(fps, 2) if fps > 0 else None,
            "plates": plates,
            "plate_count": len(plates),
        }
    finally:
        if capture is not None:
            capture.release()
        gc.collect()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as exc:
                logger.warning("Could not delete temp video %s: %s", tmp_path, exc)


# --------------------------------------------------------------- static files

# Mounted last so it cannot shadow the API routes above.
if SERVE_FRONTEND and FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info("Serving the dashboard from %s", FRONTEND_DIR)
else:
    if SERVE_FRONTEND:
        logger.warning("Frontend directory %s not found; running API-only.", FRONTEND_DIR)
    else:
        logger.info("SERVE_FRONTEND is off; running API-only.")

    @app.get("/", include_in_schema=False)
    def service_info() -> dict:
        """Root metadata for an API-only deployment, so "/" is not a bare 404."""
        return {
            "service": "Number Plate AI API",
            "version": app.version,
            "mode": "api-only",
            "endpoints": ["/health", "/rto-dataset", "/detect", "/docs"],
            "allowed_origins": list(CORS_ALLOWED_ORIGINS),
        }
