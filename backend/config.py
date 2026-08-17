"""Central configuration for the Number Plate AI backend.

Every tunable lives here so that no detection/OCR threshold is buried as a
magic number inside the pipeline. All values can be overridden with
environment variables, which is what the Docker image and CI use.
"""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------- data files

#: Candidate locations for the RTO master dataset, highest priority first.
#: The dataset is a single CSV; there is deliberately no second copy to drift.
RTO_DATASET_CANDIDATES: tuple[Path, ...] = tuple(
    p for p in (
        Path(_env_str("RTO_DATASET_PATH", "")) if os.environ.get("RTO_DATASET_PATH") else None,
        PROJECT_ROOT / "India_RTO_Registration_Dataset_New.csv",
        BACKEND_DIR / "data" / "India_RTO_Registration_Dataset_New.csv",
    ) if p is not None
)

CASCADE_PATH = BACKEND_DIR / "cascades" / "haarcascade_russian_plate_number.xml"

#: Static dashboard. Serving it from the API means the browser talks to the same
#: origin, so there is no CORS setup and no http-from-https mixed content.
FRONTEND_DIR = Path(_env_str("FRONTEND_DIR", str(PROJECT_ROOT / "frontend")))

#: Optional YOLO weights. When absent the geometric detector is used instead.
YOLO_WEIGHTS_PATH = Path(_env_str("YOLO_WEIGHTS_PATH", str(BACKEND_DIR / "models" / "plate_yolo.pt")))

# ------------------------------------------------------------------- uploads

#: Hard ceiling on upload size. The deployed free-tier service keeps this low
#: because encoded size is not the same as decoded image memory.
MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024)

#: Reject compressed images that expand beyond this many pixels before OpenCV
#: decodes them. Twelve megapixels is already well above the detector input size.
MAX_IMAGE_PIXELS = _env_int("MAX_IMAGE_PIXELS", 12_000_000)

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ----------------------------------------------------------------- detection

#: Images larger than this on the long edge are downscaled before detection and
#: the resulting boxes are scaled back. Keeps thresholds resolution independent.
DETECTION_MAX_EDGE = _env_int("DETECTION_MAX_EDGE", 800)

#: Plate geometry limits. Ratios are relative to image area/width so that the
#: same numbers work for a 480p frame and a 12 MP phone photo.
MIN_PLATE_ASPECT = _env_float("MIN_PLATE_ASPECT", 1.6)
MAX_PLATE_ASPECT = _env_float("MAX_PLATE_ASPECT", 8.0)
MIN_PLATE_AREA_RATIO = _env_float("MIN_PLATE_AREA_RATIO", 0.00015)
MAX_PLATE_AREA_RATIO = _env_float("MAX_PLATE_AREA_RATIO", 0.45)
MIN_PLATE_WIDTH_PX = _env_int("MIN_PLATE_WIDTH_PX", 48)
MIN_PLATE_HEIGHT_PX = _env_int("MIN_PLATE_HEIGHT_PX", 16)

NMS_IOU_THRESHOLD = _env_float("NMS_IOU_THRESHOLD", 0.3)

#: Padding added around a candidate box before it is handed to OCR.
ROI_PAD_RATIO = _env_float("ROI_PAD_RATIO", 0.06)

#: Upper bound on regions sent to OCR per frame. The geometric detector proposes
#: many boxes on chrome trim and reflections; without a cap every one costs an
#: OCR pass. Candidates are ranked before the cap is applied.
MAX_OCR_CANDIDATES = _env_int("MAX_OCR_CANDIDATES", 6)

#: When true (the default) only readings that validate as a real Indian
#: registration are returned. This is what keeps manufacturer badges ("FORD"),
#: dealer stickers ("Planet Ford"), site watermarks and unreadable trim boxes
#: out of the results. Set to 0 to see every candidate region for debugging.
ONLY_VALID_PLATES = _env_flag("ONLY_VALID_PLATES", True)

#: Require the state+RTO prefix to exist in the dataset before a reading is
#: reported. Character repair can turn noise into something that merely *looks*
#: like a plate: the watermark "Team-BHP.com" read as "TG8BHPCO" and repaired to
#: "TG88HPC0", which is grammatical but TG-88 is not a real RTO. Bharat-series
#: and defence plates are exempt because they have no RTO office.
REQUIRE_KNOWN_RTO = _env_flag("REQUIRE_KNOWN_RTO", True)

#: Reject readings that needed more than this many ambiguous-character repairs.
#: A genuine plate needs none or one; several repairs means the text was probably
#: not a plate to begin with.
MAX_OCR_CORRECTIONS = _env_int("MAX_OCR_CORRECTIONS", 2)

# ----------------------------------------------------------------------- OCR

#: Minimum per-word confidence (0-1) accepted from a neural OCR engine.
OCR_MIN_CONFIDENCE = _env_float("OCR_MIN_CONFIDENCE", 0.30)

#: Engine preference order. Unavailable engines are skipped at runtime.
OCR_ENGINE_ORDER = tuple(
    e.strip() for e in _env_str("OCR_ENGINE_ORDER", "rapidocr,easyocr,tesseract,template").split(",") if e.strip()
)

#: Extra Tesseract binary locations probed on Windows installs.
TESSERACT_CANDIDATE_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
)

# --------------------------------------------------------------------- video

VIDEO_FRAME_STRIDE = _env_int("VIDEO_FRAME_STRIDE", 30)
VIDEO_MAX_FRAMES_SCANNED = _env_int("VIDEO_MAX_FRAMES_SCANNED", 12)

#: Stop synchronous media work before the hosting proxy/worker timeout. This is
#: checked between sampled video frames; images get the same frontend deadline.
PROCESSING_TIMEOUT_SECONDS = _env_float("PROCESSING_TIMEOUT_SECONDS", 75.0)

# ----------------------------------------------------------------------- API

#: Comma separated list of allowed browser origins. "*" disables credentials.
CORS_ALLOWED_ORIGINS = tuple(
    o.strip() for o in _env_str("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()
)

#: Serve the dashboard from this process. This is enabled for the combined
#: Render service so browser and API requests share one origin.
SERVE_FRONTEND = _env_flag("SERVE_FRONTEND", True)

LOG_LEVEL = _env_str("LOG_LEVEL", "INFO").upper()
