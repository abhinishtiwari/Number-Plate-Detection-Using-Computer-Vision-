# Number Plate AI — local ANPR pipeline

Detects Indian number plates in images and videos, reads the registration with a
local OCR engine, and resolves it to a state, RTO code and city using a local
CSV dataset.

Everything runs on your machine. No OpenAI, Gemini, paid API or cloud service is
involved, and no result is ever hard-coded or faked: an unreadable plate is
reported as unreadable, and an RTO number that is not in the dataset is reported
as not in the dataset.

---

## Pipeline

```
Upload (image / video)
        │
        ▼
Plate localisation ─────── YOLO weights (optional)
        │                  Text-anchored boxes (OCR text detection)
        │                  Contour + Haar cascade proposals
        ▼                  → merged with NMS + containment
Crop + pad
        │
        ▼
OpenCV preprocessing ───── grayscale → upscale → CLAHE → sharpen
        │
        ▼
Local OCR ──────────────── RapidOCR → EasyOCR → Tesseract → template matcher
        │
        ▼
Plate grammar validation ─ repair ambiguous chars only where the format demands,
        │                  reject readings whose state code does not exist
        ▼
RTO lookup ─────────────── India_RTO_Registration_Dataset_New.csv
        │
        ▼
JSON result → dashboard
```

### Why detection and OCR are linked

The OCR engine's text-detection pass doubles as a plate locator. Because the box
comes from the text itself, the box drawn on the image is always the region the
text was read from. Contour and Haar proposals still run, and a nested duplicate
is merged into the larger box so one plate is reported once.

### Only plates come back

A photo of a car bumper contains far more text than the plate: a manufacturer
badge, a dealer sticker, a site watermark, and a dozen high-contrast chrome
edges. A region is reported only when all of these hold:

1. the text validates against an Indian registration grammar,
2. it needed no more than `MAX_OCR_CORRECTIONS` ambiguous-character repairs, and
3. its state+RTO prefix exists in the dataset (Bharat-series and defence plates
   are exempt, having no RTO office).

Rule 3 matters more than it looks. Character repair can turn noise into
something merely plate-shaped: the watermark "Team-BHP.com" was read as
`TG8BHPCO` and repaired into `TG88HPC0`, which is grammatical — but TG-88 is not
a real RTO, so it is dropped. Set `ONLY_VALID_PLATES=0` to see every candidate
region with its reading when debugging.

---

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000/> — the API serves the dashboard from `frontend/`,
so the browser talks to the same origin and there is no CORS setup.

> Run uvicorn from the repository root as `backend.main:app`. `cd backend &&
> uvicorn main:app` fails, because the modules use package-relative imports.

### Command line

```bash
python main.py --image samples/car.jpg --output output/car.jpg
python main.py --video clip.mp4 --csv output/detections.csv
python main.py --image car.jpg --json
```

### Tests

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

The suite covers dataset integrity, plate grammar, detection at several
resolutions, and the full upload → JSON path. The end-to-end tests skip
automatically when no neural OCR engine is installed.

---

## OCR engines

| Engine | Install | Notes |
| --- | --- | --- |
| **RapidOCR** (default) | `pip install rapidocr-onnxruntime` | ONNX Runtime, CPU only, models bundled. Recommended. |
| EasyOCR | `pip install easyocr` | Needs a working PyTorch build. |
| Tesseract | `pip install pytesseract` + system `tesseract` | Needs the binary on PATH. |
| Template matcher | built in | Dependency-free last resort. Only used when its reading validates as a real registration, so it cannot emit noise as a plate. |

`GET /health` reports which engines actually loaded and why the others did not.
The dashboard shows this, so the UI never implies an engine that is missing.

Change the order with `OCR_ENGINE_ORDER=rapidocr,tesseract,template`.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Dataset stats, OCR engine availability, detection sources, upload limit. |
| `GET` | `/rto-dataset` | Full RTO dataset for the reference table. |
| `POST` | `/detect` | Multipart `file` upload. Returns detected plates. |

### `/detect` response (image)

```json
{
  "media_type": "image",
  "image_size": { "width": 860, "height": 520 },
  "plate_count": 1,
  "plates": [
    {
      "box": [230, 275, 445, 148],
      "text": "MH12DE1433",
      "is_valid_format": true,
      "plate_format": "standard",
      "ocr_corrections": 0,
      "raw_ocr_text": "MH12DE1433",
      "confidence": 81.8,
      "confidence_basis": "ocr:rapidocr",
      "ocr_engine": "rapidocr",
      "detection_source": "haar",
      "detector_confidence": null,
      "state_name": "Maharashtra",
      "state_code": "MH",
      "full_rto_code": "MH-12",
      "city": "Pune",
      "rto_match_level": "exact"
    }
  ]
}
```

Fields worth knowing:

- **`confidence` / `confidence_basis`** — the OCR engine's own score when text was
  read (`ocr:<engine>`), or the model probability when a plate was located but
  not read (`detector:<source>`), or `null` when nothing produced a calibrated
  score (`unscored`). It is never a synthesised blend or a floor value.
- **`ocr_corrections`** — how many ambiguous characters were repaired. `0` means
  the engine's reading was already a legal registration.
- **`rto_match_level`** — `exact` (prefix in the dataset), `state` (state known,
  RTO number absent), `national` (Bharat series), or `none`.
- **`is_valid_format`** — whether the text matches an Indian registration grammar.
  Text that does not is still returned, flagged, and never mapped to a state.

Video responses add `frames_scanned`, `frame_stride`, `fps`, and per plate
`frame_index`, `timestamp_seconds` and `times_seen`. Clicking a video result in
the dashboard seeks the player to that frame.

---

## RTO dataset

`India_RTO_Registration_Dataset_New.csv` at the repository root is the single
source of truth — 1,146 rows covering 38 state/UT codes.

```csv
state_code,state_name,rto_code,full_rto_code,registration_prefix,city
MH,Maharashtra,12,MH-12,MH12,Pune
TS,Telangana,09,TS-09,TS09,Hyderabad Central (Khairatabad)
```

Included beyond the original file: Telangana under both `TS` and the current
`TG` code, Ladakh (`LA`), the Andhra Pradesh codes that were absent (16–19,
28–29, 33–34, 39, 40), and factual labels for the 16 rows that had an empty city
(for example `BR20` is "Not allotted (transferred to Jharkhand, 2000)").

This is a **lookup table, not training data**. The YOLO dataset config lives
separately in `backend/dataset/data.yaml`.

Point the app at a different file with `RTO_DATASET_PATH=/path/to/file.csv`. A
missing or malformed dataset raises at startup instead of silently answering
"Unknown" to every request. Individual bad rows are skipped with a warning.

---

## Optional: train YOLO weights

Detection works without YOLO. To add it, label plates in the layout described by
`backend/dataset/data.yaml`, then:

```bash
yolo task=detect mode=train model=yolov8n.pt data=backend/dataset/data.yaml epochs=50 imgsz=640
```

Put the resulting `best.pt` at `backend/models/plate_yolo.pt` (or set
`YOLO_WEIGHTS_PATH`) and install `ultralytics`. `/health` will then list `yolo`
in `detection_sources`.

---

## Configuration

All settings are environment variables with sensible defaults; see
`backend/config.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RTO_DATASET_PATH` | repo root CSV | RTO dataset location |
| `OCR_ENGINE_ORDER` | `rapidocr,easyocr,tesseract,template` | Engine preference |
| `OCR_MIN_CONFIDENCE` | `0.30` | Minimum accepted OCR word score |
| `ONLY_VALID_PLATES` | `1` | Return only real registrations. `0` shows every candidate region (debugging) |
| `REQUIRE_KNOWN_RTO` | `1` | Require the state+RTO prefix to exist in the dataset |
| `MAX_OCR_CORRECTIONS` | `2` | Reject readings needing more character repairs than this |
| `MAX_OCR_CANDIDATES` | `12` | Cap on regions sent to OCR per frame |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload ceiling (10 MB) |
| `DETECTION_MAX_EDGE` | `1600` | Long edge used for detection; boxes are scaled back |
| `VIDEO_FRAME_STRIDE` | `8` | Process every Nth frame |
| `VIDEO_MAX_FRAMES_SCANNED` | `900` | Cap on sampled frames |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated origins; credentials are only enabled for an explicit list |
| `YOLO_WEIGHTS_PATH` | `backend/models/plate_yolo.pt` | Optional weights |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Docker

Build from the repository root so the dataset is included:

```bash
docker build -f backend/Dockerfile -t number-plate-ai .
docker run -p 8000:8000 number-plate-ai
```

The build fails fast if the dataset or OCR engine is missing.

---

## Deploying: API on Render, dashboard on GitHub Pages

GitHub Pages serves static files only, so the dashboard has to call the API on
another host. Two pieces:

| Piece | Host | What it is |
| --- | --- | --- |
| `backend/` | Render (Docker web service) | API only: `/health`, `/rto-dataset`, `/detect`, `/docs` |
| `frontend/` | GitHub Pages | Static dashboard that calls the Render URL |

### 1. Backend on Render

`render.yaml` is a blueprint that creates the service with the right settings.
In the Render dashboard: **New → Blueprint → select this repository → Apply**.
It builds `backend/Dockerfile` with the repository root as the build context, so
the RTO CSV ends up in the image.

The blueprint sets `SERVE_FRONTEND=0`, so `/` returns service metadata rather
than the dashboard. Copy the service URL, for example
`https://number-plate-ai-api.onrender.com`, and confirm `/health` responds.

### 2. Point the dashboard at it

Edit `frontend/config.js`:

```js
window.NUMBER_PLATE_API_URL = "https://number-plate-ai-api.onrender.com";
```

Commit and push. `.github/workflows/deploy-frontend.yml` publishes `frontend/`
to Pages; enable it once under **Settings → Pages → Source: GitHub Actions**.
The workflow refuses to deploy while that value is empty, because the dashboard
would otherwise try to call `github.io` as its own API.

For a quick test against a different backend, no redeploy needed:
`https://<user>.github.io/<repo>/?api=https://other-service.onrender.com`

### 3. Lock CORS to your Pages origin

On Render, set `CORS_ALLOWED_ORIGINS` to the scheme and host only — no
repository path, no trailing slash:

```
CORS_ALLOWED_ORIGINS = https://<your-github-username>.github.io
```

An explicit origin also enables credentialed requests, which a `*` wildcard
cannot do.

### Free plan behaviour

A free Render instance sleeps after ~15 minutes idle, and the next request has
to boot the container and load the OCR models. The dashboard handles this: it
retries `/health` for about a minute at page load and shows "Waking up the
backend…" instead of reporting a failure.

The free plan also caps memory at 512 MB. The blueprint pins ONNX Runtime to one
thread and reduces video frame sampling to stay inside it. Long videos are the
most likely thing to exhaust memory or hit the request timeout; images are
comfortable.

### Before you share the URL publicly

`/detect` has no authentication and no rate limit. Anyone with the URL can spend
your compute on uploads. For anything beyond a demo, add an API key check and a
rate limit, and keep `CORS_ALLOWED_ORIGINS` set to your own origin.

---

## Repository layout

```
├── India_RTO_Registration_Dataset_New.csv   # RTO master dataset (single source of truth)
├── backend/
│   ├── config.py          # every tunable, env-overridable
│   ├── detector.py        # YOLO / text-anchored / geometric proposals + NMS
│   ├── ocr_engine.py      # engine chain, preprocessing, honest confidences
│   ├── plate_text.py      # plate grammar, repair, validation
│   ├── rto_lookup.py      # CSV loader + prefix resolution
│   ├── main.py            # FastAPI app, also serves the dashboard
│   ├── cascades/          # Haar cascade
│   ├── dataset/data.yaml  # YOLO training config
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/              # dashboard (HTML/CSS/JS, no build step)
├── tests/                 # pytest suite
├── main.py                # CLI over the same backend package
└── requirements.txt       # backend requirements + test tooling
```

---

## Known limitations

- The template matcher fallback is weak. Install RapidOCR for usable accuracy.
- Live camera detection is not implemented and is not shown in the UI.
- Heavily skewed, blurred or partially occluded plates may not be read; the
  result says so rather than guessing.
- Dataset city names follow the RTO office naming from public listings and may
  lag very recent district reorganisations.

## License

Open source, free for educational and commercial use.
