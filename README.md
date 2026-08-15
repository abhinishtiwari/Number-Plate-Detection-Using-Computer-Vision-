# Number Plate AI - Local Computer Vision Pipeline

A full-stack open-source **Number Plate Detection & Recognition AI** web application built with a **100% local computer vision pipeline** (YOLO + OpenCV + Local OCR + Local RTO Database).

No external APIs, no OpenAI/Gemini LLMs, no paid services, no fake hardcoded outputs.

---

## 🎯 Architecture Overview

```
[ Image / Video Upload ]
           │
           ▼
 [ YOLO Plate Detector ]  ──► Bounding Box [x, y, w, h] (Single or Multiple Plates)
           │
           ▼
[ OpenCV Preprocessing ] ──► Grayscale -> Denoise (Bilateral) -> Sharpen -> Otsu Threshold
           │
           ▼
  [ Local OCR Engine ]   ──► PyTesseract / EasyOCR Text Extraction & Normalization
           │
           ▼
  [ Local RTO Database ] ──► Exact Prefix Lookup (rto_database.json)
           │
           ▼
 [ Derived Result Card ] ──► Number Plate, State, RTO Code, City, Confidence
```

---

## 📁 Repository Structure

```
Number-Plate-Detection-Using-Computer-Vision-/
├── backend/
│   ├── data/
│   │   └── rto_database.json   # Local Indian RTO dataset
│   ├── dataset/
│   │   └── data.yaml           # YOLOv8 Training Dataset configuration
│   ├── detector.py             # YOLO & OpenCV multi-plate detector
│   ├── rto_lookup.py           # Fast prefix lookup engine
│   ├── main.py                 # FastAPI REST API endpoint (/detect, /health)
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile              # Docker container setup for Render
├── tests/
│   └── test_rto.py             # Automated unit tests for RTO dataset mapping
├── frontend/                   # Frontend copy for static hosting
├── index.html                  # Main UI dashboard (Plain HTML)
├── style.css                   # Custom responsive dark CSS design system
├── script.js                   # Client-side JavaScript controller
└── README.md                   # Project documentation
```

---

## 🚗 Local Indian RTO Dataset Schema (`rto_database.json`)

```json
[
  {
    "registration_prefix": "MP09",
    "state_code": "MP",
    "state_name": "Madhya Pradesh",
    "rto_code": "09",
    "full_rto_code": "MP-09",
    "city": "Indore"
  },
  {
    "registration_prefix": "RJ14",
    "state_code": "RJ",
    "state_name": "Rajasthan",
    "rto_code": "14",
    "full_rto_code": "RJ-14",
    "city": "Jaipur"
  }
]
```

### Example Matching Pipeline:
- **Input Image Plate**: `MP09AB1234`
- **OCR Text**: `MP09AB1234`
- **Extracted Prefix**: `MP09`
- **State Name**: `Madhya Pradesh`
- **Full RTO Code**: `MP-09`
- **Registration City**: `Indore`

If a plate cannot be read, the result cleanly outputs `"Not detected"` / `"Unknown"`.

---

## 📦 YOLO Dataset Training Guide (`data.yaml`)

To train a custom YOLOv8 model for Indian Number Plates:

```yaml
path: ./dataset
train: images/train
val: images/val
nc: 1
names:
  0: number_plate
```

Run training with Ultralytics CLI:
```bash
yolo task=detect mode=train model=yolov8n.pt data=backend/dataset/data.yaml epochs=50 imgsz=640
```

---

## 🚀 Running Locally

### 1. Start Backend Server (Python FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start Frontend Server
Open `index.html` directly in your browser or run:
```bash
python -m http.server 5500
```
Then visit: `http://127.0.0.1:5500`

---

## 🧪 Running Unit Tests

```bash
python -m unittest discover -s tests
```

---

## 📄 License
Open Source License - Free for educational and commercial use.
