# 🚗 Automatic Number Plate Recognition (ANPR) System

A powerful, end-to-end Automatic License / Number Plate Recognition (ANPR / ALPR) application built using **OpenCV**, **Computer Vision algorithms** (Haar Cascade & Contour Morphological Filtering), **OCR engines** (EasyOCR / PyTesseract), and **Streamlit**.

---

## ✨ Features

- **Dual Detection Engines**:
  - **Haar Cascade Classifier**: Detects plate patterns using trained cascade XML models.
  - **Contour Morphological Segmentation**: Detects rectangular plate geometries based on edge detection (Canny), bilateral filtering, and aspect ratio limits.
- **OCR Text Extraction**:
  - Automatically extracts alphanumeric characters from detected license plate ROIs.
  - Supports **EasyOCR**, **PyTesseract**, and built-in heuristic fallbacks.
- **Interactive Streamlit Web App (`app.py`)**:
  - Drag-and-drop vehicle image uploads.
  - Interactive parameter tuning (Aspect ratio, algorithm selection, OCR engines).
  - Side-by-side comparison of original vs annotated images with bounding boxes.
  - Preview cropped ROI images & preprocessed binary images.
  - Log exported directly as CSV reports.
- **Command-Line Interface (CLI) (`main.py`)**:
  - Process static images or real-time webcam / video streams.
  - Save output annotated media and CSV log history.
- **Automated Logging**: Timestamped logging of detected plate numbers and confidence scores.

---

## 📁 Repository Structure

```
Number-Plate-Detection-Using-Computer-Vision/
├── app.py                      # Interactive Streamlit Web GUI
├── main.py                     # CLI Script for Image / Video / Webcam processing
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── cascades/
│   └── haarcascade_russian_plate_number.xml # OpenCV Haar Cascade model
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration & hyperparameters
│   ├── detector.py             # Dual detection pipeline (Cascade + Contour)
│   ├── ocr_engine.py           # Optical Character Recognition engine wrapper
│   └── utils.py                # Preprocessing, drawing, and CSV logging helpers
└── tests/
    └── test_detector.py        # Unit test suite
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/abhinishtiwari/Number-Plate-Detection-Using-Computer-Vision-.git
cd Number-Plate-Detection-Using-Computer-Vision-
pip install -r requirements.txt
```

### 2. Run Interactive Web Dashboard (Streamlit)

Launch the Streamlit app in your browser:

```bash
streamlit run app.py
```

### 3. Run Command-Line Interface (CLI)

**Process an Image:**
```bash
python main.py --image path/to/car.jpg --output output/result.jpg --method hybrid
```

**Process a Video or Live Webcam:**
```bash
python main.py --video 0 --output output/webcam_output.mp4
```

### 4. Run Unit Tests

```bash
python -m unittest discover -s tests
```

---

## 🛠 Tech Stack

- **Python 3.8+**
- **OpenCV (`opencv-contrib-python`)**: Computer vision, bilateral filtering, Canny edge detection, Haar cascades.
- **NumPy**: Matrix computations & NMS bounding box processing.
- **Streamlit**: Web application framework.
- **EasyOCR / PyTesseract**: Optical Character Recognition.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
