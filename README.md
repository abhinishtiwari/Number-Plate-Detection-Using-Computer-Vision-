# 🚗 Number Plate Detection Web App (Split Architecture)

A full-stack, production-ready Automatic License / Number Plate Recognition (ANPR) web application designed with a **Split Architecture**:

- **Backend**: Python (FastAPI + OpenCV Haar Cascade + Tesseract OCR) running inside a **Docker** container on **Render** (free tier).
- **Frontend**: Pure **Vanilla HTML + CSS + JavaScript** (no React, no Next.js, no build step) static site hosted on **Vercel** (free tier) with automated cold-start wake-up handling.

---

## 🏗 Architecture Overview

```
                        ┌────────────────────────────────────────┐
                        │      Frontend (Static Site)            │
                        │      Hosted on Vercel CDN              │
                        │    (index.html, style.css, script.js)  │
                        └──────────────────┬─────────────────────┘
                                           │
                                     Instant Load
                                           │
                                ┌──────────┴──────────┐
                                │   GET /health       │  Polls until server awakes
                                └──────────┬──────────┘  ("Waking up server...")
                                           │
                                ┌──────────┴──────────┐
                                │   POST /detect      │  Submits image file
                                └──────────┬──────────┘
                                           │
                        ┌──────────────────▼─────────────────────┐
                        │       Backend (REST API)               │
                        │      Hosted on Render (Docker)         │
                        │  FastAPI + OpenCV + Tesseract OCR      │
                        └────────────────────────────────────────┘
```

### Why Split Architecture?
Render's free tier spins down services after 15 minutes of inactivity. When a new request arrives, it takes **30–60 seconds to wake up (cold start)**.

- If the frontend and backend were combined on Render, the entire website would appear broken or frozen during cold boot.
- By hosting the frontend as a static site on Vercel, the web page **loads instantly globally**.
- The frontend JavaScript handles cold starts gracefully by polling `GET /health` first while displaying a clear, animated *"Waking up the detection server..."* banner.

---

## 📁 Repository Structure

```
Number-Plate-Detection-Using-Computer-Vision/
├── backend/
│   ├── main.py                 # FastAPI application routes (/health, /detect)
│   ├── Dockerfile               # Slim Dockerfile (python:3.11-slim + tesseract-ocr)
│   ├── requirements.txt         # Backend Python dependencies
│   └── cascades/
│       └── haarcascade_russian_plate_number.xml # OpenCV Haar Cascade
│
├── frontend/
│   ├── index.html              # Plain HTML5 user interface
│   ├── style.css               # Clean, mobile-friendly CSS & keyframe spinner
│   └── script.js               # Vanilla JS with cold-start fetch polling & Canvas drawing
│
└── README.md                   # Complete deployment & project guide
```

---

## 🚀 Deployment Instructions

### Part 1: Deploy Backend to Render (Free Tier)

1. **Push Repository to GitHub**:
   Ensure your code is pushed to your GitHub repository:
   `https://github.com/abhinishtiwari/Number-Plate-Detection-Using-Computer-Vision-.git`

2. **Create New Web Service on Render**:
   - Log into [Render Dashboard](https://dashboard.render.com).
   - Click **New +** → **Web Service**.
   - Connect your GitHub repository.

3. **Configure Service Settings**:
   - **Name**: `number-plate-detection-backend` (or your choice)
   - **Runtime**: `Docker` (Render auto-detects `backend/Dockerfile` or specify Dockerfile Path as `backend/Dockerfile`).
   - **Root Directory**: `backend` (or leave empty if using Dockerfile Path `backend/Dockerfile`).
   - **Instance Type**: Select **Free**.

4. **Deploy Service**:
   - Click **Create Web Service**.
   - Once deployed, copy your live backend URL (e.g., `https://number-plate-backend.onrender.com`).

---

### Part 2: Deploy Frontend to Vercel (Free Tier)

1. **Set Backend API URL in `frontend/script.js`**:
   Open `frontend/script.js` and update line 7 with your actual Render backend URL:
   ```javascript
   const API_URL = "https://number-plate-backend.onrender.com";
   ```
   *(Note: Since static sites have no server environment variables, editing this constant is the standard way to set environment endpoints before deploying).*

2. **Commit and Push to GitHub**:
   ```bash
   git add frontend/script.js
   git commit -m "Configure production backend API URL"
   git push origin main
   ```

3. **Import Project into Vercel**:
   - Log into [Vercel Dashboard](https://vercel.com).
   - Click **Add New...** → **Project**.
   - Import your GitHub repository.
   - **Root Directory**: Select `frontend` (or click Edit and choose `frontend`).
   - **Framework Preset**: Select **Other** (Vercel treats it as a static HTML/CSS/JS site automatically).
   - Leave Build Command & Output Directory blank.

4. **Deploy**:
   - Click **Deploy**.
   - Vercel will deploy your site instantly to a URL like `https://your-app.vercel.app`.

---

## 💡 Pro-Tip: Preventing Cold Starts (Optional)

To keep Render's free container warm and avoid the 30–60 second cold start delay for users:

1. Register a free account on an uptime pinging service like **UptimeRobot** or **cron-job.org**.
2. Create an HTTP monitor that sends a `GET` request to your backend health endpoint every **10 to 14 minutes**:
   `https://your-backend.onrender.com/health`
3. This keeps the free Render container active without exceeding free-tier limits.

---

## 🛠 Local Development Setup

### 1. Run Backend Locally
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
API runs at: `http://localhost:8000` (Health check: `http://localhost:8000/health`).

### 2. Run Frontend Locally
Open `frontend/index.html` directly in your web browser, or serve it using Python:
```bash
cd frontend
python -m http.server 5500
```
Open `http://localhost:5500` in your browser.

---

## 📜 License
Distributed under the MIT License.
