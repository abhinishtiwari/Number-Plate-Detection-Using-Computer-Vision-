# 🚗 Number Plate Detection Web App (GitHub Pages + Render)

A full-stack Automatic License / Number Plate Recognition (ANPR) web application with:

- **Frontend**: Pure **HTML + CSS + Vanilla JavaScript** hosted for FREE on **GitHub Pages**.
- **Backend**: Python REST API (**FastAPI + OpenCV + Tesseract OCR**) hosted on **Render** (via Docker container).

---

## 🏗 Architecture & Cold-Start Handling

- **Frontend (GitHub Pages)**: Loads instantly from GitHub's CDN. No frameworks, no React, no build steps — plain `index.html`, `style.css`, and `script.js`.
- **Backend (Render Free Tier)**: Handles heavy OpenCV Haar Cascade detection and Tesseract OCR processing.
- **Cold-Start Resilience**: Render's free tier spins down after 15 minutes of inactivity. When a user clicks **Detect**, the frontend script first calls `GET /health` in a retry loop displaying a clear *"Waking up the detection server..."* banner until the server responds, ensuring the app never feels broken.

---

## 🌐 Live Deployment Instructions

### Part 1: Deploy Backend to Render (Free Tier)

1. Go to **[dashboard.render.com](https://dashboard.render.com)** → **New +** → **Web Service**.
2. Connect your GitHub repository: `abhinishtiwari/Number-Plate-Detection-Using-Computer-Vision-`.
3. Configure settings:
   - **Name**: `number-plate-backend`
   - **Runtime**: **Docker**
   - **Dockerfile Path**: `backend/Dockerfile`
   - **Instance Type**: **Free**
4. Click **Create Web Service**.
5. Once deployed, copy your live backend URL (e.g. `https://number-plate-backend.onrender.com`).

---

### Part 2: Enable GitHub Pages (Free Hosting)

1. Open `script.js` and set your Render URL on line 7:
   ```javascript
   const API_URL = "https://your-backend.onrender.com";
   ```
2. Commit and push:
   ```bash
   git add script.js
   git commit -m "Set production backend URL"
   git push origin main
   ```
3. Open your GitHub Repository in your browser:
   `https://github.com/abhinishtiwari/Number-Plate-Detection-Using-Computer-Vision-`
4. Click **Settings** (top tabs) → Click **Pages** (left sidebar menu).
5. Under **Build and deployment**:
   - **Source**: Select `Deploy from a branch`
   - **Branch**: Select `main` and folder `/ (root)`
   - Click **Save**.

Your frontend website will be live in ~1 minute at:
`https://abhinishtiwari.github.io/Number-Plate-Detection-Using-Computer-Vision-/`

---

## 📜 License
Distributed under the MIT License.
