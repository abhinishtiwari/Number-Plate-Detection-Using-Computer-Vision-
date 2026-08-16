/**
 * ===================================================================
 * NUMBER PLATE AI - REAL DYNAMIC COMPUTER VISION DASHBOARD
 * ===================================================================
 */
const PRODUCTION_API_URL = "https://your-backend.onrender.com";

const API_URL = (
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.protocol === "file:"
) ? "http://127.0.0.1:8000" : PRODUCTION_API_URL;

console.log("[Number Plate AI] Real Local API Server Connected:", API_URL);

// DOM Elements
const fileInput = document.getElementById("fileInput");
const videoInput = document.getElementById("videoInput");
const detectBtn = document.getElementById("detectBtn");
const fileNameDisplay = document.getElementById("fileNameDisplay");
const dropZone = document.getElementById("dropZone");
const statusBox = document.getElementById("statusBox");
const statusMessage = document.getElementById("statusMessage");
const spinner = document.getElementById("spinner");

const tabImageUpload = document.getElementById("tabImageUpload");
const tabVideoUpload = document.getElementById("tabVideoUpload");

const previewContainer = document.getElementById("previewContainer");
const resultCanvas = document.getElementById("resultCanvas");
const videoPreview = document.getElementById("videoPreview");

// Derived Detection Result Card Elements
const resultStatusBadge = document.getElementById("resultStatusBadge");
const resPlateText = document.getElementById("resPlateText");
const resState = document.getElementById("resState");
const resRtoCode = document.getElementById("resRtoCode");
const resCity = document.getElementById("resCity");
const resConfidence = document.getElementById("resConfidence");
const confProgressBar = document.getElementById("confProgressBar");

// Stats & Controls
const rtoSearchInput = document.getElementById("rtoSearchInput");
const rtoTableBody = document.getElementById("rtoTableBody");
const recentDetectionsList = document.getElementById("recentDetectionsList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const themeToggleBtn = document.getElementById("themeToggleBtn");

const totalDetectionsVal = document.getElementById("totalDetectionsVal");
const todayDetectionsVal = document.getElementById("todayDetectionsVal");
const uniqueVehiclesVal = document.getElementById("uniqueVehiclesVal");

let selectedFile = null;
let isVideoMode = false;
let detectionHistory = JSON.parse(localStorage.getItem("anpr_history_v4") || "[]");

// Initial Clean Idle State
resetDetectionResults();
renderRecentDetections();

// Theme Toggle
themeToggleBtn.addEventListener("click", () => {
  document.body.classList.toggle("light-theme");
  themeToggleBtn.textContent = document.body.classList.contains("light-theme") ? "🌙" : "☀️";
});

// Tab Mode Switching
tabImageUpload.addEventListener("click", () => {
  isVideoMode = false;
  tabImageUpload.classList.add("active");
  tabVideoUpload.classList.remove("active");
  fileInput.click();
});

tabVideoUpload.addEventListener("click", () => {
  isVideoMode = true;
  tabVideoUpload.classList.add("active");
  tabImageUpload.classList.remove("active");
  videoInput.click();
});

// File Handlers
fileInput.addEventListener("change", (e) => handleFileSelect(e.target.files[0]));
videoInput.addEventListener("change", (e) => handleFileSelect(e.target.files[0]));

// Drag and Drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

function resetDetectionResults() {
  resultStatusBadge.className = "badge-idle";
  resultStatusBadge.textContent = "Idle";
  resPlateText.textContent = "--";
  resState.textContent = "--";
  resRtoCode.textContent = "--";
  resCity.textContent = "--";
  resConfidence.textContent = "0%";
  confProgressBar.style.width = "0%";
}

function handleFileSelect(file) {
  if (!file) return;

  selectedFile = file;
  fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
  detectBtn.disabled = false;
  hideStatus();
  resetDetectionResults();

  if (file.type.startsWith("video/")) {
    isVideoMode = true;
    tabVideoUpload.classList.add("active");
    tabImageUpload.classList.remove("active");

    const videoURL = URL.createObjectURL(file);
    videoPreview.src = videoURL;
    videoPreview.classList.remove("hidden");
    resultCanvas.classList.add("hidden");
    previewContainer.classList.remove("hidden");
  } else {
    isVideoMode = false;
    tabImageUpload.classList.add("active");
    tabVideoUpload.classList.remove("active");

    videoPreview.classList.add("hidden");
    resultCanvas.classList.remove("hidden");
    previewContainer.classList.remove("hidden");

    // Preview raw uploaded image on canvas (without bounding boxes until detection is triggered)
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        resultCanvas.width = img.width;
        resultCanvas.height = img.height;
        const ctx = resultCanvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
}

/**
 * Polls backend GET /health to ensure server connection
 */
async function waitForBackendServer(maxWaitSeconds = 60) {
  const startTime = Date.now();
  showStatus("Checking backend server connection...", "info", true);

  while ((Date.now() - startTime) / 1000 < maxWaitSeconds) {
    try {
      const response = await fetch(`${API_URL}/health`, { method: "GET" });
      if (response.ok) {
        const data = await response.json();
        if (data.status === "ok") return true;
      }
    } catch (err) {}
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error("Server connection timed out.");
}

/**
 * Detect Trigger Action
 */
detectBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  detectBtn.disabled = true;

  try {
    await waitForBackendServer(60);
    showStatus("Running YOLO & OpenCV Local Pipeline...", "info", true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${API_URL}/detect`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error(`HTTP error ${response.status}`);

    const data = await response.json();
    showStatus("Detection complete!", "success", false);

    if (data.plates && data.plates.length > 0) {
      const primaryPlate = data.plates[0];
      updateResultCard(primaryPlate);
      renderCanvasOverlay(selectedFile, data.plates);
      addRecentDetection(primaryPlate);
    } else {
      updateResultCard({
        text: "Not detected",
        state_name: "Not detected",
        full_rto_code: "Not detected",
        city: "Not detected",
        confidence: 0.0
      });
      showStatus("No license plates detected in the uploaded file.", "info", false);
    }

  } catch (error) {
    console.error("Detection Error:", error);
    showStatus("Failed to communicate with local detection engine.", "error", false);
    resultStatusBadge.className = "badge-idle";
    resultStatusBadge.textContent = "Error";
  } finally {
    detectBtn.disabled = false;
  }
});

function updateResultCard(match) {
  const isDetected = match.text && match.text !== "Not detected";
  resultStatusBadge.className = isDetected ? "badge-success" : "badge-idle";
  resultStatusBadge.textContent = isDetected ? "Success" : "Not Detected";

  resPlateText.textContent = match.text || "Not detected";
  resState.textContent = match.state_name || "Not detected";
  resRtoCode.textContent = match.full_rto_code || "Not detected";
  resCity.textContent = match.city || "Not detected";
  
  const confVal = match.confidence ? match.confidence.toFixed(1) : "0.0";
  resConfidence.textContent = `${confVal}%`;
  confProgressBar.style.width = `${confVal}%`;
}

function renderCanvasOverlay(file, plates) {
  if (file.type.startsWith("video/")) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      resultCanvas.width = img.width;
      resultCanvas.height = img.height;
      const ctx = resultCanvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      plates.forEach((plate) => {
        const [x, y, w, h] = plate.box;
        const textLabel = (plate.text && plate.text !== "Not detected") ? plate.text : "PLATE DETECTED";

        // Draw bounding box
        ctx.strokeStyle = "#00ff66";
        ctx.lineWidth = Math.max(4, Math.round(img.width / 250));
        ctx.strokeRect(x, y, w, h);

        // Draw tag badge
        const fontSize = Math.max(18, Math.round(img.width / 35));
        ctx.font = `bold ${fontSize}px 'Plus Jakarta Sans', sans-serif`;
        const textWidth = ctx.measureText(textLabel).width;
        const padding = 8;

        const tagY = y - fontSize - padding * 2 > 0 ? y - fontSize - padding * 2 : y;
        ctx.fillStyle = "#00ff66";
        ctx.fillRect(x, tagY, textWidth + padding * 2, fontSize + padding);

        ctx.fillStyle = "#000000";
        ctx.fillText(textLabel, x + padding, tagY + fontSize);
      });
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// RTO Table Search Filtering
rtoSearchInput.addEventListener("input", (e) => {
  const query = e.target.value.toLowerCase();
  const rows = rtoTableBody.querySelectorAll("tr");
  rows.forEach((row) => {
    const text = row.textContent.toLowerCase();
    row.style.display = text.includes(query) ? "" : "none";
  });
});

// Recent Detections List & LocalStorage History
function addRecentDetection(item) {
  const plate = item.text || "Not detected";
  const state = item.state_name || "Unknown";
  const city = item.city || "Unknown";

  const newDetection = {
    plate: plate,
    location: `${state}, ${city}`,
    confidence: item.confidence ? item.confidence.toFixed(1) : "0.0",
    time: "Just now"
  };

  detectionHistory.unshift(newDetection);
  if (detectionHistory.length > 10) detectionHistory.pop();
  localStorage.setItem("anpr_history_v4", JSON.stringify(detectionHistory));
  renderRecentDetections();
}

function renderRecentDetections() {
  const total = detectionHistory.length;
  const uniqueSet = new Set(detectionHistory.map(d => d.plate).filter(p => p !== "Not detected"));

  totalDetectionsVal.textContent = total.toLocaleString();
  todayDetectionsVal.textContent = total.toLocaleString();
  uniqueVehiclesVal.textContent = uniqueSet.size.toLocaleString();

  if (!detectionHistory.length) {
    recentDetectionsList.innerHTML = `<div class="empty-recent-msg"><span>No detections recorded yet. Upload an image or video above to begin.</span></div>`;
    return;
  }

  recentDetectionsList.innerHTML = "";
  detectionHistory.forEach((item) => {
    const div = document.createElement("div");
    div.className = "recent-item";
    const prefix = item.plate.length >= 4 ? item.plate.substring(0, 4) : "PLATE";
    div.innerHTML = `
      <div class="recent-badge-img">${prefix}</div>
      <div class="recent-info">
        <span class="recent-plate">${item.plate}</span>
        <span class="recent-sub">${item.location} • ${item.confidence}%</span>
      </div>
      <span class="recent-time">${item.time}</span>
    `;
    recentDetectionsList.appendChild(div);
  });
}

if (clearHistoryBtn) {
  clearHistoryBtn.addEventListener("click", (e) => {
    e.preventDefault();
    detectionHistory = [];
    localStorage.removeItem("anpr_history_v4");
    renderRecentDetections();
  });
}

// Status Helpers
function showStatus(msg, type = "info", showSpinner = false) {
  statusMessage.textContent = msg;
  spinner.className = showSpinner ? "spinner-sm" : "spinner-sm hidden";
  statusBox.classList.remove("hidden");
}

function hideStatus() {
  statusBox.classList.add("hidden");
}
