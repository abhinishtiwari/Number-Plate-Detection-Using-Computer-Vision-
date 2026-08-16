/**
 * ===================================================================
 * NUMBER PLATE AI - REAL DYNAMIC COMPUTER VISION DASHBOARD
 * ===================================================================
 */
const PRODUCTION_API_URL = "http://127.0.0.1:8000";

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

// Result Cards Container & Status
const resultStatusBadge = document.getElementById("resultStatusBadge");
const resultCardsList = document.getElementById("resultCardsList");

// Stats & Controls
const rtoSearchInput = document.getElementById("rtoSearchInput");
const rtoTableBody = document.getElementById("rtoTableBody");
const rtoRecordCountBadge = document.getElementById("rtoRecordCountBadge");
const recentDetectionsList = document.getElementById("recentDetectionsList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const themeToggleBtn = document.getElementById("themeToggleBtn");

const totalDetectionsVal = document.getElementById("totalDetectionsVal");
const todayDetectionsVal = document.getElementById("todayDetectionsVal");
const uniqueVehiclesVal = document.getElementById("uniqueVehiclesVal");

let selectedFile = null;
let isVideoMode = false;
let detectionHistory = JSON.parse(localStorage.getItem("anpr_history_v5") || "[]");
let fullRtoDataset = [];

// Initialize Clean Idle State & Fetch Dynamic RTO Dataset
resetDetectionResults();
renderRecentDetections();
fetchRtoDataset();

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

/**
 * Fetch and Render RTO Reference Dataset Dynamically
 */
async function fetchRtoDataset() {
  try {
    const response = await fetch(`${API_URL}/rto-dataset`);
    if (response.ok) {
      fullRtoDataset = await response.json();
      renderRtoTable(fullRtoDataset.slice(0, 100)); // Render first 100 entries for optimal performance
      if (rtoRecordCountBadge) {
        rtoRecordCountBadge.textContent = `${fullRtoDataset.length.toLocaleString()} Records`;
      }
    }
  } catch (err) {
    console.warn("RTO Dataset fetch note:", err);
  }
}

function renderRtoTable(records) {
  if (!rtoTableBody) return;
  rtoTableBody.innerHTML = "";

  records.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="bold-text">${row.registration_prefix}</td>
      <td>${row.state_name}</td>
      <td>${row.city}</td>
    `;
    rtoTableBody.appendChild(tr);
  });
}

// RTO Table Real-time Search Filter
rtoSearchInput.addEventListener("input", (e) => {
  const query = e.target.value.toLowerCase().trim();
  if (!query) {
    renderRtoTable(fullRtoDataset.slice(0, 100));
    return;
  }

  const filtered = fullRtoDataset.filter((row) => (
    row.registration_prefix.toLowerCase().includes(query) ||
    row.state_name.toLowerCase().includes(query) ||
    row.city.toLowerCase().includes(query)
  ));
  renderRtoTable(filtered.slice(0, 100));
});

function resetDetectionResults() {
  resultStatusBadge.className = "badge-idle";
  resultStatusBadge.textContent = "Idle";
  resultCardsList.innerHTML = `
    <div class="result-details">
      <div class="detail-row highlight-row">
        <span class="detail-label">💳 Number Plate</span>
        <span class="detail-val plate-green" id="resPlateText">--</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">🏛️ State</span>
        <span class="detail-val" id="resState">--</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">🏷️ RTO Code</span>
        <span class="detail-val" id="resRtoCode">--</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">📍 Registration City</span>
        <span class="detail-val" id="resCity">--</span>
      </div>
      <div class="detail-row flex-col">
        <div class="flex-between">
          <span class="detail-label">🎯 Detection Confidence</span>
          <span class="detail-val bold" id="resConfidence">0%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="confProgressBar" style="width: 0%;"></div>
        </div>
      </div>
    </div>
  `;
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
      renderMultiPlateResults(data.plates);
      renderCanvasOverlay(selectedFile, data.plates);
      data.plates.forEach(p => addRecentDetection(p));
    } else {
      updateSingleResultCard({
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

function renderMultiPlateResults(plates) {
  const isDetected = plates.some(p => p.text && p.text !== "Not detected");
  resultStatusBadge.className = isDetected ? "badge-success" : "badge-idle";
  resultStatusBadge.textContent = isDetected ? `Detected (${plates.length})` : "Not Detected";

  resultCardsList.innerHTML = "";

  plates.forEach((match, idx) => {
    const cardDiv = document.createElement("div");
    cardDiv.className = "result-details";
    if (idx > 0) cardDiv.style.marginTop = "14px";

    const confVal = match.confidence ? match.confidence.toFixed(1) : "0.0";
    const plateText = match.text || "Not detected";
    const stateName = match.state_name || "Not detected";
    const rtoCode = match.full_rto_code || "Not detected";
    const city = match.city || "Not detected";

    cardDiv.innerHTML = `
      ${plates.length > 1 ? `<div class="detail-row"><span class="detail-label bold">🚘 Vehicle #${idx + 1}</span></div>` : ""}
      <div class="detail-row highlight-row">
        <span class="detail-label">💳 Number Plate</span>
        <span class="detail-val plate-green">${plateText}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">🏛️ State</span>
        <span class="detail-val">${stateName}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">🏷️ RTO Code</span>
        <span class="detail-val">${rtoCode}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">📍 Registration City</span>
        <span class="detail-val">${city}</span>
      </div>
      <div class="detail-row flex-col">
        <div class="flex-between">
          <span class="detail-label">🎯 Detection Confidence</span>
          <span class="detail-val bold">${confVal}%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${confVal}%;"></div>
        </div>
      </div>
    `;
    resultCardsList.appendChild(cardDiv);
  });
}

function updateSingleResultCard(match) {
  renderMultiPlateResults([match]);
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
  localStorage.setItem("anpr_history_v5", JSON.stringify(detectionHistory));
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
    localStorage.removeItem("anpr_history_v5");
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
