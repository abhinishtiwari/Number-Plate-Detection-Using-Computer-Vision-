/**
 * Number Plate AI - dashboard controller.
 *
 * Everything rendered here comes from the /detect and /rto-dataset responses or
 * from locally stored history. There are no sample plates, no seeded history and
 * no placeholder statistics.
 */
"use strict";

/* ------------------------------------------------------------------ config */

const API_URL = window.location.origin;

const HISTORY_KEY = "anpr_history_v6";
const THEME_KEY = "anpr_theme";
const HISTORY_LIMIT = 200;
const DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

/* --------------------------------------------------------------- elements */

const el = (id) => document.getElementById(id);

const fileInput = el("fileInput");
const detectBtn = el("detectBtn");
const fileNameDisplay = el("fileNameDisplay");
const dropZone = el("dropZone");
const statusBox = el("statusBox");
const statusMessage = el("statusMessage");
const spinner = el("spinner");
const engineStatus = el("engineStatus");

const tabImageUpload = el("tabImageUpload");
const tabVideoUpload = el("tabVideoUpload");

const previewContainer = el("previewContainer");
const resultCanvas = el("resultCanvas");
const videoPreview = el("videoPreview");

const resultStatusBadge = el("resultStatusBadge");
const resultCardsList = el("resultCardsList");

const rtoSearchInput = el("rtoSearchInput");
const rtoTableBody = el("rtoTableBody");
const rtoRecordCountBadge = el("rtoRecordCountBadge");
const rtoResultCount = el("rtoResultCount");
const recentDetectionsList = el("recentDetectionsList");
const clearHistoryBtn = el("clearHistoryBtn");
const themeToggleBtn = el("themeToggleBtn");

const totalDetectionsVal = el("totalDetectionsVal");
const todayDetectionsVal = el("todayDetectionsVal");
const uniqueVehiclesVal = el("uniqueVehiclesVal");

/* ------------------------------------------------------------------ state */

let selectedFile = null;
/** Guards against a slow FileReader painting a previously selected image. */
let renderToken = 0;
let maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES;
let fullRtoDataset = [];
let detectionHistory = loadHistory();
/** Last video detections, used to seek the player to a sighting. */
let lastVideoPlates = [];

/* --------------------------------------------------------------- utilities */

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.warn("Could not read stored history; starting empty.", err);
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(detectionHistory));
  } catch (err) {
    console.warn("Could not persist history.", err);
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function relativeTime(isoString) {
  const then = new Date(isoString).getTime();
  if (Number.isNaN(then)) return "";
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? "yesterday" : `${days} days ago`;
}

/** Confidence is null when nothing scored the detection; never show a fake 0%. */
function formatConfidence(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)}%` : "n/a";
}

function confidenceWidth(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.max(0, Math.min(100, value))}%`
    : "0%";
}

/* ------------------------------------------------------------------ status */

function showStatus(message, type = "info", busy = false) {
  statusMessage.textContent = message;
  statusBox.className = `status-banner status-${type}`;
  spinner.className = busy ? "spinner-sm" : "spinner-sm hidden";
}

function hideStatus() {
  statusBox.className = "status-banner hidden";
}

/* -------------------------------------------------------------- API health */

/** Check the local API once; the page itself is served by that same process. */
async function checkBackend() {
  engineStatus.classList.remove("hidden");
  try {
    const response = await fetch(`${API_URL}/health`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return applyHealth(await response.json());
  } catch (err) {
    console.warn("Local backend health check failed:", err);
    engineStatus.className = "engine-status engine-bad";
    engineStatus.textContent =
      "Local backend is not reachable. Run python main.py from the project folder, then refresh.";
    return false;
  }
}

/** Render the /health payload into the status line. */
function applyHealth(data) {
  maxUploadBytes = data.max_upload_bytes || DEFAULT_MAX_UPLOAD_BYTES;

  const ocr = data.ocr || {};
  const dataset = data.rto_dataset || {};

  if (!data.pipeline_ready) {
    engineStatus.className = "engine-status engine-bad";
    engineStatus.textContent =
      "Backend is up but has no OCR engine. Install one with: pip install rapidocr-onnxruntime";
    return false;
  }

  const warning = ocr.neural_engine_available ? "" : " (basic engine only - accuracy will be low)";
  engineStatus.className = ocr.neural_engine_available
    ? "engine-status engine-good"
    : "engine-status engine-warn";
  engineStatus.textContent =
    `OCR: ${ocr.active || "none"}${warning} · detection: ${(data.detection_sources || []).join(", ")} · ` +
    `RTO dataset: ${(dataset.records || 0).toLocaleString()} records, ${dataset.states || 0} states/UTs`;
  return true;
}

/* ------------------------------------------------------------- RTO dataset */

async function fetchRtoDataset() {
  try {
    const response = await fetch(`${API_URL}/rto-dataset`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    fullRtoDataset = payload.records || [];
    rtoRecordCountBadge.textContent = `${fullRtoDataset.length.toLocaleString()} Records`;
    renderRtoTable(fullRtoDataset);
  } catch (err) {
    console.warn("Could not load the RTO dataset:", err);
    rtoRecordCountBadge.textContent = "unavailable";
    rtoTableBody.innerHTML =
      `<tr><td colspan="3" class="table-empty">RTO dataset could not be loaded from the backend.</td></tr>`;
  }
}

const RTO_RENDER_LIMIT = 200;

function renderRtoTable(records) {
  const shown = records.slice(0, RTO_RENDER_LIMIT);
  if (!shown.length) {
    rtoTableBody.innerHTML = `<tr><td colspan="3" class="table-empty">No matching RTO records.</td></tr>`;
  } else {
    rtoTableBody.innerHTML = shown.map((row) => `
      <tr>
        <td class="bold-text">${escapeHtml(row.registration_prefix)}</td>
        <td>${escapeHtml(row.state_name)}</td>
        <td>${escapeHtml(row.city || "—")}</td>
      </tr>`).join("");
  }
  rtoResultCount.textContent = records.length > shown.length
    ? `showing ${shown.length} of ${records.length.toLocaleString()}`
    : `${records.length.toLocaleString()} shown`;
}

rtoSearchInput.addEventListener("input", (event) => {
  const query = event.target.value.toLowerCase().trim();
  if (!query) {
    renderRtoTable(fullRtoDataset);
    return;
  }
  renderRtoTable(fullRtoDataset.filter((row) => (
    (row.registration_prefix || "").toLowerCase().includes(query) ||
    (row.state_name || "").toLowerCase().includes(query) ||
    (row.city || "").toLowerCase().includes(query)
  )));
});

/* ------------------------------------------------------------ file picking */

function validateFile(file) {
  if (file.size === 0) return "That file is empty.";
  if (file.size > maxUploadBytes) {
    return `That file is ${formatBytes(file.size)}; the limit is ${formatBytes(maxUploadBytes)}.`;
  }
  const isMedia = file.type.startsWith("image/") || file.type.startsWith("video/");
  const hasKnownSuffix = /\.(jpe?g|png|bmp|webp|mp4|avi|mov|mkv|webm)$/i.test(file.name);
  if (!isMedia && !hasKnownSuffix) return "Upload an image or a video file.";
  return null;
}

function handleFileSelect(file) {
  if (!file) return;

  const problem = validateFile(file);
  if (problem) {
    selectedFile = null;
    detectBtn.disabled = true;
    fileNameDisplay.textContent = "No file chosen";
    showStatus(problem, "error", false);
    return;
  }

  selectedFile = file;
  detectBtn.disabled = false;
  fileNameDisplay.textContent = `${file.name} (${formatBytes(file.size)})`;
  hideStatus();
  resetDetectionResults();
  lastVideoPlates = [];

  const isVideo = file.type.startsWith("video/") || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);
  setModeTabs(isVideo);
  previewContainer.classList.remove("hidden");

  const token = ++renderToken;
  if (isVideo) {
    if (videoPreview.dataset.objectUrl) URL.revokeObjectURL(videoPreview.dataset.objectUrl);
    const url = URL.createObjectURL(file);
    videoPreview.dataset.objectUrl = url;
    videoPreview.src = url;
    videoPreview.classList.remove("hidden");
    resultCanvas.classList.add("hidden");
  } else {
    videoPreview.classList.add("hidden");
    resultCanvas.classList.remove("hidden");
    drawImageOnCanvas(file, [], token);
  }
}

function setModeTabs(isVideo) {
  tabVideoUpload.classList.toggle("active", isVideo);
  tabImageUpload.classList.toggle("active", !isVideo);
}

fileInput.addEventListener("change", (event) => handleFileSelect(event.target.files[0]));

// The drop zone advertises "click to browse", so make the whole zone clickable.
dropZone.addEventListener("click", (event) => {
  if (event.target.closest("label,input")) return;
  fileInput.click();
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  handleFileSelect(event.dataTransfer?.files?.[0]);
});

tabImageUpload.addEventListener("click", () => {
  fileInput.setAttribute("accept", "image/*");
  fileInput.click();
});
tabVideoUpload.addEventListener("click", () => {
  fileInput.setAttribute("accept", "video/*");
  fileInput.click();
});

/* ---------------------------------------------------------------- detection */

detectBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const token = ++renderToken;
  detectBtn.disabled = true;
  showStatus("Running the local detection pipeline...", "info", true);

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${API_URL}/detect`, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.detail || `Request failed with HTTP ${response.status}`);
    }
    if (token !== renderToken) return; // a newer file was selected mid-request

    const plates = payload.plates || [];
    renderResults(plates, payload);

    if (payload.media_type === "video") {
      lastVideoPlates = plates;
      const scanned = payload.frames_scanned ?? 0;
      showStatus(
        plates.length
          ? `Found ${plates.length} plate(s) across ${scanned} sampled frame(s).`
          : `No plates found in ${scanned} sampled frame(s).`,
        plates.length ? "success" : "info", false,
      );
    } else {
      drawImageOnCanvas(selectedFile, plates, token);
      showStatus(
        plates.length ? `Found ${plates.length} plate region(s).` : "No number plate found in this image.",
        plates.length ? "success" : "info", false,
      );
    }

    plates.forEach(addToHistory);
  } catch (error) {
    console.error("Detection failed:", error);
    const message = error instanceof TypeError
      ? "Lost connection to the local backend. Run python main.py, refresh, and try again."
      : error.message || "Detection failed.";
    showStatus(message, "error", false);
    resultStatusBadge.className = "badge-error";
    resultStatusBadge.textContent = "Error";
  } finally {
    detectBtn.disabled = false;
  }
});

/* ----------------------------------------------------------------- results */

function resetDetectionResults() {
  resultStatusBadge.className = "badge-idle";
  resultStatusBadge.textContent = "Idle";
  resultCardsList.innerHTML =
    `<div class="result-placeholder">Upload an image or video and run detection to see results here.</div>`;
}

function renderResults(plates, payload) {
  if (!plates.length) {
    resultStatusBadge.className = "badge-idle";
    resultStatusBadge.textContent = "No plate found";
    resultCardsList.innerHTML =
      `<div class="result-placeholder">No number plate was found in this ${escapeHtml(payload.media_type || "file")}.</div>`;
    return;
  }

  const readable = plates.filter((p) => p.text);
  resultStatusBadge.className = readable.length ? "badge-success" : "badge-warn";
  resultStatusBadge.textContent = readable.length
    ? `Read ${readable.length} of ${plates.length}`
    : `${plates.length} region(s), text unreadable`;

  resultCardsList.innerHTML = plates.map((plate, index) => renderPlateCard(plate, index, plates.length)).join("");

  resultCardsList.querySelectorAll("[data-seek]").forEach((node) => {
    node.addEventListener("click", () => {
      const seconds = Number(node.dataset.seek);
      if (Number.isFinite(seconds)) {
        videoPreview.currentTime = seconds;
        videoPreview.pause();
      }
    });
  });
}

function renderPlateCard(plate, index, total) {
  // A plate that could not be read says so, rather than showing a guess.
  const plateText = plate.text
    ? escapeHtml(plate.text)
    : `<span class="muted">could not be read</span>`;

  const formatNote = plate.text && !plate.is_valid_format
    ? `<span class="tag tag-warn" title="Text was read but does not match an Indian registration format">unverified format</span>`
    : "";
  const correctionNote = plate.ocr_corrections > 0
    ? `<span class="tag" title="Ambiguous characters repaired using the plate format">${plate.ocr_corrections} char fix</span>`
    : "";

  const rtoRows = renderRtoRows(plate);
  const seekable = typeof plate.timestamp_seconds === "number"
    ? `<button class="link-btn" data-seek="${plate.timestamp_seconds}">
         seen at ${plate.timestamp_seconds.toFixed(2)}s (frame ${plate.frame_index})${plate.times_seen > 1 ? ` · ${plate.times_seen} frames` : ""}
       </button>`
    : "";

  return `
    <div class="result-details${index > 0 ? " stacked" : ""}">
      ${total > 1 ? `<div class="detail-row"><span class="detail-label bold">Vehicle #${index + 1}</span></div>` : ""}
      <div class="detail-row highlight-row">
        <span class="detail-label">Number Plate</span>
        <span class="detail-val plate-green">${plateText} ${formatNote} ${correctionNote}</span>
      </div>
      ${rtoRows}
      <div class="detail-row flex-col">
        <div class="flex-between">
          <span class="detail-label">Confidence</span>
          <span class="detail-val bold">${formatConfidence(plate.confidence)}</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${confidenceWidth(plate.confidence)};"></div>
        </div>
        <span class="detail-note">${escapeHtml(describeBasis(plate))}</span>
      </div>
      ${seekable}
    </div>`;
}

/** Show state/RTO/city only as far as the dataset actually resolved them. */
function renderRtoRows(plate) {
  if (!plate.text) {
    return `<div class="detail-row"><span class="detail-label">RTO lookup</span>
      <span class="detail-val muted">needs readable plate text</span></div>`;
  }

  const rows = [
    ["State", plate.state_name],
    ["RTO Code", plate.full_rto_code],
    ["Registration City", plate.city],
  ].map(([label, value]) => `
    <div class="detail-row">
      <span class="detail-label">${label}</span>
      <span class="detail-val${value ? "" : " muted"}">${value ? escapeHtml(value) : "not in dataset"}</span>
    </div>`).join("");

  const levelNote = {
    exact: "",
    state: `<div class="detail-row"><span class="detail-note">RTO number is not in the dataset; only the state could be resolved.</span></div>`,
    national: `<div class="detail-row"><span class="detail-note">Nationwide series - not tied to one RTO.</span></div>`,
    none: `<div class="detail-row"><span class="detail-note">No RTO match for this plate.</span></div>`,
  }[plate.rto_match_level] || "";

  return rows + levelNote;
}

function describeBasis(plate) {
  const basis = plate.confidence_basis || "";
  if (basis.startsWith("ocr:")) return `OCR confidence (${plate.ocr_engine}), box from ${plate.detection_source}`;
  if (basis.startsWith("detector:")) return `Detector probability (${plate.detection_source}); text not read`;
  return `Box from ${plate.detection_source}; no calibrated score available`;
}

/* ------------------------------------------------------------ canvas paint */

function drawImageOnCanvas(file, plates, token) {
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      if (token !== renderToken) return; // stale paint from an earlier file
      resultCanvas.width = img.naturalWidth;
      resultCanvas.height = img.naturalHeight;
      const ctx = resultCanvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      plates.forEach((plate) => drawPlateBox(ctx, plate, img.naturalWidth));
    };
    img.onerror = () => showStatus("That image could not be displayed.", "error", false);
    img.src = event.target.result;
  };
  reader.onerror = () => showStatus("That file could not be read.", "error", false);
  reader.readAsDataURL(file);
}

function drawPlateBox(ctx, plate, imageWidth) {
  const [x, y, w, h] = plate.box;
  // Label the box with the text that was actually read from that box.
  const label = plate.text || "unreadable";
  const readable = Boolean(plate.text);

  ctx.strokeStyle = readable ? "#00ff66" : "#ffb020";
  ctx.lineWidth = Math.max(3, Math.round(imageWidth / 300));
  ctx.strokeRect(x, y, w, h);

  const fontSize = Math.max(16, Math.round(imageWidth / 40));
  ctx.font = `bold ${fontSize}px sans-serif`;
  const padding = Math.round(fontSize * 0.35);
  const textWidth = ctx.measureText(label).width;
  const boxHeight = fontSize + padding * 2;
  // Keep the tag on screen: above the box normally, inside it when there is no room.
  const tagY = y - boxHeight >= 0 ? y - boxHeight : y;

  ctx.fillStyle = readable ? "#00ff66" : "#ffb020";
  ctx.fillRect(x, tagY, textWidth + padding * 2, boxHeight);
  ctx.fillStyle = "#000000";
  ctx.fillText(label, x + padding, tagY + fontSize + padding * 0.5);
}

/* ----------------------------------------------------------------- history */

function addToHistory(plate) {
  detectionHistory.unshift({
    plate: plate.text || null,
    isValid: Boolean(plate.is_valid_format),
    state: plate.state_name || null,
    city: plate.city || null,
    rtoCode: plate.full_rto_code || null,
    confidence: typeof plate.confidence === "number" ? plate.confidence : null,
    engine: plate.ocr_engine || null,
    detectedAt: new Date().toISOString(),
  });
  if (detectionHistory.length > HISTORY_LIMIT) {
    detectionHistory.length = HISTORY_LIMIT;
  }
  saveHistory();
  renderHistory();
}

function renderHistory() {
  // Every statistic below is computed from stored detections.
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);

  const todayCount = detectionHistory.filter(
    (item) => new Date(item.detectedAt).getTime() >= startOfToday.getTime(),
  ).length;
  const uniquePlates = new Set(detectionHistory.filter((i) => i.plate).map((i) => i.plate));

  totalDetectionsVal.textContent = detectionHistory.length.toLocaleString();
  todayDetectionsVal.textContent = todayCount.toLocaleString();
  uniqueVehiclesVal.textContent = uniquePlates.size.toLocaleString();

  if (!detectionHistory.length) {
    recentDetectionsList.innerHTML =
      `<div class="empty-recent-msg"><span>No detections recorded yet. Upload an image or video to begin.</span></div>`;
    return;
  }

  recentDetectionsList.innerHTML = detectionHistory.slice(0, 20).map((item) => {
    const label = item.plate || "unreadable";
    const badge = item.plate ? item.plate.slice(0, 4) : "—";
    const place = [item.state, item.city].filter(Boolean).join(", ") || "no RTO match";
    return `
      <div class="recent-item">
        <div class="recent-badge-img${item.plate ? "" : " muted"}">${escapeHtml(badge)}</div>
        <div class="recent-info">
          <span class="recent-plate">${escapeHtml(label)}</span>
          <span class="recent-sub">${escapeHtml(place)} · ${formatConfidence(item.confidence)}</span>
        </div>
        <span class="recent-time">${escapeHtml(relativeTime(item.detectedAt))}</span>
      </div>`;
  }).join("");
}

clearHistoryBtn.addEventListener("click", (event) => {
  event.preventDefault();
  detectionHistory = [];
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

/* ------------------------------------------------------------------- theme */

function applyTheme(theme) {
  const light = theme === "light";
  document.body.classList.toggle("light-theme", light);
  themeToggleBtn.textContent = light ? "🌙" : "☀️";
  themeToggleBtn.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
}

themeToggleBtn.addEventListener("click", () => {
  const next = document.body.classList.contains("light-theme") ? "dark" : "light";
  applyTheme(next);
  localStorage.setItem(THEME_KEY, next);
});

/* -------------------------------------------------------- sidebar sections */

// The sidebar links previously did nothing. Wire them to the real sections.
document.querySelectorAll(".nav-item[data-target]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    const target = document.getElementById(link.dataset.target);
    if (!target) return;
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
    link.classList.add("active");
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

/* -------------------------------------------------------------- initialise */

applyTheme(localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark");
resetDetectionResults();
renderHistory();
checkBackend().then((ok) => {
  if (ok) fetchRtoDataset();
});
