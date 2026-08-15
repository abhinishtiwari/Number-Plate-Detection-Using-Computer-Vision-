/**
 * ===================================================================
 * API CONFIGURATION
 * ===================================================================
 * Change the API_URL constant below to point to your deployed Render backend URL.
 * Example: const API_URL = "https://number-plate-backend.onrender.com";
 */
const API_URL = "https://your-backend.onrender.com";

// DOM Elements
const fileInput = document.getElementById("fileInput");
const cameraInput = document.getElementById("cameraInput");
const detectBtn = document.getElementById("detectBtn");
const fileNameDisplay = document.getElementById("fileNameDisplay");
const dropZone = document.getElementById("dropZone");
const statusBox = document.getElementById("statusBox");
const statusMessage = document.getElementById("statusMessage");
const spinner = document.getElementById("spinner");
const resultCard = document.getElementById("resultCard");
const resultCanvas = document.getElementById("resultCanvas");
const resultsTable = document.getElementById("resultsTable");

let selectedFile = null;

// Event Listeners for File Selection
fileInput.addEventListener("change", (e) => handleFileSelect(e.target.files[0]));
cameraInput.addEventListener("change", (e) => handleFileSelect(e.target.files[0]));

// Drag and Drop Handlers
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

function handleFileSelect(file) {
  if (!file || !file.type.startsWith("image/")) {
    showStatus("Please select a valid image file.", "error");
    return;
  }
  selectedFile = file;
  fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  detectBtn.disabled = false;
  hideStatus();
  resultCard.classList.add("hidden");
}

/**
 * Polls the backend health check endpoint GET /health until the server is awake.
 * Handles Render free-tier cold starts (spins down after inactivity, takes 30-60s to boot).
 */
async function waitForBackendServer(maxWaitSeconds = 60) {
  const startTime = Date.now();
  showStatus("Waking up the detection server — this can take up to a minute on first use...", "info", true);

  while ((Date.now() - startTime) / 1000 < maxWaitSeconds) {
    try {
      const response = await fetch(`${API_URL}/health`, { method: "GET" });
      if (response.ok) {
        const data = await response.json();
        if (data.status === "ok") {
          return true;
        }
      }
    } catch (err) {
      // Backend is still spinning up or cold starting, retry after delay
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }

  throw new Error("Server wakeup timed out. Please try again.");
}

/**
 * Main Detection Trigger
 */
detectBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  detectBtn.disabled = true;
  resultCard.classList.add("hidden");

  try {
    // Step 1: Ensure backend server is awake (handles Render cold start)
    await waitForBackendServer(60);

    // Step 2: Call detection endpoint POST /detect
    showStatus("Detecting number plate...", "info", true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${API_URL}/detect`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    const data = await response.json();
    showStatus("Detection complete!", "success", false);

    // Step 3: Render results onto canvas
    renderResultsOnCanvas(selectedFile, data.plates);

  } catch (error) {
    console.error("Detection Error:", error);
    showStatus("Couldn't reach the detection server or processing failed. Please try again.", "error", false);
  } finally {
    detectBtn.disabled = false;
  }
});

/**
 * Renders the uploaded image onto HTML5 canvas with bounding boxes & labels
 */
function renderResultsOnCanvas(file, plates) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      // Set canvas dimensions to match original image
      resultCanvas.width = img.width;
      resultCanvas.height = img.height;

      const ctx = resultCanvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      resultsTable.innerHTML = "";

      if (!plates || plates.length === 0) {
        resultsTable.innerHTML = `<div class="plate-badge"><span style="color:#94a3b8;">No license plates detected in this image.</span></div>`;
        resultCard.classList.remove("hidden");
        return;
      }

      plates.forEach((plate, idx) => {
        const [x, y, w, h] = plate.box;
        const text = plate.text || `Plate #${idx + 1}`;

        // Draw Bounding Box
        ctx.strokeStyle = "#00ff66";
        ctx.lineWidth = Math.max(3, Math.round(img.width / 300));
        ctx.strokeRect(x, y, w, h);

        // Draw Text Background Tag
        const fontSize = Math.max(16, Math.round(img.width / 40));
        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        const textWidth = ctx.measureText(text).width;
        const padding = 8;

        ctx.fillStyle = "#00ff66";
        ctx.fillRect(x, y - fontSize - padding * 2 > 0 ? y - fontSize - padding * 2 : y, textWidth + padding * 2, fontSize + padding);

        // Draw Text Label
        ctx.fillStyle = "#000000";
        ctx.fillText(
          text,
          x + padding,
          y - fontSize - padding * 2 > 0 ? y - padding : y + fontSize
        );

        // Populate Table List
        const plateRow = document.createElement("div");
        plateRow.className = "plate-badge";
        plateRow.innerHTML = `
          <div>
            <span class="plate-text">${text}</span>
          </div>
          <div class="plate-bbox">
            Box: (${x}, ${y}, ${w}, ${h})
          </div>
        `;
        resultsTable.appendChild(plateRow);
      });

      resultCard.classList.remove("hidden");
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// UI Helper Functions
function showStatus(msg, type = "info", showSpinner = false) {
  statusBox.className = `status-box ${type}`;
  statusMessage.textContent = msg;
  if (showSpinner) {
    spinner.classList.remove("hidden");
  } else {
    spinner.classList.add("hidden");
  }
  statusBox.classList.remove("hidden");
}

function hideStatus() {
  statusBox.classList.add("hidden");
}
