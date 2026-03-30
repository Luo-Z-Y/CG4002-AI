const gestureLabel = document.getElementById("gesture-label");
const gestureConfidence = document.getElementById("gesture-confidence");
const gestureSamples = document.getElementById("gesture-samples");
const gestureSource = document.getElementById("gesture-source");
const gestureStatus = document.getElementById("gesture-status");
const gestureProbs = document.getElementById("gesture-probs");

const voiceLabel = document.getElementById("voice-label");
const voiceConfidence = document.getElementById("voice-confidence");
const voiceFileName = document.getElementById("voice-file-name");
const voiceSource = document.getElementById("voice-source");
const voiceStatus = document.getElementById("voice-status");
const voiceProbs = document.getElementById("voice-probs");
const voiceFile = document.getElementById("voice-file");
const recordToggle = document.getElementById("record-toggle");
const voicePlayerWrap = document.getElementById("voice-player-wrap");
const voicePlayer = document.getElementById("voice-player");
const voiceProcessedPlayer = document.getElementById("voice-processed-player");
const dashboardGrid = document.getElementById("dashboard-grid");
const viewBothButton = document.getElementById("view-both");
const viewGestureButton = document.getElementById("view-gesture");
const viewVoiceButton = document.getElementById("view-voice");

const gestureRawCanvas = document.getElementById("gesture-raw");
const gestureProcessedCanvas = document.getElementById("gesture-processed");
const gestureRawTable = document.getElementById("gesture-raw-table");
const gestureProcessedTable = document.getElementById("gesture-processed-table");
const voiceWaveCanvas = document.getElementById("voice-waveform");
const voiceNormalizedCanvas = document.getElementById("voice-normalized");
const voiceMfccCanvas = document.getElementById("voice-mfcc");

const channelColours = ["#d76b39", "#0f7c82", "#e6b13a", "#6e4a2f", "#9a5633", "#2d5258"];

let lastGestureUpdate = null;
let lastVoiceUpdate = null;
let mediaRecorder = null;
let recordedChunks = [];
let currentVoiceUrl = null;
let currentProcessedVoiceUrl = null;
let currentProcessedVoiceBase64 = null;

function setDashboardView(mode) {
  dashboardGrid.classList.remove("focus-gesture", "focus-voice");
  viewBothButton.classList.toggle("is-active", mode === "both");
  viewGestureButton.classList.toggle("is-active", mode === "gesture");
  viewVoiceButton.classList.toggle("is-active", mode === "voice");

  if (mode === "gesture") {
    dashboardGrid.classList.add("focus-gesture");
  } else if (mode === "voice") {
    dashboardGrid.classList.add("focus-voice");
  }
}

function setChip(element, text, background) {
  element.textContent = text;
  element.style.background = background;
}

function setVoicePlayback(blob) {
  if (currentVoiceUrl) {
    URL.revokeObjectURL(currentVoiceUrl);
  }
  currentVoiceUrl = URL.createObjectURL(blob);
  voicePlayer.src = currentVoiceUrl;
  voicePlayer.load();
  voicePlayerWrap.hidden = false;
}

function base64ToBlob(base64Text, contentType = "application/octet-stream") {
  const binary = atob(base64Text);
  const bytes = new Uint8Array(binary.length);
  for (let idx = 0; idx < binary.length; idx += 1) {
    bytes[idx] = binary.charCodeAt(idx);
  }
  return new Blob([bytes], {type: contentType});
}

function setProcessedVoicePlayback(wavBase64) {
  if (!wavBase64 || wavBase64 === currentProcessedVoiceBase64) {
    return;
  }
  if (currentProcessedVoiceUrl) {
    URL.revokeObjectURL(currentProcessedVoiceUrl);
  }
  const wavBlob = base64ToBlob(wavBase64, "audio/wav");
  currentProcessedVoiceUrl = URL.createObjectURL(wavBlob);
  currentProcessedVoiceBase64 = wavBase64;
  voiceProcessedPlayer.src = currentProcessedVoiceUrl;
  voiceProcessedPlayer.load();
  voicePlayerWrap.hidden = false;
}

function renderProbabilities(container, probabilities) {
  container.innerHTML = "";
  for (const item of probabilities || []) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span>${item.label}</span>
      <div class="prob-bar"><div class="prob-fill" style="width:${(item.probability * 100).toFixed(1)}%"></div></div>
      <span>${(item.probability * 100).toFixed(1)}%</span>
    `;
    container.appendChild(row);
  }
}

function renderMatrixTable(table, matrix, labels) {
  if (!table) return;
  if (!matrix || !matrix.length) {
    table.innerHTML = "";
    return;
  }

  const header = `
    <thead>
      <tr>
        <th>t</th>
        ${labels.map((label) => `<th>${label}</th>`).join("")}
      </tr>
    </thead>
  `;

  const bodyRows = matrix.map((row, index) => `
    <tr>
      <td>${index}</td>
      ${row.map((value) => `<td>${Number(value).toFixed(2)}</td>`).join("")}
    </tr>
  `).join("");

  table.innerHTML = `${header}<tbody>${bodyRows}</tbody>`;
}

function drawMultiSeries(canvas, matrix, labels) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!matrix || !matrix.length) {
    return;
  }

  const rows = matrix.length;
  const cols = matrix[0].length;
  const flat = matrix.flat();
  const maxAbs = Math.max(...flat.map((value) => Math.abs(value)), 1);
  const pad = 22;
  const width = canvas.width - pad * 2;
  const height = canvas.height - pad * 2;

  ctx.strokeStyle = "rgba(23, 35, 33, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad + height / 2);
  ctx.lineTo(pad + width, pad + height / 2);
  ctx.stroke();

  for (let col = 0; col < cols; col += 1) {
    ctx.strokeStyle = channelColours[col % channelColours.length];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let row = 0; row < rows; row += 1) {
      const x = pad + (row / Math.max(rows - 1, 1)) * width;
      const y = pad + height / 2 - (matrix[row][col] / maxAbs) * (height * 0.44);
      if (row === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  ctx.font = "12px Avenir Next, sans-serif";
  ctx.fillStyle = "rgba(23, 35, 33, 0.65)";
  labels.forEach((label, idx) => {
    ctx.fillStyle = channelColours[idx % channelColours.length];
    ctx.fillText(label, pad + idx * 62, 14);
  });
}

function drawWaveform(canvas, values) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!values || !values.length) {
    return;
  }

  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 1e-6);
  const pad = 20;
  const width = canvas.width - pad * 2;
  const height = canvas.height - pad * 2;

  ctx.strokeStyle = "rgba(23, 35, 33, 0.12)";
  ctx.beginPath();
  ctx.moveTo(pad, pad + height / 2);
  ctx.lineTo(pad + width, pad + height / 2);
  ctx.stroke();

  ctx.strokeStyle = "#0f7c82";
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = pad + (index / Math.max(values.length - 1, 1)) * width;
    const y = pad + height / 2 - (value / maxAbs) * (height * 0.44);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawHeatmap(canvas, matrix) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!matrix || !matrix.length || !matrix[0].length) {
    return;
  }

  const rows = matrix.length;
  const cols = matrix[0].length;
  const flat = matrix.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const span = Math.max(max - min, 1e-6);
  const pad = 14;
  const cellW = (canvas.width - pad * 2) / cols;
  const cellH = (canvas.height - pad * 2) / rows;

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const t = (matrix[r][c] - min) / span;
      const hue = 210 - t * 190;
      const light = 92 - t * 42;
      ctx.fillStyle = `hsl(${hue}, 72%, ${light}%)`;
      ctx.fillRect(pad + c * cellW, pad + r * cellH, cellW + 0.5, cellH + 0.5);
    }
  }
}

function updateGestureView(gesture) {
  if (!gesture) return;
  gestureLabel.textContent = gesture.label;
  gestureConfidence.textContent = `${(gesture.confidence * 100).toFixed(1)}%`;
  gestureSamples.textContent = `${gesture.sample_count ?? "-"}`;
  gestureSource.textContent = gesture.source || "hardware-client";
  setChip(gestureStatus, "Live", "#d76b39");
  renderProbabilities(gestureProbs, gesture.probabilities);
  drawMultiSeries(gestureRawCanvas, gesture.raw_window, ["gx", "gy", "gz", "ax", "ay", "az"]);
  drawMultiSeries(gestureProcessedCanvas, gesture.processed_window, ["gx", "gy", "gz", "ax", "ay", "az"]);
  renderMatrixTable(gestureRawTable, gesture.raw_window, ["gx", "gy", "gz", "ax", "ay", "az"]);
  renderMatrixTable(gestureProcessedTable, gesture.processed_window, ["gx", "gy", "gz", "ax", "ay", "az"]);
}

function updateVoiceView(voice) {
  if (!voice) return;
  voiceLabel.textContent = voice.label;
  voiceConfidence.textContent = `${(voice.confidence * 100).toFixed(1)}%`;
  voiceFileName.textContent = voice.filename || "Recorded clip";
  voiceSource.textContent = voice.source || "voice-web";
  setChip(voiceStatus, "Analysed", "#0f7c82");
  renderProbabilities(voiceProbs, voice.probabilities);
  drawWaveform(voiceWaveCanvas, voice.raw_waveform);
  drawWaveform(voiceNormalizedCanvas, voice.normalized_waveform);
  drawHeatmap(voiceMfccCanvas, voice.mfcc);
  if (voice.normalized_waveform_wav_base64) {
    setProcessedVoicePlayback(voice.normalized_waveform_wav_base64);
  }
}

async function pollState() {
  try {
    const response = await fetch("/api/state");
    const state = await response.json();
    if (state.gesture && state.gesture.updated_at !== lastGestureUpdate) {
      lastGestureUpdate = state.gesture.updated_at;
      updateGestureView(state.gesture);
    }
    if (state.voice && state.voice.updated_at !== lastVoiceUpdate) {
      lastVoiceUpdate = state.voice.updated_at;
      updateVoiceView(state.voice);
    }
  } catch (error) {
    setChip(gestureStatus, "Offline", "#8b5e3c");
    setChip(voiceStatus, "Offline", "#8b5e3c");
  }
}

async function uploadVoiceBlob(blob, filename) {
  setChip(voiceStatus, "Uploading", "#e6b13a");
  const bytes = await blob.arrayBuffer();
  const binary = new Uint8Array(bytes);
  let text = "";
  binary.forEach((value) => {
    text += String.fromCharCode(value);
  });
  const payload = {
    filename,
    content_type: blob.type,
    audio_base64: btoa(text),
    source: "voice-web",
  };

  const response = await fetch("/api/voice/infer", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || "Voice inference failed");
  }
  lastVoiceUpdate = result.updated_at ?? lastVoiceUpdate;
  setVoicePlayback(blob);
  updateVoiceView(result);
}

voiceFile.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    await uploadVoiceBlob(file, file.name);
  } catch (error) {
    setChip(voiceStatus, "Error", "#8b5e3c");
    alert(error.message);
  } finally {
    voiceFile.value = "";
  }
});

recordToggle.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
    };
    mediaRecorder.onstop = async () => {
      setChip(voiceStatus, "Processing", "#e6b13a");
      recordToggle.textContent = "Start recording";
      const blob = new Blob(recordedChunks, {type: mediaRecorder.mimeType || "audio/webm"});
      stream.getTracks().forEach((track) => track.stop());
      try {
        await uploadVoiceBlob(blob, "recorded-clip.webm");
      } catch (error) {
        setChip(voiceStatus, "Error", "#8b5e3c");
        alert(error.message);
      }
    };
    mediaRecorder.start();
    recordToggle.textContent = "Stop recording";
    setChip(voiceStatus, "Recording", "#d76b39");
  } catch (error) {
    alert(`Unable to start recording: ${error.message}`);
  }
});

viewBothButton.addEventListener("click", () => setDashboardView("both"));
viewGestureButton.addEventListener("click", () => setDashboardView("gesture"));
viewVoiceButton.addEventListener("click", () => setDashboardView("voice"));

setDashboardView("both");
pollState();
setInterval(pollState, 1200);
