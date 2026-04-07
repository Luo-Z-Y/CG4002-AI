const gestureLabel = document.getElementById("gesture-label");
const gestureConfidence = document.getElementById("gesture-confidence");
const gestureSamples = document.getElementById("gesture-samples");
const gestureSource = document.getElementById("gesture-source");
const gestureStatus = document.getElementById("gesture-status");
const gestureProbs = document.getElementById("gesture-probs");
const gestureReviewStatus = document.getElementById("gesture-review-status");
const gestureMarkCorrect = document.getElementById("gesture-mark-correct");
const gestureCorrectLabel = document.getElementById("gesture-correct-label");
const gestureMarkWrong = document.getElementById("gesture-mark-wrong");

const voiceLabel = document.getElementById("voice-label");
const voiceConfidence = document.getElementById("voice-confidence");
const voiceFileName = document.getElementById("voice-file-name");
const voiceSource = document.getElementById("voice-source");
const voiceStatus = document.getElementById("voice-status");
const voiceProbs = document.getElementById("voice-probs");
const voiceReviewStatus = document.getElementById("voice-review-status");
const voiceMarkCorrect = document.getElementById("voice-mark-correct");
const voiceCorrectLabel = document.getElementById("voice-correct-label");
const voiceMarkWrong = document.getElementById("voice-mark-wrong");
const voiceFile = document.getElementById("voice-file");
const recordToggle = document.getElementById("record-toggle");
const voicePlayerWrap = document.getElementById("voice-player-wrap");
const voicePlayer = document.getElementById("voice-player");
const voiceProcessedPlayer = document.getElementById("voice-processed-player");
const dashboardGrid = document.getElementById("dashboard-grid");
const viewBothButton = document.getElementById("view-both");
const viewGestureButton = document.getElementById("view-gesture");
const viewVoiceButton = document.getElementById("view-voice");
const viewCleanupButton = document.getElementById("view-cleanup");
const sessionDir = document.getElementById("session-dir");
const cleanupStatus = document.getElementById("cleanup-status");
const cleanupRefresh = document.getElementById("cleanup-refresh");
const cleanupSummary = document.getElementById("cleanup-summary");
const cleanupMeta = document.getElementById("cleanup-meta");
const cleanupList = document.getElementById("cleanup-list");

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
let currentGestureSampleId = null;
let currentVoiceSampleId = null;
let gestureLabels = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"];
let voiceLabels = ["Bulbasaur", "Charizard", "Greninja", "Lugia", "Mewtwo", "Pikachu"];

function populateLabelOptions(select, labels) {
  select.innerHTML = '<option value="">Correct label</option>';
  for (const label of labels) {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label;
    select.appendChild(option);
  }
}

function setDashboardView(mode) {
  dashboardGrid.classList.remove("focus-gesture", "focus-voice", "focus-cleanup");
  viewBothButton.classList.toggle("is-active", mode === "both");
  viewGestureButton.classList.toggle("is-active", mode === "gesture");
  viewVoiceButton.classList.toggle("is-active", mode === "voice");
  viewCleanupButton.classList.toggle("is-active", mode === "cleanup");

  if (mode === "gesture") {
    dashboardGrid.classList.add("focus-gesture");
  } else if (mode === "voice") {
    dashboardGrid.classList.add("focus-voice");
  } else if (mode === "cleanup") {
    dashboardGrid.classList.add("focus-cleanup");
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

function renderReviewState(kind, sample) {
  const isGesture = kind === "gesture";
  const statusNode = isGesture ? gestureReviewStatus : voiceReviewStatus;
  const correctButton = isGesture ? gestureMarkCorrect : voiceMarkCorrect;
  const labelSelect = isGesture ? gestureCorrectLabel : voiceCorrectLabel;
  const wrongButton = isGesture ? gestureMarkWrong : voiceMarkWrong;
  const labels = isGesture ? gestureLabels : voiceLabels;

  populateLabelOptions(labelSelect, labels);

  if (!sample || !sample.sample_id) {
    statusNode.textContent = `Waiting for first ${kind} sample`;
    correctButton.disabled = true;
    labelSelect.disabled = true;
    wrongButton.disabled = true;
    return;
  }

  const review = sample.review || {};
  if (review.status === "reviewed") {
    if (review.is_correct) {
      statusNode.textContent = `Saved: prediction confirmed for ${sample.sample_id}`;
    } else {
      statusNode.textContent = `Saved: corrected to ${review.correct_label} for ${sample.sample_id}`;
      labelSelect.value = review.correct_label || "";
    }
  } else {
    statusNode.textContent = `Sample ${sample.sample_id} captured. Mark whether ${sample.label} is correct.`;
  }

  correctButton.disabled = false;
  labelSelect.disabled = false;
  wrongButton.disabled = false;
}

async function submitReview(kind, isCorrect) {
  const sampleId = kind === "gesture" ? currentGestureSampleId : currentVoiceSampleId;
  const labelSelect = kind === "gesture" ? gestureCorrectLabel : voiceCorrectLabel;
  const statusNode = kind === "gesture" ? gestureReviewStatus : voiceReviewStatus;

  if (!sampleId) {
    return;
  }

  const payload = {
    kind,
    sample_id: sampleId,
    is_correct: isCorrect,
  };
  if (!isCorrect) {
    if (!labelSelect.value) {
      alert("Choose the correct label first.");
      return;
    }
    payload.correct_label = labelSelect.value;
  }

  statusNode.textContent = "Saving review...";
  const response = await fetch("/api/review", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || "Failed to save review");
  }

  const stateKey = kind === "gesture" ? "gesture" : "voice";
  const latest = result.review;
  if (kind === "gesture" && window.__latestGesture) {
    window.__latestGesture.review = latest;
  }
  if (kind === "voice" && window.__latestVoice) {
    window.__latestVoice.review = latest;
  }
  renderReviewState(kind, kind === "gesture" ? window.__latestGesture : window.__latestVoice);
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
  window.__latestGesture = gesture;
  currentGestureSampleId = gesture.sample_id || null;
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
  renderReviewState("gesture", gesture);
}

function updateVoiceView(voice) {
  if (!voice) return;
  window.__latestVoice = voice;
  currentVoiceSampleId = voice.sample_id || null;
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
  renderReviewState("voice", voice);
}

function renderCleanupMeta(payload) {
  cleanupMeta.innerHTML = "";
  if (!payload) {
    return;
  }
  const rows = [
    ["Artefact dir", payload.artifact_dir || "-"],
    ["Predictions CSV", payload.predictions_path || "-"],
    ["Deleted log", payload.deleted_log_path || "-"],
  ];
  rows.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "cleanup-meta-item";
    item.innerHTML = `<span class="label">${label}</span><code>${value}</code>`;
    cleanupMeta.appendChild(item);
  });
}

function renderCleanupList(items) {
  cleanupList.innerHTML = "";
  if (!items || !items.length) {
    const empty = document.createElement("div");
    empty.className = "cleanup-empty";
    empty.textContent = "No misclassified clips to review in the current predictions file.";
    cleanupList.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "cleanup-card";
    const top = document.createElement("div");
    top.className = "cleanup-card-top";
    top.innerHTML = `
      <div>
        <div class="cleanup-title">manifest_idx=${item.manifest_idx ?? "-"}</div>
        <div class="cleanup-subtitle">true=${item.true_label ?? "-"} | pred=${item.pred_label ?? "-"} | q88=${item.q88_pred_label ?? "-"}</div>
      </div>
      <button class="button cleanup-delete"${item.exists ? "" : " disabled"}>${item.exists ? "Delete clip" : "Missing"}</button>
    `;

    const meta = document.createElement("div");
    meta.className = "cleanup-card-meta";
    meta.innerHTML = `
      <span>speaker=${item.speaker_id || "-"}</span>
      <span>utterance=${item.utterance_id || "-"}</span>
      <span>source=${item.source || "-"}</span>
      <span>variant=${item.model_variant || "-"}</span>
      <span>split_seed=${item.split_seed || "-"}</span>
    `;

    const pathNode = document.createElement("code");
    pathNode.className = "cleanup-path";
    pathNode.textContent = item.path;

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = `/api/voice-cleanup/audio?path=${encodeURIComponent(item.path)}`;

    const status = document.createElement("div");
    status.className = "cleanup-row-status";

    const deleteButton = top.querySelector(".cleanup-delete");
    deleteButton.addEventListener("click", async () => {
      if (!item.exists) {
        return;
      }
      const confirmed = window.confirm(`Delete this clip?\n\n${item.path}`);
      if (!confirmed) {
        return;
      }
      deleteButton.disabled = true;
      deleteButton.textContent = "Deleting...";
      status.textContent = "Deleting clip...";
      try {
        const response = await fetch("/api/voice-cleanup/delete", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({path: item.path}),
        });
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.error || "Failed to delete clip");
        }
        card.remove();
        status.textContent = "";
        await fetchCleanupList();
      } catch (error) {
        deleteButton.disabled = false;
        deleteButton.textContent = "Delete clip";
        status.textContent = error.message;
      }
    });

    card.appendChild(top);
    card.appendChild(meta);
    card.appendChild(pathNode);
    card.appendChild(audio);
    card.appendChild(status);
    cleanupList.appendChild(card);
  });
}

async function fetchCleanupList() {
  cleanupRefresh.disabled = true;
  setChip(cleanupStatus, "Loading", "#e6b13a");
  cleanupSummary.textContent = "Refreshing misclassified clips...";
  try {
    const response = await fetch("/api/voice-cleanup/list?limit=60");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Failed to load cleanup list");
    }
    renderCleanupMeta(payload);
    renderCleanupList(payload.items || []);
    if (payload.error) {
      cleanupSummary.textContent = payload.error;
      setChip(cleanupStatus, "Waiting", "#8b5e3c");
    } else {
      cleanupSummary.textContent = `${payload.total || 0} misclassified clips found.`;
      setChip(cleanupStatus, "Ready", "#0f7c82");
    }
  } catch (error) {
    cleanupSummary.textContent = error.message;
    cleanupMeta.innerHTML = "";
    cleanupList.innerHTML = "";
    setChip(cleanupStatus, "Error", "#8b5e3c");
  } finally {
    cleanupRefresh.disabled = false;
  }
}

async function pollState() {
  try {
    const response = await fetch("/api/state");
    const state = await response.json();
    if (Array.isArray(state.gesture_labels) && state.gesture_labels.length) {
      gestureLabels = state.gesture_labels;
    }
    if (Array.isArray(state.voice_labels) && state.voice_labels.length) {
      voiceLabels = state.voice_labels;
    }
    if (state.session_dir) {
      sessionDir.textContent = state.session_dir;
    }
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
viewCleanupButton.addEventListener("click", async () => {
  setDashboardView("cleanup");
  await fetchCleanupList();
});
cleanupRefresh.addEventListener("click", fetchCleanupList);
gestureMarkCorrect.addEventListener("click", async () => {
  try {
    await submitReview("gesture", true);
  } catch (error) {
    gestureReviewStatus.textContent = "Failed to save review";
    alert(error.message);
  }
});
gestureMarkWrong.addEventListener("click", async () => {
  try {
    await submitReview("gesture", false);
  } catch (error) {
    gestureReviewStatus.textContent = "Failed to save review";
    alert(error.message);
  }
});
voiceMarkCorrect.addEventListener("click", async () => {
  try {
    await submitReview("voice", true);
  } catch (error) {
    voiceReviewStatus.textContent = "Failed to save review";
    alert(error.message);
  }
});
voiceMarkWrong.addEventListener("click", async () => {
  try {
    await submitReview("voice", false);
  } catch (error) {
    voiceReviewStatus.textContent = "Failed to save review";
    alert(error.message);
  }
});

setDashboardView("both");
renderReviewState("gesture", null);
renderReviewState("voice", null);
setChip(cleanupStatus, "Idle", "#8b5e3c");
pollState();
setInterval(pollState, 1200);
