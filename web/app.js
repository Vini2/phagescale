const state = {
  pyodide: null,
  imageLoaded: false,
  imageName: "",
  imageSize: { width: 0, height: 0 },
  scaleBar: null,
  nmPerPx: null,
  mode: "straight",
  zoom: 1,
  isDrawing: false,
  activePath: [],
  measurements: [],
};

const els = {
  exampleButton: document.querySelector("#exampleButton"),
  dismissHelpButton: document.querySelector("#dismissHelpButton"),
  helpPanel: document.querySelector("#helpPanel"),
  imageInput: document.querySelector("#imageInput"),
  canvasStage: document.querySelector("#canvasStage"),
  canvasStack: document.querySelector("#canvasStack"),
  imageCanvas: document.querySelector("#imageCanvas"),
  overlayCanvas: document.querySelector("#overlayCanvas"),
  emptyState: document.querySelector("#emptyState"),
  detectButton: document.querySelector("#detectButton"),
  scaleNm: document.querySelector("#scaleNm"),
  barPx: document.querySelector("#barPx"),
  calibration: document.querySelector("#calibration"),
  scaleMessage: document.querySelector("#scaleMessage"),
  straightMode: document.querySelector("#straightMode"),
  wavyMode: document.querySelector("#wavyMode"),
  zoomOutButton: document.querySelector("#zoomOutButton"),
  zoomSlider: document.querySelector("#zoomSlider"),
  zoomInButton: document.querySelector("#zoomInButton"),
  resetZoomButton: document.querySelector("#resetZoomButton"),
  zoomValue: document.querySelector("#zoomValue"),
  undoButton: document.querySelector("#undoButton"),
  clearButton: document.querySelector("#clearButton"),
  saveImageButton: document.querySelector("#saveImageButton"),
  exportButton: document.querySelector("#exportButton"),
  measurementList: document.querySelector("#measurementList"),
};

const imageCtx = els.imageCanvas.getContext("2d", { willReadFrequently: true });
const overlayCtx = els.overlayCanvas.getContext("2d");
const LINE_COLORS = ["#ff8a00", "#00d5ff", "#ff3fb4", "#9cff00", "#8f5cff", "#ff3b30", "#2f7bff", "#ffd400"];
const EXAMPLE_IMAGE_URL = "./examples/MarsHill.jpeg";

function pyResultToObject(result) {
  if (result && typeof result.toJs === "function") {
    const converted = result.toJs({ dict_converter: Object.fromEntries });
    result.destroy();
    return converted;
  }
  return result;
}

async function initPyodide() {
  try {
    state.pyodide = await loadPyodide();
    await state.pyodide.loadPackage("numpy");
    const response = await fetch(`./phagescale_pyodide.py?v=${Date.now()}`, { cache: "no-store" });
    state.pyodide.runPython(await response.text());
    refreshButtons();
  } catch (error) {
    els.scaleMessage.textContent = error.message;
    els.scaleMessage.classList.add("warning");
  }
}

function setCanvasSize(width, height) {
  for (const canvas of [els.imageCanvas, els.overlayCanvas]) {
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
  els.canvasStack.style.width = `${width}px`;
  els.canvasStack.style.height = `${height}px`;
  applyZoom(1);
}

function refreshButtons() {
  const canDetect = Boolean(state.pyodide && state.imageLoaded);
  els.detectButton.disabled = !canDetect;
  els.zoomOutButton.disabled = !state.imageLoaded || state.zoom <= 0.25;
  els.zoomSlider.disabled = !state.imageLoaded;
  els.zoomInButton.disabled = !state.imageLoaded || state.zoom >= 5;
  els.resetZoomButton.disabled = !state.imageLoaded || state.zoom === 1;
  els.undoButton.disabled = state.measurements.length === 0;
  els.clearButton.disabled = state.measurements.length === 0;
  els.saveImageButton.disabled = !state.imageLoaded || state.measurements.length === 0;
  els.exportButton.disabled = state.measurements.length === 0;
}

function applyZoom(zoom, anchor = null) {
  const previousZoom = state.zoom;
  const nextZoom = Math.min(5, Math.max(0.25, zoom));
  const frame = els.canvasStage.parentElement;
  const frameRect = frame.getBoundingClientRect();
  const anchorX = anchor?.clientX ?? frameRect.left + frameRect.width / 2;
  const anchorY = anchor?.clientY ?? frameRect.top + frameRect.height / 2;
  const contentX = (frame.scrollLeft + anchorX - frameRect.left) / previousZoom;
  const contentY = (frame.scrollTop + anchorY - frameRect.top) / previousZoom;

  state.zoom = nextZoom;
  els.canvasStage.style.width = `${els.imageCanvas.width * nextZoom}px`;
  els.canvasStage.style.height = `${els.imageCanvas.height * nextZoom}px`;
  els.canvasStack.style.transform = `scale(${nextZoom})`;
  els.zoomSlider.value = Math.round(nextZoom * 100);
  els.zoomValue.textContent = `${Math.round(nextZoom * 100)}%`;
  frame.scrollLeft = contentX * nextZoom - (anchorX - frameRect.left);
  frame.scrollTop = contentY * nextZoom - (anchorY - frameRect.top);
  refreshButtons();
}

function stepZoom(direction) {
  const current = Math.round(state.zoom * 100);
  const next = current + direction * (current < 100 ? 10 : 25);
  applyZoom(next / 100);
}

function updateCalibration() {
  if (!state.scaleBar) {
    state.nmPerPx = null;
    els.barPx.textContent = "-";
    els.calibration.textContent = "-";
    return;
  }

  const scaleNm = Number.parseFloat(els.scaleNm.value);
  if (!Number.isFinite(scaleNm) || scaleNm <= 0) {
    state.nmPerPx = null;
    els.calibration.textContent = "Enter nm value";
    return;
  }

  state.nmPerPx = scaleNm / state.scaleBar.lengthPx;
  els.barPx.textContent = `${state.scaleBar.lengthPx.toFixed(1)} px`;
  els.calibration.textContent = `${state.nmPerPx.toFixed(4)} nm/px`;
}

function canvasPoint(event) {
  const rect = els.overlayCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * els.overlayCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * els.overlayCanvas.height,
  };
}

function labelAnchor(path) {
  const point = path[0] ?? { x: 12, y: 12 };
  return {
    x: Math.min(els.overlayCanvas.width - 28, Math.max(12, point.x + 10)),
    y: Math.min(els.overlayCanvas.height - 12, Math.max(22, point.y - 10)),
  };
}

function drawLineLabel(path, measurement) {
  const anchor = labelAnchor(path);
  overlayCtx.save();
  overlayCtx.fillStyle = measurement.color;
  overlayCtx.strokeStyle = "#07100d";
  overlayCtx.lineWidth = 3;
  overlayCtx.beginPath();
  overlayCtx.arc(anchor.x, anchor.y, 12, 0, Math.PI * 2);
  overlayCtx.fill();
  overlayCtx.stroke();
  overlayCtx.fillStyle = "#07100d";
  overlayCtx.font = "800 14px ui-monospace, SFMono-Regular, Menlo, monospace";
  overlayCtx.textAlign = "center";
  overlayCtx.textBaseline = "middle";
  overlayCtx.fillText(String(measurement.id), anchor.x, anchor.y + 0.5);
  overlayCtx.restore();
}

function drawPath(path, options = {}) {
  if (path.length < 2) return;
  overlayCtx.save();
  overlayCtx.lineCap = "round";
  overlayCtx.lineJoin = "round";
  overlayCtx.lineWidth = options.preview ? 3 : 4;
  overlayCtx.strokeStyle = options.color ?? (options.preview ? "#ffffff" : LINE_COLORS[0]);
  if (options.preview) overlayCtx.setLineDash([8, 6]);
  overlayCtx.beginPath();
  overlayCtx.moveTo(path[0].x, path[0].y);
  for (const point of path.slice(1)) overlayCtx.lineTo(point.x, point.y);
  overlayCtx.stroke();
  overlayCtx.restore();
}

function drawScaleBar() {
  if (!state.scaleBar) return;
  const { x, y, width, height } = state.scaleBar.bbox;
  overlayCtx.save();
  overlayCtx.strokeStyle = "#ff4f5e";
  overlayCtx.lineWidth = 3;
  overlayCtx.setLineDash([10, 6]);
  overlayCtx.strokeRect(x - 4, y - 4, width + 8, height + 8);
  overlayCtx.restore();
}

function redrawOverlay() {
  overlayCtx.clearRect(0, 0, els.overlayCanvas.width, els.overlayCanvas.height);
  drawScaleBar();
  for (const measurement of state.measurements) {
    drawPath(measurement.path, { color: measurement.color });
    drawLineLabel(measurement.path, measurement);
  }
  drawPath(state.activePath, { preview: true });
}

async function loadImage(file) {
  const bitmap = await createImageBitmap(file);
  state.imageName = file.name;
  state.imageLoaded = true;
  state.scaleBar = null;
  state.measurements = [];
  setCanvasSize(bitmap.width, bitmap.height);
  imageCtx.clearRect(0, 0, bitmap.width, bitmap.height);
  imageCtx.drawImage(bitmap, 0, 0);
  els.canvasStage.hidden = false;
  els.emptyState.hidden = true;
  els.scaleMessage.textContent = "Image loaded. Detect the scale bar, then draw measurements.";
  els.scaleMessage.classList.remove("warning");
  updateCalibration();
  redrawOverlay();
  refreshButtons();
}

async function loadExampleImage() {
  els.exampleButton.disabled = true;
  els.scaleMessage.textContent = "Loading example TEM image...";
  els.scaleMessage.classList.remove("warning");
  try {
    const response = await fetch(EXAMPLE_IMAGE_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`Example image could not be loaded (${response.status}).`);
    const blob = await response.blob();
    const file = new File([blob], "MarsHill.jpeg", { type: blob.type || "image/jpeg" });
    await loadImage(file);
  } catch (error) {
    els.scaleMessage.textContent = error.message;
    els.scaleMessage.classList.add("warning");
  } finally {
    els.exampleButton.disabled = false;
  }
}

async function detectScaleBar() {
  if (!state.pyodide || !state.imageLoaded) return;
  els.detectButton.disabled = true;
  els.scaleMessage.textContent = "Detecting scale bar in Pyodide...";

  const { width, height } = state.imageSize;
  const imageData = imageCtx.getImageData(0, 0, width, height);
  const detect = state.pyodide.globals.get("detect_scale_bar");
  const result = pyResultToObject(detect(imageData.data, width, height));
  detect.destroy();

  if (result.found) {
    state.scaleBar = result;
    els.scaleMessage.textContent = `${result.message} Check the red box before measuring.`;
  } else {
    state.scaleBar = null;
    els.scaleMessage.textContent = result.message;
  }
  els.scaleMessage.classList.toggle("warning", !result.found);

  updateCalibration();
  redrawOverlay();
  refreshButtons();
}

async function finishMeasurement() {
  state.isDrawing = false;
  if (state.activePath.length < 2 || !state.nmPerPx) {
    state.activePath = [];
    redrawOverlay();
    return;
  }

  const measure = state.pyodide.globals.get("measure_path");
  const result = pyResultToObject(measure(state.activePath, state.nmPerPx));
  measure.destroy();

  state.measurements.push({
    id: state.measurements.length + 1,
    mode: state.mode,
    color: LINE_COLORS[state.measurements.length % LINE_COLORS.length],
    path: [...state.activePath],
    lengthPx: result.lengthPx,
    lengthNm: result.lengthNm,
  });
  state.activePath = [];
  renderMeasurements();
  redrawOverlay();
  refreshButtons();
}

function renderMeasurements() {
  els.measurementList.replaceChildren();
  for (const measurement of state.measurements) {
    const lengthNm = state.nmPerPx ? measurement.lengthPx * state.nmPerPx : measurement.lengthNm;
    const item = document.createElement("li");
    item.style.setProperty("--line-color", measurement.color);
    item.innerHTML = `
      <span>Line ${measurement.id}</span>
      <strong>${lengthNm.toFixed(2)} nm</strong>
      <small>${measurement.lengthPx.toFixed(1)} px</small>
    `;
    els.measurementList.append(item);
  }
}

function setMode(mode) {
  state.mode = mode;
  els.straightMode.classList.toggle("active", mode === "straight");
  els.wavyMode.classList.toggle("active", mode === "wavy");
}

function exportCsv() {
  const rows = [
    ["image", "line_id", "mode", "color", "length_px", "length_nm", "points_json"],
    ...state.measurements.map((measurement) => [
      state.imageName,
      measurement.id,
      measurement.mode,
      measurement.color,
      measurement.lengthPx.toFixed(3),
      (state.nmPerPx ? measurement.lengthPx * state.nmPerPx : measurement.lengthNm).toFixed(3),
      JSON.stringify(measurement.path),
    ]),
  ];
  const csv = rows
    .map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(","))
    .join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "phagescale-measurements.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function downloadBlobUrl(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
}

function safeStem(filename) {
  return (filename || "tem-image").replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "") || "tem-image";
}

function saveAnnotatedImage() {
  if (!state.imageLoaded || state.measurements.length === 0) return;
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = els.imageCanvas.width;
  exportCanvas.height = els.imageCanvas.height;
  const exportCtx = exportCanvas.getContext("2d");
  exportCtx.drawImage(els.imageCanvas, 0, 0);
  exportCtx.drawImage(els.overlayCanvas, 0, 0);
  exportCanvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    downloadBlobUrl(url, `${safeStem(state.imageName)}_annotated.png`);
    URL.revokeObjectURL(url);
  }, "image/png");
}

els.imageInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) loadImage(file);
});

els.exampleButton.addEventListener("click", loadExampleImage);
els.dismissHelpButton.addEventListener("click", () => {
  els.helpPanel.hidden = true;
});
els.detectButton.addEventListener("click", detectScaleBar);
els.scaleNm.addEventListener("input", () => {
  updateCalibration();
  renderMeasurements();
});
els.straightMode.addEventListener("click", () => setMode("straight"));
els.wavyMode.addEventListener("click", () => setMode("wavy"));
els.zoomOutButton.addEventListener("click", () => stepZoom(-1));
els.zoomInButton.addEventListener("click", () => stepZoom(1));
els.resetZoomButton.addEventListener("click", () => applyZoom(1));
els.zoomSlider.addEventListener("input", () => applyZoom(Number.parseInt(els.zoomSlider.value, 10) / 100));
els.undoButton.addEventListener("click", () => {
  state.measurements.pop();
  renderMeasurements();
  redrawOverlay();
  refreshButtons();
});
els.clearButton.addEventListener("click", () => {
  state.measurements = [];
  renderMeasurements();
  redrawOverlay();
  refreshButtons();
});
els.exportButton.addEventListener("click", exportCsv);
els.saveImageButton.addEventListener("click", saveAnnotatedImage);

els.canvasStage.parentElement.addEventListener(
  "wheel",
  (event) => {
    if (!state.imageLoaded || !(event.ctrlKey || event.metaKey)) return;
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    applyZoom(state.zoom * factor, event);
  },
  { passive: false },
);

els.overlayCanvas.addEventListener("pointerdown", (event) => {
  if (!state.imageLoaded) return;
  if (!state.scaleBar) {
    els.scaleMessage.textContent = "Detect the scale bar before drawing measurements.";
    els.scaleMessage.classList.add("warning");
    return;
  }
  if (!state.nmPerPx) {
    els.scaleMessage.textContent = "Enter a valid scale bar value in nm before drawing measurements.";
    els.scaleMessage.classList.add("warning");
    return;
  }
  els.overlayCanvas.setPointerCapture(event.pointerId);
  state.isDrawing = true;
  const point = canvasPoint(event);
  state.activePath = [point, point];
  redrawOverlay();
});

els.overlayCanvas.addEventListener("pointermove", (event) => {
  if (!state.isDrawing) return;
  const point = canvasPoint(event);
  if (state.mode === "straight") {
    state.activePath[1] = point;
  } else {
    const last = state.activePath[state.activePath.length - 1];
    if (Math.hypot(point.x - last.x, point.y - last.y) >= 2) state.activePath.push(point);
  }
  redrawOverlay();
});

els.overlayCanvas.addEventListener("pointerup", finishMeasurement);
els.overlayCanvas.addEventListener("pointercancel", () => {
  state.isDrawing = false;
  state.activePath = [];
  redrawOverlay();
});

const resizeObserver = new ResizeObserver(() => redrawOverlay());
resizeObserver.observe(els.overlayCanvas);

Object.defineProperty(state, "imageSize", {
  get() {
    return { width: els.imageCanvas.width, height: els.imageCanvas.height };
  },
});

initPyodide();
