import * as THREE from "three";
import {
  camera,
  controls,
  main as viewerMain,
  origin_colors,
  points,
  renderer,
  replacePointCloud,
} from "/static/viewer.js";

const LABELS = {
  0: { name: "Erase", color: [0.62, 0.62, 0.62] },
  1: { name: "Plant", color: [0, 1, 0] },
  2: { name: "Ground", color: [1, 0, 0] },
};

let activeLabel = 1;
let appMode = "row";
let activeTarget = { kind: "stem" };
let promptIsPositive = true;
let toolMode = "brush";
let promptIndices = [];
let promptLabels = [];
let previewMask = null;
let samCandidates = [];
let samSelected = 0;
let previewContext = null;
let persistentLabels = [];
let persistentOtype = [];
let persistentLeafid = [];
let persistentPlantId = [];
let allDates = [];
let earlyDates = [];
let separationDates = [];
let ghostedPlantIds = new Set();
let drawPoints = [];
let isDrawing = false;

const overlay = document.getElementById("draw-overlay");
const overlayCtx = overlay.getContext("2d");

function syncOverlaySize() {
  const rect = overlay.getBoundingClientRect();
  overlay.width = Math.round(rect.width);
  overlay.height = Math.round(rect.height);
}

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

function labelColor(label) {
  return LABELS[label]?.color || null;
}

function leafColor(leafid) {
  const palette = [
    [0.00, 0.45, 1.00], [1.00, 0.55, 0.00], [0.00, 0.70, 0.20], [0.95, 0.00, 0.85],
    [0.00, 0.75, 0.85], [0.95, 0.85, 0.00], [0.55, 0.20, 1.00], [1.00, 0.15, 0.15],
    [0.00, 0.95, 0.55], [1.00, 0.35, 0.65], [0.35, 0.70, 1.00], [0.70, 0.45, 0.00],
  ];
  return palette[(leafid - 1) % palette.length];
}

function instanceColor(id) {
  const palette = [
    [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.50, 0.50, 0.50],
    [0.74, 0.74, 0.13], [0.09, 0.75, 0.81], [0.65, 0.30, 0.00], [0.30, 0.65, 1.00],
    [0.85, 0.20, 0.55], [0.10, 0.80, 0.45],
  ];
  return palette[id % palette.length];
}

function cssRgb(color) {
  return `rgb(${Math.round(color[0] * 255)}, ${Math.round(color[1] * 255)}, ${Math.round(color[2] * 255)})`;
}

function isGhostedPlantId(pid) {
  return appMode === "separation" && Number.isInteger(pid) && ghostedPlantIds.has(pid);
}

function isSelectableIndex(index) {
  return !(appMode === "separation" && isGhostedPlantId(persistentPlantId[index]));
}

function plantPointColor(i) {
  const otype = persistentOtype[i] || 0;
  if (otype === 1) return [0.45, 0.24, 0.08];
  if (otype === 2) return leafColor(persistentLeafid[i] || 1);
  return [0.36, 0.36, 0.36];
}

function activeColor() {
  return labelColor(activeLabel) || [0.62, 0.62, 0.62];
}

function targetDisplayName(data = null) {
  if (appMode === "separation") {
    if (activeTarget.kind === "unassign" || activeTarget.kind === "eraser") return "Unassign";
    if (activeTarget.kind === "new_plant") return data?.target_plant_id !== undefined ? `Plant ${data.target_plant_id}` : "New plant";
    return `Plant ${activeTarget.plant_id}`;
  }
  if (appMode !== "plant") return LABELS[activeLabel].name;
  if (activeTarget.kind === "stem") return "Stem";
  if (activeTarget.kind === "eraser") return "Eraser";
  if (activeTarget.kind === "leaf") return `Leaf ${activeTarget.leafid}`;
  const leaf = data?.target_leafid || "";
  return leaf ? `Leaf ${leaf}` : "New leaf";
}

function adoptNewLeafTarget(data) {
  if (appMode === "plant" && activeTarget.kind === "new_leaf" && data?.target_leafid > 0) {
    activeTarget = { kind: "leaf", leafid: data.target_leafid };
    syncTargetButtons();
  }
}

function updateCounts(meta) {
  if (!meta || !meta.counts) return;
  appMode = meta.mode || appMode;
  updateModeVisibility();
  const crop = meta.crop_parent_count ? ` cropped from ${meta.crop_parent_count.toLocaleString()}` : "";
  const fullText = meta.row_count && meta.row_count !== meta.n ? ` of ${meta.row_count.toLocaleString()} full points` : "";
  if (appMode === "separation") {
    document.getElementById("counts").textContent =
      `unassigned ${meta.counts.unassigned.toLocaleString()} | ` +
      `assigned ${meta.counts.assigned.toLocaleString()} | ` +
      `plants ${meta.counts.plants}`;
    document.getElementById("sample-info").textContent =
      `${meta.date} row vegetation: ${meta.n.toLocaleString()} points${fullText}`;
    const newPlantBtn = document.getElementById("target-new-plant");
    if (newPlantBtn && Number.isInteger(meta.next_plant_id)) {
      newPlantBtn.textContent = `New plant ${meta.next_plant_id}`;
    }
    renderPlantList(meta.plant_instances || []);
  } else if (appMode === "plant") {
    document.getElementById("counts").textContent =
      `unlabelled ${meta.counts.unlabeled.toLocaleString()} | ` +
      `stem ${meta.counts.stem.toLocaleString()} | ` +
      `leaf ${meta.counts.leaf.toLocaleString()}`;
    document.getElementById("sample-info").textContent =
      `${meta.date} plant ${meta.plant}: ${meta.n.toLocaleString()} points${fullText}`;
    renderLeafList(meta.leaves || []);
  } else {
    document.getElementById("counts").textContent =
      `unlabelled ${meta.counts.unlabeled.toLocaleString()} | ` +
      `plant ${meta.counts.plant.toLocaleString()} | ` +
      `ground ${meta.counts.ground.toLocaleString()}`;
    document.getElementById("sample-info").textContent =
      `${meta.date}: ${meta.n.toLocaleString()} points${crop} ` +
      `(${meta.row_count.toLocaleString()} full row)`;
  }
}

function ingestLabels(data) {
  if (!data) return;
  if (data.labels) persistentLabels = data.labels;
  if (data.otype) persistentOtype = data.otype;
  if (data.leafid) persistentLeafid = data.leafid;
  if (data.plant_id) persistentPlantId = data.plant_id;
}

function repaint(data = null) {
  if (!points || !origin_colors) return;
  if (Array.isArray(data)) persistentLabels = data;
  else ingestLabels(data);
  const colors = points.geometry.attributes.color;

  for (let i = 0; i < origin_colors.count; i++) {
    if (appMode === "separation") {
      const pid = persistentPlantId[i];
      const c = pid >= 0 ? (isGhostedPlantId(pid) ? [0.87, 0.87, 0.87] : instanceColor(pid)) : [0.33, 0.33, 0.33];
      colors.setXYZ(i, c[0], c[1], c[2]);
      continue;
    }
    if (appMode === "plant") {
      const c = plantPointColor(i);
      colors.setXYZ(i, c[0], c[1], c[2]);
      continue;
    }
    const classColor = labelColor(persistentLabels[i] || 0);
    if (classColor && persistentLabels[i]) {
      colors.setXYZ(i, classColor[0], classColor[1], classColor[2]);
    } else {
      colors.setXYZ(i, 0.58, 0.58, 0.58);
    }
  }

  for (let j = 0; j < promptIndices.length; j++) {
    colors.setXYZ(promptIndices[j], promptLabels[j] > 0 ? 1 : 0, promptLabels[j] > 0 ? 1 : 0, 0);
  }
  colors.needsUpdate = true;
}

function renderPreviewMask(seg, context = null) {
  previewMask = Array.isArray(seg) ? seg : null;
  previewContext = context;
  repaint();
  if (!previewMask || !points) return;
  const colors = points.geometry.attributes.color;
  for (let i = 0; i < previewMask.length && i < origin_colors.count; i++) {
    if (previewMask[i]) colors.setXYZ(i, 0.0, 0.95, 1.0);
  }
  for (let j = 0; j < promptIndices.length; j++) {
    colors.setXYZ(promptIndices[j], promptLabels[j] > 0 ? 1 : 0, promptLabels[j] > 0 ? 1 : 0, 0);
  }
  colors.needsUpdate = true;
}

function clearPreviewMask() {
  previewMask = null;
  previewContext = null;
  repaint();
}

function setActiveLabel(label) {
  activeLabel = label;
  for (const id of ["class-plant", "class-ground", "class-erase"]) {
    document.getElementById(id).classList.remove("active");
  }
  if (label === 1) document.getElementById("class-plant").classList.add("active");
  if (label === 2) document.getElementById("class-ground").classList.add("active");
  if (label === 0) document.getElementById("class-erase").classList.add("active");
  if (appMode === "separation") {
    activeTarget = label === 0 ? { kind: "unassign" } : { kind: "new_plant" };
    syncTargetButtons();
  } else if (appMode === "plant") {
    activeTarget = label === 1 ? { kind: "stem" } : label === 0 ? { kind: "eraser" } : { kind: "new_leaf" };
    syncTargetButtons();
  }
  setDefaultLayerPolicy();
  repaint();
}

function syncTargetButtons() {
  for (const id of ["target-new-plant", "target-unassign"]) {
    const el = document.getElementById(id);
    if (el) el.classList.remove("active");
  }
  document.querySelectorAll(".plant-select").forEach((el) => el.classList.remove("active"));
  for (const id of ["target-stem", "target-new-leaf", "target-eraser"]) {
    document.getElementById(id).classList.remove("active");
  }
  document.querySelectorAll(".leaf-select").forEach((el) => el.classList.remove("active"));
  if (activeTarget.kind === "stem") document.getElementById("target-stem").classList.add("active");
  if (activeTarget.kind === "new_leaf") document.getElementById("target-new-leaf").classList.add("active");
  if (activeTarget.kind === "eraser") document.getElementById("target-eraser").classList.add("active");
  if (activeTarget.kind === "leaf") {
    const btn = document.querySelector(`.leaf-select[data-leafid="${activeTarget.leafid}"]`);
    if (btn) btn.classList.add("active");
  }
  if (appMode === "separation") {
    if (activeTarget.kind === "new_plant") document.getElementById("target-new-plant")?.classList.add("active");
    if (activeTarget.kind === "unassign") document.getElementById("target-unassign")?.classList.add("active");
    if (activeTarget.kind === "plant") {
      const btn = document.querySelector(`.plant-select[data-plantid="${activeTarget.plant_id}"]`);
      if (btn) btn.classList.add("active");
    }
    activeLabel = activeTarget.kind === "unassign" ? 0 : 1;
  } else {
    activeLabel = activeTarget.kind === "stem" ? 1 : activeTarget.kind === "eraser" ? 0 : 2;
  }
}

function updateModeVisibility() {
  document.getElementById("mode-select").value = appMode;
  document.querySelectorAll(".plant-only").forEach((el) => {
    el.style.display = appMode === "plant" ? "" : "none";
  });
  document.querySelectorAll(".separation-only").forEach((el) => {
    el.style.display = appMode === "separation" ? "" : "none";
  });
  document.querySelectorAll(".row-only").forEach((el) => {
    el.style.display = appMode === "plant" || appMode === "separation" ? "none" : "";
  });
  if ((appMode === "plant" || appMode === "separation") && toolMode === "crop") {
    setToolMode("brush");
  }
  document.getElementById("export-labels").textContent = appMode === "plant" ? "Export plant" : appMode === "separation" ? "Export separation" : "Export";
  populateDateSelect(document.getElementById("date-select").value);
}

function populateDateSelect(preferredDate = null) {
  const dateSelect = document.getElementById("date-select");
  if (!dateSelect) return;
  const showAll = appMode !== "plant" || document.getElementById("plant-all-dates")?.checked;
  const dates = appMode === "separation" ? (separationDates.length ? separationDates : allDates) : showAll ? allDates : earlyDates;
  if (!dates.length) return;
  const fallback = appMode === "plant" || appMode === "separation" ? "230619" : preferredDate;
  const selected = dates.includes(preferredDate) ? preferredDate : dates.includes(fallback) ? fallback : dates[0];
  dateSelect.innerHTML = "";
  for (const date of dates) {
    const option = document.createElement("option");
    option.value = date;
    option.textContent = date;
    if (date === selected) option.selected = true;
    dateSelect.appendChild(option);
  }
}

function populatePlotSelect(plots = [], active = null) {
  const sel = document.getElementById("plot-select");
  if (!sel) return;
  sel.innerHTML = "";
  for (const plot of plots) {
    // Accept either {id,label} objects or bare "PlotNN" strings.
    const id = typeof plot === "string" ? plot : plot.id;
    const label = typeof plot === "string" ? plot : plot.label;
    const option = document.createElement("option");
    option.value = id;
    option.textContent = label;
    if (id === active) option.selected = true;
    sel.appendChild(option);
  }
  sel.style.display = plots.length > 1 ? "" : "none";
}

function populatePlantSelect(plants = [], preferred = "06") {
  const plantSelect = document.getElementById("plant-select");
  if (!plantSelect) return;
  plantSelect.innerHTML = "";
  for (const plant of plants) {
    const option = document.createElement("option");
    option.value = plant;
    option.textContent = plant;
    if (plant === preferred) option.selected = true;
    plantSelect.appendChild(option);
  }
}

function adoptDatasetMeta(data, preferredDate = null) {
  allDates = data.dates || [];
  earlyDates = data.early_dates || allDates;
  separationDates = data.separation_dates || allDates;
  populateDateSelect(preferredDate || data.date);
  populatePlantSelect(data.plants || [], document.getElementById("plant-select")?.value || "06");
}

async function populateDatasetSelect() {
  const response = await fetch("/datasets");
  const data = await response.json();
  const select = document.getElementById("dataset-select");
  const input = document.getElementById("dataset-root");
  if (!select || !input) return;
  select.innerHTML = "";
  for (const ds of data.datasets || []) {
    const option = document.createElement("option");
    option.value = ds.root;
    option.textContent = `${ds.plot}: ${ds.root}`;
    if (ds.active) {
      option.selected = true;
      populatePlotSelect(ds.plots || [], ds.plot);
    }
    select.appendChild(option);
  }
  input.value = data.active?.root || select.value || "";
  select.onchange = () => {
    input.value = select.value;
    const chosen = (data.datasets || []).find((d) => d.root === select.value);
    if (chosen) populatePlotSelect(chosen.plots || [], chosen.plot);
  };
}

async function setDataset(plot = null) {
  const root = document.getElementById("dataset-root").value.trim();
  if (!root) {
    setStatus("Dataset path is empty");
    return;
  }
  setStatus(`Switching dataset to ${root}${plot ? " / " + plot : ""}...`);
  const response = await fetch("/set_dataset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ root, plot }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Dataset switch failed");
    return;
  }
  promptIndices = [];
  promptLabels = [];
  samCandidates = [];
  samSelected = 0;
  previewMask = null;
  previewContext = null;
  ghostedPlantIds.clear();
  populatePlotSelect(data.plots || [], data.dataset.plot);
  adoptDatasetMeta(data);
  await populateDatasetSelect();
  const plotCount = (data.plots || []).length;
  setStatus(
    `Dataset set: ${data.dataset.plot}` +
      `${plotCount > 1 ? ` (1 of ${plotCount} plots)` : ""}, ` +
      `${data.dates.length} dates, ${data.plants.length} plants`
  );
}

function renderLeafList(leaves) {
  const container = document.getElementById("leaf-list");
  container.innerHTML = "";
  for (const leaf of leaves) {
    const row = document.createElement("div");
    row.className = "leaf-row";
    const swatch = document.createElement("span");
    swatch.className = "leaf-swatch";
    swatch.style.backgroundColor = cssRgb(leafColor(leaf.id));
    const btn = document.createElement("button");
    btn.className = "leaf-select";
    btn.dataset.leafid = leaf.id;
    btn.textContent = `Leaf ${leaf.id} (${leaf.points})`;
    btn.onclick = () => {
      activeTarget = { kind: "leaf", leafid: leaf.id };
      syncTargetButtons();
      setStatus(`Active target: leaf ${leaf.id}`);
    };
    const del = document.createElement("button");
    del.className = "leaf-delete";
    del.textContent = "X";
    del.onclick = () => deleteLeaf(leaf.id);
    const redo = document.createElement("button");
    redo.className = "leaf-redo";
    redo.textContent = "Redo";
    redo.onclick = () => redoLeaf(leaf.id);
    row.appendChild(swatch);
    row.appendChild(btn);
    row.appendChild(redo);
    row.appendChild(del);
    container.appendChild(row);
  }
  syncTargetButtons();
}

function renderPlantList(plants) {
  const container = document.getElementById("plant-list");
  if (!container) return;
  container.innerHTML = "";
  if (ghostedPlantIds.size) {
    const clear = document.createElement("button");
    clear.className = "plant-ghost-clear";
    clear.textContent = "Show all plants";
    clear.onclick = () => {
      ghostedPlantIds.clear();
      repaint();
      renderPlantList(plants);
      setStatus("Showing all plants");
    };
    container.appendChild(clear);
  }
  for (const plant of plants) {
    const plantId = Number(plant.id);
    const row = document.createElement("div");
    row.className = "plant-row";
    if (ghostedPlantIds.has(plantId)) row.classList.add("ghosted");
    const swatch = document.createElement("span");
    swatch.className = "leaf-swatch";
    swatch.style.backgroundColor = cssRgb(ghostedPlantIds.has(plantId) ? [0.87, 0.87, 0.87] : instanceColor(plantId));
    const btn = document.createElement("button");
    btn.className = "plant-select";
    btn.dataset.plantid = plantId;
    btn.textContent = `Plant ${plantId} (${plant.points})`;
    btn.onclick = () => {
      activeTarget = { kind: "plant", plant_id: plantId };
      activeLabel = 1;
      syncTargetButtons();
      setStatus(`Active target: plant ${plantId}`);
    };
    const ghost = document.createElement("button");
    ghost.className = "plant-ghost";
    ghost.textContent = ghostedPlantIds.has(plantId) ? "Show" : "Ghost";
    ghost.onclick = () => {
      if (ghostedPlantIds.has(plantId)) {
        ghostedPlantIds.delete(plantId);
        setStatus(`Showing plant ${plantId}`);
      } else {
        ghostedPlantIds.add(plantId);
        setStatus(`Ghosted plant ${plantId}`);
      }
      repaint();
      renderPlantList(plants);
    };
    row.appendChild(swatch);
    row.appendChild(btn);
    row.appendChild(ghost);
    container.appendChild(row);
  }
  syncTargetButtons();
}

function setDefaultLayerPolicy() {
  const plantPolicy = document.getElementById("sam-policy-plant");
  const groundPolicy = document.getElementById("sam-policy-ground");
  if (!plantPolicy || !groundPolicy) return;
  plantPolicy.value = activeLabel === 1 ? "override" : "protect";
  groundPolicy.value = activeLabel === 2 ? "override" : "protect";
}

function layerPolicyPayload() {
  return {
    1: document.getElementById("sam-policy-plant").value,
    2: document.getElementById("sam-policy-ground").value,
  };
}

function setSamHelp(show) {
  document.getElementById("sam-help")?.classList.toggle("hidden", !show);
}

function setToolMode(mode) {
  toolMode = mode;
  for (const id of ["tool-brush", "tool-lasso", "tool-sam", "tool-crop", "tool-delete"]) {
    document.getElementById(id).classList.remove("active");
  }
  document.getElementById(`tool-${mode}`).classList.add("active");
  setSamHelp(mode === "sam");
  setStatus(
    mode === "brush"
      ? "Brush: click or drag over points to paint the active class"
      : mode === "lasso"
      ? "Lasso: drag around points to paint the active class"
      : mode === "crop"
      ? "Crop: drag a rectangle around the area you want to focus on"
      : mode === "delete"
      ? "Delete Points: lasso points to remove them from the cloud and export"
      : "Point-SAM: click a point to apply ML mask to the active class"
  );
}

function setPromptMode(isPositive) {
  promptIsPositive = isPositive;
  document.getElementById("annotate-positive").classList.toggle("active", promptIsPositive);
  document.getElementById("annotate-negative").classList.toggle("active", !promptIsPositive);
}

function canvasPoint(event) {
  const rect = overlay.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function drawOverlay() {
  syncOverlaySize();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (drawPoints.length < 2) return;

  overlayCtx.strokeStyle =
    toolMode === "delete" ? "#00df00" : activeLabel === 2 ? "#ff1f1f" : activeLabel === 1 ? "#00df00" : "#999";
  overlayCtx.fillStyle =
    toolMode === "crop"
      ? "rgba(255, 255, 255, 0.16)"
      : toolMode === "delete"
      ? "rgba(0, 220, 0, 0.16)"
      : "rgba(255, 255, 255, 0.08)";
  overlayCtx.lineWidth = 2;
  overlayCtx.beginPath();
  overlayCtx.moveTo(drawPoints[0].x, drawPoints[0].y);
  if (toolMode === "crop") {
    const a = drawPoints[0];
    const b = drawPoints[drawPoints.length - 1];
    overlayCtx.rect(a.x, a.y, b.x - a.x, b.y - a.y);
  } else {
    for (const p of drawPoints.slice(1)) overlayCtx.lineTo(p.x, p.y);
    overlayCtx.closePath();
  }
  overlayCtx.fill();
  overlayCtx.stroke();
}

function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  drawPoints = [];
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function selectedIndicesFromDrawnShape() {
  const rect = overlay.getBoundingClientRect();
  const positions = points.geometry.attributes.position;
  const selected = [];
  const p = new THREE.Vector3();

  let xmin, xmax, ymin, ymax;
  if (toolMode === "crop") {
    const a = drawPoints[0];
    const b = drawPoints[drawPoints.length - 1];
    xmin = Math.min(a.x, b.x);
    xmax = Math.max(a.x, b.x);
    ymin = Math.min(a.y, b.y);
    ymax = Math.max(a.y, b.y);
  }

  for (let i = 0; i < positions.count; i++) {
    if (!isSelectableIndex(i)) continue;
    p.fromBufferAttribute(positions, i);
    p.project(camera);
    const sx = ((p.x + 1) / 2) * rect.width;
    const sy = ((-p.y + 1) / 2) * rect.height;
    if (toolMode === "crop") {
      if (sx >= xmin && sx <= xmax && sy >= ymin && sy <= ymax && isSelectableIndex(i)) selected.push(i);
    } else if (pointInPolygon(sx, sy, drawPoints)) {
      if (isSelectableIndex(i)) selected.push(i);
    }
  }
  return selected;
}

function distancePointToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function selectedIndicesFromBrushStroke() {
  const radius = Number(document.getElementById("brush-size").value);
  const rect = overlay.getBoundingClientRect();
  const positions = points.geometry.attributes.position;
  const selected = [];
  const p = new THREE.Vector3();
  const stroke = drawPoints.length === 1 ? [drawPoints[0], drawPoints[0]] : drawPoints;

  for (let i = 0; i < positions.count; i++) {
    if (!isSelectableIndex(i)) continue;
    p.fromBufferAttribute(positions, i);
    p.project(camera);
    const sx = ((p.x + 1) / 2) * rect.width;
    const sy = ((-p.y + 1) / 2) * rect.height;
    for (let j = 1; j < stroke.length; j++) {
      if (distancePointToSegment(sx, sy, stroke[j - 1].x, stroke[j - 1].y, stroke[j].x, stroke[j].y) <= radius) {
        selected.push(i);
        break;
      }
    }
  }
  return selected;
}

function nearestPointAtCanvasPoint(canvasPt, maxPixels = 14) {
  const rect = overlay.getBoundingClientRect();
  const positions = points.geometry.attributes.position;
  const p = new THREE.Vector3();
  let bestIndex = -1;
  let bestDist2 = maxPixels * maxPixels;

  for (let i = 0; i < positions.count; i++) {
    p.fromBufferAttribute(positions, i);
    p.project(camera);
    if (p.z < -1 || p.z > 1) continue;
    const sx = ((p.x + 1) / 2) * rect.width;
    const sy = ((-p.y + 1) / 2) * rect.height;
    const dx = sx - canvasPt.x;
    const dy = sy - canvasPt.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestDist2) {
      bestDist2 = d2;
      bestIndex = i;
    }
  }

  if (bestIndex < 0) return null;
  const clicked = new THREE.Vector3().fromBufferAttribute(positions, bestIndex);
  return { index: bestIndex, point: clicked, dist: Math.sqrt(bestDist2) };
}

async function assignIndices(indices) {
  if (!indices.length) {
    setStatus("No points selected");
    return;
  }
  setStatus(`Painting ${indices.length.toLocaleString()} points as ${targetDisplayName()}...`);
  const response = await fetch("/assign_indices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label: activeLabel, target: activeTarget, indices }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Lasso failed");
    return;
  }
  if (appMode === "separation" && activeTarget.kind === "new_plant" && data.target_plant_id >= 0) {
    activeTarget = { kind: "plant", plant_id: data.target_plant_id };
    syncTargetButtons();
  }
  adoptNewLeafTarget(data);
  repaint(data);
  updateCounts(data);
  setStatus(`Painted ${data.changed.toLocaleString()} points as ${targetDisplayName(data)}`);
}

async function cropToIndices(indices) {
  if (!indices.length) {
    setStatus("No points selected for crop");
    return;
  }
  setStatus(`Cropping to ${indices.length.toLocaleString()} points...`);
  const response = await fetch("/crop_indices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ indices }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Crop failed");
    return;
  }
  replacePointCloud(data);
  ingestLabels(data);
  promptIndices = [];
  promptLabels = [];
  previewMask = null;
  previewContext = null;
  repaint();
  updateCounts(data);
  setStatus(`Cropped to ${data.n.toLocaleString()} points`);
}

async function deleteIndices(indices) {
  if (!indices.length) {
    setStatus("No points selected for deletion");
    return;
  }
  setStatus(`Deleting ${indices.length.toLocaleString()} points...`);
  const response = await fetch("/delete_indices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ indices }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Delete failed");
    return;
  }
  replacePointCloud(data);
  ingestLabels(data);
  promptIndices = [];
  promptLabels = [];
  previewMask = null;
  previewContext = null;
  repaint();
  updateCounts(data);
  setStatus(`Deleted ${data.deleted.toLocaleString()} points`);
}

async function onSamClick(event) {
  const hit = nearestPointAtCanvasPoint(canvasPoint(event));
  if (!hit) {
    setStatus("No point under cursor");
    return;
  }

  promptIndices.push(hit.index);
  promptLabels.push(promptIsPositive ? 1 : 0);
  setStatus(`Smart Mask from nearest point (${hit.dist.toFixed(1)} px)...`);
  const response = await fetch("/segment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt_point: hit.point.toArray(),
      prompt_label: promptIsPositive,
      active_label: activeLabel,
      target: activeTarget,
      use_label_context: document.getElementById("sam-context-enabled")?.checked ?? false,
      use_height_prior: document.getElementById("sam-height-enabled").checked,
      height_threshold: Number(document.getElementById("sam-height-cm").value) / 100.0,
      flood_points: Number(document.getElementById("sam-flood-points").value || 0),
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Smart Mask failed");
    return;
  }

  samCandidates = data.candidates || [];
  samSelected = data.selected ?? 0;
  const pos = promptLabels.filter((x) => x > 0).length;
  const neg = promptLabels.length - pos;
  const maskCount = data.seg.filter(Boolean).length;
  renderPreviewMask(data.seg, {
    source: "Smart Mask",
    maskCount,
    elapsed: data.elapsed,
    contextPrompts: data.context_prompts,
    heightRemoved: data.height_prior_removed,
    floodKept: data.flood_kept,
    floodDropped: data.flood_dropped,
  });
  const candText = samCandidates.length > 1 ? ` [mask ${samSelected + 1}/${samCandidates.length}, Tab cycles]` : "";
  setStatus(
    `Preview ${maskCount.toLocaleString()} points; prompts ${promptLabels.length} ` +
      `(+${pos}/-${neg}); iou ${Number(data.iou || 0).toFixed(3)}.${candText} Enter accepts, Esc cancels.`
  );
}

async function cycleMask(delta) {
  if (samCandidates.length < 2) return;
  samSelected = (samSelected + delta + samCandidates.length) % samCandidates.length;
  const response = await fetch("/select_mask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index: samSelected }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Mask cycle failed");
    return;
  }
  renderPreviewMask(data.seg, previewContext);
  setStatus(
    `Mask ${samSelected + 1}/${data.num_candidates} — ${Number(data.count).toLocaleString()} points; ` +
      `iou ${Number(data.iou || 0).toFixed(3)}. Tab cycles, Enter accepts, Esc cancels.`
  );
}

async function commitMask(context = null) {
  const response = await fetch("/commit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      label: activeLabel,
      target: activeTarget,
      layer_policy: layerPolicyPayload(),
      ghosted_plant_ids: Array.from(ghostedPlantIds),
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "No active mask to commit");
    return;
  }
  promptIndices = [];
  promptLabels = [];
  samCandidates = [];
  samSelected = 0;
  clearPreviewMask();
  if (appMode === "separation" && activeTarget.kind === "new_plant" && data.target_plant_id >= 0) {
    activeTarget = { kind: "plant", plant_id: data.target_plant_id };
    syncTargetButtons();
  }
  adoptNewLeafTarget(data);
  repaint(data);
  updateCounts(data);
  const protectedText = data.skipped ? `; protected ${data.skipped.toLocaleString()} old labels` : "";
  const prefix = context?.source ? `${context.source}: ` : "";
  const maskText = context?.maskCount ? ` from ${context.maskCount.toLocaleString()} mask points` : "";
  const contextText = context?.contextPrompts ? ` using ${context.contextPrompts} old-label prompts` : "";
  const heightText = context?.heightRemoved ? `; height filter removed ${context.heightRemoved.toLocaleString()}` : "";
  const floodText = context?.floodKept
    ? `; local flood kept ${context.floodKept.toLocaleString()} nearest points`
    : "";
  const targetName = targetDisplayName(data);
  const replacedText = data.replaced ? `; replaced ${data.replaced.toLocaleString()} old ${targetName} points` : "";
  setStatus(
    data.changed
      ? `${prefix}saved ${data.changed.toLocaleString()} ${targetName} points${maskText}${contextText}${heightText}${floodText}${protectedText}${replacedText}`
      : `${prefix}did not change labels${heightText}${floodText}${protectedText}${replacedText}`
  );
}

async function clearPrompts() {
  promptIndices = [];
  promptLabels = [];
  samCandidates = [];
  samSelected = 0;
  previewMask = null;
  previewContext = null;
  await fetch("/clear", { method: "POST" });
  repaint();
  setStatus("Cleared Point-SAM prompts");
}

async function undoCommit() {
  const response = await fetch("/undo", { method: "POST" });
  const data = await response.json();
  if (data.cloud_changed) {
    replacePointCloud(data);
    ingestLabels(data);
    promptIndices = [];
    promptLabels = [];
    repaint();
  } else {
    repaint(data);
  }
  updateCounts(data);
  setStatus(data.status === "empty" ? "Nothing to undo" : "Undid last action");
}

async function resetLabels() {
  const response = await fetch("/reset_labels", { method: "POST" });
  const data = await response.json();
  promptIndices = [];
  promptLabels = [];
  repaint(data);
  updateCounts(data);
  setStatus("Reset labels");
}

async function resetCrop() {
  setStatus("Reloading full row...");
  const response = await fetch("/reset_crop", { method: "POST" });
  const data = await response.json();
  replacePointCloud(data);
  persistentLabels = data.labels;
  promptIndices = [];
  promptLabels = [];
  repaint();
  updateCounts(data);
  setStatus("Full row restored");
}

async function exportLabels() {
  const fullres = document.getElementById("export-fullres").checked;
  setStatus(appMode === "plant" ? "Exporting plant labels..." : appMode === "separation" ? "Exporting manual separation..." : fullres ? "Exporting sample and full-res LAS..." : "Exporting LAS...");
  const response = await fetch("/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fullres }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Export failed");
    return;
  }
  updateCounts(data);
  if (appMode === "separation") {
    const warn = data.leaf_label_warnings?.length ? `; backed up ${data.leaf_label_warnings.length} existing leaf-label files` : "";
    setStatus(`Exported separation QC ${data.qc_png}${warn}`);
  } else if (appMode === "plant") {
    setStatus(`Exported ${data.gt_otype}, ${data.gt_leafid}, and QC PNG`);
  } else {
    setStatus(data.fullres ? `Exported ${data.las} and full-res LAS` : `Exported ${data.las}`);
  }
}

async function loadDate() {
  const date = document.getElementById("date-select").value;
  const n = Number(document.getElementById("density-select").value);
  const densityText = n > 0 ? `${n.toLocaleString()} points` : "full resolution";
  setStatus(`Loading ${date} at ${densityText}...`);
  const response = await fetch(`/load_date/${date}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || `Failed to load ${date}`);
    return;
  }
  appMode = "row";
  replacePointCloud(data);
  ingestLabels(data);
  promptIndices = [];
  promptLabels = [];
  previewMask = null;
  previewContext = null;
  repaint();
  updateCounts(data);
  setStatus(`Loaded ${date}`);
}

async function loadRowVeg() {
  const date = document.getElementById("date-select").value;
  const seedAuto = document.getElementById("seed-auto-separation").checked;
  const n = Number(document.getElementById("density-select").value);
  const densityText = n > 0 ? `${n.toLocaleString()} points` : "full resolution";
  setStatus(`Loading row vegetation ${date} at ${densityText}${seedAuto ? " seeded from auto" : ""}...`);
  const response = await fetch("/load_row_veg", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date, seed_auto: seedAuto, n }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || `Failed to load row vegetation ${date}`);
    return;
  }
  appMode = "separation";
  ghostedPlantIds.clear();
  replacePointCloud(data);
  promptIndices = [];
  promptLabels = [];
  previewMask = null;
  previewContext = null;
  ingestLabels(data);
  repaint();
  updateCounts(data);
  setToolMode("brush");
  setStatus(`Loaded row vegetation ${date}: ${data.loaded_labels_from || "blank"}`);
}

async function loadPlant() {
  const date = document.getElementById("date-select").value;
  const plant = document.getElementById("plant-select").value;
  const n = Number(document.getElementById("density-select").value);
  const densityText = n > 0 ? `${n.toLocaleString()} points` : "full resolution";
  setStatus(`Loading plant ${plant} @ ${date} at ${densityText}...`);
  const response = await fetch("/load_plant", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plant_id: plant, date, n }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || `Failed to load plant ${plant}`);
    return;
  }
  appMode = "plant";
  replacePointCloud(data);
  promptIndices = [];
  promptLabels = [];
  previewMask = null;
  previewContext = null;
  ingestLabels(data);
  repaint();
  updateCounts(data);
  setStatus(data.loaded_labels_from ? `Loaded plant ${plant} with saved labels` : `Loaded blank plant ${plant}`);
}

async function renumberLeaves() {
  const response = await fetch("/renumber_leaves", { method: "POST" });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Renumber failed");
    return;
  }
  ingestLabels(data);
  repaint();
  updateCounts(data);
  setStatus("Renumbered leaves by insertion height");
}

async function deleteLeaf(leafid) {
  const response = await fetch("/delete_leaf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ leafid }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Delete leaf failed");
    return;
  }
  ingestLabels(data);
  repaint();
  updateCounts(data);
  setStatus(`Deleted leaf ${leafid}`);
}

async function redoLeaf(leafid) {
  const response = await fetch("/redo_leaf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ leafid }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Redo leaf failed");
    return;
  }
  activeTarget = { kind: "leaf", leafid };
  activeLabel = 2;
  ingestLabels(data);
  repaint();
  updateCounts(data);
  syncTargetButtons();
  setStatus(`Cleared leaf ${leafid}; repaint it now`);
}

function stepPlant(delta) {
  const select = document.getElementById("plant-select");
  const next = Math.max(0, Math.min(select.options.length - 1, select.selectedIndex + delta));
  select.selectedIndex = next;
  loadPlant();
}

async function switchMode(mode) {
  appMode = mode;
  updateModeVisibility();
  if (mode === "separation") {
    activeTarget = { kind: "unassign" };
    syncTargetButtons();
    await loadRowVeg();
  } else if (mode === "plant") {
    await loadPlant();
  } else {
    await loadDate();
  }
}

function bindButtons(initialData) {
  const dateSelect = document.getElementById("date-select");
  adoptDatasetMeta(initialData, initialData.date);
  document.getElementById("density-select").value = String(initialData.n_target || initialData.n);
  dateSelect.onchange = () => appMode === "plant" ? loadPlant() : appMode === "separation" ? loadRowVeg() : loadDate();
  document.getElementById("density-select").onchange = () => appMode === "plant" ? loadPlant() : appMode === "separation" ? loadRowVeg() : loadDate();
  document.getElementById("plant-all-dates").onchange = () => populateDateSelect(dateSelect.value);
  document.getElementById("load-row-veg").onclick = loadRowVeg;
  document.getElementById("target-new-plant").onclick = () => {
    activeTarget = { kind: "new_plant" };
    activeLabel = 1;
    syncTargetButtons();
  };
  document.getElementById("target-unassign").onclick = () => {
    activeTarget = { kind: "unassign" };
    activeLabel = 0;
    syncTargetButtons();
  };

  document.getElementById("set-dataset").onclick = () => setDataset();
  document.getElementById("plot-select").onchange = (e) => setDataset(e.target.value);
  document.getElementById("mode-select").onchange = (event) => switchMode(event.target.value);
  document.getElementById("load-plant").onclick = loadPlant;
  document.getElementById("prev-plant").onclick = () => stepPlant(-1);
  document.getElementById("next-plant").onclick = () => stepPlant(1);
  document.getElementById("target-stem").onclick = () => {
    activeTarget = { kind: "stem" };
    activeLabel = 1;
    syncTargetButtons();
  };
  document.getElementById("target-new-leaf").onclick = () => {
    activeTarget = { kind: "new_leaf" };
    activeLabel = 2;
    syncTargetButtons();
  };
  document.getElementById("target-eraser").onclick = () => {
    activeTarget = { kind: "eraser" };
    activeLabel = 0;
    syncTargetButtons();
  };
  document.getElementById("renumber-leaves").onclick = renumberLeaves;

  document.getElementById("tool-brush").onclick = () => setToolMode("brush");
  document.getElementById("tool-lasso").onclick = () => setToolMode("lasso");
  document.getElementById("tool-sam").onclick = () => setToolMode("sam");
  document.getElementById("sam-help-close").onclick = () => setSamHelp(false);
  document.getElementById("tool-crop").onclick = () => setToolMode("crop");
  document.getElementById("tool-delete").onclick = () => setToolMode("delete");
  document.getElementById("class-plant").onclick = () => setActiveLabel(1);
  document.getElementById("class-ground").onclick = () => setActiveLabel(2);
  document.getElementById("class-erase").onclick = () => setActiveLabel(0);
  document.getElementById("annotate-positive").onclick = () => setPromptMode(true);
  document.getElementById("annotate-negative").onclick = () => setPromptMode(false);
  document.getElementById("commit-mask").onclick = () => commitMask(previewContext);
  document.getElementById("undo-commit").onclick = undoCommit;
  document.getElementById("clear-prompts").onclick = clearPrompts;
  document.getElementById("reset-labels").onclick = resetLabels;
  document.getElementById("reset-crop").onclick = resetCrop;
  document.getElementById("export-labels").onclick = exportLabels;
  document.getElementById("export-fullres").checked = initialData.export_fullres_default;
}

function bindCanvasEvents() {
  renderer.domElement.addEventListener("mousedown", async (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    if (toolMode === "sam") {
      await onSamClick(event);
      return;
    }
    isDrawing = true;
    controls.enabled = false;
    drawPoints = [canvasPoint(event)];
    if (toolMode !== "brush") drawOverlay();
  });

  window.addEventListener("mousemove", (event) => {
    if (!isDrawing || (toolMode !== "brush" && toolMode !== "lasso" && toolMode !== "crop" && toolMode !== "delete")) return;
    if (toolMode === "crop") {
      drawPoints = [drawPoints[0], canvasPoint(event)];
    } else if (toolMode === "brush") {
      drawPoints.push(canvasPoint(event));
    } else {
      drawPoints.push(canvasPoint(event));
    }
    if (toolMode !== "brush") drawOverlay();
  });

  window.addEventListener("mouseup", async () => {
    if (!isDrawing) return;
    isDrawing = false;
    controls.enabled = true;
    if (drawPoints.length < 1) {
      clearOverlay();
      return;
    }
    const indices = toolMode === "brush" ? selectedIndicesFromBrushStroke() : selectedIndicesFromDrawnShape();
    clearOverlay();
    if (toolMode === "crop") {
      await cropToIndices(indices);
    } else if (toolMode === "delete") {
      await deleteIndices(indices);
    } else {
      await assignIndices(indices);
    }
  });

window.addEventListener("resize", syncOverlaySize);
window.addEventListener("contextmenu", (event) => event.preventDefault());
window.addEventListener("keydown", async (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    await undoCommit();
  } else if (event.key === "?" && toolMode === "sam") {
    event.preventDefault();
    setSamHelp(document.getElementById("sam-help")?.classList.contains("hidden"));
  } else if (event.key === "Tab" && toolMode === "sam" && samCandidates.length > 1) {
    event.preventDefault();
    await cycleMask(event.shiftKey ? -1 : 1);
  } else if (event.key === "Enter" && toolMode === "sam") {
    event.preventDefault();
    await commitMask(previewContext);
  } else if (event.key === "Escape") {
    event.preventDefault();
    await clearPrompts();
  }
});
}

const initialData = await viewerMain();
syncOverlaySize();
ingestLabels(initialData);
repaint();
updateCounts(initialData);
bindButtons(initialData);
await populateDatasetSelect();
bindCanvasEvents();
setActiveLabel(1);
setPromptMode(true);
setToolMode("brush");
updateModeVisibility();
