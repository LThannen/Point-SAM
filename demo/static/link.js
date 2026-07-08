import * as THREE from "three";

const MODE = "link";
const CLOUD_POINTS = 50_000;
const MIN_VIEW_WIDTH = 300;
const VIEW_GAP = 10;
const PICK_RADIUS_PX = 14;
const POINT_SIZE = 4;

const SELECTED = [1.0, 0.82, 0.05];
const STEM = [0.45, 0.24, 0.08];
const OTHER = [0.48, 0.50, 0.54];

const state = {
  plant: "01",
  loadedPlant: null,
  dates: [],
  clouds: {},
  chains: [],
  draft: {},
  editingIndex: -1,
  hasManual: false,
  seedLabel: "auto seed",
  views: [],
  panel: null,
  canvas: null,
  renderer: null,
  wrap: null,
  labels: null,
  resizeObserver: null,
  renderQueued: false,
  loadToken: 0,
  yaw: 0,
  zoom: 1,
};

const scratch = new THREE.Vector3();

function leafColor(leafid) {
  const palette = [
    [0.00, 0.45, 1.00], [1.00, 0.55, 0.00], [0.00, 0.70, 0.20], [0.95, 0.00, 0.85],
    [0.00, 0.75, 0.85], [0.95, 0.85, 0.00], [0.55, 0.20, 1.00], [1.00, 0.15, 0.15],
    [0.00, 0.95, 0.55], [1.00, 0.35, 0.65], [0.35, 0.70, 1.00], [0.70, 0.45, 0.00],
  ];
  return palette[(leafid - 1) % palette.length];
}

function rankColor(rank) {
  const c = new THREE.Color();
  c.setHSL((rank * 0.38196601125) % 1, 0.72, 0.46);
  return [c.r, c.g, c.b];
}

function cssColor(rgb) {
  return `rgb(${rgb.map((v) => Math.round(v * 255)).join(", ")})`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function currentPlant() {
  const sel = document.getElementById("plant-select");
  const value = sel && sel.value ? sel.value : "1";
  return value.toString().padStart(2, "0");
}

function cleanObs(obs) {
  const out = {};
  for (const [date, leafid] of Object.entries(obs || {})) {
    const lid = parseInt(leafid, 10);
    if (Number.isFinite(lid) && lid > 0) out[date] = lid;
  }
  return out;
}

function normalizeChains(chains) {
  return (chains || [])
    .map((chain, index) => ({ rank: index + 1, obs: cleanObs(chain.obs) }))
    .filter((chain) => Object.keys(chain.obs).length > 0);
}

function renumberChains() {
  state.chains.forEach((chain, index) => {
    chain.rank = index + 1;
  });
}

function dateOrder(obs = {}) {
  const seen = new Set(state.dates);
  const extras = Object.keys(obs).filter((date) => !seen.has(date)).sort();
  return [...state.dates, ...extras];
}

function formatObs(obs) {
  const parts = [];
  for (const date of dateOrder(obs)) {
    if (obs[date]) parts.push(`${date.slice(-4)}:L${obs[date]}`);
  }
  return parts.join(" ");
}

function draftRank() {
  return state.editingIndex >= 0 ? state.editingIndex + 1 : state.chains.length + 1;
}

function hasDraft() {
  return Object.keys(state.draft).length > 0;
}

function chainRankForLeaf(date, leafid) {
  for (let i = 0; i < state.chains.length; i += 1) {
    if (i !== state.editingIndex && state.chains[i].obs[date] === leafid) return i + 1;
  }
  return 0;
}

function pointColor(date, otype, leafid) {
  if (otype === 2 && state.draft[date] === leafid) return SELECTED;
  const rank = otype === 2 ? chainRankForLeaf(date, leafid) : 0;
  if (rank) return rankColor(rank);
  if (otype === 1) return STEM;
  if (otype === 2 && leafid > 0) return leafColor(leafid);
  return OTHER;
}

function setStatus(text) {
  const n = document.getElementById("link-status");
  if (n) n.textContent = text || "";
}

function setBusy(text) {
  const n = document.getElementById("link-loading");
  if (!n) return;
  n.hidden = !text;
  n.textContent = text || "";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function flattenXYZ(xyz) {
  if (!xyz || !xyz.length) return new Float32Array();
  if (Array.isArray(xyz[0])) {
    const out = new Float32Array(xyz.length * 3);
    for (let i = 0; i < xyz.length; i += 1) {
      out[i * 3] = xyz[i][0];
      out[i * 3 + 1] = xyz[i][1];
      out[i * 3 + 2] = xyz[i][2];
    }
    return out;
  }
  return Float32Array.from(xyz);
}

function boundsFor(positions) {
  const b = {
    minX: Infinity, minY: Infinity, minZ: Infinity,
    maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity,
  };
  for (let i = 0; i < positions.length; i += 3) {
    b.minX = Math.min(b.minX, positions[i]);
    b.maxX = Math.max(b.maxX, positions[i]);
    b.minY = Math.min(b.minY, positions[i + 1]);
    b.maxY = Math.max(b.maxY, positions[i + 1]);
    b.minZ = Math.min(b.minZ, positions[i + 2]);
    b.maxZ = Math.max(b.maxZ, positions[i + 2]);
  }
  if (!Number.isFinite(b.minX)) {
    b.minX = b.minY = b.minZ = -1;
    b.maxX = b.maxY = b.maxZ = 1;
  }
  b.cx = (b.minX + b.maxX) / 2;
  b.cy = (b.minY + b.maxY) / 2;
  b.cz = (b.minZ + b.maxZ) / 2;
  b.sizeX = Math.max(1e-3, b.maxX - b.minX);
  b.sizeY = Math.max(1e-3, b.maxY - b.minY);
  b.sizeZ = Math.max(1e-3, b.maxZ - b.minZ);
  b.size = Math.max(b.sizeX, b.sizeY, b.sizeZ, 1);
  return b;
}

function disposeViews() {
  for (const view of state.views) {
    view.geometry.dispose();
    view.material.dispose();
  }
  state.views = [];
}

function makeView(date, cloud) {
  const positions = flattenXYZ(cloud.xyz);
  const count = Math.floor(positions.length / 3);
  if (!count) return null;

  const otype = Uint8Array.from(cloud.otype || []);
  const leafid = Int16Array.from(cloud.leafid || []);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));

  const material = new THREE.PointsMaterial({
    size: POINT_SIZE,
    sizeAttenuation: false,
    vertexColors: true,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8fafc);
  scene.add(points);

  return {
    date,
    cloud,
    scene,
    camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100000),
    geometry,
    material,
    points,
    otype,
    leafid,
    bounds: boundsFor(positions),
    viewport: { x: 0, y: 0, w: 1, h: 1 },
  };
}

function buildViews() {
  disposeViews();
  for (const date of state.dates) {
    const view = makeView(date, state.clouds[date]);
    if (view) state.views.push(view);
  }
  updateAllColors();
}

function updateViewColors(view) {
  const colors = view.geometry.attributes.color.array;
  for (let i = 0; i < view.leafid.length; i += 1) {
    const c = pointColor(view.date, view.otype[i] || 0, view.leafid[i] || 0);
    colors[i * 3] = c[0];
    colors[i * 3 + 1] = c[1];
    colors[i * 3 + 2] = c[2];
  }
  view.geometry.attributes.color.needsUpdate = true;
}

function updateAllColors() {
  state.views.forEach(updateViewColors);
  requestRender();
}

function fitCamera(view) {
  const { w, h } = view.viewport;
  const aspect = Math.max(0.1, w / Math.max(1, h));
  const b = view.bounds;
  const horizontal = Math.max(b.sizeX, b.sizeY);
  const halfHeight = Math.max(b.sizeZ / 2, horizontal / (2 * aspect), 1) * 1.16 / state.zoom;
  const halfWidth = halfHeight * aspect;
  const cam = view.camera;
  cam.left = -halfWidth;
  cam.right = halfWidth;
  cam.top = halfHeight;
  cam.bottom = -halfHeight;
  cam.near = 0.01;
  cam.far = b.size * 10 + 1000;
  const dist = b.size * 3 + 10;
  cam.position.set(
    b.cx + Math.sin(state.yaw) * dist,
    b.cy + Math.cos(state.yaw) * dist,
    b.cz
  );
  cam.up.set(0, 0, 1);
  cam.lookAt(b.cx, b.cy, b.cz);
  cam.updateProjectionMatrix();
}

function fitAllCameras() {
  state.views.forEach(fitCamera);
}

function layoutViews(width, height) {
  const n = state.views.length;
  if (!n) return;
  const viewWidth = Math.floor((width - VIEW_GAP * (n - 1)) / n);
  for (let i = 0; i < n; i += 1) {
    state.views[i].viewport = {
      x: i * (viewWidth + VIEW_GAP),
      y: 0,
      w: viewWidth,
      h: height,
    };
  }
}

function shortCount(n) {
  if (n >= 10000) return `${Math.round(n / 1000)}k`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return `${n || 0}`;
}

function updateDateLabels() {
  if (!state.labels) return;
  state.labels.innerHTML = state.views.map((view) => {
    const v = view.viewport;
    const sample = shortCount(view.cloud.sample_count || view.leafid.length);
    const full = shortCount(view.cloud.full_count || view.leafid.length);
    return `<div class="lk-date-label" style="left:${v.x}px;width:${v.w}px">
      <b>${escapeHtml(view.date)}</b><span>${sample}/${full} pts</span>
    </div>`;
  }).join("");
}

function resizeCanvas() {
  if (!state.renderer || !state.wrap || !state.canvas || !state.panel?.classList.contains("on")) return;
  const rect = state.wrap.getBoundingClientRect();
  const visibleWidth = Math.max(320, Math.floor(rect.width));
  const height = Math.max(300, Math.floor(rect.height));
  const n = Math.max(1, state.views.length);
  const width = Math.max(visibleWidth, n * MIN_VIEW_WIDTH + (n - 1) * VIEW_GAP);

  state.canvas.style.width = `${width}px`;
  state.canvas.style.height = `${height}px`;
  if (state.labels) {
    state.labels.style.width = `${width}px`;
    state.labels.style.height = `${height}px`;
  }
  state.renderer.setSize(width, height, false);
  layoutViews(width, height);
  fitAllCameras();
  updateDateLabels();
  requestRender();
}

function requestRender() {
  if (state.renderQueued) return;
  state.renderQueued = true;
  requestAnimationFrame(renderFrame);
}

function renderFrame() {
  state.renderQueued = false;
  if (!state.renderer || !state.panel?.classList.contains("on")) return;
  const renderer = state.renderer;
  const size = renderer.getSize(new THREE.Vector2());
  renderer.setScissorTest(false);
  renderer.setClearColor(0xe8edf3, 1);
  renderer.clear();
  renderer.setScissorTest(true);

  for (const view of state.views) {
    const v = view.viewport;
    const y = size.y - v.y - v.h;
    renderer.setViewport(v.x, y, v.w, v.h);
    renderer.setScissor(v.x, y, v.w, v.h);
    renderer.render(view.scene, view.camera);
  }
  renderer.setScissorTest(false);
}

function viewAtCanvasPoint(pt) {
  return state.views.find((view) => {
    const v = view.viewport;
    return pt.x >= v.x && pt.x <= v.x + v.w && pt.y >= v.y && pt.y <= v.y + v.h;
  });
}

function nearestPoint(view, pt, maxPixels = PICK_RADIUS_PX) {
  const positions = view.geometry.attributes.position;
  let bestIndex = -1;
  let bestDist2 = maxPixels * maxPixels;
  const v = view.viewport;

  for (let i = 0; i < positions.count; i += 1) {
    scratch.fromBufferAttribute(positions, i);
    scratch.project(view.camera);
    if (scratch.z < -1 || scratch.z > 1) continue;
    const sx = v.x + ((scratch.x + 1) / 2) * v.w;
    const sy = v.y + ((-scratch.y + 1) / 2) * v.h;
    const dx = sx - pt.x;
    const dy = sy - pt.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestDist2) {
      bestDist2 = d2;
      bestIndex = i;
    }
  }

  return bestIndex < 0 ? null : { index: bestIndex, dist: Math.sqrt(bestDist2) };
}

function selectAtEvent(event) {
  if (event.button !== 0) return;
  const rect = state.canvas.getBoundingClientRect();
  const pt = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  const view = viewAtCanvasPoint(pt);
  if (!view) return;
  const hit = nearestPoint(view, pt);
  if (!hit) {
    setStatus(`${view.date}: no point close enough`);
    return;
  }
  const leafid = view.leafid[hit.index] || 0;
  const otype = view.otype[hit.index] || 0;
  if (otype !== 2 || leafid <= 0) {
    setStatus(`${view.date}: clicked point is not a labeled leaf`);
    return;
  }
  state.draft[view.date] = leafid;
  updateAllColors();
  renderUi();
  setStatus(`${view.date}: leaf ${leafid} selected for rank ${draftRank()}`);
}

function zoomBy(factor) {
  state.zoom = THREE.MathUtils.clamp(state.zoom * factor, 0.35, 8);
  fitAllCameras();
  requestRender();
}

function rotateBy(delta) {
  state.yaw += delta;
  fitAllCameras();
  requestRender();
}

function resetView() {
  state.yaw = 0;
  state.zoom = 1;
  fitAllCameras();
  requestRender();
}

function picksHtml(obs) {
  const clean = cleanObs(obs);
  const parts = [];
  for (const date of dateOrder(clean)) {
    if (!clean[date]) continue;
    parts.push(`<span class="lk-pick">${escapeHtml(date)} <b>L${clean[date]}</b></span>`);
  }
  return parts.length ? parts.join("") : '<span class="lk-empty">No leaf picks</span>';
}

function renderUi() {
  const meta = document.getElementById("link-meta");
  if (meta) {
    meta.textContent = `plant ${state.plant} - ${state.dates.length} dates - ${state.chains.length} chains - ${state.seedLabel}`;
  }

  const draftTitle = document.getElementById("link-draft-title");
  if (draftTitle) {
    draftTitle.textContent = state.editingIndex >= 0 ? `Editing rank ${draftRank()}` : `New rank ${draftRank()}`;
  }
  const draftPicks = document.getElementById("link-draft-picks");
  if (draftPicks) draftPicks.innerHTML = picksHtml(state.draft);

  const commit = document.querySelector('[data-link-action="commit"]');
  if (commit) {
    commit.textContent = state.editingIndex >= 0 ? `Update rank ${draftRank()}` : `Commit rank ${draftRank()}`;
    commit.disabled = !hasDraft();
  }

  const chainList = document.getElementById("link-chain-list");
  if (chainList) {
    chainList.innerHTML = state.chains.length ? state.chains.map((chain, index) => {
      const color = cssColor(rankColor(index + 1));
      const active = index === state.editingIndex ? " active" : "";
      return `<div class="lk-chain-row${active}">
        <button type="button" class="lk-chain-main" data-link-action="edit-chain" data-index="${index}">
          <span class="lk-rank" style="background:${color}">r${index + 1}</span>
          <span class="lk-obs">${escapeHtml(formatObs(chain.obs))}</span>
        </button>
        <button type="button" data-link-action="move-up" data-index="${index}" ${index === 0 ? "disabled" : ""}>Up</button>
        <button type="button" data-link-action="move-down" data-index="${index}" ${index === state.chains.length - 1 ? "disabled" : ""}>Down</button>
        <button type="button" data-link-action="delete-chain" data-index="${index}">Del</button>
      </div>`;
    }).join("") : '<div class="lk-empty">No committed chains</div>';
  }
}

function removeDuplicateClaims(obs, keepIndex) {
  const claims = new Set(Object.entries(obs).map(([date, leafid]) => `${date}:${leafid}`));
  for (let i = 0; i < state.chains.length; i += 1) {
    if (i === keepIndex) continue;
    for (const [date, leafid] of Object.entries(state.chains[i].obs)) {
      if (claims.has(`${date}:${leafid}`)) delete state.chains[i].obs[date];
    }
  }
  const kept = [];
  let newEditing = -1;
  for (let i = 0; i < state.chains.length; i += 1) {
    if (Object.keys(state.chains[i].obs).length) {
      if (i === state.editingIndex) newEditing = kept.length;
      kept.push(state.chains[i]);
    }
  }
  state.chains = kept;
  state.editingIndex = newEditing;
}

function commitDraft() {
  const obs = cleanObs(state.draft);
  if (!Object.keys(obs).length) {
    setStatus("Pick at least one leaf before committing");
    return;
  }
  const wasEditing = state.editingIndex >= 0;
  removeDuplicateClaims(obs, state.editingIndex);
  const editIndex = wasEditing ? state.editingIndex : -1;
  let committedRank;
  if (editIndex >= 0 && state.chains[editIndex]) {
    state.chains[editIndex] = { rank: editIndex + 1, obs };
    committedRank = editIndex + 1;
  } else {
    state.chains.push({ rank: state.chains.length + 1, obs });
    committedRank = state.chains.length;
  }
  state.draft = {};
  state.editingIndex = -1;
  renumberChains();
  updateAllColors();
  renderUi();
  setStatus(`Committed rank ${committedRank}`);
}

function editChain(index) {
  const chain = state.chains[index];
  if (!chain) return;
  state.editingIndex = index;
  state.draft = { ...chain.obs };
  updateAllColors();
  renderUi();
  setStatus(`Editing rank ${index + 1}`);
}

function moveChain(index, delta) {
  const next = index + delta;
  if (next < 0 || next >= state.chains.length) return;
  [state.chains[index], state.chains[next]] = [state.chains[next], state.chains[index]];
  if (state.editingIndex === index) state.editingIndex = next;
  else if (state.editingIndex === next) state.editingIndex = index;
  renumberChains();
  updateAllColors();
  renderUi();
}

function deleteChain(index) {
  if (!state.chains[index]) return;
  state.chains.splice(index, 1);
  if (state.editingIndex === index) {
    state.editingIndex = -1;
    state.draft = {};
  } else if (state.editingIndex > index) {
    state.editingIndex -= 1;
  }
  renumberChains();
  updateAllColors();
  renderUi();
  setStatus(`Deleted rank ${index + 1}`);
}

function clearDraft() {
  state.draft = {};
  state.editingIndex = -1;
  updateAllColors();
  renderUi();
}

function clearAllChains() {
  if (state.chains.length && !window.confirm("Clear all unsaved link chains?")) return;
  state.chains = [];
  clearDraft();
  setStatus("Cleared chains");
}

async function loadData(fresh = false) {
  state.plant = currentPlant();
  state.loadedPlant = state.plant;
  const token = ++state.loadToken;
  setBusy(`Loading plant ${state.plant}...`);
  setStatus("");
  try {
    const suffix = fresh ? "&fresh=1" : "";
    const [clouds, chains] = await Promise.all([
      fetchJson(`/link/clouds?plant=${encodeURIComponent(state.plant)}&max_points=${CLOUD_POINTS}`),
      fetchJson(`/link/chains?plant=${encodeURIComponent(state.plant)}${suffix}`),
    ]);
    if (token !== state.loadToken) return;
    state.clouds = clouds.clouds || {};
    state.dates = (clouds.dates || Object.keys(state.clouds)).filter((date) => state.clouds[date]);
    state.chains = normalizeChains(chains.chains || []);
    state.draft = {};
    state.editingIndex = -1;
    state.hasManual = !!chains.has_manual;
    state.seedLabel = fresh ? "auto seed" : (state.hasManual ? "manual saved" : "auto seed");
    buildViews();
    renderUi();
    resizeCanvas();
    setStatus(state.views.length ? "Click leaf points to build the current rank" : "No handlabel clouds found for this plant");
  } catch (err) {
    if (token === state.loadToken) setStatus(`Link load failed: ${err.message}`);
  } finally {
    if (token === state.loadToken) setBusy("");
  }
}

async function saveChains() {
  if (hasDraft()) {
    setStatus("Commit or clear the current rank before saving");
    return;
  }
  renumberChains();
  try {
    const data = await fetchJson("/link/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plant: state.plant, chains: state.chains }),
    });
    state.hasManual = true;
    state.seedLabel = "manual saved";
    renderUi();
    setStatus(`Saved ${data.n} chains to manual_chains.json`);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`);
  }
}

function handleAction(event) {
  const button = event.target.closest("button[data-link-action]");
  if (!button) return;
  const action = button.dataset.linkAction;
  const index = Number.parseInt(button.dataset.index || "-1", 10);

  if (action === "reload") loadData(false);
  else if (action === "fresh") loadData(true);
  else if (action === "save") saveChains();
  else if (action === "commit") commitDraft();
  else if (action === "new-draft") clearDraft();
  else if (action === "clear-draft") clearDraft();
  else if (action === "clear-all") clearAllChains();
  else if (action === "edit-chain") editChain(index);
  else if (action === "move-up") moveChain(index, -1);
  else if (action === "move-down") moveChain(index, 1);
  else if (action === "delete-chain") deleteChain(index);
  else if (action === "yaw-left") rotateBy(-Math.PI / 12);
  else if (action === "yaw-right") rotateBy(Math.PI / 12);
  else if (action === "zoom-in") zoomBy(1.18);
  else if (action === "zoom-out") zoomBy(1 / 1.18);
  else if (action === "reset-view") resetView();
}

function ensureRenderer() {
  if (state.renderer) return;
  state.renderer = new THREE.WebGLRenderer({ canvas: state.canvas, antialias: true });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
}

function ensurePanel() {
  let panel = document.getElementById("link-panel");
  if (!panel) {
    panel = document.createElement("div");
    panel.id = "link-panel";
    panel.innerHTML = `
      <div id="link-toolbar">
        <div class="lk-title"><b>LINK</b><span id="link-meta"></span></div>
        <div class="lk-actions">
          <button type="button" data-link-action="reload">Reload</button>
          <button type="button" data-link-action="fresh">Auto seed</button>
          <button type="button" data-link-action="clear-all">Clear all</button>
          <button type="button" data-link-action="yaw-left">View -15</button>
          <button type="button" data-link-action="yaw-right">View +15</button>
          <button type="button" data-link-action="zoom-out">Zoom out</button>
          <button type="button" data-link-action="zoom-in">Zoom in</button>
          <button type="button" data-link-action="reset-view">Reset view</button>
          <button type="button" class="primary" data-link-action="save">Save</button>
        </div>
      </div>
      <div id="link-body">
        <div id="link-view-wrap">
          <canvas id="link-viewer"></canvas>
          <div id="link-date-labels"></div>
          <div id="link-loading" hidden></div>
        </div>
        <aside id="link-sidebar">
          <section>
            <div class="lk-section-head">
              <span id="link-draft-title">New rank 1</span>
              <button type="button" data-link-action="new-draft">New</button>
            </div>
            <div id="link-draft-picks"></div>
            <div class="lk-side-actions">
              <button type="button" class="primary" data-link-action="commit">Commit rank 1</button>
              <button type="button" data-link-action="clear-draft">Clear draft</button>
            </div>
          </section>
          <section class="lk-chain-section">
            <div class="lk-section-head"><span>Ranks</span></div>
            <div id="link-chain-list"></div>
          </section>
          <div id="link-status"></div>
        </aside>
      </div>`;
    (document.getElementById("preview") || document.body).appendChild(panel);
    panel.addEventListener("click", handleAction);
  }

  state.panel = panel;
  state.canvas = document.getElementById("link-viewer");
  state.wrap = document.getElementById("link-view-wrap");
  state.labels = document.getElementById("link-date-labels");
  ensureRenderer();

  state.canvas.addEventListener("click", selectAtEvent);
  state.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12);
  }, { passive: false });
  state.canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  if (!state.resizeObserver && state.wrap) {
    state.resizeObserver = new ResizeObserver(resizeCanvas);
    state.resizeObserver.observe(state.wrap);
  }
}

function installModeOption() {
  const sel = document.getElementById("mode-select");
  if (!sel) return null;
  if (!Array.from(sel.options).some((option) => option.value === MODE)) {
    const option = document.createElement("option");
    option.value = MODE;
    option.textContent = "Link leaves (cross-date)";
    sel.appendChild(option);
  }
  return sel;
}

function togglePanel() {
  const sel = document.getElementById("mode-select");
  if (!state.panel || !sel) return;
  if (sel.value === MODE) {
    state.panel.classList.add("on");
    const plant = currentPlant();
    if (plant !== state.loadedPlant || !state.views.length) loadData(false);
    else resizeCanvas();
  } else {
    state.panel.classList.remove("on");
  }
}

function setup() {
  ensurePanel();
  renderUi();
  const modeSelect = installModeOption();
  if (modeSelect) {
    modeSelect.addEventListener("change", togglePanel);
  }
  const plantSelect = document.getElementById("plant-select");
  if (plantSelect) {
    plantSelect.addEventListener("change", () => {
      if (state.panel?.classList.contains("on")) loadData(false);
    });
  }
  window.addEventListener("resize", resizeCanvas);
  togglePanel();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", setup);
else setup();
