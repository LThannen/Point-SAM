// Cross-date leaf LINK mode — GUIDED 1:1, two stages at a time.
// Shows exactly two consecutive stages side by side. You track one leaf in the
// left stage (highlighted yellow) and click its match in the right stage; the
// tool records the link and steps to the next leaf, then the next stage pair.
// Chains (one physical leaf across dates) + basal->apical rank are saved to
// manual_chains.json (consumed by relink_leaf_identity.py). Seeded from the
// automatic tracker so you CORRECT its guesses rather than start blank.
import * as THREE from "three";

const MODE = "link";
const CLOUD_POINTS = 50_000;
const VIEW_GAP = 14;
const PICK_RADIUS_PX = 18;
const POINT_SIZE = 4;

const YELLOW = [1.0, 0.82, 0.05];
const STEM = [0.42, 0.24, 0.10];
const PALE = [0.66, 0.70, 0.75];
const FAINT = [0.82, 0.84, 0.87];

const state = {
  plant: "01", loadedPlant: null, loadedContext: null,
  dates: [], clouds: {}, allViews: {},
  chains: [],                 // [{obs:{date:leafid}}], order == rank-1
  pairIndex: 0, leftCursor: 0,
  panel: null, canvas: null, renderer: null, wrap: null, labels: null,
  views: [], yaw: 0, pitch: 0, zoom: 1, rotatePointer: null,
  renderQueued: false, loadToken: 0, resizeObserver: null,
};
const scratch = new THREE.Vector3();

// ---------- colour + small utils ----------
function rankColor(rank) {
  const c = new THREE.Color();
  c.setHSL((rank * 0.38196601125) % 1, 0.72, 0.5);
  return [c.r, c.g, c.b];
}
function dim(rgb) { return rgb.map((v) => 0.45 * v + 0.5); }   // pastel version for non-active
function cssColor(rgb) { return `rgb(${rgb.map((v) => Math.round(v * 255)).join(", ")})`; }
function escapeHtml(v) { return String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function currentPlant() {
  const lp = document.getElementById("link-plant");     // the linker's own picker (visible in link mode)
  if (lp && lp.value) return lp.value;
  const sel = document.getElementById("plant-select");
  const v = sel && sel.value ? sel.value : "1";
  return v.toString().padStart(2, "0");
}
async function populatePlants(token = state.loadToken) {
  const lp = document.getElementById("link-plant");
  if (!lp) return false;
  try {
    const data = await fetchJson("/link/plants");
    if (token !== state.loadToken) return false;
    const plants = data.plants || [];
    const opts = plants.map((p) => p.plant);
    const prev = lp.value;
    const side = ((document.getElementById("plant-select") || {}).value || "").padStart(2, "0");
    lp.innerHTML = plants.length
      ? plants.map((p) => `<option value="${p.plant}">plant ${p.plant} · ${p.n_dates} dates</option>`).join("")
      : `<option value="">no labeled plants</option>`;
    if (opts.length) lp.value = opts.includes(prev) ? prev : (opts.includes(side) ? side : opts[0]);
    return true;
  } catch (err) { return false; }
}
function cleanObs(obs) {
  const out = {};
  for (const [d, l] of Object.entries(obs || {})) { const n = parseInt(l, 10); if (n > 0) out[d] = n; }
  return out;
}
function normalizeChains(chains) {
  return (chains || []).map((c) => ({ obs: cleanObs(c.obs) })).filter((c) => Object.keys(c.obs).length);
}
async function fetchJson(url, options) {
  const r = await fetch(url, options);
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
  return d;
}
function setStatus(t) { const n = document.getElementById("link-status"); if (n) n.textContent = t || ""; }
function setBusy(t) { const n = document.getElementById("link-loading"); if (n) { n.hidden = !t; n.textContent = t || ""; } }
function shortCount(n) { return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n || 0}`; }

// ---------- geometry / three plumbing (from the prior renderer) ----------
function flattenXYZ(xyz) {
  if (!xyz || !xyz.length) return new Float32Array();
  if (Array.isArray(xyz[0])) {
    const out = new Float32Array(xyz.length * 3);
    for (let i = 0; i < xyz.length; i += 1) { out[i * 3] = xyz[i][0]; out[i * 3 + 1] = xyz[i][1]; out[i * 3 + 2] = xyz[i][2]; }
    return out;
  }
  return Float32Array.from(xyz);
}
function boundsFor(p) {
  const b = { minX: Infinity, minY: Infinity, minZ: Infinity, maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity };
  for (let i = 0; i < p.length; i += 3) {
    b.minX = Math.min(b.minX, p[i]); b.maxX = Math.max(b.maxX, p[i]);
    b.minY = Math.min(b.minY, p[i + 1]); b.maxY = Math.max(b.maxY, p[i + 1]);
    b.minZ = Math.min(b.minZ, p[i + 2]); b.maxZ = Math.max(b.maxZ, p[i + 2]);
  }
  if (!Number.isFinite(b.minX)) { b.minX = b.minY = b.minZ = -1; b.maxX = b.maxY = b.maxZ = 1; }
  b.cx = (b.minX + b.maxX) / 2; b.cy = (b.minY + b.maxY) / 2; b.cz = (b.minZ + b.maxZ) / 2;
  b.sizeX = Math.max(1e-3, b.maxX - b.minX); b.sizeY = Math.max(1e-3, b.maxY - b.minY); b.sizeZ = Math.max(1e-3, b.maxZ - b.minZ);
  b.size = Math.max(b.sizeX, b.sizeY, b.sizeZ, 1);
  return b;
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
  const material = new THREE.PointsMaterial({ size: POINT_SIZE, sizeAttenuation: false, vertexColors: true });
  const points = new THREE.Points(geometry, material); points.frustumCulled = false;
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xf6f8fb); scene.add(points);
  return {
    date, cloud, scene, points, geometry, material, otype, leafid,
    camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100000),
    bounds: boundsFor(positions), viewport: { x: 0, y: 0, w: 1, h: 1 },
  };
}
function disposeAll() {
  for (const v of Object.values(state.allViews)) { v.geometry.dispose(); v.material.dispose(); }
  state.allViews = {}; state.views = [];
}
function fitCamera(view) {
  const { w, h } = view.viewport;
  const aspect = Math.max(0.1, w / Math.max(1, h));
  const b = view.bounds;
  const horiz = Math.max(b.sizeX, b.sizeY);
  const halfH = Math.max(b.sizeZ / 2, horiz / (2 * aspect), 1) * 1.16 / state.zoom;
  const halfW = halfH * aspect;
  const cam = view.camera;
  cam.left = -halfW; cam.right = halfW; cam.top = halfH; cam.bottom = -halfH;
  cam.near = 0.01; cam.far = b.size * 10 + 1000;
  const dist = b.size * 3 + 10;
  const horizontal = Math.cos(state.pitch) * dist;
  cam.position.set(
    b.cx + Math.sin(state.yaw) * horizontal,
    b.cy + Math.cos(state.yaw) * horizontal,
    b.cz + Math.sin(state.pitch) * dist,
  );
  cam.up.set(0, 0, 1); cam.lookAt(b.cx, b.cy, b.cz); cam.updateProjectionMatrix();
}
function layoutViews(width, height) {
  const n = state.views.length; if (!n) return;
  const vw = Math.floor((width - VIEW_GAP * (n - 1)) / n);
  state.views.forEach((v, i) => { v.viewport = { x: i * (vw + VIEW_GAP), y: 0, w: vw, h: height }; });
}
function updateDateLabels() {
  if (!state.labels) return;
  const roles = ["◀ earlier stage", "later stage ▶"];
  state.labels.innerHTML = state.views.map((view, i) => {
    const v = view.viewport;
    return `<div class="lk-date-label" style="left:${v.x}px;width:${v.w}px">
      <b>${escapeHtml(view.date)}</b><span>${roles[i] || ""} · ${shortCount(view.cloud.sample_count || view.leafid.length)} pts</span></div>`;
  }).join("");
}
function resizeCanvas() {
  if (!state.renderer || !state.wrap || !state.canvas || !state.panel?.classList.contains("on")) return;
  const rect = state.wrap.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(300, Math.floor(rect.height));
  state.canvas.style.width = `${width}px`; state.canvas.style.height = `${height}px`;
  if (state.labels) { state.labels.style.width = `${width}px`; state.labels.style.height = `${height}px`; }
  state.renderer.setSize(width, height, false);
  layoutViews(width, height); state.views.forEach(fitCamera); updateDateLabels(); requestRender();
}
function requestRender() { if (!state.renderQueued) { state.renderQueued = true; requestAnimationFrame(renderFrame); } }
function renderFrame() {
  state.renderQueued = false;
  if (!state.renderer || !state.panel?.classList.contains("on")) return;
  const r = state.renderer, size = r.getSize(new THREE.Vector2());
  r.setScissorTest(false); r.setClearColor(0xe8edf3, 1); r.clear(); r.setScissorTest(true);
  for (const view of state.views) {
    const v = view.viewport, y = size.y - v.y - v.h;
    r.setViewport(v.x, y, v.w, v.h); r.setScissor(v.x, y, v.w, v.h); r.render(view.scene, view.camera);
  }
  r.setScissorTest(false);
}
function viewAtCanvasPoint(pt) {
  return state.views.find((v) => pt.x >= v.viewport.x && pt.x <= v.viewport.x + v.viewport.w && pt.y >= v.viewport.y && pt.y <= v.viewport.y + v.viewport.h);
}
function nearestPoint(view, pt, maxPixels = PICK_RADIUS_PX) {
  const pos = view.geometry.attributes.position; let best = -1, bd2 = maxPixels * maxPixels; const v = view.viewport;
  for (let i = 0; i < pos.count; i += 1) {
    scratch.fromBufferAttribute(pos, i); scratch.project(view.camera);
    if (scratch.z < -1 || scratch.z > 1) continue;
    const sx = v.x + ((scratch.x + 1) / 2) * v.w, sy = v.y + ((-scratch.y + 1) / 2) * v.h;
    const d2 = (sx - pt.x) ** 2 + (sy - pt.y) ** 2;
    if (d2 < bd2) { bd2 = d2; best = i; }
  }
  return best < 0 ? null : best;
}

// ---------- guided-flow model ----------
function leftDate() { return state.dates[state.pairIndex]; }
function rightDate() { return state.dates[state.pairIndex + 1]; }
function leavesOf(date) {
  const view = state.allViews[date]; if (!view) return [];
  const s = new Set();
  for (let i = 0; i < view.leafid.length; i += 1) { const l = view.leafid[i]; if (l > 0 && (view.otype[i] === 2)) s.add(l); }
  return [...s].sort((a, b) => a - b);   // leafid ascending == basal->apical
}
function leftLeaves() { return leavesOf(leftDate()); }
function currentLeftLeaf() { return leftLeaves()[state.leftCursor]; }
function chainIndexFor(date, leafid) { return state.chains.findIndex((c) => c.obs[date] === leafid); }
function assignMap(date) { const m = {}; state.chains.forEach((c, i) => { if (c.obs[date]) m[c.obs[date]] = i; }); return m; }
function activeChainIndex() { const l = currentLeftLeaf(); return l ? chainIndexFor(leftDate(), l) : -1; }

function colorView(view, role) {
  const colors = view.geometry.attributes.color.array;
  const date = view.date;
  const amap = assignMap(date);
  const li = currentLeftLeaf();
  const ci = activeChainIndex();
  const activeRight = role === "right" && ci >= 0 ? state.chains[ci].obs[rightDate()] : null;
  for (let i = 0; i < view.leafid.length; i += 1) {
    const ot = view.otype[i] || 0, lf = view.leafid[i] || 0;
    let c;
    if (ot === 1) c = STEM;
    else if (ot !== 2 || lf <= 0) c = FAINT;
    else if (role === "left" && lf === li) c = YELLOW;                 // the leaf you are tracking
    else if (role === "right" && lf === activeRight) c = YELLOW;       // its current match
    else if (amap[lf] !== undefined) c = dim(rankColor(amap[lf] + 1)); // assigned to some chain
    else c = PALE;                                                     // unassigned candidate
    colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
  }
  view.geometry.attributes.color.needsUpdate = true;
}
function refreshColors() {
  if (state.views[0]) colorView(state.views[0], "left");
  if (state.views[1]) colorView(state.views[1], "right");
  requestRender();
}

function showPair() {
  const lv = state.allViews[leftDate()], rv = state.allViews[rightDate()];
  state.views = [lv, rv].filter(Boolean);
  const leaves = leftLeaves();
  if (state.leftCursor >= leaves.length) state.leftCursor = Math.max(0, leaves.length - 1);
  resizeCanvas(); refreshColors(); renderUi();
}
function nextLeaf() { const n = leftLeaves().length; if (state.leftCursor < n - 1) { state.leftCursor += 1; refreshColors(); renderUi(); } else setStatus("Last leaf of this stage — use “Next stage ▶”."); }
function prevLeaf() { if (state.leftCursor > 0) { state.leftCursor -= 1; refreshColors(); renderUi(); } }
function linkRight(rj) {
  const li = currentLeftLeaf(); if (!li) return;
  const rd = rightDate();
  // a right leaf matches only one left leaf: drop it from any other chain first
  state.chains.forEach((c) => { if (c.obs[rd] === rj) delete c.obs[rd]; });
  let ci = activeChainIndex();
  if (ci < 0) { state.chains.push({ obs: { [leftDate()]: li } }); ci = state.chains.length - 1; }
  state.chains[ci].obs[rd] = rj;
  setStatus(`Linked ${leftDate()} L${li} ↔ ${rd} L${rj}`);
  nextLeafOrHold();
}
function markNotPresent() {
  const ci = activeChainIndex();
  if (ci >= 0) delete state.chains[ci].obs[rightDate()];
  setStatus(`${leftDate()} L${currentLeftLeaf()} marked absent in ${rightDate()}`);
  nextLeafOrHold();
}
function nextLeafOrHold() {
  const n = leftLeaves().length;
  if (state.leftCursor < n - 1) state.leftCursor += 1;
  refreshColors(); renderUi();
}
function newRightLeaves() {
  const amap = assignMap(rightDate());
  return leavesOf(rightDate()).filter((l) => amap[l] === undefined);
}
function nextPair() {
  // any still-unmatched right leaf becomes a new chain starting here
  for (const rj of newRightLeaves()) state.chains.push({ obs: { [rightDate()]: rj } });
  if (state.pairIndex >= state.dates.length - 2) { refreshColors(); renderUi(); setStatus("Last stage pair reached."); return; }
  state.pairIndex += 1; state.leftCursor = 0; showPair();
}
function prevPair() { if (state.pairIndex > 0) { state.pairIndex -= 1; state.leftCursor = 0; showPair(); } }

function jumpLeftTo(leafid) { const idx = leftLeaves().indexOf(leafid); if (idx >= 0) { state.leftCursor = idx; refreshColors(); renderUi(); } }

function selectAtEvent(event) {
  if (event.button !== 0) return;
  const rect = state.canvas.getBoundingClientRect();
  const pt = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  const view = viewAtCanvasPoint(pt); if (!view) return;
  const idx = nearestPoint(view, pt);
  if (idx == null) { setStatus(`${view.date}: no point close enough`); return; }
  const lf = view.leafid[idx] || 0, ot = view.otype[idx] || 0;
  if (ot !== 2 || lf <= 0) { setStatus(`${view.date}: not a labeled leaf`); return; }
  if (view === state.views[0]) jumpLeftTo(lf);      // left panel: choose which leaf to track
  else linkRight(lf);                                // right panel: set its match
}

// ---------- camera controls ----------
function zoomBy(f) { state.zoom = THREE.MathUtils.clamp(state.zoom * f, 0.35, 8); state.views.forEach(fitCamera); requestRender(); }
function rotateBy(yaw, pitch = 0) {
  state.yaw += yaw;
  state.pitch = THREE.MathUtils.clamp(state.pitch + pitch, -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
  state.views.forEach(fitCamera); requestRender();
}
function startRotate(event) {
  if (event.button !== 1) return;
  event.preventDefault();
  state.rotatePointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
  state.canvas.setPointerCapture(event.pointerId);
  state.canvas.classList.add("dragging");
}
function dragRotate(event) {
  const drag = state.rotatePointer;
  if (!drag || drag.id !== event.pointerId) return;
  rotateBy(-(event.clientX - drag.x) * 0.01, -(event.clientY - drag.y) * 0.01);
  drag.x = event.clientX; drag.y = event.clientY;
}
function stopRotate(event) {
  if (!state.rotatePointer || state.rotatePointer.id !== event.pointerId) return;
  state.rotatePointer = null;
  state.canvas.classList.remove("dragging");
}
function resetView() { state.yaw = 0; state.pitch = 0; state.zoom = 1; state.views.forEach(fitCamera); requestRender(); }

// ---------- sidebar ----------
function renderUi() {
  const meta = document.getElementById("link-meta");
  if (meta) meta.textContent = `plant ${state.plant} · ${state.dates.length} stages · ${state.chains.length} leaves`;
  const nPairs = Math.max(1, state.dates.length - 1);
  const stepEl = document.getElementById("link-step");
  if (stepEl) {
    const leaves = leftLeaves(), li = currentLeftLeaf();
    const ci = activeChainIndex();
    const rd = rightDate();
    const rmatch = ci >= 0 ? state.chains[ci].obs[rd] : undefined;
    stepEl.innerHTML = `
      <div class="lk-pair">Stage pair <b>${state.pairIndex + 1}/${nPairs}</b>:
        <span class="lk-d">${escapeHtml(leftDate() || "-")}</span> → <span class="lk-d">${escapeHtml(rd || "-")}</span></div>
      <div class="lk-track">Tracking <b>leaf ${state.leftCursor + 1}/${leaves.length}</b> —
        <span class="lk-yellow">${escapeHtml(leftDate() || "")} L${li ?? "-"}</span>
        ${ci >= 0 ? `(rank ${ci + 1})` : `(new)`}</div>
      <div class="lk-match">match in ${escapeHtml(rd || "")}:
        ${rmatch ? `<span class="lk-yellow">L${rmatch}</span>` : `<i>none — click it in the right panel</i>`}</div>`;
  }
  const list = document.getElementById("link-chain-list");
  if (list) {
    list.innerHTML = state.chains.length ? state.chains.map((c, i) => {
      const obs = [...state.dates].filter((d) => c.obs[d]).map((d) => `${d.slice(-4)}:L${c.obs[d]}`).join(" ");
      const active = i === activeChainIndex() ? " lk-active" : "";
      return `<div class="lk-chain${active}">
        <button class="lk-rank" style="background:${cssColor(rankColor(i + 1))}" data-link-action="focus-chain" data-index="${i}">r${i + 1}</button>
        <span class="lk-obs">${obs || "(empty)"}</span>
        <button data-link-action="move-up" data-index="${i}" ${i === 0 ? "disabled" : ""}>↑</button>
        <button data-link-action="move-down" data-index="${i}" ${i === state.chains.length - 1 ? "disabled" : ""}>↓</button>
        <button data-link-action="delete-chain" data-index="${i}">✕</button></div>`;
    }).join("") : `<div class="lk-empty">no leaves yet</div>`;
  }
}

// ---------- chain edits ----------
function moveChain(i, d) {
  const j = i + d; if (j < 0 || j >= state.chains.length) return;
  [state.chains[i], state.chains[j]] = [state.chains[j], state.chains[i]];
  refreshColors(); renderUi();
}
function deleteChain(i) { if (state.chains[i]) { state.chains.splice(i, 1); refreshColors(); renderUi(); } }
function focusChain(i) {
  // jump the guided cursor to this chain's leaf in the current left stage (if present)
  const c = state.chains[i]; if (!c) return;
  const l = c.obs[leftDate()];
  if (l) jumpLeftTo(l); else setStatus(`rank ${i + 1} has no leaf in ${leftDate()}`);
}
function chainBaseHeight(chain) {
  let base = Infinity;
  for (const [date, leafid] of Object.entries(chain.obs)) {
    const view = state.allViews[date]; if (!view) continue;
    const attachment = view.cloud.attachment_z?.[leafid];
    if (Number.isFinite(attachment)) base = Math.min(base, attachment);
  }
  return base;
}
function renumberByHeight() {
  state.chains.sort((a, b) => chainBaseHeight(a) - chainBaseHeight(b));
  refreshColors(); renderUi();
  setStatus("Renumbered ranks by stem attachment height — click Save to keep.");
}

// ---------- load + save ----------
async function loadData(fresh = false) {
  state.plant = currentPlant(); state.loadedPlant = null; state.loadedContext = null;
  const token = ++state.loadToken; setBusy(fresh ? `Auto-linking plant ${state.plant} (CPD, ~10s)…` : `Loading plant ${state.plant}…`); setStatus("");
  try {
    const [clouds, chains] = await Promise.all([
      fetchJson(`/link/clouds?plant=${encodeURIComponent(state.plant)}&max_points=${CLOUD_POINTS}`),
      fetchJson(`/link/chains?plant=${encodeURIComponent(state.plant)}${fresh ? "&fresh=1" : ""}`),
    ]);
    if (token !== state.loadToken) return;
    if (!clouds.context || clouds.context !== chains.context) throw new Error("dataset changed while links loaded");
    state.clouds = clouds.clouds || {};
    disposeAll();
    const dates = (clouds.dates || Object.keys(state.clouds)).filter((d) => state.clouds[d]);
    for (const d of dates) { const v = makeView(d, state.clouds[d]); if (v) state.allViews[d] = v; }
    // keep only stages that actually have >=1 labeled leaf
    state.dates = dates.filter((d) => leavesOf(d).length > 0);
    state.chains = normalizeChains(chains.chains || []);
    if (!state.chains.length && state.dates.length) {
      state.chains = leavesOf(state.dates[0]).map((leaf) => ({ obs: { [state.dates[0]]: leaf } }));
    }
    state.loadedPlant = state.plant; state.loadedContext = clouds.context;
    const auto = document.querySelector('[data-link-action="fresh"]');
    if (auto) { auto.disabled = !chains.auto_available; auto.title = chains.auto_available ? "Reset to the automatic CPD draft" : "Automatic linking is not configured for this plant"; }
    // Show ALL hand-labeled stages (the earliest anchor the basal leaves).
    // The auto seed only covers the comparison dates; earlier-stage leaves start
    // unassigned and you link them by hand.
    state.pairIndex = 0; state.leftCursor = 0;
    if (state.dates.length < 2) { setStatus("Need at least two stages with labeled leaves."); state.views = []; renderUi(); return; }
    showPair();
    setStatus("Left panel: track the yellow leaf. Click its match on the right; middle-drag either panel to rotate both.");
  } catch (err) {
    if (token === state.loadToken) setStatus(`Link load failed: ${err.message}`);
  } finally { if (token === state.loadToken) setBusy(""); }
}
async function saveChains() {
  if (!state.loadedContext) { setStatus("Save blocked: reload links for the current dataset."); return; }
  const payload = state.chains.map((c, i) => ({ rank: i + 1, obs: c.obs }));
  try {
    const data = await fetchJson("/link/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ plant: state.plant, context: state.loadedContext, chains: payload }) });
    setStatus(`Saved ${data.n} leaves → manual_chains.json`);
  } catch (err) { setStatus(`Save failed: ${err.message}`); }
}

// ---------- wiring ----------
function handleAction(event) {
  const b = event.target.closest("button[data-link-action]"); if (!b) return;
  const a = b.dataset.linkAction, i = Number.parseInt(b.dataset.index || "-1", 10);
  ({
    "reload": () => loadData(false), "fresh": () => loadData(true), "save": saveChains,
    "next-leaf": nextLeaf, "prev-leaf": prevLeaf, "not-present": markNotPresent,
    "next-pair": nextPair, "prev-pair": prevPair,
    "move-up": () => moveChain(i, -1), "move-down": () => moveChain(i, 1),
    "delete-chain": () => deleteChain(i), "focus-chain": () => focusChain(i),
    "renumber-height": renumberByHeight,
    "yaw-left": () => rotateBy(-Math.PI / 12), "yaw-right": () => rotateBy(Math.PI / 12),
    "zoom-in": () => zoomBy(1.18), "zoom-out": () => zoomBy(1 / 1.18), "reset-view": resetView,
  }[a] || (() => {}))();
}
function ensureRenderer() {
  if (state.renderer) return;
  state.renderer = new THREE.WebGLRenderer({ canvas: state.canvas, antialias: true });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
}
function ensurePanel() {
  let panel = document.getElementById("link-panel");
  if (!panel) {
    panel = document.createElement("div"); panel.id = "link-panel";
    panel.innerHTML = `
      <div id="link-toolbar">
        <div class="lk-title"><b>LINK</b><select id="link-plant" class="lk-plant" title="plant"></select><span id="link-meta"></span></div>
        <div class="lk-actions">
          <button data-link-action="reload">Reload</button>
          <button data-link-action="fresh" title="Automatic linking loads after the plant" disabled>⚡ Auto-link (CPD)</button>
          <button data-link-action="yaw-left">⟲</button>
          <button data-link-action="yaw-right">⟳</button>
          <button data-link-action="zoom-out">–</button>
          <button data-link-action="zoom-in">+</button>
          <button data-link-action="reset-view">Reset view</button>
          <button data-link-action="renumber-height">↕ Renumber by height</button>
          <label class="lk-width">Chain panel <input id="link-sidebar-width" type="range" min="280" max="900" step="20" value="340"></label>
          <button class="primary" data-link-action="save">Save</button>
        </div>
      </div>
      <div id="link-body">
        <div id="link-view-wrap">
          <canvas id="link-viewer"></canvas>
          <div id="link-date-labels"></div>
          <div id="link-loading" hidden></div>
        </div>
        <aside id="link-sidebar">
          <div id="link-step"></div>
          <div class="lk-guide-actions">
            <button class="primary" data-link-action="not-present">Not present ▸</button>
            <button data-link-action="prev-leaf">◂ Prev leaf</button>
            <button data-link-action="next-leaf">Next leaf ▸</button>
          </div>
          <div class="lk-pair-actions">
            <button data-link-action="prev-pair">◀ Prev stage</button>
            <button class="primary" data-link-action="next-pair">Next stage ▶</button>
          </div>
          <div class="lk-section-head"><span>Leaves (rank order — click ↑↓ to fix basal→apical)</span></div>
          <div id="link-chain-list"></div>
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
  const width = document.getElementById("link-sidebar-width");
  if (width && !width.dataset.wired) {
    width.dataset.wired = "1";
    width.addEventListener("input", () => panel.style.setProperty("--link-sidebar-width", `${width.value}px`));
  }
  state.canvas.addEventListener("click", selectAtEvent);
  state.canvas.addEventListener("pointerdown", startRotate);
  state.canvas.addEventListener("pointermove", dragRotate);
  state.canvas.addEventListener("pointerup", stopRotate);
  state.canvas.addEventListener("pointercancel", stopRotate);
  state.canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12); }, { passive: false });
  state.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  const lp = document.getElementById("link-plant");
  if (lp && !lp.dataset.wired) { lp.dataset.wired = "1"; lp.addEventListener("change", () => loadData(false)); }
  if (!state.resizeObserver && state.wrap) { state.resizeObserver = new ResizeObserver(resizeCanvas); state.resizeObserver.observe(state.wrap); }
}
function installModeOption() {
  const sel = document.getElementById("mode-select"); if (!sel) return null;
  if (!Array.from(sel.options).some((o) => o.value === MODE)) {
    const o = document.createElement("option"); o.value = MODE; o.textContent = "Link leaves (cross-date)"; sel.appendChild(o);
  }
  return sel;
}
function togglePanel() {
  const sel = document.getElementById("mode-select"); if (!state.panel || !sel) return;
  if (sel.value === MODE) {
    state.panel.classList.add("on");
    const token = state.loadToken;
    populatePlants(token).then((loaded) => {
      if (!loaded || token !== state.loadToken) return;
      if (currentPlant() !== state.loadedPlant || !state.views.length) loadData(false); else resizeCanvas();
    });
  } else state.panel.classList.remove("on");
}
function setup() {
  ensurePanel(); renderUi();
  const m = installModeOption(); if (m) m.addEventListener("change", togglePanel);
  const ps = document.getElementById("plant-select");
  if (ps) ps.addEventListener("change", () => { if (state.panel?.classList.contains("on")) loadData(false); });
  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("pointsam:dataset-changed", () => {
    state.loadedPlant = null; state.loadedContext = null; state.loadToken += 1;
    const token = state.loadToken;
    if (state.panel?.classList.contains("on")) {
      populatePlants(token).then((loaded) => { if (loaded && token === state.loadToken) loadData(false); });
    }
  });
  window.addEventListener("keydown", (e) => {
    if (!state.panel?.classList.contains("on")) return;
    if (e.key === "n") { markNotPresent(); } else if (e.key === "ArrowRight") { nextLeaf(); } else if (e.key === "ArrowLeft") { prevLeaf(); }
  });
  togglePanel();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", setup);
else setup();
