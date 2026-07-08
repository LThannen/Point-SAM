// Cross-date leaf LINK mode for the FP4D labeller (self-contained; no edits to
// annotate.js). Adds a "Link leaves (cross-date)" option to #mode-select and an
// overlay panel of 6 top-down azimuth plots. Trace the same physical leaf across
// dates by clicking, assign basal->apical rank, save -> manual_chains.json
// (consumed by relink_leaf_identity.py). Azimuth continuity is the identity cue.

const MODE = "link";
const state = { plant: "01", dates: [], leaves: {}, chains: [], current: {}, scale: 1, has_manual: false };

function rankColor(r) {                       // distinct hue per rank
  if (!r) return "#9aa0a6";
  const h = (r * 137.508) % 360;
  return `hsl(${h.toFixed(0)} 70% 45%)`;
}

function injectCss() {
  if (document.getElementById("link-css")) return;
  const s = document.createElement("style");
  s.id = "link-css";
  s.textContent = `
  #link-panel{position:absolute;inset:0;z-index:40;background:#12161c;color:#e6ebf0;
    display:none;flex-direction:column;font:13px/1.4 system-ui,sans-serif;overflow:auto}
  #link-panel.on{display:flex}
  #link-bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 14px;
    background:#0d1116;border-bottom:1px solid #263042;position:sticky;top:0}
  #link-bar b{color:#7fd1c0} #link-bar .cur{font-family:ui-monospace,monospace;color:#ffd479}
  #link-bar button{background:#1d2735;color:#e6ebf0;border:1px solid #34435c;border-radius:6px;
    padding:5px 10px;cursor:pointer} #link-bar button:hover{border-color:#7fd1c0}
  #link-bar button.pri{background:#1f6f5c;border-color:#2c9a80}
  #link-grid{display:flex;gap:10px;padding:12px 14px;flex-wrap:nowrap;overflow-x:auto}
  .lk-date{flex:0 0 auto;text-align:center}
  .lk-date h4{margin:0 0 4px;font-size:12px;color:#9fb0c6;font-weight:600}
  .lk-svg{background:#0d1116;border:1px solid #263042;border-radius:8px;touch-action:none}
  .lk-leaf{cursor:pointer}
  .lk-leaf:hover circle{stroke:#fff;stroke-width:2}
  #link-chains{padding:6px 14px 20px;display:flex;flex-wrap:wrap;gap:6px}
  .lk-chip{background:#0d1116;border:1px solid #263042;border-radius:6px;padding:3px 8px;
    font-family:ui-monospace,monospace;font-size:11px}
  .lk-chip .rk{font-weight:700;margin-right:5px}
  #link-note{padding:0 14px 10px;color:#9fb0c6;max-width:70ch}`;
  document.head.appendChild(s);
}

function currentPlant() {
  const sel = document.getElementById("plant-select");
  const v = sel && sel.value ? sel.value : "";
  return (v || "1").toString().padStart(2, "0");
}

function leafRank() {
  const m = {};
  state.chains.forEach((ch, i) => { for (const d in ch.obs) m[`${d}:${ch.obs[d]}`] = i + 1; });
  return m;
}

async function loadData() {
  state.plant = currentPlant();
  const [lv, ch] = await Promise.all([
    fetch(`/link/leaves?plant=${state.plant}`).then((r) => r.json()),
    fetch(`/link/chains?plant=${state.plant}`).then((r) => r.json()),
  ]);
  if (lv.error) { setLinkStatus(`error: ${lv.error}`); return; }
  state.dates = lv.dates;
  state.leaves = lv.leaves;
  state.chains = (ch.chains || []).map((c) => ({ rank: c.rank, obs: { ...c.obs } }));
  state.has_manual = !!ch.has_manual;
  state.current = {};
  // shared scale across all dates
  let m = 1;
  for (const d of state.dates) for (const L of state.leaves[d]) m = Math.max(m, Math.abs(L.cx), Math.abs(L.cy));
  state.scale = m * 1.15;
  render();
}

function setLinkStatus(t) {
  const n = document.getElementById("link-note");
  if (n) n.textContent = t;
}

const W = 190, H = 250, PAD = 16;
function px(v) { return PAD + ((v + state.scale) / (2 * state.scale)) * (W - 2 * PAD); }
function py(v) { return H - PAD - ((v + state.scale) / (2 * state.scale)) * (H - 2 * PAD); }

function svgForDate(d, lr) {
  const cx0 = px(0), cy0 = py(0);
  let s = `<svg class="lk-svg" width="${W}" height="${H}" data-date="${d}">`;
  s += `<line x1="${cx0}" y1="0" x2="${cx0}" y2="${H}" stroke="#1c2634"/>`;
  s += `<line x1="0" y1="${cy0}" x2="${W}" y2="${cy0}" stroke="#1c2634"/>`;
  for (const L of state.leaves[d]) {
    const key = `${d}:${L.leafid}`;
    const inCur = state.current[d] === L.leafid;
    const r = lr[key];
    const col = inCur ? "#111" : rankColor(r);
    const x = px(L.cx), y = py(L.cy);
    s += `<g class="lk-leaf" data-date="${d}" data-leafid="${L.leafid}">`;
    s += `<line x1="${cx0}" y1="${cy0}" x2="${x}" y2="${y}" stroke="${col}" stroke-width="1" opacity="0.6"/>`;
    s += `<circle cx="${x}" cy="${y}" r="9" fill="${col}" stroke="${inCur ? '#ffd479' : '#0d1116'}" stroke-width="${inCur ? 2.5 : 1}"/>`;
    s += `<text x="${x}" y="${y + 3}" text-anchor="middle" font-size="9" fill="#fff" font-weight="700">${L.leafid}</text>`;
    s += `</g>`;
  }
  s += `</svg>`;
  return s;
}

function render() {
  const panel = document.getElementById("link-panel");
  if (!panel) return;
  const lr = leafRank();
  const cur = state.dates.filter((d) => d in state.current).map((d) => `${d.slice(-4)}:L${state.current[d]}`).join(" ");
  const grid = state.dates.map((d) => `<div class="lk-date"><h4>${d}</h4>${svgForDate(d, lr)}</div>`).join("");
  const chips = state.chains.map((ch, i) => {
    const obs = state.dates.filter((d) => d in ch.obs).map((d) => `${d.slice(-2)}:${ch.obs[d]}`).join(" ");
    return `<span class="lk-chip" style="border-color:${rankColor(i + 1)}"><span class="rk" style="color:${rankColor(i + 1)}">r${i + 1}</span>${obs}</span>`;
  }).join("");
  panel.innerHTML = `
    <div id="link-bar">
      <b>LINK</b> plant ${state.plant} · ${state.chains.length} chains ${state.has_manual ? "· (manual saved)" : "· (auto seed)"}
      &nbsp; building r${state.chains.length + 1}: <span class="cur">[${cur || "click a leaf"}]</span>
      <button id="lk-commit" class="pri">Commit rank ${state.chains.length + 1}</button>
      <button id="lk-undo">Undo pick</button>
      <button id="lk-discard">Discard</button>
      <button id="lk-dellast">Delete last chain</button>
      <button id="lk-reseed">Re-seed auto</button>
      <button id="lk-reload">Reload plant</button>
      <button id="lk-save" class="pri">Save</button>
    </div>
    <div id="link-grid">${grid}</div>
    <div id="link-chains">${chips}</div>
    <div id="link-note">Click the same physical leaf in each date it appears (trace by azimuth = the arrow direction), then Commit. Commit the most BASAL leaf first (rank 1), then upward. Save writes manual_chains.json next to the hand-labels; relink picks it up automatically.</div>`;

  panel.querySelectorAll(".lk-leaf").forEach((g) => g.addEventListener("click", () => {
    state.current[g.dataset.date] = parseInt(g.dataset.leafid, 10);
    render();
  }));
  const on = (id, fn) => { const b = document.getElementById(id); if (b) b.addEventListener("click", fn); };
  on("lk-commit", () => { if (Object.keys(state.current).length) { state.chains.push({ rank: state.chains.length + 1, obs: { ...state.current } }); state.current = {}; render(); } });
  on("lk-undo", () => { const ks = Object.keys(state.current); if (ks.length) { delete state.current[ks[ks.length - 1]]; render(); } });
  on("lk-discard", () => { state.current = {}; render(); });
  on("lk-dellast", () => { state.chains.pop(); render(); });
  on("lk-reseed", async () => {
    const ch = await fetch(`/link/chains?plant=${state.plant}&fresh=1`).then((x) => x.json());
    state.chains = (ch.chains || []).map((c) => ({ rank: c.rank, obs: { ...c.obs } }));
    state.current = {};
    render();
  });
  on("lk-reload", loadData);
  on("lk-save", saveChains);
}

async function saveChains() {
  const r = await fetch("/link/save", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plant: state.plant, chains: state.chains }),
  }).then((x) => x.json());
  setLinkStatus(r.status === "saved" ? `saved ${r.n} chains -> ${r.path}` : `save failed: ${JSON.stringify(r)}`);
  state.has_manual = true;
}

function ensurePanel() {
  if (document.getElementById("link-panel")) return;
  const p = document.createElement("div");
  p.id = "link-panel";
  (document.getElementById("preview") || document.body).appendChild(p);
}

function setup() {
  injectCss();
  ensurePanel();
  const sel = document.getElementById("mode-select");
  if (sel && !Array.from(sel.options).some((o) => o.value === MODE)) {
    const o = document.createElement("option");
    o.value = MODE; o.textContent = "Link leaves (cross-date)";
    sel.appendChild(o);
  }
  const toggle = () => {
    const panel = document.getElementById("link-panel");
    if (!panel) return;
    if (sel && sel.value === MODE) { panel.classList.add("on"); loadData(); }
    else panel.classList.remove("on");
  };
  if (sel) sel.addEventListener("change", toggle);
  toggle();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", setup);
else setup();
