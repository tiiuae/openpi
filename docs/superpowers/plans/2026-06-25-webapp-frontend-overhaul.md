# Webapp Frontend — UX Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the eval webapp UI in the dataset-wizard style: separate Live/Replay pages, wizard theme, grouped/named charts, terminal-parity logs, reorganized + renamed config with help tooltips, a feedback form, a folder browser, and Sleep/Home buttons.

**Architecture:** Two static HTML pages (`/` Live, `/replay` Replay) served by the existing FastAPI app, sharing a `theme.css` (wizard CSS variables) and ES modules under `static/js/`. The current 159-line monolith `app.js` is split into focused modules. The frontend talks to the backend endpoints added by the companion plan `2026-06-25-webapp-backend-safety-motion.md` (`/api/files`, `/api/feedback`, WS actions `go_sleep`/`go_home`, new `status` kinds `connecting`/`connect_failed`/`firmware_error`).

**Tech Stack:** Vanilla ES modules, Chart.js 4 (+ zoom plugin via CDN), CSS, FastAPI static serving.

**Prerequisite:** The backend plan should be implemented first (this plan calls its endpoints). The reference implementation to port from is `get_inspired_from_web_wizard/viewer/` (`wizard.css`, `wizard.js`, `js/graphs.js`).

Verification is visual + manual (browser). Off-robot you can serve the UI with
the policy server down to exercise the connect-failed and stop paths:

```bash
cd examples/trossen_ai && python -m uvicorn webapp.server:app --reload --port 8001
```

---

## File Structure

- Create: `webapp/static/theme.css` — wizard CSS variables + shared components (cards, buttons, modal, log panel, tooltips).
- Create: `webapp/static/replay.html` — Dataset Replay page.
- Modify: `webapp/static/index.html` — Live Control page (restructured, themed).
- Delete: `webapp/static/styles.css`, `webapp/static/app.js` (replaced by theme.css + modules).
- Create: `webapp/static/js/api.js` — `api`, `apiPost` fetch helpers.
- Create: `webapp/static/js/ws.js` — telemetry WebSocket connect/send/reconnect + dispatch.
- Create: `webapp/static/js/charts.js` — wizard-style chart factory + live ring-buffer charts.
- Create: `webapp/static/js/config.js` — config groups, `readConfig`/`applyConfig`, presets, `FIELD_HELP` tooltips, EE-conditional visibility.
- Create: `webapp/static/js/filebrowser.js` — modal folder browser (ports wizard).
- Create: `webapp/static/js/logs.js` — terminal-parity log panel.
- Create: `webapp/static/js/controls.js` — start/stop/estop/sleep/home + session badge + status.
- Create: `webapp/static/js/feedback.js` — feedback form submit.
- Create: `webapp/static/js/live.js` — Live page wiring.
- Create: `webapp/static/js/replay.js` — Replay page wiring.
- Modify: `webapp/server.py` — add `GET /replay` route.
- Modify: `webapp/tests/test_server.py` — assert `/replay` serves HTML.

Each module has one responsibility and is imported by `live.js`/`replay.js`. No
bundler — pages load modules with `<script type="module">`.

---

## Task 1: Theme + `/replay` route + page shells

**Files:**
- Create: `webapp/static/theme.css`
- Modify: `webapp/static/index.html`, `webapp/server.py`, `webapp/tests/test_server.py`
- Create: `webapp/static/replay.html`

- [ ] **Step 1: Add a failing test for the `/replay` route**

Append to `webapp/tests/test_server.py`:

```python
def test_replay_page_served():
    client = TestClient(create_app())
    r = client.get("/replay")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
```

- [ ] **Step 2: Run it, verify failure**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_server.py::test_replay_page_served -q`
Expected: FAIL (404).

- [ ] **Step 3: Create a minimal `replay.html` stub + add the route**

Create `webapp/static/replay.html`:

```html
<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Replay</title></head><body></body></html>
```

In `webapp/server.py`, next to the `index()` route:

```python
    @app.get("/replay")
    def replay():
        return FileResponse(STATIC_DIR / "replay.html")
```

- [ ] **Step 4: Run it, verify pass**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_server.py::test_replay_page_served -q`
Expected: PASS.

- [ ] **Step 5: Create `theme.css` (port from `wizard.css`)**

Create `webapp/static/theme.css`. Copy the `:root` variables, base typography,
`.card`, form-field, button (`.btn-primary`/`.btn-outline`/`.btn-danger`/`.btn-icon`/`.btn-sm`/`.btn-close`),
modal (`.modal-overlay`/`.modal-box`/`.modal-titlebar`/`.browser-*`/`.ds-badge`),
`.field-inline`/`.field-row`/`.checkbox-label`/`.input-browse`, `.run-controls`/`.run-status`,
and scrollbar rules verbatim from `get_inspired_from_web_wizard/viewer/wizard.css`.
Then append app-specific blocks:

```css
/* ── App header / nav ─────────────────────────────────────────── */
.app-header { display:flex; align-items:center; justify-content:space-between;
  padding:0 20px; height:46px; background:var(--bg-card);
  border-bottom:1px solid var(--border); position:sticky; top:0; z-index:50; }
.app-nav a { color:var(--muted); text-decoration:none; margin-right:14px; font-size:13px; }
.app-nav a.active { color:var(--text); }
.header-actions { display:flex; gap:8px; align-items:center; }

/* ── Live page layout: config column + stacked cameras→charts→logs ─ */
.live-grid { display:grid; grid-template-columns:300px 1fr; gap:14px; padding:16px; }
.live-main { display:flex; flex-direction:column; gap:14px; }
@media (max-width:900px){ .live-grid{ grid-template-columns:1fr; } }

/* ── Cameras strip ────────────────────────────────────────────── */
.cameras-strip { display:flex; gap:8px; flex-wrap:wrap; }
.cameras-strip img { max-height:200px; border-radius:4px; border:1px solid var(--border); }

/* ── Log panel (terminal parity) ──────────────────────────────── */
.log-header { display:flex; justify-content:space-between; align-items:center; }
.log-panel { background:var(--bg); border:1px solid var(--border); border-radius:4px;
  padding:10px 12px; font:11px var(--font); line-height:1.6; height:280px;
  overflow:auto; white-space:pre-wrap; word-break:break-all; }
.log-INFO{ color:var(--text); } .log-WARNING{ color:var(--warn); }
.log-ERROR{ color:var(--err); } .log-DEBUG{ color:var(--muted); }

/* ── Help tooltip (ⓘ) ─────────────────────────────────────────── */
.help-ic { display:inline-block; width:14px; height:14px; line-height:14px;
  text-align:center; border-radius:50%; background:var(--bg-hover);
  color:var(--muted); font-size:10px; cursor:help; margin-left:5px; position:relative; }
.help-ic:hover::after { content:attr(data-help); position:absolute; left:18px; top:-4px;
  width:230px; background:var(--bg-card); border:1px solid var(--border);
  border-radius:4px; padding:7px 9px; color:var(--text); font-size:11px;
  line-height:1.4; z-index:100; white-space:normal; }

/* ── Charts grid ──────────────────────────────────────────────── */
.charts-grid { display:flex; flex-direction:column; gap:10px; }
.chart-wrap { background:var(--bg); border:1px solid var(--border); border-radius:4px; padding:8px; }
.chart-title { font-size:11px; color:var(--muted); margin-bottom:5px; }
.chart-cj-container { position:relative; height:220px; }
.section-label { font-size:10px; font-weight:600; color:var(--muted);
  text-transform:uppercase; letter-spacing:.8px; margin:10px 0 4px; }
#metrics-box { font:12px var(--font); white-space:pre-wrap; color:var(--muted); }
```

- [ ] **Step 6: Rebuild `index.html` shell (Live page)**

Replace `webapp/static/index.html` with:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trossen Control — Live</title>
  <link rel="stylesheet" href="/static/theme.css">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
</head>
<body>
  <header class="app-header">
    <div class="app-nav">
      <strong style="color:var(--text);margin-right:16px">Trossen Control</strong>
      <a href="/" class="active">Live</a><a href="/replay">Replay</a>
    </div>
    <div class="header-actions">
      <span class="badge" id="badge-rtt">rtt</span>
      <span class="badge" id="badge-session">idle</span>
      <button class="btn-outline" id="btn-home">Home</button>
      <button class="btn-outline" id="btn-sleep">Sleep</button>
    </div>
  </header>

  <div class="live-grid">
    <aside id="config-col"><!-- config cards injected by config.js --></aside>
    <main class="live-main">
      <section class="card"><h2>Cameras</h2><div class="cameras-strip" id="images-box"></div></section>
      <section class="card"><h2>Actions</h2><div class="charts-grid" id="charts-box"></div>
        <div class="section-label">Metrics</div><div id="metrics-box"></div></section>
      <section class="card">
        <div class="log-header"><h2>Logs</h2><button class="btn-sm" id="btn-clear-log">Clear</button></div>
        <div class="log-panel" id="log-box"></div>
      </section>
      <section class="card" id="feedback-card"><!-- feedback form injected by feedback.js --></section>
    </main>
  </div>

  <script type="module" src="/static/js/live.js"></script>
</body>
</html>
```

- [ ] **Step 7: Run server tests + eyeball the page**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_server.py -q` → PASS.
Serve and open `http://localhost:8001/` — header, nav, themed cards render (the
console will error on the missing `live.js` until Task 8 — expected).

- [ ] **Step 8: Commit**

```bash
git add webapp/static/theme.css webapp/static/index.html webapp/static/replay.html \
        webapp/server.py webapp/tests/test_server.py
git commit -m "feat(webapp-ui): wizard theme, app header/nav, /replay route, page shells"
```

---

## Task 2: Core modules — `api.js`, `ws.js`

**Files:**
- Create: `webapp/static/js/api.js`, `webapp/static/js/ws.js`

- [ ] **Step 1: Create `api.js`** (ported from wizard `wizard.js` helpers)

```javascript
export async function api(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
  return res.json();
}
export async function apiPost(url, body) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
  return res.json();
}
```

- [ ] **Step 2: Create `ws.js`** (telemetry socket + dispatch registry)

```javascript
// Telemetry WebSocket: reconnecting, with a handler registry keyed by event type.
let ws = null;
const handlers = {};            // type -> fn(evt); "__close__" runs on socket close.
let onOpenCb = () => {};

export function onMessage(type, fn) { handlers[type] = fn; }
export function onOpen(fn) { onOpenCb = fn; }

export function connect() {
  ws = new WebSocket(`ws://${location.host}/ws/telemetry`);
  ws.onopen = () => onOpenCb();
  ws.onmessage = (e) => {
    const evt = JSON.parse(e.data);
    (handlers[evt.type] || (() => {}))(evt);
  };
  ws.onclose = () => { (handlers.__close__ || (() => {}))(); setTimeout(connect, 1000); };
}

export function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
```

- [ ] **Step 3: Syntax sanity (browser console)**

Serve, open `/`, in DevTools: `import('/static/js/ws.js').then(m=>console.log(Object.keys(m)))`
Expected: `["onMessage","onOpen","connect","send"]`, no parse error.

- [ ] **Step 4: Commit**

```bash
git add webapp/static/js/api.js webapp/static/js/ws.js
git commit -m "feat(webapp-ui): api + reconnecting telemetry websocket modules"
```

---

## Task 3: Config module — groups, rename, help tooltips, presets

**Files:**
- Create: `webapp/static/js/config.js`

- [ ] **Step 1: Create `config.js`** — single `GROUPS` source of truth. Each field
declares the backend `key` (unchanged), the approved `label`, `type`, optional
`options`, `def`, and `help`. `renderConfig` builds cards; `readConfig`/`applyConfig`
map to/from backend keys; EE-only group toggles on `adapter`.

```javascript
import { api, apiPost } from "./api.js";

const GROUPS = [
  { title: "Connection", fields: [
    { key:"policy_host", label:"Policy server host", type:"text", def:"192.168.50.174",
      help:"IP/hostname of the OpenPI policy server." },
    { key:"policy_port", label:"Policy server port", type:"number", def:8800,
      help:"TCP port the policy server listens on." },
    { key:"control_freq", label:"Control rate (Hz)", type:"number", def:25,
      help:"Control steps per second the loop targets." },
    { key:"max_steps", label:"Max steps", type:"number", def:1000,
      help:"Episode ends after this many control steps." },
    { key:"connect_timeout", label:"Connect timeout (s)", type:"number", def:15,
      help:"Give up waiting for the policy server after this many seconds." },
  ]},
  { title: "Policy", fields: [
    { key:"adapter", label:"Action space", type:"select",
      options:[["joint","Joint"],["ee","End-effector"]], def:"joint",
      help:"Joint = raw 14-D joints. End-effector = EE poses solved to joints via IK." },
    { key:"task_prompt", label:"Task prompt", type:"text", def:"move the arm to the left",
      help:"Natural-language instruction sent to the policy." },
    { key:"starvla", label:"StarVLA input mode (224×224 RGB)", type:"checkbox", def:false,
      help:"Preprocess camera images to 224×224 RGB (PIL) for StarVLA-family policies. Off uses the default training resize." },
  ]},
  { title: "Action smoothing", fields: [
    { key:"ensemble_type", label:"Action smoothing", type:"select",
      options:[["exp","Exponential"],["cogact","CogACT"],["none","None"]], def:"exp",
      help:"Blends overlapping predicted action chunks. Exponential = recency-weighted; CogACT = learned weighting; None = latest only." },
    { key:"cogact_mode", label:"CogACT blend mode", type:"select",
      options:[["cogact","cogact"],["latest","latest"],["hybrid","hybrid"]], def:"cogact",
      help:"Weighting variant used only when smoothing = CogACT." },
    { key:"rate_of_inference", label:"Inference interval (steps)", type:"number", def:20,
      help:"Run a new policy inference every N control steps." },
    { key:"async_inference", label:"Async inference", type:"checkbox", def:false,
      help:"Run inference in a background thread (requires a smoothing method other than None)." },
  ]},
  { title: "Arms", fields: [
    { key:"use_left_arm_only", label:"Left arm only", type:"checkbox", def:false,
      help:"Drive only the left arm; right arm holds." },
    { key:"use_right_arm_only", label:"Right arm only", type:"checkbox", def:false,
      help:"Drive only the right arm; left arm holds." },
  ]},
  { title: "IK (End-effector only)", eeOnly:true, fields: [
    { key:"ik_orientation_weight", label:"IK orientation weight", type:"number", step:0.01, def:0.01,
      help:"How strongly IK matches target orientation vs position." },
    { key:"ik_pos_tol_m", label:"IK position tolerance (m)", type:"number", step:0.001, def:0.001,
      help:"Acceptable IK position error in metres." },
  ]},
  { title: "Motion tuning (Advanced)", fields: [
    { key:"smooth_streaming", label:"Smooth streaming (feed-forward velocity)", type:"checkbox", def:false,
      help:"Send feed-forward joint velocities so the arm carries momentum through waypoints — fixes jerky/stop-start motion." },
    { key:"min_time_to_move_multiplier", label:"Goal-time multiplier", type:"number", step:0.5, def:3.0,
      help:"Driver goal_time = multiplier / loop rate. Larger = smoother but laggier; smaller = snappier but jerkier." },
    { key:"loop_rate", label:"Driver loop rate (Hz)", type:"number", def:25,
      help:"Match this to Control rate to avoid mid-motion re-planning." },
  ]},
];

const $ = (id) => document.getElementById(id);

export function renderConfig(container) {
  container.innerHTML =
    GROUPS.map(g => `<section class="card" data-group="${g.title}" ${g.eeOnly?'data-ee-only="1"':''}>
      <h2>${g.title}</h2>${g.fields.map(fieldHtml).join("")}</section>`).join("")
    + presetsHtml() + modeHtml();
  $("cf_adapter")?.addEventListener("change", updateEEVisibility);
  updateEEVisibility();
  wirePresets();
}

function fieldHtml(f) {
  const help = `<span class="help-ic" data-help="${f.help.replace(/"/g,'&quot;')}">i</span>`;
  const id = `cf_${f.key}`;
  if (f.type === "checkbox")
    return `<label class="checkbox-label"><input type="checkbox" name="${f.key}" id="${id}">${f.label}${help}</label>`;
  if (f.type === "select")
    return `<div class="field-inline"><label>${f.label}${help}</label>
      <select name="${f.key}" id="${id}">${f.options.map(([v,t])=>`<option value="${v}">${t}</option>`).join("")}</select></div>`;
  return `<div class="field-inline"><label>${f.label}${help}</label>
    <input type="${f.type}" name="${f.key}" id="${id}" ${f.step?`step="${f.step}"`:""}></div>`;
}

function modeHtml() {
  return `<section class="card"><h2>Run</h2>
    <div class="field-inline"><label>Mode</label>
      <select id="mode-select"><option value="test">test (no movement)</option><option value="autonomous">autonomous</option></select></div>
    <div class="run-controls">
      <button class="btn-primary" id="btn-start">Start Live</button>
      <button class="btn-outline" id="btn-stop">Stop</button>
      <button class="btn-danger" id="btn-estop">E-STOP</button>
    </div></section>`;
}

function presetsHtml() {
  return `<section class="card"><h2>Presets</h2>
    <div class="field-inline"><select id="preset-select"></select><button class="btn-sm" id="preset-load">Load</button></div>
    <div class="field-inline"><input id="preset-name" placeholder="preset name"><button class="btn-sm" id="preset-save">Save</button><button class="btn-sm" id="preset-delete">Delete</button></div></section>`;
}

function defaults() { const d={}; GROUPS.forEach(g=>g.fields.forEach(f=>d[f.key]=f.def)); return d; }

export function readConfig() {
  const cfg = {};
  GROUPS.forEach(g => g.fields.forEach(f => {
    const el = $(`cf_${f.key}`); if (!el) return;
    cfg[f.key] = f.type === "checkbox" ? el.checked : el.value;
  }));
  cfg.mode = $("mode-select")?.value || "test";
  return cfg;
}

export function applyConfig(cfg) {
  GROUPS.forEach(g => g.fields.forEach(f => {
    const el = $(`cf_${f.key}`); if (!el || !(f.key in cfg)) return;
    if (f.type === "checkbox") el.checked = !!cfg[f.key]; else el.value = cfg[f.key];
  }));
  if (cfg.mode && $("mode-select")) $("mode-select").value = cfg.mode;
  updateEEVisibility();
}

function updateEEVisibility() {
  const ee = $("cf_adapter")?.value === "ee";
  document.querySelectorAll('[data-ee-only="1"]').forEach(s => s.style.display = ee ? "" : "none");
}

export function setDefaults() { applyConfig(defaults()); }

async function refreshPresets() {
  const names = await api("/api/presets");
  const sel = $("preset-select"); if (!sel) return;
  sel.innerHTML = names.map(n => `<option value="${n}">${n}</option>`).join("");
}
function wirePresets() {
  $("preset-save").onclick = async () => {
    const name = $("preset-name").value.trim(); if (!name) return;
    await apiPost("/api/presets", { name, config: readConfig() }); refreshPresets();
  };
  $("preset-load").onclick = async () => {
    const name = $("preset-select").value; if (!name) return;
    applyConfig(await api("/api/presets/" + name));
  };
  $("preset-delete").onclick = async () => {
    const name = $("preset-select").value; if (!name) return;
    await fetch("/api/presets/" + name, { method: "DELETE" }); refreshPresets();
  };
  refreshPresets();
}
```

- [ ] **Step 2: Commit**

```bash
git add webapp/static/js/config.js
git commit -m "feat(webapp-ui): grouped+renamed config with help tooltips and presets"
```

---

## Task 4: Charts module — wizard style + live streaming

**Files:**
- Create: `webapp/static/js/charts.js`

- [ ] **Step 1: Create `charts.js`** porting `makeWizardChart` styling + `PALETTE`
from `get_inspired_from_web_wizard/viewer/wizard.js`, plus a streaming `LiveChart`
with a bounded ring buffer.

```javascript
// Wizard-style charts. makeSeriesChart: full-series. LiveChart: streaming (bounded).
export const PALETTE = [
  '#4a9eff','#ff5555','#50fa7b','#ffb86c','#bd93f9','#ff79c6','#8be9fd','#f1fa8c',
  '#ff6e6e','#5af78e','#caa9fa','#ffca6a','#1dc9a4','#ff92d0','#6272a4','#44bc9f',
];
const FONT = { size: 9, family: 'Consolas,Menlo,Monaco,monospace' };
const TICK = { color: '#484f58', font: FONT }, GRID = { color: '#21262d' };

function baseOptions() {
  return { animation:false, maintainAspectRatio:false, responsive:true, parsing:false,
    interaction:{ mode:'index', intersect:false },
    plugins:{ legend:{ labels:{ color:'#8b949e', boxWidth:10, font:FONT }, position:'bottom' },
      tooltip:{ backgroundColor:'#161b22', borderColor:'#30363d', borderWidth:1,
        titleColor:'#8b949e', bodyColor:'#e6edf3', bodyFont:FONT },
      zoom:{ pan:{ enabled:true, mode:'x' }, zoom:{ wheel:{ enabled:true }, mode:'x' } } },
    scales:{ x:{ type:'linear', ticks:TICK, grid:GRID }, y:{ ticks:TICK, grid:GRID } } };
}

// Full-series chart (series: [{label, points:[{x,y}]}]).
export function makeSeriesChart(canvas, series) {
  return new Chart(canvas, {
    type:'line',
    data:{ datasets: series.map((s,i)=>({ label:s.label, data:s.points,
      borderColor: PALETTE[i%PALETTE.length], borderWidth:1.5, pointRadius:0, tension:0 })) },
    options: baseOptions(),
  });
}

// Streaming chart: push(seriesIdx, x, y), bounded to maxPts.
export class LiveChart {
  constructor(canvas, labels, maxPts = 300) {
    this.maxPts = maxPts;
    this.chart = new Chart(canvas, {
      type:'line',
      data:{ datasets: labels.map((l,i)=>({ label:l, data:[],
        borderColor: PALETTE[i%PALETTE.length], borderWidth:1, pointRadius:0, tension:0 })) },
      options: baseOptions(),
    });
  }
  push(idx, x, y) {
    const d = this.chart.data.datasets[idx].data;
    d.push({ x, y }); if (d.length > this.maxPts) d.shift();
  }
  update() { this.chart.update('none'); }
}

// Split 14-D joint names into Left/Right groups for grouped charts.
export function splitArms(names) {
  const left = [], right = [];
  names.forEach((n,i)=> (n.startsWith('left_') ? left : right).push({ i, name:n.replace(/^(left_|right_)/,'') }));
  return { left, right };
}
```

- [ ] **Step 2: Commit**

```bash
git add webapp/static/js/charts.js
git commit -m "feat(webapp-ui): wizard-style chart factory + live streaming charts"
```

---

## Task 5: Folder browser module

**Files:**
- Create: `webapp/static/js/filebrowser.js`

- [ ] **Step 1: Create `filebrowser.js`** porting the wizard browser (calls `/api/files`).

```javascript
import { api } from "./api.js";

let target = null, curPath = null;
const $ = (id) => document.getElementById(id);

export function setupFileBrowser() {
  $("btn-cancel-browser")?.addEventListener("click", close);
  $("btn-select-dir")?.addEventListener("click", selectCurrent);
  $("modal-filebrowser")?.addEventListener("click", e => { if (e.target === e.currentTarget) close(); });
}
export function openFileBrowser(inputEl) {
  target = inputEl; $("modal-filebrowser").style.display = "";
  browse(inputEl.value.trim() || "~");
}
function close() { $("modal-filebrowser").style.display = "none"; target = null; }
function selectCurrent() {
  if (target && curPath) { target.value = curPath; target.dispatchEvent(new Event("change")); }
  close();
}
async function browse(path) {
  const box = $("browser-entries"); box.innerHTML = '<div class="browser-msg">Loading…</div>';
  try {
    const r = await api(`/api/files?path=${encodeURIComponent(path)}`);
    if (r.error) { box.innerHTML = `<div class="browser-msg err">${r.error}</div>`; return; }
    curPath = r.path; $("browser-current-path").textContent = r.path;
    let html = "";
    if (r.parent) html += `<div class="browser-item browser-up" data-path="${r.parent}"><span class="bi-icon">↑</span><span class="bi-name">..</span></div>`;
    for (const e of r.entries) {
      const badge = e.is_dataset ? '<span class="ds-badge">dataset</span>' : '';
      html += `<div class="browser-item${e.is_dataset?' is-dataset':''}" data-path="${e.path}"><span class="bi-icon">📁</span><span class="bi-name">${e.name}</span>${badge}</div>`;
    }
    box.innerHTML = html || '<div class="browser-msg">Empty directory</div>';
    box.querySelectorAll(".browser-item").forEach(it => it.addEventListener("click", () => browse(it.dataset.path)));
  } catch (e) { box.innerHTML = `<div class="browser-msg err">Error: ${e.message}</div>`; }
}
```

- [ ] **Step 2: Commit**

```bash
git add webapp/static/js/filebrowser.js
git commit -m "feat(webapp-ui): modal folder browser module"
```

---

## Task 6: Shared `logs.js`, `controls.js`, `feedback.js`

**Files:**
- Create: `webapp/static/js/logs.js`, `webapp/static/js/controls.js`, `webapp/static/js/feedback.js`

- [ ] **Step 1: Create `logs.js`** (terminal-parity log panel).

```javascript
import { onMessage } from "./ws.js";

const $ = (id) => document.getElementById(id);

export function setupLogs() {
  $("btn-clear-log")?.addEventListener("click", () => { $("log-box").innerHTML = ""; });
  onMessage("log", (e) => logLine(e.level, e.msg));
}

export function logLine(level, msg) {
  const box = $("log-box"); if (!box) return;
  const div = document.createElement("div");
  div.className = "log-" + (level || "INFO");
  div.textContent = `[${level}] ${msg}`;
  box.appendChild(div);
  if (box.childElementCount > 1000) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}
```

- [ ] **Step 2: Create `controls.js`** (stop/estop/sleep/home + badges + status).

```javascript
import { send, onMessage } from "./ws.js";
import { logLine } from "./logs.js";

const $ = (id) => document.getElementById(id);
function badge(id, text, cls) { const b = $(id); if (b) { b.textContent = text; b.className = "badge " + (cls || ""); } }

export function setupControls() {
  $("btn-stop")?.addEventListener("click", () => send({ action: "stop" }));
  $("btn-estop")?.addEventListener("click", () => send({ action: "estop" }));
  $("btn-sleep")?.addEventListener("click", () => guarded("go_sleep", "Send arms to SLEEP?"));
  $("btn-home")?.addEventListener("click", () => guarded("go_home", "Send arms to HOME?"));

  onMessage("status", (e) => {
    if (e.kind === "started" || e.kind === "connecting") badge("badge-session", e.kind, "");
    if (e.kind === "stopped") badge("badge-session", "idle", "");
    if (e.kind === "connect_failed") { badge("badge-session", "no server", "bad"); logLine("ERROR", e.payload?.message || "connect failed"); }
    if (e.kind === "firmware_error") { badge("badge-session", "fault → sleep", "bad"); logLine("ERROR", "Firmware fault: " + JSON.stringify(e.payload)); }
    if (e.kind === "error") { badge("badge-session", "error", "bad"); logLine("ERROR", e.payload?.message || "error"); }
  });
  onMessage("inference", (e) => badge("badge-rtt", `${e.rtt_ms.toFixed(0)}ms`, e.rtt_ms < 100 ? "ok" : "bad"));
}

function guarded(action, prompt) {
  const mode = $("mode-select")?.value;
  if (mode === "autonomous" && !confirm(prompt + " (moves the REAL robot)")) return;
  send({ action, config: { mode } });
}
```

- [ ] **Step 3: Create `feedback.js`** (renders the form, POSTs `/api/feedback`).

```javascript
import { apiPost } from "./api.js";

export function renderFeedback(card) {
  if (!card) return;
  card.innerHTML = `<h2>Feedback</h2>
    <div class="field-inline"><label>Name</label><input id="fb-name" type="text"></div>
    <div class="field-inline"><label>Email</label><input id="fb-email" type="text"></div>
    <div class="field-row align-top"><label>Feedback</label><textarea id="fb-text" rows="3" style="width:100%"></textarea></div>
    <div class="run-controls"><button class="btn-primary" id="fb-submit">Submit</button>
      <span class="run-status" id="fb-status"></span></div>`;
  card.querySelector("#fb-submit").addEventListener("click", async () => {
    const s = card.querySelector("#fb-status");
    try {
      await apiPost("/api/feedback", {
        name: card.querySelector("#fb-name").value,
        email: card.querySelector("#fb-email").value,
        feedback: card.querySelector("#fb-text").value,
      });
      s.textContent = "Thanks!"; s.className = "run-status ok";
      card.querySelector("#fb-text").value = "";
    } catch (e) { s.textContent = "Failed: " + e.message; s.className = "run-status err"; }
  });
}
```

- [ ] **Step 4: Commit**

```bash
git add webapp/static/js/logs.js webapp/static/js/controls.js webapp/static/js/feedback.js
git commit -m "feat(webapp-ui): shared logs, controls (sleep/home/status), feedback modules"
```

---

## Task 7: Replay page + `replay.js`

**Files:**
- Modify: `webapp/static/replay.html` (replace the Task 1 stub)
- Create: `webapp/static/js/replay.js`

- [ ] **Step 1: Build `replay.html`** — themed page with header/nav, dataset
folder-browser input, episode select, charts, logs, feedback, and the browser modal.

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trossen Control — Replay</title>
  <link rel="stylesheet" href="/static/theme.css">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
</head>
<body>
  <header class="app-header">
    <div class="app-nav"><strong style="color:var(--text);margin-right:16px">Trossen Control</strong>
      <a href="/">Live</a><a href="/replay" class="active">Replay</a></div>
    <div class="header-actions">
      <span class="badge" id="badge-session">idle</span>
      <button class="btn-outline" id="btn-home">Home</button>
      <button class="btn-outline" id="btn-sleep">Sleep</button>
    </div>
  </header>

  <div class="live-grid">
    <aside id="config-col">
      <section class="card"><h2>Dataset</h2>
        <div class="input-browse"><input id="dataset_dir" type="text" placeholder="dataset directory">
          <button class="btn-icon" id="btn-browse-dataset">📁</button></div>
        <div class="field-inline"><label>Episode</label><select id="episode-select"></select></div>
        <div class="field-inline"><label>Mode</label>
          <select id="mode-select"><option value="test">test (no movement)</option><option value="autonomous">autonomous</option></select></div>
        <div class="run-controls">
          <button class="btn-primary" id="btn-replay">Replay Episode</button>
          <button class="btn-outline" id="btn-stop">Stop</button>
          <button class="btn-danger" id="btn-estop">E-STOP</button></div>
      </section>
    </aside>
    <main class="live-main">
      <section class="card"><h2>Episode actions</h2><div class="charts-grid" id="charts-box"></div></section>
      <section class="card"><div class="log-header"><h2>Logs</h2><button class="btn-sm" id="btn-clear-log">Clear</button></div>
        <div class="log-panel" id="log-box"></div></section>
      <section class="card" id="feedback-card"></section>
    </main>
  </div>

  <div class="modal-overlay" id="modal-filebrowser" style="display:none">
    <div class="modal-box">
      <div class="modal-titlebar">Select dataset folder <button class="btn-close" id="btn-cancel-browser">×</button></div>
      <div class="browser-path-bar"><span class="browser-path-text" id="browser-current-path"></span></div>
      <div class="browser-entries" id="browser-entries"></div>
      <div class="modal-footer"><button class="btn-primary" id="btn-select-dir">Select this folder</button></div>
    </div>
  </div>
  <script type="module" src="/static/js/replay.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `replay.js`** wiring browser → episodes → replay → charts/logs.

```javascript
import { api } from "./api.js";
import { connect, send, onMessage, onOpen } from "./ws.js";
import { setupFileBrowser, openFileBrowser } from "./filebrowser.js";
import { LiveChart } from "./charts.js";
import { setupLogs } from "./logs.js";
import { setupControls } from "./controls.js";
import { renderFeedback } from "./feedback.js";

const $ = (id) => document.getElementById(id);
let chart = null;

document.addEventListener("DOMContentLoaded", () => {
  setupFileBrowser();
  setupLogs();
  setupControls();
  renderFeedback($("feedback-card"));

  $("charts-box").innerHTML =
    `<div class="chart-wrap"><div class="chart-title">Replayed action (joint 0)</div>
      <div class="chart-cj-container"><canvas id="chart-actions"></canvas></div></div>`;
  chart = new LiveChart($("chart-actions"), ["joint 0"]);
  setInterval(() => chart.update(), 200);

  $("btn-browse-dataset").addEventListener("click", () => openFileBrowser($("dataset_dir")));
  $("dataset_dir").addEventListener("change", loadEpisodes);
  $("btn-replay").addEventListener("click", () => {
    const cfg = { dataset_dir: $("dataset_dir").value, episode_index: $("episode-select").value,
                  mode: $("mode-select").value };
    if (cfg.mode === "autonomous" && !confirm("Replay will move the REAL robot. Continue?")) return;
    send({ action: "start_replay", config: cfg });
  });

  onMessage("action", (e) => { if (e.action?.length) chart.push(0, e.step, e.action[0]); });
  onOpen(() => {});
  connect();
});

async function loadEpisodes() {
  const dir = $("dataset_dir").value; if (!dir) return;
  try {
    const info = await api(`/api/episodes?dataset_dir=${encodeURIComponent(dir)}`);
    $("episode-select").innerHTML = Array.from({ length: info.total_episodes }, (_, i) =>
      `<option value="${i}">Episode ${i}</option>`).join("");
  } catch (e) { /* errors surface via the logs panel / network tab */ }
}
```

> Note: `/api/episodes` returns `{fps, total_episodes}`, so the episode picker is
> populated by count. Full per-frame replay preview (like the wizard) would need a
> `/api/frames` endpoint — out of scope; replay streams live actions during playback.

- [ ] **Step 3: Commit**

```bash
git add webapp/static/replay.html webapp/static/js/replay.js
git commit -m "feat(webapp-ui): replay page with folder browser and episode picker"
```

---

## Task 8: Live page wiring (`live.js`) + cleanup

**Files:**
- Create: `webapp/static/js/live.js`
- Delete: `webapp/static/app.js`, `webapp/static/styles.css`

- [ ] **Step 1: Create `live.js`** tying the modules together for the Live page.

```javascript
import { connect, send, onMessage, onOpen } from "./ws.js";
import { renderConfig, readConfig, setDefaults } from "./config.js";
import { LiveChart } from "./charts.js";
import { setupLogs } from "./logs.js";
import { setupControls } from "./controls.js";
import { renderFeedback } from "./feedback.js";

const $ = (id) => document.getElementById(id);
let actionChart = null;
const f = (v) => (v == null ? "—" : Number(v).toFixed(1));

document.addEventListener("DOMContentLoaded", () => {
  renderConfig($("config-col"));
  setDefaults();
  setupLogs();
  setupControls();
  renderFeedback($("feedback-card"));

  $("charts-box").innerHTML =
    `<div class="chart-wrap"><div class="chart-title">Action (joint 0)</div>
      <div class="chart-cj-container"><canvas id="chart-actions"></canvas></div></div>`;
  actionChart = new LiveChart($("chart-actions"), ["joint 0"]);
  setInterval(() => actionChart.update(), 200);

  onMessage("action", (e) => { if (e.action?.length) actionChart.push(0, e.step, e.action[0]); });
  onMessage("images", (e) => {
    const ib = $("images-box"); ib.innerHTML = "";
    for (const [name, b64] of Object.entries(e.images)) {
      const img = new Image(); img.src = "data:image/jpeg;base64," + b64; img.title = name; ib.appendChild(img);
    }
  });
  onMessage("metrics", (e) => {
    $("metrics-box").textContent =
      `RTT last/p50/p95: ${f(e.rtt_last)}/${f(e.rtt_p50)}/${f(e.rtt_p95)} ms  ` +
      `Loop Hz: ${f(e.loop_hz)}  Jitter: ${f(e.jitter)}  Drops: ${e.drops}`;
  });

  $("btn-start").addEventListener("click", () => {
    const cfg = readConfig();
    if (cfg.mode === "autonomous" && !confirm("Autonomous mode moves the REAL robot. Continue?")) return;
    send({ action: "start_live", config: cfg });
  });

  onOpen(() => {});
  connect();
});
```

> Per-arm grouped live charts (via `splitArms`) can be added later once joint
> names are sent over telemetry; the single-joint streaming chart matches existing
> behavior, restyled. This is intentionally the minimum viable view.

- [ ] **Step 2: Manual verification (browser)**

Serve: `cd examples/trossen_ai && python -m uvicorn webapp.server:app --reload --port 8001`.
With the policy server **down**, click **Start Live** → badge `connecting` then
`no server`, ERROR log line appears; **Stop** returns to `idle` without hanging.
On `/replay`, browse to a dataset dir (can leave the repo) and pick an episode.
Submit feedback → "Thanks!" and a file lands in `webapp/feedback/`. Hover ⓘ →
tooltips. Confirm wizard colors/fonts on both pages.

- [ ] **Step 3: Delete the obsolete files**

```bash
git rm webapp/static/app.js webapp/static/styles.css
```

- [ ] **Step 4: Run backend tests (routes/static still serve)**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add webapp/static/js/live.js
git commit -m "feat(webapp-ui): live page wiring; remove legacy app.js/styles.css"
```

---

## Final verification

- [ ] **All backend tests green:** `cd examples/trossen_ai && python -m pytest webapp/tests -q`
- [ ] **Manual UI pass** (Task 8 Step 2) on `/` and `/replay`: theme, nav, config
  groups + tooltips, charts render, logs stream and color, stop works without
  hanging, folder browser traverses outside the repo, feedback saves, Sleep/Home
  send their actions.
- [ ] **Update `webapp/README.md`** to describe the two pages + module layout, then commit:

```bash
git add webapp/README.md
git commit -m "docs(webapp): document Live/Replay pages and frontend module layout"
```
