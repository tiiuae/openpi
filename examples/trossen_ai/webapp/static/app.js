const $ = (id) => document.getElementById(id);
const MAXPTS = 300;

function mkChart(id, datasets) {
  return new Chart($(id), {
    type: "line",
    data: { datasets },
    options: { animation: false, responsive: true, parsing: false,
      scales: { x: { type: "linear", display: false }, y: { ticks: { color: "#9fb3c8" } } },
      plugins: { legend: { labels: { color: "#9fb3c8" } } } },
  });
}
const ds = (label, color) => ({ label, borderColor: color, data: [], pointRadius: 0, borderWidth: 1 });

const charts = {
  actions: mkChart("chart-actions", [ds("joint", "#4ad")]),
  rawvs: mkChart("chart-rawvs", [ds("raw", "#f80"), ds("smoothed", "#4ad")]),
  overlap: mkChart("chart-overlap", [ds("overlaps", "#7c7")]),
  buffer: mkChart("chart-buffer", [ds("buffer size", "#c7f")]),
};

// Per-prediction blend weights for the latest step (bar chart, oldest->newest).
const weightsChart = new Chart($("chart-weights"), {
  type: "bar",
  data: { labels: [], datasets: [{ label: "weight", data: [], backgroundColor: "#fa4" }] },
  options: { animation: false, responsive: true,
    scales: { x: { ticks: { color: "#9fb3c8" } }, y: { min: 0, max: 1, ticks: { color: "#9fb3c8" } } },
    plugins: { legend: { labels: { color: "#9fb3c8" } } } },
});

function push(chart, dsIndex, x, y) {
  const d = chart.data.datasets[dsIndex].data;
  d.push({ x, y });
  if (d.length > MAXPTS) d.shift();
}
setInterval(() => Object.values(charts).forEach((c) => c.update("none")), 200);

function jointIdx() { return parseInt($("joint-index").value || "0", 10); }

// ---- WebSocket ----
let ws;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws/telemetry`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { setBadge("badge-session", "disconnected", "bad"); setTimeout(connect, 1000); };
}
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function logLine(level, msg) {
  const box = $("log-box");
  const div = document.createElement("div");
  div.className = "log-" + level;
  div.textContent = `[${level}] ${msg}`;
  box.appendChild(div);
  if (box.childElementCount > 500) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}
function setBadge(id, text, cls) { const b = $(id); b.textContent = text; b.className = "badge " + (cls || ""); }
const fmt = (v) => (v == null ? "—" : Number(v).toFixed(1));

function handle(evt) {
  switch (evt.type) {
    case "log": logLine(evt.level, evt.msg); break;
    case "action": {
      const j = jointIdx();
      push(charts.actions, 0, evt.step, evt.action[j]);
      if (evt.raw) push(charts.rawvs, 0, evt.step, evt.raw[j]);
      push(charts.rawvs, 1, evt.step, evt.action[j]);
      break;
    }
    case "overlap": push(charts.overlap, 0, evt.step, evt.count); break;
    case "weights": {
      const w = evt.weights || [];
      weightsChart.data.labels = w.map((_, i) => i);  // 0 = oldest prediction
      weightsChart.data.datasets[0].data = w;
      weightsChart.update("none");
      break;
    }
    case "status":
      if (evt.kind === "buffer") { push(charts.buffer, 0, Date.now() / 1000, evt.payload.size); }
      else {
        logLine("INFO", `status: ${evt.kind} ${JSON.stringify(evt.payload)}`);
        if (evt.kind === "started") setBadge("badge-session", "running", "ok");
        if (evt.kind === "stopped") setBadge("badge-session", "idle", "");
      }
      break;
    case "inference": setBadge("badge-rtt", `${evt.rtt_ms.toFixed(0)}ms`, evt.rtt_ms < 100 ? "ok" : "bad"); break;
    case "images": {
      const box = $("images-box"); box.innerHTML = "";
      for (const [name, b64] of Object.entries(evt.images)) {
        const img = new Image(); img.src = "data:image/jpeg;base64," + b64; img.title = name;
        box.appendChild(img);
      }
      break;
    }
    case "metrics":
      $("metrics-box").textContent =
        `RTT last/p50/p95: ${fmt(evt.rtt_last)}/${fmt(evt.rtt_p50)}/${fmt(evt.rtt_p95)} ms\n` +
        `Loop Hz: ${fmt(evt.loop_hz)}   Jitter: ${fmt(evt.jitter)}   Drops: ${evt.drops}`;
      break;
  }
}

// ---- config form ----
function readConfig() {
  const f = $("config-form"); const cfg = {};
  for (const el of f.elements) {
    if (!el.name) continue;
    cfg[el.name] = el.type === "checkbox" ? el.checked : el.value;
  }
  cfg.mode = $("mode-select").value;
  return cfg;
}
function applyConfig(cfg) {
  const f = $("config-form");
  for (const el of f.elements) {
    if (!el.name || !(el.name in cfg)) continue;
    if (el.type === "checkbox") el.checked = !!cfg[el.name]; else el.value = cfg[el.name];
  }
  if (cfg.mode) $("mode-select").value = cfg.mode;
}

// ---- controls ----
$("btn-start").onclick = () => {
  const cfg = readConfig();
  if (cfg.mode === "autonomous" && !confirm("Autonomous mode moves the REAL robot. Continue?")) return;
  send({ action: "start_live", config: cfg });
};
$("btn-replay").onclick = () => {
  const cfg = readConfig();
  if (cfg.mode === "autonomous" && !confirm("Replay will move the REAL robot. Continue?")) return;
  send({ action: "start_replay", config: cfg });
};
$("btn-stop").onclick = () => send({ action: "stop" });
$("btn-estop").onclick = () => send({ action: "estop" });

// ---- presets ----
async function refreshPresets() {
  const names = await (await fetch("/api/presets")).json();
  const sel = $("preset-select"); sel.innerHTML = "";
  names.forEach((n) => { const o = document.createElement("option"); o.value = o.textContent = n; sel.appendChild(o); });
}
$("preset-save").onclick = async () => {
  const name = $("preset-name").value.trim(); if (!name) return;
  await fetch("/api/presets", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, config: readConfig() }) });
  refreshPresets();
};
$("preset-load").onclick = async () => {
  const name = $("preset-select").value; if (!name) return;
  applyConfig(await (await fetch("/api/presets/" + name)).json());
};
$("preset-delete").onclick = async () => {
  const name = $("preset-select").value; if (!name) return;
  await fetch("/api/presets/" + name, { method: "DELETE" }); refreshPresets();
};

connect();
refreshPresets();
