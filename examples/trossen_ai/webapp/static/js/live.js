import { connect, send, onMessage, onOpen } from "./ws.js";
import { renderConfig, readConfig, setDefaults } from "./config.js";
import { makeJointCharts } from "./charts.js";
import { setupLogs } from "./logs.js";
import { setupControls, setSessionActive } from "./controls.js";
import { setupSmoothing } from "./smoothing.js";
import { onRunStatus } from "./runlog.js";
import { setupModelServing } from "./model_serving.js";

const $ = (id) => document.getElementById(id);
let jointCharts = null;
const f = (v) => (v == null ? "—" : Number(v).toFixed(1));

document.addEventListener("DOMContentLoaded", () => {
  renderConfig($("config-col"));
  setDefaults();
  setupLogs();
  setupControls();
  setupModelServing($("config-col"));

  const smoothing = setupSmoothing($("smoothing-box"));
  setInterval(() => smoothing.update(), 200);

  try {
    jointCharts = makeJointCharts($("charts-box"));
    setInterval(() => jointCharts.update(), 200);
  } catch (err) {
    console.error("Chart init failed (charts disabled, telemetry still runs):", err);
    $("charts-box").innerHTML = '<div class="browser-msg err">Charts unavailable (chart library failed to load).</div>';
  }

  onMessage("action", (e) => { if (jointCharts) jointCharts.pushAction(e.step, e.action); smoothing.onAction(e); });
  onMessage("chunk", (e) => smoothing.onChunk(e));
  onMessage("overlap", (e) => smoothing.onOverlap(e));
  onMessage("weights", (e) => smoothing.onWeights(e));
  onMessage("status", (e) => onRunStatus(e));
  // Reuse one <img> per camera and just swap its .src each frame. Rebuilding the
  // DOM every frame (old behaviour) tore down + re-laid-out the strip on every
  // inference, which read as a fast full-page "refresh"/flicker.
  const camImgs = {};
  onMessage("images", (e) => {
    const ib = $("images-box");
    for (const [name, b64] of Object.entries(e.images)) {
      let img = camImgs[name];
      if (!img) {
        img = new Image(); img.title = name; img.className = "cam-frame";
        camImgs[name] = img; ib.appendChild(img);
      }
      img.src = "data:image/jpeg;base64," + b64;
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
    // Fresh run → wipe live viz so nothing carries over from the last run, and
    // size the action charts to hold the whole episode (max_steps).
    if (jointCharts) { jointCharts.setMaxPts(Number(cfg.max_steps) || 1000); jointCharts.clear(); }
    smoothing.reset();
    setSessionActive(true);  // optimistic; status events keep it in sync
    send({ action: "start_live", config: cfg });
  });

  onOpen(() => {});
  connect();
});
