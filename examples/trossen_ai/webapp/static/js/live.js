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
