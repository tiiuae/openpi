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

  try {
    $("charts-box").innerHTML =
      `<div class="chart-wrap"><div class="chart-title">Replayed action (joint 0)</div>
        <div class="chart-cj-container"><canvas id="chart-actions"></canvas></div></div>`;
    chart = new LiveChart($("chart-actions"), ["joint 0"]);
    setInterval(() => chart.update(), 200);
  } catch (err) {
    console.error("Chart init failed (charts disabled, telemetry still runs):", err);
    $("charts-box").innerHTML = '<div class="browser-msg err">Charts unavailable (chart library failed to load).</div>';
  }

  $("btn-browse-dataset").addEventListener("click", () => openFileBrowser($("dataset_dir")));
  $("dataset_dir").addEventListener("change", loadEpisodes);
  $("btn-replay").addEventListener("click", () => {
    const cfg = { dataset_dir: $("dataset_dir").value, episode_index: $("episode-select").value,
                  mode: $("mode-select").value };
    if (cfg.mode === "autonomous" && !confirm("Replay will move the REAL robot. Continue?")) return;
    send({ action: "start_replay", config: cfg });
  });

  onMessage("action", (e) => { if (chart && e.action?.length) chart.push(0, e.step, e.action[0]); });
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
