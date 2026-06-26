import { api } from "./api.js";
import { connect, send, onMessage, onOpen } from "./ws.js";
import { setupFileBrowser, openFileBrowser } from "./filebrowser.js";
import { setupLogs } from "./logs.js";
import { setupControls } from "./controls.js";
import { renderFeedback } from "./feedback.js";
import { UrdfView } from "./urdf_view.js";
import { buildTrajectoryCharts, Transport } from "./trajectory.js";
import { JOINT_NAMES_14 } from "./charts.js";

const $ = (id) => document.getElementById(id);

let view = null;          // UrdfView
let traj = null;          // last fetched trajectory payload
let charts = null;        // { charts, setCursor }
let transport = null;     // Transport

document.addEventListener("DOMContentLoaded", () => {
  setupFileBrowser();
  setupLogs();
  setupControls();
  renderFeedback($("feedback-card"));

  try {
    view = new UrdfView($("urdf-canvas"));
    view.load().catch((e) => console.warn("URDF load failed:", e));
  } catch (e) {
    console.warn("3D viewer unavailable:", e);
  }

  $("cam-views").addEventListener("click", (e) => {
    const v = e.target.dataset.view;
    if (v && view) view.snapView(v);
  });

  $("btn-browse-dataset").addEventListener("click", () => openFileBrowser($("dataset_dir")));
  $("dataset_dir").addEventListener("change", loadEpisodes);
  $("btn-preview").addEventListener("click", buildPreview);
  $("btn-replay").addEventListener("click", onReplayClick);

  // Spike modal wiring.
  $("spike-ack").addEventListener("change", (e) => { $("spike-confirm").disabled = !e.target.checked; });
  $("spike-cancel").addEventListener("click", () => ($("modal-spike").style.display = "none"));
  $("spike-confirm").addEventListener("click", () => { $("modal-spike").style.display = "none"; startReplay(); });

  // During a real hardware replay, advance the transport cursor from telemetry.
  onMessage("action", (e) => { if (transport && e.step != null) transport.seek(e.step); });
  onOpen(() => {});
  connect();
});

async function loadEpisodes() {
  const dir = $("dataset_dir").value; if (!dir) return;
  try {
    const info = await api(`/api/episodes?dataset_dir=${encodeURIComponent(dir)}`);
    $("episode-select").innerHTML = Array.from({ length: info.total_episodes }, (_, i) =>
      `<option value="${i}">Episode ${i}</option>`).join("");
  } catch (e) { /* surfaced via logs */ }
}

// Read the Replay-config card. Numbers come back as strings; the backend casts.
function readReplayConfig() {
  const num = (id) => $(id)?.value ?? "";
  return {
    control_freq: num("cf_control_freq") || "0",   // 0 -> backend uses episode fps
    max_joint_speed: num("cf_max_joint_speed") || "3.0",
    ik_orientation_weight: num("cf_ik_orientation_weight") || "0.01",
    ik_pos_tol_m: num("cf_ik_pos_tol_m") || "0.001",
    min_time_to_move_multiplier: num("cf_min_time_to_move_multiplier") || "3.0",
    loop_rate: num("cf_loop_rate") || "30",
    smooth_streaming: $("cf_smooth_streaming")?.checked ? "1" : "",
  };
}

function trajUrl() {
  const c = readReplayConfig();
  const p = new URLSearchParams({
    dataset_dir: $("dataset_dir").value,
    episode_index: $("episode-select").value || "0",
    control_freq: c.control_freq,
    max_joint_speed: c.max_joint_speed,
    ik_orientation_weight: c.ik_orientation_weight,
    ik_pos_tol_m: c.ik_pos_tol_m,
  });
  return `/api/episode_trajectory?${p}`;
}

async function fetchTrajectory() {
  const data = await api(trajUrl());
  data._dataset = $("dataset_dir").value;
  data._ep = $("episode-select").value;
  traj = data;
  return data;
}

async function buildPreview() {
  $("btn-preview").textContent = "Computing…"; $("btn-preview").disabled = true;
  try {
    if (charts) { charts.charts.forEach((c) => c.destroy()); charts = null; }
    if (transport) { transport.destroy(); transport = null; }
    const data = await fetchTrajectory();
    renderSpikeBanner(data);
    charts = buildTrajectoryCharts($("chart-joints"), $("chart-ee"), data);
    transport = new Transport({
      nFrames: data.n_frames, fps: data.fps,
      els: { play: $("tp-play"), stop: $("tp-stop"), slider: $("tp-slider"),
             readout: $("tp-readout"), speed: $("tp-speed") },
      onFrame: (i) => {
        charts.setCursor(i);
        if (view) view.setFrameJoints(data.joints_clamped[i]);
      },
    });
  } catch (e) {
    console.error("Preview failed:", e);
  } finally {
    $("btn-preview").textContent = "Preview"; $("btn-preview").disabled = false;
  }
}

function renderSpikeBanner(data) {
  const b = $("spike-banner");
  if (data.spikes && data.spikes.length) {
    const p = peakSpike(data);
    b.textContent =
      `⚠ ${data.spikes.length} velocity spike(s) — peak ${p.value.toFixed(1)} rad/s` +
      ` on ${p.joint} at frame ${p.frame} (limit ${data.max_joint_speed} rad/s).` +
      ` Replay may damage the robot.`;
    b.style.display = "block";
  } else {
    b.style.display = "none";
  }
}

// Peak per-joint velocity across all spike frames, with which joint and frame.
function peakSpike(data) {
  let value = 0, frame = data.spikes[0] ?? 0, jointIdx = 0;
  for (const t of data.spikes) {
    for (let j = 0; j < data.velocity[t].length; j++) {
      if (data.velocity[t][j] > value) { value = data.velocity[t][j]; frame = t; jointIdx = j; }
    }
  }
  return { value, frame, joint: JOINT_NAMES_14[jointIdx] ?? `joint ${jointIdx}` };
}

function worstVelocity(data) { return peakSpike(data).value; }

// Replay button: ensure spike data exists, gate if spiky, else go.
async function onReplayClick() {
  $("btn-replay").disabled = true;
  try {
    try {
      const stale = !traj || traj._dataset !== $("dataset_dir").value ||
                    String(traj._ep) !== String($("episode-select").value);
      if (stale) { traj = null; await fetchTrajectory(); }
    } catch (e) { console.warn("Spike pre-check failed; replay will be blocked:", e); }

    if (!traj) {
      console.error("Cannot verify trajectory safety; replay blocked.");
      alert("Could not compute the episode trajectory to check for velocity spikes. Replay is blocked until a preview succeeds.");
      return;
    }
    if (traj.spikes && traj.spikes.length) {
      $("spike-msg").textContent =
        `${traj.spikes.length} frame(s) exceed the ${traj.max_joint_speed} rad/s limit ` +
        `(worst ${worstVelocity(traj).toFixed(1)} rad/s).`;
      $("spike-ack").checked = false; $("spike-confirm").disabled = true;
      $("modal-spike").style.display = "flex";
      return;
    }
    startReplay();
  } finally {
    $("btn-replay").disabled = false;
  }
}

function startReplay() {
  const cfg = { dataset_dir: $("dataset_dir").value, episode_index: $("episode-select").value,
                mode: $("mode-select").value, ...readReplayConfig() };
  if (cfg.mode === "autonomous" && !confirm("Replay will move the REAL robot. Continue?")) return;
  send({ action: "start_replay", config: cfg });
}
