import { connect, send, onMessage, onOpen } from "./ws.js";
import { setupLogs } from "./logs.js";
import { UrdfView } from "./urdf_view.js";
import { readInput, readEdges, hasGamepad } from "./gamepad.js";

const $ = (id) => document.getElementById(id);
let view = null, active = false, sendTimer = null;

function readConfig() {
  return {
    mode: $("cf_mode").value,
    control_freq: +$("cf_control_freq").value,
    max_lin: +$("cf_max_lin").value,
    max_ang: +$("cf_max_ang").value,
    grip_rate: +$("cf_grip_rate").value,
    deadzone: +$("cf_deadzone").value,
    ik_orientation_weight: +$("cf_ik_orientation_weight").value,
    ik_pos_tol_m: +$("cf_ik_pos_tol_m").value,
  };
}

const BINDINGS = [
  ["Left stick / WASD", "translate X-Y"],
  ["RB·LB / Q·E", "translate Z up/down"],
  ["Right stick / IJKL", "pitch / yaw"],
  ["RT·LT / U·O", "roll"],
  ["A·B / Z·C", "gripper close / open"],
  ["X / Tab", "switch active arm"],
  ["D-pad ↑ / H", "Home pose"],
  ["D-pad ↓ / P", "Sleep pose"],
  ["Back / Space", "E-STOP"],
  ["Start / Esc", "Stop session"],
];

function renderKeybindings() {
  $("keybindings").innerHTML =
    "<table class='kb'>" +
    BINDINGS.map(([k, a]) => `<tr><td><kbd>${k}</kbd></td><td>${a}</td></tr>`).join("") +
    "</table>";
}

function startSendLoop(cfg) {
  const period = 1000 / Math.max(10, Math.min(50, cfg.control_freq));
  sendTimer = setInterval(() => {
    if (!active) return;
    const inp = readInput(cfg.deadzone);
    const { estop, stop, ...moves } = readEdges();
    $("badge-pad").textContent = inp.hasPad ? "gamepad ✓" : "keyboard only";
    if (estop) { active = false; send({ action: "estop" }); return; }
    if (stop) { active = false; send({ action: "stop" }); return; }
    send({ action: "teleop_input", payload: { axes: inp.axes, grip: inp.grip, ...moves } });
  }, period);
}

document.addEventListener("DOMContentLoaded", async () => {
  setupLogs();
  renderKeybindings();
  view = new UrdfView($("urdf-canvas"));
  await view.load();
  $("cam-views").addEventListener("click", (e) => {
    const v = e.target?.dataset?.view; if (v) view.snapView(v);
  });

  onMessage("action", (e) => view.setFrameJoints(e.action));
  onMessage("images", (e) => {
    const ib = $("images-box"); ib.innerHTML = "";
    for (const [name, b64] of Object.entries(e.images)) {
      const img = new Image(); img.src = "data:image/jpeg;base64," + b64; img.title = name; ib.appendChild(img);
    }
  });
  onMessage("status", (e) => {
    if (e.kind === "teleop_started") $("badge-session").textContent = "teleop";
    if (e.kind === "teleop_stopped" || e.kind === "stopped") { $("badge-session").textContent = "idle"; active = false; }
    if (e.kind === "teleop_move") $("badge-arm").textContent = "moving → " + e.payload.target;
  });

  $("btn-start").addEventListener("click", () => {
    const cfg = readConfig();
    if (cfg.mode === "autonomous" && !confirm("Autonomous mode moves the REAL robot. Continue?")) return;
    active = true; $("badge-session").textContent = "teleop";
    send({ action: "start_teleop", config: cfg });
    if (!sendTimer) startSendLoop(cfg);
  });
  $("btn-switch").addEventListener("click", () => {
    send({ action: "switch_arm" });
    const b = $("badge-arm"); b.textContent = b.textContent.includes("left") ? "arm: right" : "arm: left";
  });
  $("btn-stop").addEventListener("click", () => { active = false; send({ action: "stop" }); });
  $("btn-estop").addEventListener("click", () => { active = false; send({ action: "estop" }); });
  window.addEventListener("keydown", (e) => {
    if (e.key === " ") { active = false; send({ action: "estop" }); }
    if (e.key === "Escape") { active = false; send({ action: "stop" }); }
  });

  onOpen(() => { $("badge-pad").textContent = hasGamepad() ? "gamepad ✓" : "keyboard only"; });
  connect();
});
