import { send, onMessage } from "./ws.js";
import { logLine } from "./logs.js";

const $ = (id) => document.getElementById(id);
function badge(id, text, cls) { const b = $(id); if (b) { b.textContent = text; b.className = "badge " + (cls || ""); } }

// Single source of truth for whether a session is active. Toggles which
// buttons are clickable so you can't double-start or stop nothing.
let sessionActive = false;
let liveStartBlocked = false;
const sessionActiveListeners = new Set();

export function setSessionActive(active) {
  const changed = sessionActive !== active;
  sessionActive = active;
  updateSessionControls();
  if (changed) sessionActiveListeners.forEach((listener) => listener(active));
}

export function onSessionActiveChange(listener) {
  sessionActiveListeners.add(listener);
  listener(sessionActive);
  return () => sessionActiveListeners.delete(listener);
}

// Block Eval while a managed child exists but is not ready. External
// policy-server workflows remain available when no managed child is running.
export function setLiveStartBlocked(blocked) {
  liveStartBlocked = blocked;
  updateSessionControls();
}

function updateSessionControls() {
  const start = $("btn-start"), stop = $("btn-stop");
  if (start) start.disabled = sessionActive || liveStartBlocked;
  if (stop) stop.disabled = !sessionActive;
  // Home/Sleep/Replay also occupy the single session slot, so disable them
  // while a session runs (Stop + E-STOP stay enabled). Preview is off-robot,
  // so it stays clickable.
  ["btn-home", "btn-sleep", "btn-replay"].forEach((id) => {
    const button = $(id);
    if (button) button.disabled = sessionActive;
  });
}

export function setupControls() {
  setSessionActive(false);  // initial: Stop disabled, Start enabled

  $("btn-stop")?.addEventListener("click", () => { logLine("INFO", "Stop clicked"); send({ action: "stop" }); });
  $("btn-estop")?.addEventListener("click", () => { logLine("WARNING", "E-STOP clicked"); send({ action: "estop" }); });
  $("btn-sleep")?.addEventListener("click", () => guarded("go_sleep", "Send arms to SLEEP?", "Sleep"));
  $("btn-home")?.addEventListener("click", () => guarded("go_home", "Send arms to HOME?", "Home"));

  onMessage("status", (e) => {
    if (e.kind === "connecting") { badge("badge-session", "connecting", ""); setSessionActive(true); }
    if (e.kind === "started") { badge("badge-session", "running", ""); setSessionActive(true); }
    if (e.kind === "stopped") { badge("badge-session", "idle", ""); setSessionActive(false); }
    if (e.kind === "connect_failed") { badge("badge-session", "no server", "bad"); setSessionActive(false); logLine("ERROR", e.payload?.message || "connect failed"); }
    if (e.kind === "firmware_error") { badge("badge-session", "fault → sleep", "bad"); setSessionActive(false); logLine("ERROR", "Firmware fault: " + JSON.stringify(e.payload)); }
    if (e.kind === "error") { badge("badge-session", "error", "bad"); setSessionActive(false); logLine("ERROR", e.payload?.message || "error"); }
  });
  onMessage("inference", (e) => badge("badge-rtt", `${e.rtt_ms.toFixed(0)}ms`, e.rtt_ms < 100 ? "ok" : "bad"));
}

// Home/Sleep always physically move the arm regardless of the page's mode
// selector, so always confirm and never forward `mode` (the mover ignores it).
function guarded(action, prompt, label) {
  if (!confirm(prompt + " (moves the REAL robot)")) return;
  logLine("INFO", `${label} clicked`);
  setSessionActive(true);
  send({ action });
}
