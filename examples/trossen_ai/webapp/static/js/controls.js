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
