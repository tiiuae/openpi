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
