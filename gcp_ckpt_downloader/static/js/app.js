// Controller: wires up the page on load, and owns the download button +
// progress-polling flow. Browsing/selection state lives in gcs_browser.js;
// the destination-picker modal lives in local_browser.js.
import { api, apiPost, escapeHtml } from "./api.js";
import { initGcsBrowser, browse, getSelected } from "./gcs_browser.js";
import { setupLocalBrowser, openLocalBrowser } from "./local_browser.js";

const $ = (id) => document.getElementById(id);

let pollTimer = null;

async function init() {
  setupLocalBrowser();
  initGcsBrowser();

  let bucket = "";
  try {
    const cfg = await api("/api/config");
    bucket = cfg.bucket;
    $("bucket-input").value = cfg.bucket;
    $("dest-input").value = cfg.dest;
  } catch (e) {
    setAuthChip(false, `config load failed: ${e.message}`);
  }

  loadAuth();

  $("btn-refresh").addEventListener("click", () => browse($("bucket-input").value.trim()));
  $("btn-browse-dest").addEventListener("click", openLocalBrowser);
  $("btn-download").addEventListener("click", onDownloadClick);

  if (bucket) browse(bucket);
}

async function loadAuth() {
  try {
    const auth = await api("/api/gcs/auth");
    if (auth.can_list) {
      setAuthChip(true, auth.active_account ? `✓ ${auth.active_account}` : "✓ authenticated");
    } else {
      setAuthChip(false, auth.error ? `✗ ${auth.error}` : "✗ cannot list bucket");
    }
  } catch (e) {
    setAuthChip(false, `✗ ${e.message}`);
  }
}

function setAuthChip(ok, text) {
  const chip = $("auth-chip");
  chip.textContent = text;
  chip.className = ok ? "badge badge-ok" : "badge badge-err";
}

async function onDownloadClick() {
  const items = getSelected();
  if (!items.length) return;

  const dest = $("dest-input").value.trim();
  if (!dest) {
    alert("Choose a destination directory first.");
    return;
  }

  // Warn-per-item: list the specific selected folders flagged as already
  // existing at the destination (the ⚠ badge rendered by gcs_browser.js),
  // and let the user back out before overwriting anything.
  const selectedNames = new Set(items.map((it) => it.name));
  const overwriteNames = [...document.querySelectorAll("#gcs-tree .warn-badge")]
    .filter((b) => b.style.display !== "none" && selectedNames.has(b.dataset.name))
    .map((b) => b.dataset.name);

  if (overwriteNames.length) {
    const ok = confirm(`These will be overwritten:\n${overwriteNames.join("\n")}\n\nContinue?`);
    if (!ok) return;
  }

  $("btn-download").disabled = true;
  try {
    const job = await apiPost("/api/gcs/download", { dest, items });
    startPolling(job.id);
    renderProgress(job);
  } catch (e) {
    alert(`Failed to start download: ${e.message}`);
    $("btn-download").disabled = getSelected().length === 0;
  }
}

function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const job = await api(`/api/gcs/download/${jobId}`);
      renderProgress(job);
      if (job.state === "done" || job.state === "error") {
        stopPolling();
        // Re-enable Download (gated on current selection) so the user can
        // retry/re-run without having to touch a checkbox first.
        $("btn-download").disabled = getSelected().length === 0;
      }
    } catch (e) {
      stopPolling();
      $("btn-download").disabled = getSelected().length === 0;
      const list = $("progress-list");
      list.innerHTML += `<div class="browser-msg err">Lost track of job: ${escapeHtml(e.message)}</div>`;
    }
  }, 1500);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function renderProgress(job) {
  const panel = $("progress");
  const list = $("progress-list");
  panel.style.display = "";

  let html = `<div class="progress-state">Job ${escapeHtml(job.id)} — <strong>${escapeHtml(job.state)}</strong> (dest: ${escapeHtml(job.dest)})</div>`;
  for (const item of job.items) {
    html += `
      <div class="progress-item">
        <span class="badge badge-${escapeHtml(item.state)}">${escapeHtml(item.state)}</span>
        <span class="bi-name">${escapeHtml(item.name)}</span>
        ${item.message ? `<div class="browser-msg err">${escapeHtml(item.message)}</div>` : ""}
        ${item.log_tail ? `<details class="log-tail"><summary>log tail</summary><pre>${escapeHtml(item.log_tail)}</pre></details>` : ""}
      </div>`;
  }
  list.innerHTML = html;
}

init();
