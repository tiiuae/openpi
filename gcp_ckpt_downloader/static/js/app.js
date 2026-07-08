// Controller: wires up the page on load, and owns the download button +
// progress-polling flow. Browsing/selection state lives in gcs_browser.js;
// the destination-picker modal lives in local_browser.js.
//
// #btn-download.disabled has exactly one writer: updateDownloadButtonState()
// below, which is the single source of truth combining BOTH "is anything
// selected" (getSelected(), owned by gcs_browser.js) and "is a job currently
// running" (jobActive, owned here). gcs_browser.js only ever *notifies* this
// module (via setOnSelectionChange) that the selection may have changed --
// it never touches the DOM element itself, so a tree re-render (drill,
// breadcrumb click, Refresh) can't accidentally re-enable Download mid-job.
import { api, apiPost, escapeHtml } from "./api.js";
import { initGcsBrowser, browse, getSelected, setOnSelectionChange } from "./gcs_browser.js";
import { setupLocalBrowser, openLocalBrowser } from "./local_browser.js";

const $ = (id) => document.getElementById(id);

let pollTimer = null;
let jobActive = false;

async function init() {
  setupLocalBrowser();
  initGcsBrowser();
  setOnSelectionChange(updateDownloadButtonState);

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

  updateDownloadButtonState();
  if (bucket) browse(bucket);
}

function updateDownloadButtonState() {
  $("btn-download").disabled = jobActive || getSelected().length === 0;
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

  // Warn-per-item: re-check /api/local/existing against the FULL current
  // selection, not just whatever folder happens to be rendered in #gcs-tree
  // right now. The selection Map in gcs_browser.js survives drill in/out
  // specifically so a user can pick items from multiple folders -- reading
  // ⚠ badges out of the DOM here would silently miss warnings for anything
  // selected in a previously-viewed folder, since renderTree() replaces
  // #gcs-tree's innerHTML (and its badges) on every browse().
  const overwriteNames = await namesThatExistAtDest(dest, items.map((it) => it.name));
  if (overwriteNames.length) {
    const ok = confirm(`These will be overwritten:\n${overwriteNames.join("\n")}\n\nContinue?`);
    if (!ok) return;
  }

  jobActive = true;
  updateDownloadButtonState();
  try {
    const job = await apiPost("/api/gcs/download", { dest, items });
    startPolling(job.id);
    renderProgress(job);
  } catch (e) {
    jobActive = false;
    updateDownloadButtonState();
    alert(`Failed to start download: ${e.message}`);
  }
}

async function namesThatExistAtDest(dest, names) {
  if (!names.length) return [];
  try {
    const r = await api(`/api/local/existing?dest=${encodeURIComponent(dest)}&names=${encodeURIComponent(names.join(","))}`);
    return r.existing || [];
  } catch (e) {
    // Non-fatal, same as gcs_browser.js's checkExisting(): if the probe
    // itself fails, fall back to no warning rather than blocking download.
    console.error("existing-check before download failed:", e);
    return [];
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
        // Job is no longer active: re-enable Download (still gated on
        // current selection via updateDownloadButtonState) so the user can
        // retry/re-run without having to touch a checkbox first.
        jobActive = false;
        updateDownloadButtonState();
      }
    } catch (e) {
      stopPolling();
      jobActive = false;
      updateDownloadButtonState();
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
