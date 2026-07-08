// Local destination-directory picker modal. Adapted from the reference
// webapp's static/js/filebrowser.js: same modal/browse pattern, but hits
// /api/local/ls (entries have no "is_dataset" flag here) and always writes
// the chosen path into #dest-input (this app only ever browses for one
// thing -- the download destination -- so there's no need for the
// reference's generic "target input element" parameter).
import { api, escapeHtml } from "./api.js";

const $ = (id) => document.getElementById(id);
let curPath = null;

export function setupLocalBrowser() {
  $("btn-cancel-browser")?.addEventListener("click", close);
  $("btn-select-dir")?.addEventListener("click", selectCurrent);
  $("modal-localbrowser")?.addEventListener("click", (e) => {
    if (e.target === e.currentTarget) close();
  });
}

export function openLocalBrowser() {
  $("modal-localbrowser").style.display = "";
  browse($("dest-input")?.value.trim() || "~");
}

function close() {
  $("modal-localbrowser").style.display = "none";
}

function selectCurrent() {
  if (curPath) {
    const input = $("dest-input");
    input.value = curPath;
    // Dispatch "change" so gcs_browser.js re-checks existing-folder badges
    // against the newly-picked destination.
    input.dispatchEvent(new Event("change"));
  }
  close();
}

async function browse(path) {
  const box = $("browser-entries");
  box.innerHTML = '<div class="browser-msg">Loading…</div>';
  try {
    const r = await api(`/api/local/ls?path=${encodeURIComponent(path)}`);
    if (r.error) {
      box.innerHTML = `<div class="browser-msg err">${escapeHtml(r.error)}</div>`;
      return;
    }
    curPath = r.path;
    $("browser-current-path").textContent = r.path;
    let html = "";
    if (r.parent) {
      html += `<div class="browser-item browser-up" data-path="${escapeHtml(r.parent)}"><span class="bi-icon">↑</span><span class="bi-name">..</span></div>`;
    }
    for (const e of r.entries) {
      html += `<div class="browser-item" data-path="${escapeHtml(e.path)}"><span class="bi-icon">📁</span><span class="bi-name">${escapeHtml(e.name)}</span></div>`;
    }
    box.innerHTML = html || '<div class="browser-msg">Empty directory</div>';
    box.querySelectorAll(".browser-item").forEach((it) => {
      it.addEventListener("click", () => browse(it.dataset.path));
    });
  } catch (e) {
    box.innerHTML = `<div class="browser-msg err">Error: ${escapeHtml(e.message)}</div>`;
  }
}
