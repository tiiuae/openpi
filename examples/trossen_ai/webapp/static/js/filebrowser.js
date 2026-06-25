import { api } from "./api.js";

let target = null, curPath = null;
const $ = (id) => document.getElementById(id);

export function setupFileBrowser() {
  $("btn-cancel-browser")?.addEventListener("click", close);
  $("btn-select-dir")?.addEventListener("click", selectCurrent);
  $("modal-filebrowser")?.addEventListener("click", e => { if (e.target === e.currentTarget) close(); });
}
export function openFileBrowser(inputEl) {
  target = inputEl; $("modal-filebrowser").style.display = "";
  browse(inputEl.value.trim() || "~");
}
function close() { $("modal-filebrowser").style.display = "none"; target = null; }
function selectCurrent() {
  if (target && curPath) { target.value = curPath; target.dispatchEvent(new Event("change")); }
  close();
}
async function browse(path) {
  const box = $("browser-entries"); box.innerHTML = '<div class="browser-msg">Loading…</div>';
  try {
    const r = await api(`/api/files?path=${encodeURIComponent(path)}`);
    if (r.error) { box.innerHTML = `<div class="browser-msg err">${r.error}</div>`; return; }
    curPath = r.path; $("browser-current-path").textContent = r.path;
    let html = "";
    if (r.parent) html += `<div class="browser-item browser-up" data-path="${r.parent}"><span class="bi-icon">↑</span><span class="bi-name">..</span></div>`;
    for (const e of r.entries) {
      const badge = e.is_dataset ? '<span class="ds-badge">dataset</span>' : '';
      html += `<div class="browser-item${e.is_dataset?' is-dataset':''}" data-path="${e.path}"><span class="bi-icon">📁</span><span class="bi-name">${e.name}</span>${badge}</div>`;
    }
    box.innerHTML = html || '<div class="browser-msg">Empty directory</div>';
    box.querySelectorAll(".browser-item").forEach(it => it.addEventListener("click", () => browse(it.dataset.path)));
  } catch (e) { box.innerHTML = `<div class="browser-msg err">Error: ${e.message}</div>`; }
}
