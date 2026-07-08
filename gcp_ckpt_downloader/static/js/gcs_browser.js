// Owns the GCS tree: browsing/drilling via /api/gcs/ls, the breadcrumb, the
// checked-selection state (kept across drill-in/out), and the "already
// exists at destination" warning badges.
//
// This module deliberately does NOT own #btn-download's `disabled` state:
// that also depends on whether a download job is currently running, which
// is state only app.js (the controller) knows about. Owning both "is
// anything selected" and "is a job running" in two different places was a
// real bug (see setOnSelectionChange below) -- so instead this module just
// notifies app.js whenever the selection changes, and app.js is the single
// place that writes to #btn-download.disabled.
import { api, escapeHtml } from "./api.js";

const $ = (id) => document.getElementById(id);

// uri -> name, for every currently-checked folder row (survives drilling in
// and out, since it's independent of what's currently rendered).
const selected = new Map();

let currentPrefix = null;

// Callback invoked whenever the selection Set changes size/contents (a
// checkbox toggled, or a re-render re-applied previously-checked state).
// app.js registers one via setOnSelectionChange() to recompute
// #btn-download.disabled from BOTH selection size and job-active state.
let onSelectionChange = null;

export function setOnSelectionChange(cb) {
  onSelectionChange = cb;
}

export function initGcsBrowser() {
  // Re-check "exists at destination" badges whenever the dest path changes
  // (including when local_browser.js writes a newly-picked path into
  // #dest-input and dispatches "change").
  $("dest-input")?.addEventListener("change", recheckExisting);
  // Event delegation for breadcrumb segment clicks (breadcrumb is
  // re-rendered on every browse(), so a single delegated listener avoids
  // having to re-wire it each time).
  $("breadcrumb")?.addEventListener("click", (e) => {
    const seg = e.target.closest("[data-prefix]");
    if (seg) browse(seg.dataset.prefix);
  });
}

export async function browse(prefix) {
  const tree = $("gcs-tree");
  tree.innerHTML = '<div class="browser-msg">Loading…</div>';
  try {
    const r = await api(`/api/gcs/ls?prefix=${encodeURIComponent(prefix)}`);
    if (r.error) {
      tree.innerHTML = `<div class="browser-msg err">${escapeHtml(r.error)}</div>`;
      return;
    }
    currentPrefix = r.prefix;
    renderBreadcrumb(r.prefix);
    renderTree(r);
    await checkExisting(r.folders.map((f) => f.name));
  } catch (e) {
    tree.innerHTML = `<div class="browser-msg err">Error: ${escapeHtml(e.message)}</div>`;
  }
}

export function getSelected() {
  return [...selected.entries()].map(([uri, name]) => ({ uri, name }));
}

// Re-run the "exists at destination" check for whatever folders are
// currently rendered, without re-browsing (used after #dest-input changes).
export function recheckExisting() {
  if (!currentPrefix) return;
  const names = [...document.querySelectorAll("#gcs-tree .gcs-folder")].map((el) => el.dataset.name);
  checkExisting(names);
}

function renderBreadcrumb(prefix) {
  const bc = $("breadcrumb");
  const withoutScheme = prefix.replace(/^gs:\/\//, "");
  const trimmed = withoutScheme.endsWith("/") ? withoutScheme.slice(0, -1) : withoutScheme;
  const parts = trimmed.split("/").filter(Boolean);
  let acc = "gs://";
  const segs = parts.map((part) => {
    acc += `${part}/`;
    return `<span class="crumb" data-prefix="${escapeHtml(acc)}">${escapeHtml(part)}</span>`;
  });
  bc.innerHTML = segs.join('<span class="crumb-sep">/</span>');
}

function renderTree(r) {
  const tree = $("gcs-tree");
  let html = "";
  if (r.parent) {
    html += rowUp(r.parent);
  }
  for (const f of r.folders) {
    const checked = selected.has(f.uri) ? "checked" : "";
    html += `
      <div class="browser-item gcs-folder" data-uri="${escapeHtml(f.uri)}" data-name="${escapeHtml(f.name)}">
        <input type="checkbox" class="gcs-check" data-uri="${escapeHtml(f.uri)}" data-name="${escapeHtml(f.name)}" ${checked}>
        <span class="bi-icon folder-drill" data-prefix="${escapeHtml(f.uri)}">📁</span>
        <span class="bi-name folder-drill" data-prefix="${escapeHtml(f.uri)}">${escapeHtml(f.name)}</span>
        <span class="badge warn-badge" data-name="${escapeHtml(f.name)}" style="display:none" title="A folder with this name already exists at the destination">⚠</span>
      </div>`;
  }
  for (const o of r.objects) {
    html += `
      <div class="browser-item gcs-object">
        <span class="bi-icon">📄</span>
        <span class="bi-name">${escapeHtml(o.name)}</span>
      </div>`;
  }
  tree.innerHTML = html || '<div class="browser-msg">Empty</div>';

  tree.querySelectorAll(".folder-drill, .browser-up").forEach((el) => {
    el.addEventListener("click", () => browse(el.dataset.prefix));
  });
  tree.querySelectorAll(".gcs-check").forEach((cb) => {
    cb.addEventListener("change", () => {
      if (cb.checked) selected.set(cb.dataset.uri, cb.dataset.name);
      else selected.delete(cb.dataset.uri);
      onSelectionChange?.();
    });
  });
  onSelectionChange?.();
}

function rowUp(parentPrefix) {
  return `<div class="browser-item browser-up" data-prefix="${escapeHtml(parentPrefix)}"><span class="bi-icon">↑</span><span class="bi-name">..</span></div>`;
}

async function checkExisting(names) {
  if (!names.length) return;
  const dest = $("dest-input")?.value?.trim();
  if (!dest) return;
  try {
    const r = await api(`/api/local/existing?dest=${encodeURIComponent(dest)}&names=${encodeURIComponent(names.join(","))}`);
    const existing = new Set(r.existing || []);
    document.querySelectorAll("#gcs-tree .warn-badge").forEach((b) => {
      b.style.display = existing.has(b.dataset.name) ? "" : "none";
    });
  } catch {
    // Non-fatal: a failed "exists" probe shouldn't block browsing/selecting.
  }
}
