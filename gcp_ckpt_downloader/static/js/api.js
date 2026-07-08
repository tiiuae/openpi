// Copied verbatim from the reference webapp's static/js/api.js -- these two
// helpers already do the right thing (surface FastAPI's {"detail": ...} on
// non-2xx responses so callers can show *why* a request failed).
export async function api(url) {
  const res = await fetch(url);
  if (!res.ok) {
    // Surface the server's error detail (FastAPI {"detail": ...}) if present,
    // so callers can show *why* a request failed instead of a bare status.
    let detail = "";
    try { detail = (await res.json())?.detail || ""; } catch { /* non-JSON body */ }
    throw new Error(detail ? `HTTP ${res.status} — ${detail}` : `HTTP ${res.status} — ${url}`);
  }
  return res.json();
}
export async function apiPost(url, body) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
  return res.json();
}

// Small shared helper (not in the reference file): escape text before it's
// interpolated into innerHTML. Bucket/folder names and gcloud log tails are
// free-form strings we don't control, so this keeps a stray "&"/"<" in a
// checkpoint name or a log line from corrupting the rendered DOM.
export function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
