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
