async function responseError(res, url) {
  // Surface FastAPI's {"detail": ...} response so the UI can explain
  // conflicts such as an active robot session or an already-running model.
  let detail = "";
  try { detail = (await res.json())?.detail || ""; } catch { /* non-JSON body */ }
  if (detail && typeof detail !== "string") detail = JSON.stringify(detail);
  return new Error(detail ? `HTTP ${res.status} — ${detail}` : `HTTP ${res.status} — ${url}`);
}

export async function api(url) {
  const res = await fetch(url);
  if (!res.ok) throw await responseError(res, url);
  return res.json();
}
export async function apiPost(url, body) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await responseError(res, url);
  return res.json();
}
