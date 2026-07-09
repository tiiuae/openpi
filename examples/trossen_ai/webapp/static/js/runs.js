// webapp/static/js/runs.js
import { makeSeriesChart, JOINT_NAMES_14 } from "./charts.js";

const $ = (id) => document.getElementById(id);
const selected = new Set();
let deltaChart = null, overlapChart = null, jointChart = null, rttChart = null, eeChart = null;

const ESCAPE_HTML = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ESCAPE_HTML[ch]);
}

async function loadList() {
  const runs = await (await fetch("/api/runs")).json();
  const box = $("runs-list");
  if (!runs.length) { box.innerHTML = '<em class="muted">No runs recorded yet.</em>'; return; }
  box.innerHTML = runs.map(r => {
    const id = String(r.run_id);
    return `
      <div class="run-item">
        <label class="run-select">
          <input type="checkbox" value="${escapeHtml(id)}" ${selected.has(id) ? "checked" : ""}>
          <span class="run-id">${escapeHtml(id)}</span>
          <span class="run-model">${escapeHtml(r.model_name || "—")}</span>
          <span class="run-metric">${escapeHtml(r.steps ?? "—")} steps</span>
          <span class="run-metric">${escapeHtml(r.end_reason)}</span>
          <span class="run-metric">${escapeHtml(r.rating ? r.rating.success : "unrated")}</span>
        </label>
        <button type="button" class="btn-sm run-delete" data-run-id="${escapeHtml(id)}">Delete</button>
      </div>`;
  }).join("");
  box.querySelectorAll("input[type=checkbox]").forEach(cb =>
    cb.addEventListener("change", () => {
      cb.checked ? selected.add(cb.value) : selected.delete(cb.value);
      refresh();
    }));
  box.querySelectorAll(".run-delete").forEach(btn =>
    btn.addEventListener("click", () => deleteRun(btn.dataset.runId)));

  const sel = $("runs-joint-sel");
  if (sel && !sel.options.length) {
    sel.innerHTML = JOINT_NAMES_14.map((n, i) => `<option value="${i}">${escapeHtml(n)}</option>`).join("");
    sel.addEventListener("change", refresh);
  }

  const esel = $("runs-ee-sel");
  if (esel && !esel.options.length) {
    esel.innerHTML = EE_NAMES_16.map((n, i) => `<option value="${i}">${escapeHtml(n)}</option>`).join("");
    esel.addEventListener("change", refresh);
  }
}

async function deleteRun(id) {
  if (!id || !window.confirm(`Delete run ${id}? This cannot be undone.`)) return;
  const res = await fetch(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) {
    window.alert(`Could not delete run ${id}.`);
    return;
  }
  selected.delete(id);
  await loadList();
  await refresh();
}

async function fetchRun(id) {
  return (await fetch(`/api/runs/${encodeURIComponent(id)}?events=1`)).json();
}

function configDiff(runs) {
  // runs: [{id, manifest}]. Build union of config keys; highlight rows that differ.
  const keys = new Set();
  runs.forEach(r => Object.keys(r.manifest.config || {}).forEach(k => keys.add(k)));
  const rows = [...keys].sort().map(k => {
    const vals = runs.map(r => String((r.manifest.config || {})[k] ?? ""));
    const differ = new Set(vals).size > 1;
    return `<tr class="${differ ? "diff" : ""}"><td>${escapeHtml(k)}</td>${
      vals.map(v => `<td>${escapeHtml(v)}</td>`).join("")}</tr>`;
  }).join("");
  return `<table class="runs-diff"><thead><tr><th>key</th>${
    runs.map(r => `<th>${escapeHtml(r.id)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
}

// From an events array, series of {x:step, y:value}. delta from action events
// (‖action − raw‖), overlap from overlap events.
function deltaSeries(events) {
  const pts = [];
  for (const e of events) {
    if (e.type !== "action" || !e.raw || !e.action) continue;
    let s = 0; for (let i = 0; i < e.action.length; i++) { const d = e.action[i] - e.raw[i]; s += d * d; }
    pts.push({ x: e.step, y: Math.sqrt(s) });
  }
  return pts;
}
function overlapSeries(events) {
  return events.filter(e => e.type === "overlap").map(e => ({ x: e.step, y: e.count }));
}

// action[jointIdx] vs step, from action events.
function jointSeries(events, jointIdx) {
  return events
    .filter(e => e.type === "action" && Array.isArray(e.action) && e.action.length > jointIdx)
    .map(e => ({ x: e.step, y: e.action[jointIdx] }));
}
// 16-D EE target names: per arm [x,y,z,qw,qx,qy,qz,grip].
const EE_NAMES_16 = ["l_x","l_y","l_z","l_qw","l_qx","l_qy","l_qz","l_grip",
                     "r_x","r_y","r_z","r_qw","r_qx","r_qy","r_qz","r_grip"];
function hasEE(events) { return events.some(e => e.type === "ee"); }
function eeSeries(events, dim) {
  return events
    .filter(e => e.type === "ee" && Array.isArray(e.ee) && e.ee.length > dim)
    .map(e => ({ x: e.step, y: e.ee[dim] }));
}
// inference RTT vs event index (inference events carry no step).
function rttSeries(events) {
  let i = 0;
  return events.filter(e => e.type === "inference").map(e => ({ x: i++, y: e.rtt_ms }));
}
// summary tiles: mean RTT (ms), effective Hz (from action ts deltas), mean overlap.
function metricsSummary(events) {
  const rtts = events.filter(e => e.type === "inference").map(e => e.rtt_ms);
  const ots = events.filter(e => e.type === "overlap").map(e => e.count);
  const ts = events.filter(e => e.type === "action" && typeof e.ts === "number").map(e => e.ts);
  const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
  let hz = null;
  if (ts.length > 1) { const span = ts[ts.length - 1] - ts[0]; if (span > 0) hz = (ts.length - 1) / span; }
  return { rtt: mean(rtts), hz, overlap: mean(ots) };
}

async function refresh() {
  const ids = [...selected];
  if (ids.length < 1) {
    $("runs-diff").innerHTML = '<em class="muted">Select 1+ runs.</em>';
    $("runs-metrics").innerHTML = '<em class="muted">Select runs above.</em>';
    for (const c of [deltaChart, overlapChart, jointChart, rttChart, eeChart]) { if (c) c.destroy(); }
    deltaChart = overlapChart = jointChart = rttChart = eeChart = null;
    const eeCard = $("runs-ee-card"); if (eeCard) eeCard.style.display = "none";
    return;
  }
  const runs = await Promise.all(ids.map(async id => ({ id, ...(await fetchRun(id)) })));
  $("runs-diff").innerHTML = configDiff(runs);

  const jointIdx = Number($("runs-joint-sel").value || 0);
  const deltaData = runs.map(r => ({ label: r.id, points: deltaSeries(r.events) }));
  const overlapData = runs.map(r => ({ label: r.id, points: overlapSeries(r.events) }));
  const jointData = runs.map(r => ({ label: r.id, points: jointSeries(r.events, jointIdx) }));
  const rttData = runs.map(r => ({ label: r.id, points: rttSeries(r.events) }));

  if (deltaChart) deltaChart.destroy();
  if (overlapChart) overlapChart.destroy();
  if (jointChart) jointChart.destroy();
  if (rttChart) rttChart.destroy();
  deltaChart = makeSeriesChart($("runs-chart-delta"), deltaData);
  overlapChart = makeSeriesChart($("runs-chart-overlap"), overlapData);
  jointChart = makeSeriesChart($("runs-chart-joint"), jointData);
  rttChart = makeSeriesChart($("runs-chart-rtt"), rttData);

  $("runs-metrics").innerHTML = runs.map(r => {
    const m = metricsSummary(r.events);
    const f = (v, d) => v == null ? "—" : v.toFixed(d);
    return `<div class="runs-metric-row"><b>${escapeHtml(r.id)}</b>
      <span>RTT ${f(m.rtt, 1)} ms</span>
      <span>${f(m.hz, 1)} Hz</span>
      <span>overlap ${f(m.overlap, 2)}</span></div>`;
  }).join("");

  const eeRuns = runs.filter(r => hasEE(r.events));
  const eeCard = $("runs-ee-card");
  if (eeChart) { eeChart.destroy(); eeChart = null; }
  if (eeRuns.length) {
    eeCard.style.display = "";
    const eeDim = Number($("runs-ee-sel").value || 0);
    const eeData = eeRuns.map(r => ({ label: r.id, points: eeSeries(r.events, eeDim) }));
    eeChart = makeSeriesChart($("runs-chart-ee"), eeData);
  } else {
    eeCard.style.display = "none";
  }
}

loadList();
