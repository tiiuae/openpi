// webapp/static/js/runs.js
import { makeSeriesChart, JOINT_NAMES_14 } from "./charts.js";

const $ = (id) => document.getElementById(id);
const selected = new Set();
let deltaChart = null, overlapChart = null, jointChart = null, rttChart = null;

async function loadList() {
  const runs = await (await fetch("/api/runs")).json();
  const box = $("runs-list");
  if (!runs.length) { box.innerHTML = '<em class="muted">No runs recorded yet.</em>'; return; }
  box.innerHTML = runs.map(r => `
    <label class="run-item">
      <input type="checkbox" value="${r.run_id}">
      <span class="run-id">${r.run_id}</span>
      <span class="run-model">${r.model_name || "—"}</span>
      <span class="run-metric">${r.steps ?? "—"} steps</span>
      <span class="run-metric">${r.end_reason}</span>
      <span class="run-metric">${r.rating ? r.rating.success : "unrated"}</span>
    </label>`).join("");
  box.querySelectorAll("input[type=checkbox]").forEach(cb =>
    cb.addEventListener("change", () => {
      cb.checked ? selected.add(cb.value) : selected.delete(cb.value);
      refresh();
    }));

  const sel = $("runs-joint-sel");
  if (sel && !sel.options.length) {
    sel.innerHTML = JOINT_NAMES_14.map((n, i) => `<option value="${i}">${n}</option>`).join("");
    sel.addEventListener("change", refresh);
  }
}

async function fetchRun(id) {
  return (await fetch(`/api/runs/${id}?events=1`)).json();
}

function configDiff(runs) {
  // runs: [{id, manifest}]. Build union of config keys; highlight rows that differ.
  const keys = new Set();
  runs.forEach(r => Object.keys(r.manifest.config || {}).forEach(k => keys.add(k)));
  const rows = [...keys].sort().map(k => {
    const vals = runs.map(r => String((r.manifest.config || {})[k] ?? ""));
    const differ = new Set(vals).size > 1;
    return `<tr class="${differ ? "diff" : ""}"><td>${k}</td>${vals.map(v => `<td>${v}</td>`).join("")}</tr>`;
  }).join("");
  return `<table class="runs-diff"><thead><tr><th>key</th>${
    runs.map(r => `<th>${r.id}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
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
    return `<div class="runs-metric-row"><b>${r.id}</b>
      <span>RTT ${f(m.rtt, 1)} ms</span>
      <span>${f(m.hz, 1)} Hz</span>
      <span>overlap ${f(m.overlap, 2)}</span></div>`;
  }).join("");
}

loadList();
