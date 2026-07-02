// webapp/static/js/runs.js
import { makeSeriesChart } from "./charts.js";

const $ = (id) => document.getElementById(id);
const selected = new Set();
let deltaChart = null, overlapChart = null;

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

async function refresh() {
  const ids = [...selected];
  if (ids.length < 2) {
    $("runs-diff").innerHTML = '<em class="muted">Select 2+ runs.</em>';
    return;
  }
  const runs = await Promise.all(ids.map(async id => ({ id, ...(await fetchRun(id)) })));
  $("runs-diff").innerHTML = configDiff(runs);

  const deltaData = runs.map(r => ({ label: r.id, points: deltaSeries(r.events) }));
  const overlapData = runs.map(r => ({ label: r.id, points: overlapSeries(r.events) }));
  if (deltaChart) deltaChart.destroy();
  if (overlapChart) overlapChart.destroy();
  deltaChart = makeSeriesChart($("runs-chart-delta"), deltaData);
  overlapChart = makeSeriesChart($("runs-chart-overlap"), overlapData);
}

loadList();
