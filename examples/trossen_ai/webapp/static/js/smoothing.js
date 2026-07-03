// Action-smoothing visualization — pure SVG/DOM, no chart.js.
//
// The temporal ensemble blends several overlapping predicted chunks into one
// commanded action. This box shows, live:
//   • Smoothing Δ  — ‖smoothed − raw‖ over the 14 joints: how far the blended
//                    command sits from the newest raw prediction (how much work
//                    the ensemble is doing). 0 = no smoothing / single chunk.
//   • Accumulation — overlap count: how many chunks predict the current step
//                    (buffer depth feeding the blend).
//   • Blend weights — the ensemble weight per accumulated chunk (oldest→newest),
//                    as a stacked bar, so you can see how the blend is split.
//
// Data comes straight off the telemetry stream:
//   action  → { step, action:[14], raw:[14]|null }   (raw = newest chunk @ step)
//   overlap → { step, count }
//   weights → { step, weights:[N] }                   (oldest-query first)

const NS = "http://www.w3.org/2000/svg";
const MAXPTS = 240;

// Kept in sync with charts.js PALETTE so blend segments match the joint charts.
const PALETTE = [
  "#4a9eff", "#ff5555", "#50fa7b", "#ffb86c", "#bd93f9", "#ff79c6", "#8be9fd", "#f1fa8c",
  "#ff6e6e", "#5af78e", "#caa9fa", "#ffca6a", "#1dc9a4", "#ff92d0", "#6272a4", "#44bc9f",
];

function el(tag, attrs) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

// Fixed-viewBox sparkline. Rebuilds one <polyline> (+ filled area) from `vals`.
class Spark {
  constructor(host, color, { w = 300, h = 40 } = {}) {
    this.w = w; this.h = h; this.vals = [];
    this.svg = el("svg", { viewBox: `0 0 ${w} ${h}`, class: "sm-spark", preserveAspectRatio: "none" });
    this.area = el("polyline", { fill: color, "fill-opacity": "0.12", stroke: "none", points: "" });
    this.line = el("polyline", { fill: "none", stroke: color, "stroke-width": "1.5",
      "stroke-linejoin": "round", "stroke-linecap": "round", points: "" });
    this.svg.append(this.area, this.line);
    host.appendChild(this.svg);
  }
  push(v) { this.vals.push(Number(v) || 0); if (this.vals.length > MAXPTS) this.vals.shift(); }
  draw() {
    const n = this.vals.length;
    if (n < 2) { this.line.setAttribute("points", ""); this.area.setAttribute("points", ""); return; }
    let lo = Math.min(...this.vals), hi = Math.max(...this.vals);
    if (hi - lo < 1e-9) { hi = lo + 1; }           // flat series → don't divide by 0
    const pad = 3, span = this.h - 2 * pad;
    const pts = this.vals.map((v, i) => {
      const x = (i / (n - 1)) * this.w;
      const y = this.h - pad - ((v - lo) / (hi - lo)) * span;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    this.line.setAttribute("points", pts.join(" "));
    this.area.setAttribute("points", `0,${this.h} ${pts.join(" ")} ${this.w},${this.h}`);
  }
}

const l2 = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - (b ? b[i] : 0); s += d * d; }
  return Math.sqrt(s);
};

// Live "chunk overlap map" — the diagram version of what the ensemble does.
// Rows = the recent predicted chunks (one policy inference each). Columns =
// control steps. A chunk queried at step qs owns cells at steps qs … qs+len-1;
// cell a[qs,k] is that chunk's prediction for step qs+k. Opacity fades with k
// (exp(-λk)) so "just predicted" reads bold and "predicted long ago" reads
// faint — the recency weighting. The current step column is highlighted; the
// stack of lit cells in that column is exactly the overlap the ensemble blends.
class ChunkGrid {
  constructor(host, { cols = 25, maxRows = 10, lambda = 0.35 } = {}) {
    this.host = host; this.cols = cols; this.maxRows = maxRows; this.lambda = lambda;
    this.chunks = [];      // [{qs, len, color}]  newest last
    this.curStep = 0; this.ci = 0;
  }
  reset() {
    this.chunks = []; this.ci = 0; this.curStep = 0;
    this.host.innerHTML = '<div class="cg-empty">Waiting for action chunks…</div>';
  }
  addChunk(qs, len) {
    this.chunks.push({ qs, len, color: PALETTE[this.ci % PALETTE.length] });
    this.ci++;
    if (this.chunks.length > 32) this.chunks.shift();
  }
  setStep(s) { this.curStep = s; }
  draw() {
    // Newest maxRows chunks (the ones actually feeding the current blend).
    const rows = this.chunks.slice(-this.maxRows);
    if (!rows.length) { this.host.innerHTML = '<div class="cg-empty">Waiting for action chunks…</div>'; return; }
    // Window: current step sits a few columns from the right so upcoming
    // (still-unconsumed) predictions stay visible on the right edge.
    const winStart = Math.max(0, this.curStep - (this.cols - 3));
    const steps = Array.from({ length: this.cols }, (_, i) => winStart + i);

    const head = `<div class="cg-rowlbl cg-corner"></div>` +
      steps.map(s => `<div class="cg-head${s === this.curStep ? " cg-now" : ""}">${s}</div>`).join("");

    const body = rows.map((ch) => {
      const rowlbl = `<div class="cg-rowlbl">chunk&nbsp;@&nbsp;${ch.qs}</div>`;
      const cells = steps.map((s) => {
        const k = s - ch.qs;
        const nowCls = s === this.curStep ? " cg-col-now" : "";
        if (k < 0 || k >= ch.len) return `<div class="cg-cell cg-off${nowCls}"></div>`;
        const w = Math.exp(-this.lambda * k);                 // recency weight
        return `<div class="cg-cell cg-fill${nowCls}" title="a[${ch.qs},${k}] → step ${s} (weight≈${w.toFixed(2)})" `
          + `style="background:${ch.color};opacity:${(0.18 + 0.82 * w).toFixed(3)}">a[${ch.qs},${k}]</div>`;
      }).join("");
      return rowlbl + cells;
    }).join("");

    // Fixed-width columns (not 1fr) so the full 25-step map keeps its size and
    // the container scrolls horizontally instead of squashing cells to nothing.
    this.host.style.gridTemplateColumns = `max-content repeat(${this.cols}, 34px)`;
    this.host.innerHTML = head + body;
  }
}

const BOX_HELP = "Live view of the temporal ensemble. One policy inference returns a " +
  "whole CHUNK of future actions; the loop re-queries before the old chunk runs out, so several " +
  "chunks predict the same step. The ensemble blends those overlapping predictions into one " +
  "command. Smoothing Δ = how far the blend sits from the newest raw prediction (how much work " +
  "smoothing does). Accumulation = how many chunks overlap at the current step. Blend weights = " +
  "the share each overlapping chunk gets. The overlap map below shows it as a grid.";

export function setupSmoothing(container) {
  if (!container) return { onAction() {}, onOverlap() {}, onWeights() {}, onChunk() {}, update() {}, reset() {} };
  container.innerHTML = `
    <div class="sm-head">Temporal ensemble<span class="help-ic" data-help="${BOX_HELP.replace(/"/g, "&quot;")}">i</span></div>
    <div class="sm-stats">
      <div class="sm-stat"><div class="sm-val" id="sm-delta">—</div>
        <div class="sm-lbl">Smoothing Δ<span class="sm-sub">‖blended − raw‖ (rad)</span></div></div>
      <div class="sm-stat"><div class="sm-val" id="sm-accum">—</div>
        <div class="sm-lbl">Accumulation<span class="sm-sub">chunks blended @ step</span></div></div>
    </div>
    <div class="sm-row"><span class="sm-cap">Smoothing Δ over time</span><div id="sm-spark-delta" class="sm-sparkbox"></div></div>
    <div class="sm-row"><span class="sm-cap">Accumulation depth</span><div id="sm-spark-accum" class="sm-sparkbox"></div></div>
    <div class="sm-row"><span class="sm-cap">Blend weights <em>(oldest→newest)</em></span><div id="sm-weights" class="sm-wbar"></div></div>
    <div class="sm-row"><span class="sm-cap">Chunk overlap map <em>(rows = chunks · columns = steps · fade = recency)</em></span>
      <div id="sm-grid" class="cg-grid"></div></div>
    <div class="sm-note" id="sm-note">Waiting for actions… (needs temporal smoothing ON)</div>`;

  const $ = (id) => container.querySelector("#" + id);
  const sparkDelta = new Spark($("sm-spark-delta"), "#4a9eff");
  const sparkAccum = new Spark($("sm-spark-accum"), "#50fa7b");
  const grid = new ChunkGrid($("sm-grid"));
  const deltaWin = [];      // recent Δ for the mean readout
  let lastDelta = null, lastAccum = null, gotAny = false;

  return {
    // Wipe all live-smoothing state so a new run starts clean instead of
    // continuing from where the last one stopped.
    reset() {
      grid.reset();
      sparkDelta.vals = []; sparkAccum.vals = [];
      deltaWin.length = 0;
      lastDelta = null; lastAccum = null; gotAny = false;
      $("sm-delta").textContent = "—"; $("sm-accum").textContent = "—";
      $("sm-accum").className = "sm-val";
      $("sm-weights").innerHTML = "";
      sparkDelta.draw(); sparkAccum.draw();
      const note = $("sm-note"); if (note) note.style.display = "";
    },
    onAction(e) {
      gotAny = true;
      grid.setStep(e.step);
      if (Array.isArray(e.raw) && Array.isArray(e.action) && e.raw.length === e.action.length) {
        const d = l2(e.action, e.raw);
        lastDelta = d;
        sparkDelta.push(d);
        deltaWin.push(d); if (deltaWin.length > MAXPTS) deltaWin.shift();
      } else {
        // No raw prediction for this step (smoothing off, or async warm-up) →
        // there is nothing to smooth against; hold the trace at 0.
        sparkDelta.push(0);
      }
    },
    onChunk(e) { grid.addChunk(e.query_step, e.len); },
    onOverlap(e) { lastAccum = e.count; sparkAccum.push(e.count); },
    onWeights(e) {
      const w = e.weights || [];
      const bar = $("sm-weights");
      if (!w.length) { bar.innerHTML = ""; return; }
      const total = w.reduce((a, b) => a + b, 0) || 1;
      bar.innerHTML = w.map((wi, i) => {
        const pct = (wi / total) * 100;
        const col = PALETTE[i % PALETTE.length];
        const lbl = pct >= 12 ? `${pct.toFixed(0)}%` : "";
        return `<span class="sm-wseg" style="width:${pct}%;background:${col}" title="chunk ${i}: ${pct.toFixed(1)}%">${lbl}</span>`;
      }).join("");
    },
    update() {
      if (lastDelta != null) {
        const mean = deltaWin.reduce((a, b) => a + b, 0) / (deltaWin.length || 1);
        $("sm-delta").textContent = lastDelta.toFixed(3);
        $("sm-delta").title = `mean ${mean.toFixed(3)} rad over last ${deltaWin.length}`;
      }
      if (lastAccum != null) {
        $("sm-accum").textContent = String(lastAccum);
        $("sm-accum").className = "sm-val " + (lastAccum >= 3 ? "ok" : lastAccum <= 1 ? "warn" : "");
      }
      const note = $("sm-note");
      if (note) note.style.display = (gotAny && lastDelta != null) ? "none" : "";
      sparkDelta.draw();
      sparkAccum.draw();
      grid.draw();
    },
  };
}
