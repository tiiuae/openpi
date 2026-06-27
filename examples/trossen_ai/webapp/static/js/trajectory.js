// Static trajectory charts + media-player transport for the Replay preview.
import { PALETTE, JOINT_NAMES_14 } from "./charts.js";

// Chart.js plugin: a vertical cursor line at chart.$frameX (data x value).
const cursorPlugin = {
  id: "frameCursor",
  afterDraw(chart) {
    const fx = chart.$frameX;
    if (fx == null) return;
    const x = chart.scales.x.getPixelForValue(fx);
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, top); ctx.lineTo(x, bottom);
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#ff5555";
    ctx.stroke();
    ctx.restore();
  },
};

// Chart.js plugin: faint vertical lines at every velocity-spike frame in
// chart.$spikes (a list of x data values). Marks exactly where the decoded
// joint trajectory exceeds the velocity limit, full chart height.
const spikePlugin = {
  id: "spikeMarkers",
  beforeDatasetsDraw(chart) {
    const spikes = chart.$spikes;
    if (!spikes || !spikes.length) return;
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(255,85,85,0.35)";
    for (const t of spikes) {
      const x = chart.scales.x.getPixelForValue(t);
      ctx.beginPath();
      ctx.moveTo(x, top); ctx.lineTo(x, bottom);
      ctx.stroke();
    }
    ctx.restore();
  },
};
if (typeof Chart !== "undefined") Chart.register(cursorPlugin, spikePlugin);

// Pin the hover tooltip to the chart's top-left corner instead of letting it
// follow the cursor. With many series the default floating box lands in the
// middle of the plot and hides the data underneath; a fixed corner keeps the
// readout visible without covering the trace.
if (typeof Chart !== "undefined" && Chart.Tooltip) {
  Chart.Tooltip.positioners.corner = function () {
    const area = this.chart.chartArea;
    return { x: area.left + 6, y: area.top + 6 };
  };
}

const FONT = { size: 9, family: "Consolas,Menlo,Monaco,monospace" };
function baseOpts(title) {
  return {
    animation: false, maintainAspectRatio: false, responsive: true, parsing: false,
    // Nearest single-point hover: an index tooltip over 14×2 series renders a
    // giant box that covers the whole chart. Nearest shows just the series
    // under the cursor, so the readout stays small and out of the way.
    interaction: { mode: "nearest", axis: "x", intersect: false },
    plugins: {
      legend: { labels: { color: "#8b949e", boxWidth: 10, font: FONT }, position: "bottom" },
      title: { display: true, text: title, color: "#8b949e", font: FONT },
      tooltip: {
        position: "corner",  // pinned top-left; never covers the plotted trace
        backgroundColor: "#161b22", borderColor: "#30363d", borderWidth: 1,
        titleColor: "#8b949e", bodyColor: "#e6edf3", bodyFont: FONT, caretPadding: 8,
      },
      // Wheel-zoom + drag-pan on the time axis (matches the Live charts).
      zoom: {
        pan: { enabled: true, mode: "x" },
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "x" },
      },
    },
    scales: {
      x: { type: "linear", ticks: { color: "#484f58", font: FONT }, grid: { color: "#21262d" } },
      y: { ticks: { color: "#484f58", font: FONT }, grid: { color: "#21262d" } },
    },
  };
}

// series: [{label, raw:[y], clamped:[y]}]; spikeSeries: [{x,y}] or null.
function buildChart(canvas, title, series, spikeSeries) {
  const datasets = [];
  series.forEach((s, i) => {
    const color = PALETTE[i % PALETTE.length];
    datasets.push({
      label: s.label, borderColor: color, borderWidth: 1.5, pointRadius: 0, tension: 0,
      data: s.clamped.map((y, x) => ({ x, y })),
    });
    datasets.push({
      label: `${s.label} (raw)`, borderColor: color, borderWidth: 1, pointRadius: 0,
      borderDash: [4, 3], tension: 0,
      data: s.raw.map((y, x) => ({ x, y })),
    });
  });
  if (spikeSeries) {
    datasets.push({
      label: "⚠ spike", type: "scatter", showLine: false,
      pointRadius: 4, pointStyle: "triangle",
      backgroundColor: "#ff5555", borderColor: "#ff5555",
      data: spikeSeries,
    });
  }
  return new Chart(canvas, { type: "line", data: { datasets }, options: baseOpts(title) });
}

// Joint-velocity chart: 14 raw per-step speed series, a dashed horizontal limit
// line at max_joint_speed, and a triangle marker on each spike frame (placed at
// that frame's peak joint speed). Vertical spike markers come from $spikes.
function buildVelocityChart(canvas, data) {
  const N = data.n_frames;
  const limit = data.max_joint_speed;
  const datasets = JOINT_NAMES_14.map((label, j) => ({
    label, borderColor: PALETTE[j % PALETTE.length], borderWidth: 1, pointRadius: 0, tension: 0,
    data: Array.from({ length: N }, (_, t) => ({ x: t, y: data.velocity[t][j] })),
  }));
  // Dashed limit line spanning the full time axis.
  datasets.push({
    label: `limit ${limit}`, borderColor: "#ff5555", borderWidth: 1.5,
    borderDash: [6, 4], pointRadius: 0,
    data: [{ x: 0, y: limit }, { x: Math.max(0, N - 1), y: limit }],
  });
  // One triangle per spike frame, sitting at that frame's worst joint speed.
  const spikePoints = data.spikes.map((t) => ({
    x: t, y: Math.max(...data.velocity[t]),
  }));
  if (spikePoints.length) {
    datasets.push({
      label: "⚠ spike", type: "scatter", showLine: false,
      pointRadius: 4, pointStyle: "triangle",
      backgroundColor: "#ff5555", borderColor: "#ff5555",
      data: spikePoints,
    });
  }
  const chart = new Chart(canvas, { type: "line", data: { datasets }, options: baseOpts("Joint velocity (rad/s)") });
  chart.$spikes = data.spikes;  // vertical spike-marker plugin
  return chart;
}

// Build the preview charts from a /api/episode_trajectory payload.
// Returns { charts:[Chart], setCursor(frameIdx) }.
export function buildTrajectoryCharts(jointsCanvas, velCanvas, eeCanvas, data) {
  if (typeof Chart !== "undefined") Chart.register(cursorPlugin, spikePlugin);
  const N = data.n_frames;
  // Joints chart: 14 series, raw vs clamped. Spikes plotted on joint 0's clamped y.
  const jointSeries = JOINT_NAMES_14.map((label, j) => ({
    label,
    clamped: Array.from({ length: N }, (_, t) => data.joints_clamped[t][j]),
    raw: Array.from({ length: N }, (_, t) => data.joints_raw[t][j]),
  }));
  const spikePoints = data.spikes.map((t) => ({ x: t, y: data.joints_clamped[t][0] }));
  const jointsChart = buildChart(jointsCanvas, "Joint angles (rad)", jointSeries, spikePoints);
  jointsChart.$spikes = data.spikes;  // drives the vertical spike-marker plugin

  // EE position chart: 6 series (L/R × x/y/z). Recorded EE, so clamped == raw.
  const axes = ["x", "y", "z"];
  const eeSeries = [];
  for (const arm of ["left", "right"]) {
    axes.forEach((ax, k) => {
      const vals = Array.from({ length: N }, (_, t) => data.ee_pos[arm][t][k]);
      eeSeries.push({ label: `${arm} ${ax}`, clamped: vals, raw: vals });
    });
  }
  const eeChart = buildChart(eeCanvas, "EE position (m)", eeSeries, null);

  const velChart = buildVelocityChart(velCanvas, data);

  const charts = [jointsChart, velChart, eeChart];

  // Mirror zoom/pan across all charts: a wheel-zoom or drag-pan on any one
  // applies the same x-window (time axis) to the others, so the joint, velocity
  // and EE traces stay aligned. `syncing` guards against the re-entrancy that
  // would otherwise loop (applying to others fires their complete callbacks).
  let syncing = false;
  const syncX = (src) => {
    if (syncing) return;
    syncing = true;
    const { min, max } = src.scales.x;
    for (const c of charts) {
      if (c === src) continue;
      c.options.scales.x.min = min;
      c.options.scales.x.max = max;
      c.update("none");
    }
    syncing = false;
  };
  for (const c of charts) {
    const z = c.options.plugins.zoom;
    z.zoom.onZoomComplete = ({ chart }) => syncX(chart);
    z.pan.onPanComplete = ({ chart }) => syncX(chart);
  }

  return {
    charts,
    setCursor(frameIdx) {
      for (const c of charts) { c.$frameX = frameIdx; c.update("none"); }
    },
    // Reset every chart back to the full, un-zoomed time window.
    resetZoom() {
      for (const c of charts) {
        c.options.scales.x.min = undefined;
        c.options.scales.x.max = undefined;
        c.resetZoom();
      }
    },
  };
}

// Media-player transport over a fixed-length trajectory.
// onFrame(i) is called for every frame change (drives 3D + chart cursor + readout).
export class Transport {
  constructor({ nFrames, fps, els, onFrame }) {
    this.n = nFrames;
    this.fps = fps || 30;
    this.els = els;            // { play, stop, slider, readout, speed }
    this.onFrame = onFrame;
    this.idx = 0;
    this.playing = false;
    this.timer = null;

    els.slider.min = 0; els.slider.max = Math.max(0, nFrames - 1); els.slider.value = 0;
    this._onPlay = () => (this.playing ? this.pause() : this.play());
    this._onStop = () => this.stop();
    this._onSlider = (e) => { this.pause(); this.seek(+e.target.value); };
    els.play.addEventListener("click", this._onPlay);
    els.stop.addEventListener("click", this._onStop);
    els.slider.addEventListener("input", this._onSlider);
    this.seek(0);
  }

  destroy() {
    this.pause();
    this.els.play.removeEventListener("click", this._onPlay);
    this.els.stop.removeEventListener("click", this._onStop);
    this.els.slider.removeEventListener("input", this._onSlider);
  }

  _fmt(i) {
    const secs = i / this.fps;
    const m = Math.floor(secs / 60), s = Math.floor(secs % 60);
    return `${i + 1} / ${this.n} · ${m}:${String(s).padStart(2, "0")}`;
  }

  seek(i) {
    this.idx = Math.max(0, Math.min(i, this.n - 1));
    this.els.slider.value = this.idx;
    this.els.readout.textContent = this._fmt(this.idx);
    this.onFrame(this.idx);
  }

  play() {
    if (this.playing || this.n === 0) return;
    this.playing = true;
    this.els.play.textContent = "⏸ Pause";
    const step = () => {
      if (!this.playing) return;
      if (this.idx >= this.n - 1) { this.pause(); return; }
      this.seek(this.idx + 1);
      const speed = parseFloat(this.els.speed?.value || "1");
      this.timer = setTimeout(step, 1000 / (this.fps * speed));
    };
    step();
  }

  pause() {
    this.playing = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    this.els.play.textContent = "▶ Play";
  }

  stop() { this.pause(); this.seek(0); }
}
