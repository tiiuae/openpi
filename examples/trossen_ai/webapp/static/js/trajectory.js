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
Chart.register(cursorPlugin);

const FONT = { size: 9, family: "Consolas,Menlo,Monaco,monospace" };
function baseOpts(title) {
  return {
    animation: false, maintainAspectRatio: false, responsive: true, parsing: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { color: "#8b949e", boxWidth: 10, font: FONT }, position: "bottom" },
      title: { display: true, text: title, color: "#8b949e", font: FONT },
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

// Build the two preview charts from a /api/episode_trajectory payload.
// Returns { charts:[Chart], setCursor(frameIdx) }.
export function buildTrajectoryCharts(jointsCanvas, eeCanvas, data) {
  const N = data.n_frames;
  // Joints chart: 14 series, raw vs clamped. Spikes plotted on joint 0's clamped y.
  const jointSeries = JOINT_NAMES_14.map((label, j) => ({
    label,
    clamped: Array.from({ length: N }, (_, t) => data.joints_clamped[t][j]),
    raw: Array.from({ length: N }, (_, t) => data.joints_raw[t][j]),
  }));
  const spikePoints = data.spikes.map((t) => ({ x: t, y: data.joints_clamped[t][0] }));
  const jointsChart = buildChart(jointsCanvas, "Joint angles (rad)", jointSeries, spikePoints);

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

  const charts = [jointsChart, eeChart];
  return {
    charts,
    setCursor(frameIdx) {
      for (const c of charts) { c.$frameX = frameIdx; c.update("none"); }
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
    els.play.addEventListener("click", () => (this.playing ? this.pause() : this.play()));
    els.stop.addEventListener("click", () => this.stop());
    els.slider.addEventListener("input", (e) => { this.pause(); this.seek(+e.target.value); });
    this.seek(0);
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
