// Wizard-style charts. makeSeriesChart: full-series. LiveChart: streaming (bounded).
export const PALETTE = [
  '#4a9eff','#ff5555','#50fa7b','#ffb86c','#bd93f9','#ff79c6','#8be9fd','#f1fa8c',
  '#ff6e6e','#5af78e','#caa9fa','#ffca6a','#1dc9a4','#ff92d0','#6272a4','#44bc9f',
];
const FONT = { size: 9, family: 'Consolas,Menlo,Monaco,monospace' };
const TICK = { color: '#484f58', font: FONT }, GRID = { color: '#21262d' };

function baseOptions() {
  return { animation:false, maintainAspectRatio:false, responsive:true, parsing:false,
    interaction:{ mode:'index', intersect:false },
    plugins:{ legend:{ labels:{ color:'#8b949e', boxWidth:10, font:FONT }, position:'bottom' },
      tooltip:{ backgroundColor:'#161b22', borderColor:'#30363d', borderWidth:1,
        titleColor:'#8b949e', bodyColor:'#e6edf3', bodyFont:FONT },
      zoom:{ pan:{ enabled:true, mode:'x' }, zoom:{ wheel:{ enabled:true }, mode:'x' } } },
    scales:{ x:{ type:'linear', ticks:TICK, grid:GRID }, y:{ ticks:TICK, grid:GRID } } };
}

// Full-series chart (series: [{label, points:[{x,y}]}]).
export function makeSeriesChart(canvas, series) {
  return new Chart(canvas, {
    type:'line',
    data:{ datasets: series.map((s,i)=>({ label:s.label, data:s.points,
      borderColor: PALETTE[i%PALETTE.length], borderWidth:1.5, pointRadius:0, tension:0 })) },
    options: baseOptions(),
  });
}

// Streaming chart: push(seriesIdx, x, y), bounded to maxPts.
export class LiveChart {
  constructor(canvas, labels, maxPts = 300) {
    this.maxPts = maxPts;
    this.chart = new Chart(canvas, {
      type:'line',
      data:{ datasets: labels.map((l,i)=>({ label:l, data:[],
        borderColor: PALETTE[i%PALETTE.length], borderWidth:1, pointRadius:0, tension:0 })) },
      options: baseOptions(),
    });
  }
  push(idx, x, y) {
    const d = this.chart.data.datasets[idx].data;
    d.push({ x, y }); if (d.length > this.maxPts) d.shift();
  }
  update() { this.chart.update('none'); }
}

// Split 14-D joint names into Left/Right groups for grouped charts.
export function splitArms(names) {
  const left = [], right = [];
  names.forEach((n,i)=> (n.startsWith('left_') ? left : right).push({ i, name:n.replace(/^(left_|right_)/,'') }));
  return { left, right };
}
