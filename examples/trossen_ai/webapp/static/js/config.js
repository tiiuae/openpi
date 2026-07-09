import { api, apiPost } from "./api.js";

const GROUPS = [
  { title: "Connection", fields: [
    { key:"policy_host", label:"Policy server host", type:"text", def:"192.168.50.174",
      help:"IP/hostname of the OpenPI policy server." },
    { key:"policy_port", label:"Policy server port", type:"number", def:8800,
      help:"TCP port the policy server listens on." },
    { key:"control_freq", label:"Control rate (Hz)", type:"number", def:25,
      help:"Control steps per second the loop targets." },
    { key:"max_steps", label:"Max steps", type:"number", def:1000,
      help:"Episode ends after this many control steps." },
    { key:"connect_timeout", label:"Connect timeout (s)", type:"number", def:15,
      help:"Give up waiting for the policy server after this many seconds." },
  ]},
  { title: "Policy", fields: [
    { key:"model_name", label:"Model name / tag", type:"text", def:"",
      help:"Name this policy checkpoint (e.g. pi0-trossen-joint-v3) so recorded runs can be compared by model. Saved into each run's log." },
    { key:"adapter", label:"Action space", type:"select",
      options:[["joint","Joint"],["ee","End-effector"]], def:"joint",
      help:"Joint = raw 14-D joints. End-effector = EE poses solved to joints via IK." },
    { key:"task_prompt", label:"Task prompt", type:"textarea", rows:3,
      def:"Pick up the yellow object and place it in the fixed orange basket",
      help:"Natural-language instruction sent to the policy." },
  ]},
  { title: "Action smoothing", fields: [
    { key:"smoothing", label:"Action smoothing", type:"checkbox", def:true,
      help:"Blend overlapping predicted action chunks to smooth motion. Off = use the latest chunk's action directly." },
    { key:"smoothing_method", label:"Smoothing method", type:"select", def:"temporal",
      options:[["temporal","Temporal (exp-decay)"],["cogact","CogACT (consensus)"]],
      disabledWhen:(c)=>!c.smoothing,
      help:"How overlapping chunks are weighted. Temporal = exponential decay by age (ACT-style). CogACT = cosine-similarity consensus that down-weights outlier predictions (arXiv 2411.19650)." },
    { key:"smoothing_decay", label:"Smoothing decay", type:"number", step:0.1, def:1.0,
      disabledWhen:(c)=>!c.smoothing || c.smoothing_method === "cogact",
      help:"Temporal method only. Higher = trust older, already-committed predictions more (smoother, less reactive). 0 = plain average of overlaps." },
    { key:"cogact_mode", label:"CogACT mode", type:"select", def:"cogact",
      options:[["cogact","cogact (consensus)"],["latest","latest (anchor newest)"],["hybrid","hybrid"]],
      disabledWhen:(c)=>!c.smoothing || c.smoothing_method !== "cogact",
      help:"CogACT weighting variant. cogact = mean pairwise agreement; latest = agreement with the newest prediction; hybrid = blend of both." },
    { key:"rate_of_inference", label:"Inference interval (steps)", type:"number", def:20,
      disabledWhen:(c)=>c.async_inference,
      help:"Run a new policy inference every N control steps. Ignored with async inference (async queries every step)." },
    { key:"async_inference", label:"Async inference", type:"checkbox", def:false,
      disabledWhen:(c)=>!c.smoothing,
      help:"Run inference in a background thread. Requires smoothing on. Works in both Joint and EE mode (IK runs in the worker thread)." },
  ]},
  { title: "Arms", fields: [
    { key:"use_left_arm_only", label:"Left arm only", type:"checkbox", def:false,
      help:"Drive only the left arm; right arm holds." },
    { key:"use_right_arm_only", label:"Right arm only", type:"checkbox", def:false,
      help:"Drive only the right arm; left arm holds." },
  ]},
  { title: "IK (End-effector only)", eeOnly:true, fields: [
    { key:"ik_orientation_weight", label:"IK orientation weight", type:"number", step:0.01, def:0.01,
      help:"How strongly IK matches target orientation vs position." },
    { key:"ik_pos_tol_m", label:"IK position tolerance (m)", type:"number", step:0.001, def:0.001,
      help:"Acceptable IK position error in metres." },
    { key:"ik_max_joint_jump_deg", label:"Branch-flip guard (deg)", type:"number", step:5, def:0,
      help:"Reject any single IK solve whose joint angle jumps more than this vs the previous frame (a branch flip) and hold the last good pose. 0 disables. Catches chaotic flips that orientation weight can't. Start with a dry run before enabling on the robot." },
  ]},
  { title: "Motion tuning (Advanced)", fields: [
    { key:"smooth_streaming", label:"Smooth streaming (feed-forward velocity)", type:"checkbox", def:false,
      help:"Send feed-forward joint velocities so the arm carries momentum through waypoints — fixes jerky/stop-start motion." },
    { key:"min_time_to_move_multiplier", label:"Goal-time multiplier", type:"number", step:0.5, def:3.0,
      help:"Driver goal_time = multiplier / loop rate. Larger = smoother but laggier; smaller = snappier but jerkier." },
    { key:"loop_rate", label:"Driver loop rate (Hz)", type:"number", def:25,
      help:"Match this to Control rate to avoid mid-motion re-planning." },
    { key:"max_joint_speed", label:"Max joint speed (rad/s)", type:"number", step:0.5, def:3.0,
      help:"Safety cap on per-joint velocity. A policy/IK jump is spread over several control steps instead of tripping the firmware velocity limit (~9.4 rad/s). Lower = gentler; 0 disables." },
  ]},
];

const $ = (id) => document.getElementById(id);

export function renderConfig(container) {
  container.innerHTML =
    GROUPS.map(g => `<section class="card" data-group="${g.title}" ${g.eeOnly?'data-ee-only="1"':''}>
      <h2>${g.title}</h2>${g.fields.map(fieldHtml).join("")}</section>`).join("")
    + presetsHtml() + modeHtml();
  // Any field change can flip a dependent field's enabled state (e.g. async
  // depends on smoothing), so re-evaluate on every change.
  container.addEventListener("change", () => { updateEEVisibility(); updateDependencies(); });
  updateEEVisibility();
  updateDependencies();
  wirePresets();
}

// Enable/disable fields whose `disabledWhen(cfg)` predicate is true, dimming the
// row. Disabled checkboxes are forced off so an invalid combo (e.g. async with
// smoothing off) is never submitted.
function updateDependencies() {
  const cfg = readConfig();
  GROUPS.forEach(g => g.fields.forEach(f => {
    if (!f.disabledWhen) return;
    const el = $(`cf_${f.key}`); if (!el) return;
    const dis = !!f.disabledWhen(cfg);
    el.disabled = dis;
    if (dis && f.type === "checkbox" && el.checked) el.checked = false;
    const row = el.closest(".field-inline, .field-col, .checkbox-label");
    if (row) row.classList.toggle("row-disabled", dis);
  }));
}

function fieldHtml(f) {
  const help = `<span class="help-ic" data-help="${f.help.replace(/"/g,'&quot;')}">i</span>`;
  const id = `cf_${f.key}`;
  if (f.type === "checkbox")
    return `<label class="checkbox-label"><input type="checkbox" name="${f.key}" id="${id}">${f.label}${help}</label>`;
  if (f.type === "textarea")
    return `<div class="field-col"><label>${f.label}${help}</label>
      <textarea name="${f.key}" id="${id}" rows="${f.rows || 3}"></textarea></div>`;
  if (f.type === "select")
    return `<div class="field-inline"><label>${f.label}${help}</label>
      <select name="${f.key}" id="${id}">${f.options.map(([v,t])=>`<option value="${v}">${t}</option>`).join("")}</select></div>`;
  return `<div class="field-inline"><label>${f.label}${help}</label>
    <input type="${f.type}" name="${f.key}" id="${id}" ${f.step?`step="${f.step}"`:""}></div>`;
}

function modeHtml() {
  return `<section class="card run-card"><h2>Run</h2>
    <div class="run-mode-row"><label>Mode</label>
      <select id="mode-select"><option value="test">Dry run — no robot</option><option value="autonomous">Run on real robot</option></select></div>
    <div class="run-action-group">
      <span class="field-hint">Evaluation</span>
      <div class="status-strip">
        <span class="badge" id="badge-session">idle</span>
        <span class="badge" id="badge-rtt">rtt</span>
      </div>
      <div class="run-controls">
        <button class="btn-primary" id="btn-start">Start Live</button>
        <button class="btn-outline" id="btn-stop">Stop</button>
        <button class="btn-danger" id="btn-estop">E-STOP</button>
      </div>
    </div>
    <div class="run-action-group">
      <span class="field-hint">Robot pose</span>
      <div class="pose-controls">
        <button class="btn-outline" id="btn-home">Home</button>
        <button class="btn-outline" id="btn-sleep">Sleep</button>
      </div>
    </div></section>`;
}

function presetsHtml() {
  return `<section class="card"><h2>Presets</h2>
    <div class="field-inline"><select id="preset-select"></select><button class="btn-sm" id="preset-load">Load</button></div>
    <div class="field-inline"><input id="preset-name" placeholder="preset name"><button class="btn-sm" id="preset-save">Save</button><button class="btn-sm" id="preset-delete">Delete</button></div></section>`;
}

function defaults() { const d={}; GROUPS.forEach(g=>g.fields.forEach(f=>d[f.key]=f.def)); return d; }

export function readConfig() {
  const cfg = {};
  GROUPS.forEach(g => g.fields.forEach(f => {
    const el = $(`cf_${f.key}`); if (!el) return;
    cfg[f.key] = f.type === "checkbox" ? el.checked : el.value;
  }));
  cfg.mode = $("mode-select")?.value || "test";
  return cfg;
}

export function applyConfig(cfg) {
  GROUPS.forEach(g => g.fields.forEach(f => {
    const el = $(`cf_${f.key}`); if (!el || !(f.key in cfg)) return;
    if (f.type === "checkbox") el.checked = !!cfg[f.key]; else el.value = cfg[f.key];
  }));
  if (cfg.mode && $("mode-select")) $("mode-select").value = cfg.mode;
  updateEEVisibility();
  updateDependencies();
}

function updateEEVisibility() {
  const ee = $("cf_adapter")?.value === "ee";
  document.querySelectorAll('[data-ee-only="1"]').forEach(s => s.style.display = ee ? "" : "none");
}

export function setDefaults() { applyConfig(defaults()); }

async function refreshPresets() {
  const names = await api("/api/presets");
  const sel = $("preset-select"); if (!sel) return;
  sel.innerHTML = names.map(n => `<option value="${n}">${n}</option>`).join("");
}
function wirePresets() {
  $("preset-save").onclick = async () => {
    const name = $("preset-name").value.trim(); if (!name) return;
    await apiPost("/api/presets", { name, config: readConfig() }); refreshPresets();
  };
  $("preset-load").onclick = async () => {
    const name = $("preset-select").value; if (!name) return;
    applyConfig(await api("/api/presets/" + name));
  };
  $("preset-delete").onclick = async () => {
    const name = $("preset-select").value; if (!name) return;
    await fetch("/api/presets/" + name, { method: "DELETE" }); refreshPresets();
  };
  refreshPresets();
}
