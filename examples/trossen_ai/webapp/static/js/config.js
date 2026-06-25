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
    { key:"adapter", label:"Action space", type:"select",
      options:[["joint","Joint"],["ee","End-effector"]], def:"joint",
      help:"Joint = raw 14-D joints. End-effector = EE poses solved to joints via IK." },
    { key:"task_prompt", label:"Task prompt", type:"text", def:"move the arm to the left",
      help:"Natural-language instruction sent to the policy." },
    { key:"starvla", label:"StarVLA input mode (224×224 RGB)", type:"checkbox", def:false,
      help:"Preprocess camera images to 224×224 RGB (PIL) for StarVLA-family policies. Off uses the default training resize." },
  ]},
  { title: "Action smoothing", fields: [
    { key:"ensemble_type", label:"Action smoothing", type:"select",
      options:[["exp","Exponential"],["cogact","CogACT"],["none","None"]], def:"exp",
      help:"Blends overlapping predicted action chunks. Exponential = recency-weighted; CogACT = learned weighting; None = latest only." },
    { key:"cogact_mode", label:"CogACT blend mode", type:"select",
      options:[["cogact","cogact"],["latest","latest"],["hybrid","hybrid"]], def:"cogact",
      help:"Weighting variant used only when smoothing = CogACT." },
    { key:"rate_of_inference", label:"Inference interval (steps)", type:"number", def:20,
      help:"Run a new policy inference every N control steps." },
    { key:"async_inference", label:"Async inference", type:"checkbox", def:false,
      help:"Run inference in a background thread (requires a smoothing method other than None)." },
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
  ]},
  { title: "Motion tuning (Advanced)", fields: [
    { key:"smooth_streaming", label:"Smooth streaming (feed-forward velocity)", type:"checkbox", def:false,
      help:"Send feed-forward joint velocities so the arm carries momentum through waypoints — fixes jerky/stop-start motion." },
    { key:"min_time_to_move_multiplier", label:"Goal-time multiplier", type:"number", step:0.5, def:3.0,
      help:"Driver goal_time = multiplier / loop rate. Larger = smoother but laggier; smaller = snappier but jerkier." },
    { key:"loop_rate", label:"Driver loop rate (Hz)", type:"number", def:25,
      help:"Match this to Control rate to avoid mid-motion re-planning." },
  ]},
];

const $ = (id) => document.getElementById(id);

export function renderConfig(container) {
  container.innerHTML =
    GROUPS.map(g => `<section class="card" data-group="${g.title}" ${g.eeOnly?'data-ee-only="1"':''}>
      <h2>${g.title}</h2>${g.fields.map(fieldHtml).join("")}</section>`).join("")
    + presetsHtml() + modeHtml();
  $("cf_adapter")?.addEventListener("change", updateEEVisibility);
  updateEEVisibility();
  wirePresets();
}

function fieldHtml(f) {
  const help = `<span class="help-ic" data-help="${f.help.replace(/"/g,'&quot;')}">i</span>`;
  const id = `cf_${f.key}`;
  if (f.type === "checkbox")
    return `<label class="checkbox-label"><input type="checkbox" name="${f.key}" id="${id}">${f.label}${help}</label>`;
  if (f.type === "select")
    return `<div class="field-inline"><label>${f.label}${help}</label>
      <select name="${f.key}" id="${id}">${f.options.map(([v,t])=>`<option value="${v}">${t}</option>`).join("")}</select></div>`;
  return `<div class="field-inline"><label>${f.label}${help}</label>
    <input type="${f.type}" name="${f.key}" id="${id}" ${f.step?`step="${f.step}"`:""}></div>`;
}

function modeHtml() {
  return `<section class="card"><h2>Run</h2>
    <div class="field-inline"><label>Mode</label>
      <select id="mode-select"><option value="test">test (no movement)</option><option value="autonomous">autonomous</option></select></div>
    <div class="run-controls">
      <button class="btn-primary" id="btn-start">Start Live</button>
      <button class="btn-outline" id="btn-stop">Stop</button>
      <button class="btn-danger" id="btn-estop">E-STOP</button>
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
