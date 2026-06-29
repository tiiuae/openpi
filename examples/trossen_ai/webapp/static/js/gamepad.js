// Reads the Gamepad API and keyboard into a normalized teleop input snapshot.
// axes6 = [tx, ty, tz, roll, pitch, yaw] in [-1,1]; grip in [-1,1] (+ open).
// Edges (switch_arm/go_home/go_sleep) latch until consumed by readEdges().
const keys = new Set();
window.addEventListener("keydown", (e) => { keys.add(e.key.toLowerCase()); if (e.key === "Tab") e.preventDefault(); });
window.addEventListener("keyup",   (e) => keys.delete(e.key.toLowerCase()));

let edges = { switch_arm: false, go_home: false, go_sleep: false };
let prevPadButtons = [];
let prevTabKey = false, prevHomeKey = false, prevSleepKey = false;

const ax = (k) => (keys.has(k) ? 1 : 0);
const clamp = (v) => Math.max(-1, Math.min(1, v));

export function hasGamepad() { return [...navigator.getGamepads()].some((p) => p); }

export function readInput(deadzone) {
  const pad = [...navigator.getGamepads()].find((p) => p) || null;
  const dz = (v) => (Math.abs(v) < deadzone ? 0 : v);
  let lsx = 0, lsy = 0, rsx = 0, rsy = 0, lt = 0, rt = 0;
  let zUp = ax("q"), zDown = ax("e"), gOpen = ax("c"), gClose = ax("z");
  let roll = ax("u") - ax("o");

  if (pad) {
    lsx = dz(pad.axes[0] || 0); lsy = dz(pad.axes[1] || 0);
    rsx = dz(pad.axes[2] || 0); rsy = dz(pad.axes[3] || 0);
    lt = pad.buttons[6]?.value || 0; rt = pad.buttons[7]?.value || 0;
    zUp = zUp || (pad.buttons[5]?.pressed ? 1 : 0);   // RB
    zDown = zDown || (pad.buttons[4]?.pressed ? 1 : 0); // LB
    gClose = gClose || (pad.buttons[0]?.pressed ? 1 : 0); // A
    gOpen = gOpen || (pad.buttons[1]?.pressed ? 1 : 0);   // B
    roll += rt - lt;
    const btn = (i) => pad.buttons[i]?.pressed;
    if (btn(2) && !prevPadButtons[2]) edges.switch_arm = true;   // X
    if (btn(12) && !prevPadButtons[12]) edges.go_home = true;    // D-up
    if (btn(13) && !prevPadButtons[13]) edges.go_sleep = true;   // D-down
    prevPadButtons = pad.buttons.map((b) => b.pressed);
  }
  const tab = keys.has("tab"), h = keys.has("h"), p = keys.has("p");
  if (tab && !prevTabKey) edges.switch_arm = true;
  if (h && !prevHomeKey) edges.go_home = true;
  if (p && !prevSleepKey) edges.go_sleep = true;
  prevTabKey = tab; prevHomeKey = h; prevSleepKey = p;

  const tx = dz(-lsy) + (ax("w") - ax("s"));
  const ty = dz(lsx) + (ax("d") - ax("a"));
  const tz = (zUp ? 1 : 0) - (zDown ? 1 : 0);
  const pitch = dz(-rsy) + (ax("i") - ax("k"));
  const yaw = dz(rsx) + (ax("j") - ax("l"));
  const grip = (gOpen ? 1 : 0) - (gClose ? 1 : 0);

  return {
    axes: [clamp(tx), clamp(ty), clamp(tz), clamp(roll), clamp(pitch), clamp(yaw)],
    grip: clamp(grip),
    hasPad: !!pad,
  };
}

export function readEdges() {
  const e = edges;
  edges = { switch_arm: false, go_home: false, go_sleep: false };
  return e;
}
