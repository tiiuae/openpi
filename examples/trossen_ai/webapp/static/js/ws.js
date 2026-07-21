// Telemetry WebSocket: reconnecting, with a handler registry keyed by event type.
let ws = null;
const handlers = {};            // type -> [fn(evt)]; "__close__" runs on socket close.
let onOpenCb = () => {};

// Multiple modules may subscribe to the same event type (e.g. controls.js and
// live.js both watch "status"); fan out to every registered handler so one
// subscription can't clobber another.
export function onMessage(type, fn) { (handlers[type] || (handlers[type] = [])).push(fn); }
export function onOpen(fn) { onOpenCb = fn; }

export function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${scheme}://${location.host}/ws/telemetry`);
  ws.onopen = () => onOpenCb();
  ws.onmessage = (e) => {
    const evt = JSON.parse(e.data);
    (handlers[evt.type] || []).forEach((fn) => fn(evt));
  };
  ws.onclose = () => { (handlers.__close__ || []).forEach((fn) => fn()); setTimeout(connect, 1000); };
}

export function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
