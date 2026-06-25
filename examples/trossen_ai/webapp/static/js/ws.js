// Telemetry WebSocket: reconnecting, with a handler registry keyed by event type.
let ws = null;
const handlers = {};            // type -> fn(evt); "__close__" runs on socket close.
let onOpenCb = () => {};

export function onMessage(type, fn) { handlers[type] = fn; }
export function onOpen(fn) { onOpenCb = fn; }

export function connect() {
  ws = new WebSocket(`ws://${location.host}/ws/telemetry`);
  ws.onopen = () => onOpenCb();
  ws.onmessage = (e) => {
    const evt = JSON.parse(e.data);
    (handlers[evt.type] || (() => {}))(evt);
  };
  ws.onclose = () => { (handlers.__close__ || (() => {}))(); setTimeout(connect, 1000); };
}

export function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
