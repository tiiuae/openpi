import { api, apiPost } from "./api.js";
import { onSessionActiveChange, setLiveStartBlocked } from "./controls.js";
import { logLine } from "./logs.js";

const $ = (id) => document.getElementById(id);
const FAST_POLL_MS = 1000;
const IDLE_POLL_MS = 3000;

let currentStatus = null;
let statusKnown = false;
let statusError = "";
let listError = "";
let listWarning = "";
let actionError = "";
let listLoading = true;
let actionPending = false;
let actionKind = "";
let selectionTouched = false;
let robotSessionActive = false;
let statusRequestEpoch = 0;
let syncedReadyKey = "";
let pollTimer = null;

export function setupModelServing(container) {
  if (!container || $("checkpoint-serving-card")) return;

  const card = document.createElement("section");
  card.id = "checkpoint-serving-card";
  card.className = "card run-card model-serving-card";
  card.dataset.group = "Checkpoint serving";
  card.innerHTML = `
    <h2>Checkpoint serving</h2>
    <div class="field-inline">
      <label for="checkpoint-select">Checkpoint</label>
      <div class="checkpoint-select-row">
        <select id="checkpoint-select" disabled>
          <option value="">Loading checkpoints…</option>
        </select>
        <button class="btn-sm" id="btn-refresh-checkpoints" type="button">Refresh</button>
      </div>
    </div>
    <div class="run-action-group">
      <span class="field-hint">Inference server</span>
      <div class="status-strip model-status-strip" role="status" aria-live="polite" aria-atomic="true">
        <span class="badge" id="badge-model">checking</span>
        <span class="run-status model-status-detail" id="model-status-detail">Checking the model server…</span>
      </div>
      <div class="run-controls">
        <button class="btn-primary" id="btn-serve-checkpoint" disabled>Serve checkpoint</button>
        <button class="btn-outline" id="btn-stop-checkpoint" disabled>Stop serving</button>
      </div>
      <div class="browser-msg model-serving-error" id="model-serving-error" role="alert" hidden></div>
    </div>`;
  container.prepend(card);

  $("checkpoint-select").addEventListener("change", () => {
    selectionTouched = true;
    actionError = "";
    renderError();
    updateControls();
  });
  $("btn-refresh-checkpoints").addEventListener("click", loadCheckpoints);
  $("btn-serve-checkpoint").addEventListener("click", serveSelectedCheckpoint);
  $("btn-stop-checkpoint").addEventListener("click", stopCheckpoint);
  onSessionActiveChange((active) => {
    robotSessionActive = active;
    updateControls();
  });

  void initialize();
}

async function initialize() {
  await Promise.all([loadCheckpoints(), refreshStatus()]);
  schedulePoll();
}

async function loadCheckpoints() {
  const select = $("checkpoint-select");
  if (!select) return;

  listLoading = true;
  listError = "";
  listWarning = "";
  updateControls();
  try {
    const payload = await api("/api/model/checkpoints");
    const checkpoints = Array.isArray(payload?.checkpoints)
      ? payload.checkpoints.filter((item) => item && typeof item.id === "string" && item.id)
      : [];
    const preferred = currentStatus?.pid != null
      ? currentStatus.checkpoint_id
      : select.value || currentStatus?.checkpoint_id || "";
    select.replaceChildren();
    for (const checkpoint of checkpoints) {
      const option = document.createElement("option");
      option.value = checkpoint.id;
      // IDs include the path relative to the configured root, which keeps
      // checkpoints with the same final directory name distinguishable.
      option.textContent = checkpoint.id;
      select.appendChild(option);
    }
    if (!checkpoints.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No checkpoints found";
      select.appendChild(option);
    }
    if (preferred) {
      if (!hasOption(select, preferred) && currentStatus?.pid != null) {
        addCurrentCheckpointOption(select, preferred);
      }
      if (hasOption(select, preferred)) select.value = preferred;
    }

    if (payload?.discovery?.error) {
      listError = String(payload.discovery.error);
    } else if (payload?.discovery?.truncated) {
      listWarning = "Checkpoint search reached its safety limit; only the discovered checkpoints are shown.";
    }
  } catch (err) {
    select.replaceChildren();
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Could not load checkpoints";
    select.appendChild(option);
    listError = errorMessage(err);
    logLine("ERROR", `Checkpoint list failed: ${listError}`);
  } finally {
    listLoading = false;
    renderError();
    updateControls();
  }
}

async function refreshStatus() {
  const requestEpoch = statusRequestEpoch;
  try {
    const status = await api("/api/model/status");
    if (requestEpoch !== statusRequestEpoch) return false;
    statusKnown = true;
    statusError = "";
    applyStatus(status);
  } catch (err) {
    if (requestEpoch !== statusRequestEpoch) return false;
    statusKnown = false;
    statusError = errorMessage(err);
    renderStatus();
    renderError();
    updateControls();
  }
  return true;
}

function schedulePoll() {
  window.clearTimeout(pollTimer);
  const busy = currentStatus && ["starting", "stopping"].includes(currentStatus.state);
  pollTimer = window.setTimeout(async () => {
    if (await refreshStatus()) schedulePoll();
  }, busy ? FAST_POLL_MS : IDLE_POLL_MS);
}

async function serveSelectedCheckpoint() {
  const checkpointId = $("checkpoint-select")?.value;
  if (!checkpointId || actionPending) return;

  beginAction("serve");
  try {
    const status = await apiPost("/api/model/start", { checkpoint_id: checkpointId });
    actionError = "";
    selectionTouched = false;
    applyStatus(status);
    logLine("INFO", `Serving checkpoint ${checkpointId}`);
  } catch (err) {
    actionError = errorMessage(err);
    logLine("ERROR", `Could not serve ${checkpointId}: ${actionError}`);
  } finally {
    endAction();
    schedulePoll();
  }
}

async function stopCheckpoint() {
  if (actionPending || currentStatus?.pid == null) return;

  const checkpointId = currentStatus.checkpoint_id;
  beginAction("stop");
  try {
    const status = await apiPost("/api/model/stop", {});
    actionError = "";
    applyStatus(status);
    logLine("INFO", checkpointId ? `Stopped checkpoint ${checkpointId}` : "Stopped checkpoint server");
  } catch (err) {
    actionError = errorMessage(err);
    logLine("ERROR", `Could not stop checkpoint: ${actionError}`);
  } finally {
    endAction();
    schedulePoll();
  }
}

function beginAction(kind) {
  window.clearTimeout(pollTimer);
  statusRequestEpoch += 1;
  actionPending = true;
  actionKind = kind;
  actionError = "";
  renderStatus();
  renderError();
  updateControls();
}

function endAction() {
  actionPending = false;
  actionKind = "";
  renderStatus();
  renderError();
  updateControls();
}

function applyStatus(status) {
  statusKnown = true;
  statusError = "";
  currentStatus = status;
  const select = $("checkpoint-select");
  if (select && status?.pid == null) {
    select.querySelector('option[data-managed-current="true"]')?.remove();
  }
  if (status?.pid != null && status.checkpoint_id && select) {
    if (!hasOption(select, status.checkpoint_id)) {
      addCurrentCheckpointOption(select, status.checkpoint_id);
    }
    if (hasOption(select, status.checkpoint_id)) select.value = status.checkpoint_id;
    selectionTouched = false;
  } else if (!selectionTouched && status?.checkpoint_id && select && hasOption(select, status.checkpoint_id)) {
    select.value = status.checkpoint_id;
  }
  const readyKey = status?.state === "ready"
    ? [status.checkpoint_id, status.pid, status.started_at, status.ready_at].join("|")
    : "";
  if (readyKey && readyKey !== syncedReadyKey) {
    syncedReadyKey = readyKey;
    syncEvalConfig(status);
  }
  renderStatus();
  renderError();
  updateControls();
}

function renderStatus() {
  const badge = $("badge-model");
  const detail = $("model-status-detail");
  if (!badge || !detail) return;

  let label = "checking";
  let badgeClass = "busy";
  let detailClass = "running";
  let message = "Checking the model server…";

  if (actionPending) {
    label = actionKind === "stop" ? "stopping" : "starting";
    message = actionKind === "stop"
      ? "Stopping the inference server…"
      : `Starting ${$("checkpoint-select")?.value || "the selected checkpoint"}…`;
  } else if (!statusKnown) {
    label = "unavailable";
    badgeClass = "bad";
    detailClass = "err";
    message = "Model serving status is unavailable.";
  } else {
    const status = currentStatus || {};
    const checkpoint = status.checkpoint_id || "checkpoint";
    const endpoint = inferenceEndpoint(status.inference);
    switch (status.state) {
      case "stopped":
        label = "idle";
        badgeClass = "";
        detailClass = "";
        message = "No checkpoint is being served.";
        break;
      case "starting":
        label = "starting";
        message = `Loading ${checkpoint}${endpoint ? ` for ${endpoint}` : ""}…`;
        break;
      case "ready":
        label = "ready";
        badgeClass = "ok";
        detailClass = "ok";
        message = `Serving ${checkpoint}${endpoint ? ` at ${endpoint}` : ""}.`;
        break;
      case "stopping":
        label = "stopping";
        message = `Stopping ${checkpoint}…`;
        break;
      case "failed":
        label = "failed";
        badgeClass = "bad";
        detailClass = "err";
        message = status.error || `${checkpoint} failed to start.`;
        break;
      default:
        label = status.state || "unknown";
        badgeClass = "bad";
        detailClass = "err";
        message = "The model server returned an unknown state.";
    }
  }

  const nextBadgeClass = `badge ${badgeClass}`.trim();
  const nextDetailClass = `run-status model-status-detail ${detailClass}`.trim();
  if (badge.textContent !== label) badge.textContent = label;
  if (badge.className !== nextBadgeClass) badge.className = nextBadgeClass;
  if (detail.textContent !== message) detail.textContent = message;
  if (detail.className !== nextDetailClass) detail.className = nextDetailClass;
}

function renderError() {
  const box = $("model-serving-error");
  if (!box) return;
  const isError = Boolean(actionError || statusError || listError);
  const message = actionError || statusError || listError || listWarning;
  const nextClass = `browser-msg model-serving-error${isError ? " err" : ""}`;
  const nextRole = isError ? "alert" : "status";
  const shouldHide = !message;
  if (box.textContent !== message) box.textContent = message;
  if (box.className !== nextClass) box.className = nextClass;
  if (box.getAttribute("role") !== nextRole) box.setAttribute("role", nextRole);
  if (box.hidden !== shouldHide) box.hidden = shouldHide;
}

function updateControls() {
  const select = $("checkpoint-select");
  const refresh = $("btn-refresh-checkpoints");
  const serve = $("btn-serve-checkpoint");
  const stop = $("btn-stop-checkpoint");
  if (!select || !refresh || !serve || !stop) return;

  const ownsProcess = currentStatus?.pid != null;
  const statusUsable = statusKnown && !statusError;
  const managedModelBlocksEval = actionPending || (ownsProcess && currentStatus?.state !== "ready");
  setLiveStartBlocked(managedModelBlocksEval);
  refresh.disabled = listLoading || actionPending;
  select.disabled = listLoading || actionPending || robotSessionActive || ownsProcess || !statusUsable || !select.value;
  serve.disabled = select.disabled || !select.value;
  stop.disabled = actionPending || robotSessionActive || !statusUsable || !ownsProcess
    || currentStatus?.state === "stopping";
}

function syncEvalConfig(status) {
  const host = $("cf_policy_host");
  const port = $("cf_policy_port");
  const modelName = $("cf_model_name");
  const inference = status.inference || {};
  if (host && inference.host) host.value = clientHost(inference.host);
  if (port && inference.port != null) port.value = String(inference.port);
  if (modelName && status.checkpoint_id) modelName.value = status.checkpoint_id;
}

function clientHost(host) {
  if (host === "0.0.0.0") return "127.0.0.1";
  if (host === "::") return "::1";
  return host;
}

function inferenceEndpoint(inference) {
  if (!inference?.host || inference.port == null) return "";
  return `${clientHost(inference.host)}:${inference.port}`;
}

function hasOption(select, value) {
  return Array.from(select.options).some((option) => option.value === value);
}

function addCurrentCheckpointOption(select, checkpointId) {
  const option = document.createElement("option");
  option.value = checkpointId;
  option.textContent = `${checkpointId} (current)`;
  option.dataset.managedCurrent = "true";
  select.prepend(option);
}

function errorMessage(err) {
  return err instanceof Error ? err.message : String(err);
}
