// webapp/static/js/runlog.js
// End-of-run quick rating. Dedicated to eval logging — NOT the feedback card
// (feedback is for webapp bug reports). Remembers the active run id from the
// run_started status and, when the session ends, prompts for success/score/note
// and PATCHes it to /api/runs/{id}/rating.
let activeRunId = null;

function modal() {
  let m = document.getElementById("runlog-modal");
  if (m) return m;
  m = document.createElement("div");
  m.id = "runlog-modal";
  m.className = "runlog-backdrop";
  m.innerHTML = `
    <div class="runlog-card">
      <h3>Rate this run</h3>
      <div class="runlog-row">
        <label>Result</label>
        <select id="runlog-success">
          <option value="yes">Success</option>
          <option value="partial" selected>Partial</option>
          <option value="no">Failure</option>
        </select>
      </div>
      <div class="runlog-row">
        <label>Score (1–5)</label>
        <input id="runlog-score" type="number" min="1" max="5" step="1">
      </div>
      <div class="runlog-row">
        <label>Note</label>
        <input id="runlog-note" type="text" placeholder="one-line note (optional)">
      </div>
      <div class="runlog-actions">
        <button class="btn-outline" id="runlog-skip">Skip</button>
        <button class="btn-primary" id="runlog-save">Save rating</button>
      </div>
    </div>`;
  document.body.appendChild(m);
  m.querySelector("#runlog-skip").addEventListener("click", () => { m.style.display = "none"; });
  m.querySelector("#runlog-save").addEventListener("click", async () => {
    if (!activeRunId) { m.style.display = "none"; return; }
    const body = {
      success: m.querySelector("#runlog-success").value,
      score: Number(m.querySelector("#runlog-score").value) || null,
      note: m.querySelector("#runlog-note").value.trim(),
    };
    try {
      await fetch(`/api/runs/${activeRunId}/rating`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) { console.error("rating save failed", e); }
    m.style.display = "none";
  });
  return m;
}

// Call from a status handler. run_started -> remember id; stopped/error -> prompt.
export function onRunStatus(evt) {
  if (evt.kind === "run_started") {
    activeRunId = evt.payload?.run_id || null;
    return;
  }
  if ((evt.kind === "stopped" || evt.kind === "error") && activeRunId) {
    const m = modal();
    m.querySelector("#runlog-score").value = "";
    m.querySelector("#runlog-note").value = "";
    m.style.display = "flex";
  }
}
