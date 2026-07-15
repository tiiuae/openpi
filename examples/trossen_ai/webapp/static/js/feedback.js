import { apiPost } from "./api.js";

const WEB3FORMS_ACCESS_KEY = "34ac9cd2-770d-4bf4-8218-df607c0e0781";

let feedbackUiReady = false;

function feedbackFormHtml() {
  return `<div class="modal-overlay feedback-overlay" id="modal-feedback" style="display:none">
    <div class="modal-box feedback-modal-box">
      <div class="modal-titlebar">Feedback <button class="btn-close" id="fb-close">&times;</button></div>
      <div class="feedback-modal-body">
        <div class="field-inline"><label>Name</label><input id="fb-name" type="text"></div>
        <div class="field-inline"><label>Email</label><input id="fb-email" type="email"></div>
        <div class="field-col"><label>Feedback</label><textarea id="fb-text" rows="7"></textarea></div>
        <div class="feedback-actions">
          <span class="run-status" id="fb-status"></span>
          <button class="btn-primary" id="fb-submit">Submit</button>
        </div>
      </div>
    </div>
  </div>`;
}

async function emailFeedback({ name, email, feedback }) {
  await fetch("https://api.web3forms.com/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({
      access_key: WEB3FORMS_ACCESS_KEY,
      subject: "Trossen webapp feedback",
      name: name || "Anonymous",
      email,
      from_name: name || "Anonymous",
      replyto: email,
      message: feedback,
    }),
  });
}

export function setupFeedbackButton() {
  if (feedbackUiReady) return;
  const header = document.querySelector(".app-header");
  if (!header) return;

  let actions = header.querySelector(".header-actions");
  if (!actions) {
    actions = document.createElement("div");
    actions.className = "header-actions";
    header.appendChild(actions);
  }
  actions.insertAdjacentHTML(
    "beforeend",
    '<button type="button" class="btn-outline feedback-trigger" id="btn-feedback">Feedback</button>'
  );
  document.body.insertAdjacentHTML("beforeend", feedbackFormHtml());

  const modal = document.getElementById("modal-feedback");
  const status = document.getElementById("fb-status");
  const text = document.getElementById("fb-text");
  const open = () => { modal.style.display = "flex"; text.focus(); };
  const close = () => { modal.style.display = "none"; };

  document.getElementById("btn-feedback").addEventListener("click", open);
  document.getElementById("fb-close").addEventListener("click", close);
  modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
  document.getElementById("fb-submit").addEventListener("click", async () => {
    const payload = {
      name: document.getElementById("fb-name").value,
      email: document.getElementById("fb-email").value,
      feedback: text.value,
    };
    if (!payload.feedback.trim()) {
      status.textContent = "Write feedback first.";
      status.className = "run-status err";
      return;
    }
    try {
      status.textContent = "Sending...";
      status.className = "run-status";
      await apiPost("/api/feedback", payload);
      try {
        await emailFeedback(payload);
      } catch (e) {
        // Email is best-effort; local save above already succeeded.
        console.warn("Feedback email relay failed:", e);
      }
      status.textContent = "Thanks!";
      status.className = "run-status ok";
      text.value = "";
      setTimeout(close, 500);
    } catch (e) {
      status.textContent = "Failed: " + e.message;
      status.className = "run-status err";
    }
  });
  feedbackUiReady = true;
}
