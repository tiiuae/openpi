import { apiPost } from "./api.js";

export function renderFeedback(card) {
  if (!card) return;
  card.innerHTML = `<h2>Feedback</h2>
    <div class="field-inline"><label>Name</label><input id="fb-name" type="text"></div>
    <div class="field-inline"><label>Email</label><input id="fb-email" type="text"></div>
    <div class="field-row align-top"><label>Feedback</label><textarea id="fb-text" rows="10" style="width:100%;min-height:180px;resize:vertical"></textarea></div>
    <div class="run-controls"><button class="btn-primary" id="fb-submit">Submit</button>
      <span class="run-status" id="fb-status"></span></div>`;
  card.querySelector("#fb-submit").addEventListener("click", async () => {
    const s = card.querySelector("#fb-status");
    try {
      await apiPost("/api/feedback", {
        name: card.querySelector("#fb-name").value,
        email: card.querySelector("#fb-email").value,
        feedback: card.querySelector("#fb-text").value,
      });
      s.textContent = "Thanks!"; s.className = "run-status ok";
      card.querySelector("#fb-text").value = "";
    } catch (e) { s.textContent = "Failed: " + e.message; s.className = "run-status err"; }
  });
}
