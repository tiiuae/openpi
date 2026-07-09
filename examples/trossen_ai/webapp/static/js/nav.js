// webapp/static/js/nav.js
import { setupFeedbackButton } from "./feedback.js?v=feedback-modal";

// Single source of truth for the top navigation. Active link is marked by
// pathname; visual separators are rendered between links.
const LINKS = [
  { href: "/", label: "Eval" },
  { href: "/runs", label: "Compare" },
  { href: "/replay", label: "Replay" },
  { href: "/teleop", label: "Teleop" },
];

export function renderNav(container) {
  const path = location.pathname;
  container.innerHTML =
    `<strong style="color:var(--text);margin-right:16px">Trossen Control</strong>` +
    LINKS.map((l, i) => {
      const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
      const sep = i > 0 ? `<span class="nav-sep">|</span>` : "";
      return `${sep}<a href="${l.href}"${active ? ' class="active"' : ""}>${l.label}</a>`;
    }).join("");
}

const el = document.querySelector(".app-nav");
if (el) renderNav(el);
setupFeedbackButton();
