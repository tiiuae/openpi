// webapp/static/js/nav.js
// Single source of truth for the top navigation. Ordered links with a visual
// separator between the common pair (Live, Compare) and the rarely-used pair
// (Replay, Teleop). Active link is marked by pathname.
const LINKS = [
  { href: "/", label: "Live" },
  { href: "/runs", label: "Compare" },
  { sep: true },
  { href: "/replay", label: "Replay" },
  { href: "/teleop", label: "Teleop" },
];

export function renderNav(container) {
  const path = location.pathname;
  container.innerHTML =
    `<strong style="color:var(--text);margin-right:16px">Trossen Control</strong>` +
    LINKS.map(l => {
      if (l.sep) return `<span class="nav-sep">|</span>`;
      const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
      return `<a href="${l.href}"${active ? ' class="active"' : ""}>${l.label}</a>`;
    }).join("");
}

const el = document.querySelector(".app-nav");
if (el) renderNav(el);
