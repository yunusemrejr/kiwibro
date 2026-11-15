const button = document.querySelector(".menu-button");
const nav = document.querySelector("#site-nav");
const wide = matchMedia("(min-width: 48rem)");

const here = location.pathname.split("/").pop() || "index.html";
for (const link of nav.querySelectorAll("a")) {
  if (link.getAttribute("href") === here) link.setAttribute("aria-current", "page");
}

function setOpen(open) {
  button.setAttribute("aria-expanded", String(open));
  nav.classList.toggle("open", open);
}

button.addEventListener("click", () => setOpen(button.getAttribute("aria-expanded") !== "true"));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && button.getAttribute("aria-expanded") === "true") {
    setOpen(false);
    button.focus();
  }
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".bar")) setOpen(false);
});
wide.addEventListener("change", () => setOpen(false));
document.documentElement.classList.add("js");
