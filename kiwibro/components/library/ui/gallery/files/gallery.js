const thumbs = [...document.querySelectorAll(".thumb")];
const viewer = document.querySelector("#viewer");
const stage = document.querySelector("#stage");
let index = 0;
let startX = null;

function show(next) {
  index = (next + thumbs.length) % thumbs.length;
  const picture = thumbs[index].firstElementChild.cloneNode(true);
  const label = picture.getAttribute("aria-label") || picture.getAttribute("alt") || "";
  picture.removeAttribute("role");
  picture.setAttribute("aria-hidden", "true");
  stage.replaceChildren(picture);
  document.querySelector("#caption").textContent = label;
  document.querySelector("#count").textContent = `${index + 1} of ${thumbs.length}`;
}

thumbs.forEach((thumb, position) => {
  thumb.addEventListener("click", () => {
    show(position);
    viewer.showModal();
  });
});
document.querySelector("#prev").addEventListener("click", () => show(index - 1));
document.querySelector("#next").addEventListener("click", () => show(index + 1));
document.querySelector("#close").addEventListener("click", () => viewer.close());

viewer.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") show(index - 1);
  if (event.key === "ArrowRight") show(index + 1);
});
viewer.addEventListener("click", (event) => {
  if (event.target === viewer) viewer.close();
});
viewer.addEventListener("close", () => thumbs[index].focus());

stage.addEventListener("pointerdown", (event) => { startX = event.clientX; });
stage.addEventListener("pointerup", (event) => {
  if (startX === null) return;
  const moved = event.clientX - startX;
  startX = null;
  if (Math.abs(moved) > 40) show(index + (moved < 0 ? 1 : -1));
});
stage.addEventListener("pointercancel", () => { startX = null; });
