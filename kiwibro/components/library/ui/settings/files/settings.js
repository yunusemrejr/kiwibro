const KEY = "settings";
// KIWIBRO: one default per control name in index.html; the type of each value decides how it is read
const DEFAULTS = {
  textSize: "medium",
  compact: false,
  email: true,
  sound: false,
  digest: "monday",
  displayName: "",
};

const form = document.querySelector("#settings");
const saved = document.querySelector("#saved");
const sizes = { small: "90%", medium: "100%", large: "115%" };

function load() {
  return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || "{}") };
}

function show(values) {
  for (const [name, value] of Object.entries(values)) {
    const control = form.elements[name];
    if (typeof value === "boolean") control.checked = value;
    else control.value = value;
  }
  document.documentElement.style.fontSize = sizes[values.textSize];
  document.documentElement.classList.toggle("compact", values.compact);
}

function read() {
  const values = {};
  for (const [name, fallback] of Object.entries(DEFAULTS)) {
    const control = form.elements[name];
    values[name] = typeof fallback === "boolean" ? control.checked : control.value.trim();
  }
  return values;
}

function save(values) {
  localStorage.setItem(KEY, JSON.stringify(values));
  show(values);
  saved.textContent = "Saved on this device.";
}

form.addEventListener("change", () => save(read()));
form.elements.displayName.addEventListener("input", () => save(read()));
form.addEventListener("submit", (event) => event.preventDefault());
document.querySelector("#reset").addEventListener("click", () => save({ ...DEFAULTS }));
show(load());
