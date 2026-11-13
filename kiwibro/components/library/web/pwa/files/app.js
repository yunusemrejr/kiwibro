const STORAGE_KEY = "pwa-notes";
const form = document.querySelector("#note-form");
const input = document.querySelector("#note");
const error = document.querySelector("#note-error");
const list = document.querySelector("#notes");
const connection = document.querySelector("#connection");
const installButton = document.querySelector("#install");
let installPrompt = null;

function loadNotes() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

let notes = loadNotes();

function saveNotes() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
  } catch {
    error.textContent = "Could not save on this device; notes last until the page closes.";
    error.hidden = false;
  }
}

function escapeHtml(value) {
  const chars = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(value).replace(/[&<>"']/g, (char) => chars[char]);
}

function render() {
  list.innerHTML = notes.length === 0
    ? '<li class="empty">No notes yet. Add one above; it stays on this device.</li>'
    : notes.map((note) => `<li class="note">${escapeHtml(note)}</li>`).join("");
}

function updateConnection() {
  connection.textContent = navigator.onLine ? "Online" : "Offline — saved notes still work.";
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (text === "") {
    error.textContent = "Write something before adding it.";
    error.hidden = false;
    input.setAttribute("aria-invalid", "true");
    input.focus();
    return;
  }
  notes = [text, ...notes];
  saveNotes();
  render();
  form.reset();
  error.hidden = true;
  input.removeAttribute("aria-invalid");
});

installButton.addEventListener("click", async () => {
  if (!installPrompt) return;
  installPrompt.prompt();
  await installPrompt.userChoice;
  installPrompt = null;
  installButton.hidden = true;
});

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installPrompt = event;
  installButton.hidden = false;
});

window.addEventListener("appinstalled", () => {
  installButton.hidden = true;
});

window.addEventListener("online", updateConnection);
window.addEventListener("offline", updateConnection);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => {
      connection.textContent = "Offline mode is not available in this browser.";
    });
  });
}

updateConnection();
render();
