import { addItem, createStore, parseRoute, toggleItem } from "./store.mjs";

const STORAGE_KEY = "kiwibro-app-shell";
const app = document.querySelector("#app");
const state = createStore(load());
const EMPTY = "No items yet. Add the first one above.";
let currentRoute = null;

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && Array.isArray(saved.items) && Number.isInteger(saved.nextId)) return saved;
  } catch {
    // Unreadable storage is ignored; the app starts empty.
  }
  return { items: [], nextId: 1 };
}

function save(next) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    setStatus("Could not save in this browser; items last until the page closes.");
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
}

// KIWIBRO: point this helper at the project's own API
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`Server answered ${response.status}.`);
  return response.json();
}

function itemsView() {
  return `
    <section class="view">
      <h1 tabindex="-1">Items</h1>
      <form id="add-form" novalidate>
        <label for="new-title">New item</label>
        <div class="field-row">
          <input id="new-title" name="title" type="text" maxlength="120" autocomplete="off" aria-describedby="title-error">
          <button type="submit">Add</button>
        </div>
        <p id="title-error" class="error" role="alert" hidden></p>
      </form>
      <p class="toolbar"><button type="button" id="sync" class="quiet">Check server</button></p>
      <p id="status" class="status" role="status"></p>
      <ul id="list" class="items"></ul>
    </section>`;
}

function aboutView() {
  return `
    <section class="view">
      <h1 tabindex="-1">About {{app_name}}</h1>
      <p>{{app_name}} keeps its items in this browser, so they are here the next time you open it.</p>
    </section>`;
}

function notFoundView() {
  return `
    <section class="view">
      <h1 tabindex="-1">Page not found</h1>
      <p>That view does not exist. <a href="#/items">Back to the items</a>.</p>
    </section>`;
}

function renderList() {
  const list = document.querySelector("#list");
  if (!list) return;
  const items = state.get().items;
  list.innerHTML = items.length === 0 ? `<li class="empty">${EMPTY}</li>` : items.map((item) =>
    `<li class="item${item.done ? " done" : ""}"><button type="button" class="toggle" data-id="${item.id}" aria-pressed="${item.done}"><span class="check" aria-hidden="true"></span><span>${escapeHtml(item.title)}</span></button></li>`
  ).join("");
}

function setStatus(message) {
  const status = document.querySelector("#status");
  if (status) status.textContent = message;
}

async function checkServer() {
  const button = document.querySelector("#sync");
  button.disabled = true;
  setStatus("Checking the server...");
  try {
    await api("/health");
    setStatus("Server answered. Local items are unchanged.");
  } catch {
    setStatus("Server not reachable. Items stay in this browser.");
  } finally {
    button.disabled = false;
  }
}

function render() {
  const route = parseRoute(location.hash);
  if (route !== currentRoute) {
    const visiting = currentRoute !== null;
    currentRoute = route;
    app.innerHTML = route === "items" ? itemsView() : route === "about" ? aboutView() : notFoundView();
    document.querySelectorAll("[data-nav]").forEach((link) => {
      if (link.dataset.nav === route) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    if (visiting) app.querySelector("h1")?.focus();
  }
  renderList();
}

app.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.id) state.update((current) => toggleItem(current, Number(button.dataset.id)));
  else if (button.id === "sync") checkServer();
});

app.addEventListener("submit", (event) => {
  if (event.target.id !== "add-form") return;
  event.preventDefault();
  const input = event.target.querySelector("#new-title");
  const error = event.target.querySelector("#title-error");
  try {
    state.update((current) => addItem(current, input.value));
    event.target.reset();
    error.hidden = true;
    input.removeAttribute("aria-invalid");
  } catch (problem) {
    error.textContent = problem.message;
    error.hidden = false;
    input.setAttribute("aria-invalid", "true");
    input.focus();
  }
});

window.addEventListener("hashchange", render);
state.subscribe((next) => {
  save(next);
  render();
});

if (!location.hash) history.replaceState(null, "", "#/items");
render();
