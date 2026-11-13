// State helpers with no DOM access, so node:test can import them directly.

export function parseRoute(hash) {
  const route = String(hash || "").replace(/^#\/?/, "").split(/[/?]/)[0];
  if (route === "" || route === "items") return "items";
  if (route === "about") return "about";
  return "not-found";
}

export function createStore(initial) {
  let state = initial;
  const listeners = new Set();
  return {
    get: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    update(change) {
      state = change(state);
      for (const listener of listeners) listener(state);
      return state;
    },
  };
}

export function addItem(state, title) {
  const text = typeof title === "string" ? title.trim() : "";
  if (text === "") throw new Error("Enter a title before adding an item.");
  if (text.length > 120) throw new Error("Keep the title under 120 characters.");
  return {
    items: [...state.items, { id: state.nextId, title: text, done: false }],
    nextId: state.nextId + 1,
  };
}

export function toggleItem(state, id) {
  return {
    ...state,
    items: state.items.map((item) =>
      item.id === id ? { ...item, done: !item.done } : item
    ),
  };
}
