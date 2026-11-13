import test from "node:test";
import assert from "node:assert/strict";
import { addItem, createStore, parseRoute, toggleItem } from "../store.mjs";

test("parseRoute maps hashes to views", () => {
  assert.equal(parseRoute("#/items"), "items");
  assert.equal(parseRoute(""), "items");
  assert.equal(parseRoute("#/about"), "about");
  assert.equal(parseRoute("#/missing"), "not-found");
});

test("addItem trims the title and advances the id", () => {
  const next = addItem({ items: [], nextId: 1 }, "  Buy oats  ");
  assert.deepEqual(next.items, [{ id: 1, title: "Buy oats", done: false }]);
  assert.equal(next.nextId, 2);
});

test("addItem rejects empty and oversized titles", () => {
  const state = { items: [], nextId: 1 };
  assert.throws(() => addItem(state, "   "), /Enter a title/);
  assert.throws(() => addItem(state, "x".repeat(121)), /under 120/);
});

test("toggleItem flips only the matching item", () => {
  const state = { items: [{ id: 1, done: false }, { id: 2, done: false }], nextId: 3 };
  const next = toggleItem(state, 2);
  assert.equal(next.items[0].done, false);
  assert.equal(next.items[1].done, true);
});

test("createStore notifies subscribers until unsubscribed", () => {
  const store = createStore({ items: [], nextId: 1 });
  let seen = 0;
  const unsubscribe = store.subscribe(() => { seen += 1; });
  store.update((current) => addItem(current, "One"));
  assert.equal(seen, 1);
  unsubscribe();
  store.update((current) => addItem(current, "Two"));
  assert.equal(seen, 1);
});
