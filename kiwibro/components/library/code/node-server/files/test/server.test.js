import assert from "node:assert/strict";
import test, { after, before } from "node:test";

import { createApp } from "../src/server.js";

// A real server on a free port: the tests exercise the same path a browser takes.
let server;
let base;

before(async () => {
  server = createApp();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => new Promise((resolve) => server.close(resolve)));

async function call(method, path, body) {
  const response = await fetch(`${base}${path}`, {
    method,
    headers: body === undefined ? {} : { "content-type": "application/json" },
    body: body === undefined ? undefined : typeof body === "string" ? body : JSON.stringify(body),
  });
  return { status: response.status, type: response.headers.get("content-type"), body: await response.json() };
}

test("health says the server is up", async () => {
  const answer = await call("GET", "/health");

  assert.equal(answer.status, 200);
  assert.equal(answer.body.status, "ok");
  assert.match(answer.type, /application\/json/);
});

test("a new note comes back with an id and then appears in the list", async () => {
  const created = await call("POST", "/notes", { title: "  Buy milk  ", body: "Two litres" });

  assert.equal(created.status, 201);
  assert.equal(created.body.title, "Buy milk");
  assert.equal(typeof created.body.id, "number");

  const listed = await call("GET", "/notes");
  assert.ok(listed.body.some((note) => note.id === created.body.id));

  const single = await call("GET", `/notes/${created.body.id}`);
  assert.deepEqual(single.body, created.body);
});

// One row per rejected request; add new rules here rather than copying a test.
for (const [what, payload, wanted] of [
  ["a missing title", { body: "no title" }, /"title" to be non-empty text/],
  ["a title of spaces", { title: "   " }, /"title" to be non-empty text/],
  ["an unknown field", { title: "A", owner: "B" }, /Unknown field "owner"/],
  ["a non-text body", { title: "A", body: 7 }, /"body" to be text/],
]) {
  test(`rejects ${what} with 400`, async () => {
    const answer = await call("POST", "/notes", payload);

    assert.equal(answer.status, 400);
    assert.match(answer.body.error, wanted);
  });
}

test("rejects a body that is not a JSON object", async () => {
  const answer = await call("POST", "/notes", "not json");

  assert.equal(answer.status, 400);
  assert.match(answer.body.error, /Expected a JSON object, got invalid JSON/);
});

test("an unknown note is a 404 and an unknown route too", async () => {
  assert.equal((await call("GET", "/notes/9999")).status, 404);

  const missing = await call("GET", "/nothing");
  assert.equal(missing.status, 404);
  assert.match(missing.body.error, /There is no route GET \/nothing/);
});

test("the wrong method on a known route is a 405", async () => {
  const answer = await call("DELETE", "/notes");

  assert.equal(answer.status, 405);
  assert.match(answer.body.error, /Use GET or POST for \/notes/);
});
