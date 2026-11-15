import { createServer } from "node:http";
import { fileURLToPath } from "node:url";

const PORT = Number(process.env.PORT ?? 8080);
const LONGEST_TITLE = 120;
const LONGEST_BODY_BYTES = 10_000;

// KIWIBRO: replace these sample notes with the project's own records and fields
const notes = [{ id: 1, title: "First note", body: "Written from the starter." }];

function send(response, status, payload) {
  const text = JSON.stringify(payload);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(text),
  });
  response.end(text);
}

async function readBody(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > LONGEST_BODY_BYTES) {
      throw new Error(`Expected a body under ${LONGEST_BODY_BYTES} bytes, got more.`);
    }
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error(`Expected a JSON object, got ${raw === "" ? "an empty body" : "invalid JSON"}.`);
  }
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    throw new Error(`Expected a JSON object, got ${Array.isArray(data) ? "an array" : typeof data}.`);
  }
  return data;
}

// KIWIBRO: validate the fields the project really stores, one check per rule
function newNote(data) {
  const unknown = Object.keys(data).find((key) => key !== "title" && key !== "body");
  if (unknown) throw new Error(`Unknown field "${unknown}". Send title and body only.`);
  if (typeof data.title !== "string" || data.title.trim() === "") {
    throw new Error('Expected "title" to be non-empty text, got none.');
  }
  if (data.title.trim().length > LONGEST_TITLE) {
    throw new Error(`Expected "title" under ${LONGEST_TITLE} characters, got ${data.title.trim().length}.`);
  }
  if (data.body !== undefined && typeof data.body !== "string") {
    throw new Error(`Expected "body" to be text, got ${typeof data.body}.`);
  }
  const id = notes.reduce((highest, note) => Math.max(highest, note.id), 0) + 1;
  return { id, title: data.title.trim(), body: (data.body ?? "").trim() };
}

async function handle(request, response) {
  const url = new URL(request.url, "http://localhost");
  const parts = url.pathname.split("/").filter(Boolean);
  const method = request.method;

  if (parts[0] === "health" && parts.length === 1) {
    if (method !== "GET") return send(response, 405, { error: `Use GET for /health, not ${method}.` });
    return send(response, 200, { status: "ok", notes: notes.length });
  }

  if (parts[0] === "notes" && parts.length === 1) {
    if (method === "GET") return send(response, 200, notes);
    if (method !== "POST") return send(response, 405, { error: `Use GET or POST for /notes, not ${method}.` });
    try {
      const note = newNote(await readBody(request));
      notes.push(note);
      return send(response, 201, note);
    } catch (error) {
      return send(response, 400, { error: error.message });
    }
  }

  if (parts[0] === "notes" && parts.length === 2 && /^\d+$/.test(parts[1])) {
    if (method !== "GET") return send(response, 405, { error: `Use GET for /notes/${parts[1]}, not ${method}.` });
    const note = notes.find((entry) => entry.id === Number(parts[1]));
    if (!note) return send(response, 404, { error: `There is no note ${parts[1]}.` });
    return send(response, 200, note);
  }

  return send(response, 404, { error: `There is no route ${method} ${url.pathname}.` });
}

export function createApp() {
  return createServer(handle);
}

// Only listen when this file is the program that was started; tests import createApp instead.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  createApp().listen(PORT, "0.0.0.0", () => {
    console.log(`{{name}} listening on http://127.0.0.1:${PORT}`);
  });
}
