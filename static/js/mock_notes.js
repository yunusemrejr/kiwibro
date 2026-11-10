/**
 * Fake notes for ?mock=1: the notes routes of the REST API, kept on backend.notes.
 * Every other route returns undefined so the main mock handles it.
 */

const HOUR = 3_600_000;
const DAY = 24 * HOUR;
const TITLE_MAX = 120;
const DERIVED_TITLE_MAX = 60;
const TAG_MAX = 8;

const nowIso = () => new Date().toISOString();
const agoIso = (ms) => new Date(Date.now() - ms).toISOString();

function makeNote({ id, title, body, tags = [], pinned = false, origin = "you", source = null, ago }) {
  return {
    id,
    title,
    body,
    tags,
    pinned,
    origin,
    source_conversation_id: source,
    created_at: agoIso(ago + HOUR),
    updated_at: agoIso(ago),
  };
}

function seedNotes(backend) {
  const conversationId = (backend.conversations || [])[0]?.id ?? null;
  return [
    makeNote({
      id: "n_m1f4",
      title: "Shopping list",
      body: "- Oat milk\n- Lemons\n- Coffee beans\n- Rye bread\n- AA batteries",
      tags: ["home"],
      ago: 25 * 60_000,
    }),
    makeNote({
      id: "n_m2c9",
      title: "Project ideas",
      body:
        "## Small things to try\n\n- A packing checklist that can be shared for trips\n- A tool that renames photos by date\n- A weekly review template\n\nStart with the photo tool. It is the smallest.",
      tags: ["ideas", "work"],
      pinned: true,
      ago: 2 * DAY,
    }),
    makeNote({
      id: "n_m3a7",
      title: "Weekend trip plan",
      body:
        "**Saturday:** the old mill trail in the morning, lunch by the river.\n\n**Sunday:** the market, then the train back before dark.\n\nBook the train on Friday night.",
      tags: ["travel"],
      ago: 3 * DAY,
    }),
    makeNote({
      id: "n_m4d2",
      title: "Command to remember",
      body:
        "Copy a photo folder to a backup disk and skip files that are already there:\n\n```bash\nrsync -av --progress ./photos/ /mnt/backup/photos/\n```\n\nRun it with `--dry-run` first.",
      tags: ["tools"],
      origin: "kiwibro",
      source: conversationId,
      ago: 1 * DAY,
    }),
    makeNote({
      id: "n_m5e8",
      title: "Lentil soup",
      body:
        "# Lentil soup\n\n1. Fry an onion in olive oil.\n2. Add cumin, carrot and red lentils.\n3. Cover with water and simmer for 25 minutes.\n\nSqueeze lemon on top before serving.",
      tags: ["cooking"],
      ago: 5 * DAY,
    }),
    makeNote({
      id: "n_m6b1",
      title: "Books to try",
      body: "- A short book about river deltas\n- The sequel to the lighthouse story\n- Anything the library staff recommend",
      tags: ["reading"],
      origin: "kiwibro",
      ago: 9 * DAY,
    }),
  ];
}

function cleanTags(tags) {
  if (!Array.isArray(tags)) return [];
  return tags
    .map((tag) => String(tag).trim())
    .filter(Boolean)
    .slice(0, TAG_MAX);
}

// An empty title becomes the first line of the body, without leading markdown hashes.
function titleFor(title, text) {
  const given = String(title ?? "").trim();
  if (given) return given.slice(0, TITLE_MAX);
  const first = String(text ?? "").split("\n").find((line) => line.trim()) || "";
  return first.replace(/^\s*#+\s*/, "").trim().slice(0, DERIVED_TITLE_MAX);
}

function tagCounts(notes) {
  const counts = new Map();
  for (const note of notes) {
    for (const tag of note.tags) counts.set(tag, (counts.get(tag) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
}

function listNotes(notes, url) {
  const q = (url.searchParams.get("q") || "").trim().toLowerCase();
  const tag = (url.searchParams.get("tag") || "").trim();
  const matches = notes
    .filter((note) => !q || `${note.title}\n${note.body}`.toLowerCase().includes(q))
    .filter((note) => !tag || note.tags.includes(tag))
    .sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.updated_at.localeCompare(a.updated_at));
  return { ok: true, notes: matches, total: notes.length, tags: tagCounts(notes) };
}

function createNote(notes, input, fail) {
  const text = String(input?.body ?? "");
  if (!text.trim()) fail(400, "bad_request", "A note needs some text.");
  const stamp = nowIso();
  const note = {
    id: `n_${Math.random().toString(36).slice(2, 8)}`,
    title: titleFor(input?.title, text),
    body: text,
    tags: cleanTags(input?.tags),
    pinned: false,
    origin: "you",
    source_conversation_id: null,
    created_at: stamp,
    updated_at: stamp,
  };
  notes.push(note);
  return { ok: true, note };
}

function patchNote(note, input, fail) {
  const patch = input || {};
  if ("body" in patch && !String(patch.body ?? "").trim()) {
    fail(400, "bad_request", "A note needs some text.");
  }
  if ("body" in patch) note.body = String(patch.body);
  if ("title" in patch) note.title = titleFor(patch.title, note.body);
  if ("tags" in patch) note.tags = cleanTags(patch.tags);
  if ("pinned" in patch) note.pinned = Boolean(patch.pinned);
  note.updated_at = nowIso();
  return { ok: true, note };
}

export function mockNotes(backend, { route, verb, body, url, fail }) {
  if (!Array.isArray(backend.notes)) backend.notes = seedNotes(backend);

  if (route === "/api/notes" && verb === "GET") return listNotes(backend.notes, url);
  if (route === "/api/notes" && verb === "POST") return createNote(backend.notes, body, fail);

  const match = /^\/api\/notes\/([^/]+)$/.exec(route);
  if (!match || !["GET", "PATCH", "DELETE"].includes(verb)) return undefined;

  const id = decodeURIComponent(match[1]);
  const index = backend.notes.findIndex((note) => note.id === id);
  if (index < 0) fail(404, "not_found", "Note not found.");

  if (verb === "GET") return { ok: true, note: backend.notes[index] };
  if (verb === "DELETE") {
    backend.notes.splice(index, 1);
    return { ok: true };
  }
  return patchNote(backend.notes[index], body, fail);
}
