// Places every box and link of FLOW: ranks run top to bottom, back links loop around the right.
const KINDS = ["start", "step", "choice", "end"];
const WRAP = 22;   // characters per line in a box
const LINES = 3;   // most lines in a box
const LINE = 20;   // line height of the box text, px
const GAP = 40;    // space between two boxes in one rank
const ROW = 56;    // space between two ranks
const DUMMY = 16;  // width of an invisible waypoint
const EDGE = 24;   // margin around the chart
const LOOP = 80;   // how far a back link bends past the widest box
const LANE = 20;   // space between two back links

const measurer = document.createElement("canvas").getContext("2d");
measurer.font = `15px ${getComputedStyle(document.body).fontFamily}`;

// The first mistake in the data, in one sentence; null when the data holds up.
function checkData(flow) {
  if (!Array.isArray(flow.steps) || flow.steps.length === 0) return "The flowchart needs at least one step in data.js.";
  const ids = new Set();
  for (const step of flow.steps) {
    if (!step || typeof step.id !== "string" || step.id === "") return "Every step needs an id of its own.";
    if (ids.has(step.id)) return `Two steps use the id "${step.id}".`;
    ids.add(step.id);
    if (typeof step.text !== "string" || step.text.trim() === "") return `Step "${step.id}" has no text.`;
    if (!KINDS.includes(step.kind ?? "step")) return `Step "${step.id}" has the kind "${step.kind}"; use start, step, choice or end.`;
  }
  if (!Array.isArray(flow.links)) return "The links in data.js must be a list.";
  for (const link of flow.links) {
    if (!link || typeof link !== "object") return "Every link needs from and to.";
    for (const id of [link.from, link.to]) {
      if (!ids.has(id)) return `A link names "${id}", but no step has that id.`;
    }
  }
  return null;
}

// Lines of at most WRAP characters, at most LINES of them; a longer text ends in an ellipsis.
function wrap(text) {
  const lines = [];
  for (const word of String(text).trim().split(/\s+/)) {
    const last = lines[lines.length - 1];
    if (last !== undefined && last.length + 1 + word.length <= WRAP) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  }
  if (lines.length > LINES) {
    lines.length = LINES;
    lines[LINES - 1] = `${lines[LINES - 1].slice(0, WRAP - 1)}…`;
  }
  return lines;
}

function boxSize(kind, lines) {
  const text = Math.max(...lines.map((line) => measurer.measureText(line).width));
  const height = lines.length * LINE;
  if (kind === "choice") return { w: text * 2 + 24, h: height * 2 + 24 };
  return { w: Math.max(text + 40, 120), h: Math.max(height + 24, 48) };
}

// Depth-first walk: a link to a step still open on the walk closes a loop, so it is a back link.
function backLinks(flow) {
  const outgoing = new Map(flow.steps.map((step) => [step.id, []]));
  flow.links.forEach((link, index) => outgoing.get(link.from).push(index));
  const open = new Set();
  const done = new Set();
  const back = new Set();
  const walk = (id) => {
    open.add(id);
    for (const index of outgoing.get(id)) {
      const to = flow.links[index].to;
      if (open.has(to)) back.add(index);
      else if (!done.has(to)) walk(to);
    }
    open.delete(id);
    done.add(id);
  };
  for (const step of flow.steps) if (!done.has(step.id)) walk(step.id);
  return back;
}

// Sorts one rank by the average position of its neighbours in the rank above or below.
function reorder(row, side) {
  for (const node of row) {
    const near = node[side];
    node.order = near.length ? near.reduce((sum, other) => sum + other.pos, 0) / near.length : node.pos;
  }
  row.sort((a, b) => a.order - b.order || a.pos - b.pos);
  row.forEach((node, pos) => { node.pos = pos; });
}

function layout(flow) {
  const back = backLinks(flow);
  const nodes = new Map(flow.steps.map((step) => {
    const kind = step.kind ?? "step";
    const lines = wrap(step.text);
    return [step.id, { id: step.id, kind, lines, ...boxSize(kind, lines), up: [], down: [] }];
  }));
  const incoming = new Map(flow.steps.map((step) => [step.id, []]));
  flow.links.forEach((link, index) => {
    if (!back.has(index)) incoming.get(link.to).push(link.from);
  });
  // Rank: the longest path of forward links that ends at the step.
  const rank = new Map();
  const rankOf = (id) => {
    if (!rank.has(id)) rank.set(id, Math.max(-1, ...incoming.get(id).map(rankOf)) + 1);
    return rank.get(id);
  };
  for (const node of nodes.values()) node.rank = rankOf(node.id);
  const depth = Math.max(...[...nodes.values()].map((node) => node.rank)) + 1;
  const rows = Array.from({ length: depth }, () => []);
  for (const node of nodes.values()) rows[node.rank].push(node);

  // A link across several ranks passes through an invisible waypoint in each rank between.
  const edges = flow.links.map((link, index) => {
    const edge = { link, back: back.has(index), from: nodes.get(link.from), to: nodes.get(link.to), chain: [] };
    if (!edge.back) {
      edge.chain = [edge.from];
      for (let r = edge.from.rank + 1; r < edge.to.rank; r++) {
        const waypoint = { dummy: true, rank: r, w: DUMMY, h: 0, up: [], down: [] };
        rows[r].push(waypoint);
        edge.chain.push(waypoint);
      }
      edge.chain.push(edge.to);
      for (let i = 1; i < edge.chain.length; i++) {
        edge.chain[i - 1].down.push(edge.chain[i]);
        edge.chain[i].up.push(edge.chain[i - 1]);
      }
    }
    return edge;
  });

  rows.forEach((row) => row.forEach((node, pos) => { node.pos = pos; }));
  for (let round = 0; round < 3; round++) {
    for (let r = 1; r < rows.length; r++) reorder(rows[r], "up");
    for (let r = rows.length - 2; r >= 0; r--) reorder(rows[r], "down");
  }

  // Centre every rank on the same axis; ranks stack with a fixed gap.
  const rowWidth = rows.map((row) => row.reduce((sum, node) => sum + node.w, 0) + GAP * (row.length - 1));
  const widest = Math.max(...rowWidth);
  let top = EDGE;
  rows.forEach((row, r) => {
    const height = Math.max(0, ...row.filter((node) => !node.dummy).map((node) => node.h));
    let x = EDGE + (widest - rowWidth[r]) / 2;
    for (const node of row) {
      node.x = x + node.w / 2;
      node.y = top + height / 2;
      x += node.w + GAP;
    }
    top += height + ROW;
  });
  const right = Math.max(...[...nodes.values()].map((node) => node.x + node.w / 2));
  let lane = 0;
  for (const edge of edges) if (edge.back) edge.bend = right + LOOP + LANE * lane++;
  const width = Math.max(right, ...edges.map((edge) => edge.bend ?? 0)) + EDGE;
  const height = top - ROW + EDGE;
  return { nodes: [...nodes.values()], edges, width, height };
}
