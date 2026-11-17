/**
 * How it works: small animated figures, one idea each.
 *
 * Every figure is a pure function of time: draw(t) places its parts for second t of a
 * loop, so the picture can be paused, frozen for reduced motion, or rendered at any
 * moment. Numbers a figure shows are computed from the same values that move it.
 */

import { el, icon, clear, openSheet } from "./ui.js";
import { loadSense } from "./sense.js";

const NS = "http://www.w3.org/2000/svg";
const W = 480;
const H = 280;

function s(tag, attrs = {}, text = null) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== null) node.textContent = text;
  return node;
}

const clamp01 = (x) => Math.min(1, Math.max(0, x));
const ease = (x) => x * x * (3 - 2 * x);
/** 0 before a, 1 after b, eased in between. */
const span = (t, a, b) => ease(clamp01((t - a) / (b - a)));
/** 1 only between a and b, with soft edges. */
const during = (t, a, b, edge = 0.25) => Math.min(span(t, a, a + edge), 1 - span(t, b - edge, b));
const mix = (a, b, k) => a + (b - a) * k;
const at = (node, x, y) => node.setAttribute("transform", `translate(${x.toFixed(2)} ${y.toFixed(2)})`);
const fade = (node, k) => node.setAttribute("opacity", clamp01(k).toFixed(3));
const on = (node, yes) => node.classList.toggle("is-on", Boolean(yes));

function box(root, x, y, w, h, label, cls = "", labelY = y) {
  const group = s("g", { class: `hw-node ${cls}`.trim() });
  group.append(s("rect", { x: x - w / 2, y: y - h / 2, width: w, height: h, rx: 10 }));
  if (label) group.append(s("text", { x, y: labelY + 4.5, class: "hw-t", "text-anchor": "middle" }, label));
  root.append(group);
  return group;
}

function text(root, x, y, value, cls = "hw-t-muted", anchor = "middle") {
  const node = s("text", { x, y, class: cls, "text-anchor": anchor }, value);
  root.append(node);
  return node;
}

/* -------------------------------------------------------------------------- */
/* 1. The big picture                                                         */
/* -------------------------------------------------------------------------- */

function bigPicture(root) {
  const y = 100;
  const xs = [52, 146, 240, 334, 428];
  const names = ["You", "Context", "Model", "Kiwi Sense", "Answer"];
  for (let index = 0; index < xs.length - 1; index += 1) {
    root.append(s("path", { class: "hw-line", d: `M${xs[index] + 42} ${y}H${xs[index + 1] - 42}` }));
  }
  root.append(s("path", { class: "hw-line", d: `M226 ${y + 23}V199` }), s("path", { class: "hw-line", d: `M254 199V${y + 23}` }));
  root.append(s("path", { class: "hw-arrow", d: "M222 192l4 7 4-7" }), s("path", { class: "hw-arrow", d: `M250 ${y + 30}l4-7 4 7` }));
  root.append(s("path", { class: "hw-line hw-line-tan", d: `M146 ${y + 23}V160` }));
  const cards = s("g", { class: "hw-cards" });
  cards.append(s("rect", { x: 118, y: 160, width: 24, height: 17, rx: 3 }), s("rect", { x: 150, y: 160, width: 24, height: 17, rx: 3 }));
  root.append(cards);
  text(root, 146, 194, "memory, notes");
  const nodes = names.map((name, index) => box(root, xs[index], y, 84, 46, name, index === 3 ? "hw-node-tan" : ""));
  const tools = box(root, 240, 222, 80, 46, "Tools");
  const dot = s("circle", { r: 6, class: "hw-dot" });
  root.append(dot);

  // (time, x, y): the message's trip. Inside a box the dot is hidden and the box is lit.
  const stops = [
    [0, 52, y], [0.9, 52, y], [1.7, 146, y], [2.6, 146, y], [3.4, 240, y], [4.0, 240, y], [4.3, 226, y],
    [5.0, 226, 222], [5.5, 254, 222], [6.2, 254, y], [6.5, 240, y], [6.9, 240, y], [7.7, 334, y], [8.5, 334, y],
    [9.3, 428, y], [10.4, 428, y],
  ];
  const places = [...xs.map((x) => [x, y]), [240, 222]];
  const all = [...nodes, tools];

  return {
    duration: 10.4,
    still: 7.9,
    draw(t) {
      let index = 0;
      while (index < stops.length - 2 && t >= stops[index + 1][0]) index += 1;
      const [t0, x0, y0] = stops[index];
      const [t1, x1, y1] = stops[index + 1];
      const k = ease(clamp01((t - t0) / (t1 - t0)));
      const x = mix(x0, x1, k);
      const yy = mix(y0, y1, k);
      at(dot, x, yy);
      let inside = false;
      all.forEach((node, which) => {
        const here = Math.abs(x - places[which][0]) <= 40 && Math.abs(yy - places[which][1]) <= 23;
        on(node, here);
        inside = inside || here;
      });
      fade(dot, inside ? 0 : 1);
      on(cards, t > 1.7 && t < 2.8);
    },
  };
}

/* -------------------------------------------------------------------------- */
/* 2. The model file: weights and quantization                                */
/* -------------------------------------------------------------------------- */

function modelFile(root) {
  const x0 = 28;
  const x1 = 300;
  const yc = 128;
  const amp = 72;
  const wave = (x) => 0.55 * Math.sin((x / 170) * 6.283) + 0.3 * Math.sin((x / 61) * 6.283 + 1) + 0.15 * Math.sin((x / 29) * 6.283 + 2);
  const samples = [];
  for (let x = 0; x <= x1 - x0; x += 2) samples.push([x0 + x, wave(x)]);
  const path = (levels) => {
    const step = 2 / (levels - 1);
    let d = "";
    let last = null;
    for (const [x, value] of samples) {
      const y = yc - amp * Math.round(value / step) * step;
      d += last === null ? `M${x} ${y.toFixed(2)}` : `H${x}${y === last ? "" : `V${y.toFixed(2)}`}`;
      last = y;
    }
    return d;
  };
  const smooth = samples.map(([x, value], index) => `${index ? "L" : "M"}${x} ${(yc - amp * value).toFixed(2)}`).join("");
  root.append(s("path", { class: "hw-line hw-line-tan", d: smooth }));
  const stepped = s("path", { class: "hw-stroke-green", d: smooth });
  root.append(stepped);
  const bits = text(root, x0, 236, "", "hw-t", "start");
  text(root, x0, 254, "the same numbers, stored more and more coarsely", "hw-t-muted", "start");

  const bx = 372;
  const base = 214;
  const full = 172;
  root.append(s("rect", { class: "hw-frame", x: bx, y: base - full, width: 56, height: full, rx: 6 }));
  const bar = s("rect", { class: "hw-fill-green", x: bx, width: 56, rx: 6 });
  root.append(bar);
  text(root, bx + 28, 236, "file size", "hw-t");
  const smaller = text(root, bx + 28, 254, "");

  const steps = [[0, 16], [1.6, 8], [2.8, 6], [4.0, 5], [5.2, 4]];
  let shown = 0;

  return {
    duration: 9,
    still: 7,
    draw(t) {
      let index = 0;
      while (index < steps.length - 1 && t >= steps[index + 1][0]) index += 1;
      const now = steps[index][1];
      const before = index ? steps[index - 1][1] : now;
      if (now !== shown) {
        shown = now;
        stepped.setAttribute("d", now >= 12 ? smooth : path(2 ** now));
        bits.textContent = `${now} bits per number`;
        smaller.textContent = now === 16 ? "full size" : `${(16 / now).toFixed(Number.isInteger(16 / now) ? 0 : 1)}× smaller`;
      }
      const size = mix(before, now, span(t, steps[index][0], steps[index][0] + 0.5)) / 16;
      bar.setAttribute("y", (base - full * size).toFixed(2));
      bar.setAttribute("height", (full * size).toFixed(2));
    },
  };
}

/* -------------------------------------------------------------------------- */
/* 3. The context window, tokens and compaction                               */
/* -------------------------------------------------------------------------- */

function contextWindow(root) {
  const slots = 12;
  const x0 = 40;
  const slot = 400 / slots;
  const y = 84;
  text(root, x0, 66, "context window", "hw-t", "start");
  const count = text(root, 440, 66, "", "hw-t-muted", "end");
  root.append(s("rect", { class: "hw-frame", x: x0 - 6, y: y - 6, width: 412, height: 68, rx: 12 }));
  const sx = (index) => x0 + slot * index + 3;
  const tile = (cls) => {
    const node = s("rect", { class: cls, width: slot - 6, height: 56, rx: 5, y });
    root.append(node);
    return node;
  };
  const first = Array.from({ length: 9 }, () => tile("hw-tile"));
  const summary = tile("hw-tile hw-tile-tan");
  const later = Array.from({ length: 4 }, () => tile("hw-tile"));
  const summaryLabel = text(root, sx(0), y + 76, "summary", "hw-t-tan", "start");

  const my = 196;
  root.append(s("rect", { class: "hw-track", x: x0, y: my, width: 400, height: 6, rx: 3 }));
  const meter = s("rect", { class: "hw-fill-green", x: x0, y: my, height: 6, rx: 3 });
  root.append(meter);
  const mark = s("path", { class: "hw-mark", d: `M${x0 + 300} ${my - 7}v20` });
  root.append(mark);
  text(root, x0 + 300, my + 30, "three quarters full: fold the oldest");

  return {
    duration: 11.5,
    still: 6.6,
    draw(t) {
      const fold = span(t, 5.0, 6.4);
      const out = 1 - span(t, 10.8, 11.4);
      let used = 0;
      first.forEach((node, index) => {
        const born = span(t, 0.5 * index, 0.5 * index + 0.35);
        const target = index < 6 ? 0 : index - 5;
        node.setAttribute("x", (mix(sx(index), sx(target), fold) + (1 - born) * 30).toFixed(2));
        fade(node, born * (index < 6 ? 1 - fold : 1) * out);
        used += born * (index < 6 ? 1 - fold : 1);
        on(node, born > 0 && born < 1);
      });
      summary.setAttribute("x", sx(0).toFixed(2));
      fade(summary, fold * out);
      fade(summaryLabel, fold * out);
      used += fold;
      later.forEach((node, index) => {
        const start = 6.8 + 0.6 * index;
        const born = span(t, start, start + 0.35);
        node.setAttribute("x", (sx(4 + index) + (1 - born) * 30).toFixed(2));
        fade(node, born * out);
        used += born;
        on(node, born > 0 && born < 1);
      });
      used *= out;
      meter.setAttribute("width", ((400 * used) / slots).toFixed(2));
      on(mark, t > 4.4 && t < 6.4);
      count.textContent = `${Math.round(used)} of ${slots} slots used`;
    },
  };
}

/* -------------------------------------------------------------------------- */
/* 4. The KV cache                                                            */
/* -------------------------------------------------------------------------- */

function kvCache(root) {
  const x0 = 32;
  const slot = 32;
  const y = 96;
  const sx = (index) => x0 + slot * index;
  const tiles = [];
  const marks = [];
  for (let index = 0; index < 13; index += 1) {
    const cls = index < 3 ? "hw-tile hw-tile-tan" : index >= 10 ? "hw-tile hw-tile-new" : "hw-tile";
    const tile = s("rect", { class: cls, x: sx(index) + 3, y, width: slot - 6, height: 48, rx: 5 });
    const mark = s("rect", { class: "hw-cache", x: sx(index) + 3, y: y + 56, width: slot - 6, height: 5, rx: 2.5 });
    root.append(tile, mark);
    tiles.push(tile);
    marks.push(mark);
  }
  text(root, sx(0) + 3, y - 26, "fixed start", "hw-t-tan", "start");
  const newLabel = text(root, sx(13) - 3, y - 26, "new message", "hw-t-green", "end");
  text(root, sx(0) + 3, y + 80, "already read, kept in the KV cache", "hw-t-muted", "start");
  const head = s("g", { class: "hw-head" });
  head.append(s("path", { d: `M0 ${y - 8}V${y + 54}` }), s("path", { class: "hw-head-tip", d: `M-6 ${y - 16}h12l-6 9z` }));
  root.append(head);
  const firstLine = text(root, sx(0) + 3, 222, "", "hw-t", "start");
  const secondLine = text(root, sx(0) + 3, 244, "", "hw-t", "start");
  const perTile = 0.34;

  return {
    duration: 11,
    still: 8.6,
    draw(t) {
      const firstRead = clamp01((t - 0.7) / (10 * perTile)) * 10;
      const secondRead = clamp01((t - 6.4) / (3 * perTile)) * 3;
      const arrived = span(t, 5.2, 5.8);
      const out = 1 - span(t, 10.4, 10.95);
      tiles.forEach((tile, index) => {
        if (index >= 10) {
          fade(tile, arrived * out);
          tile.setAttribute("y", (y - (1 - arrived) * 16).toFixed(2));
        } else fade(tile, out);
      });
      fade(newLabel, arrived * out);
      const read = firstRead + secondRead;
      marks.forEach((mark, index) => on(mark, read >= index + 1));
      marks.forEach((mark) => fade(mark, out));
      at(head, sx(0) + slot * read, 0);
      fade(head, (t < 0.5 ? span(t, 0.1, 0.5) : 1) * out);
      firstLine.textContent = firstRead > 0 ? `1st message: read ${Math.round(firstRead)} tiles` : "";
      secondLine.textContent = t > 6.4 ? `2nd message: read ${Math.round(secondRead)}, not ${10 + 3}` : "";
      fade(firstLine, out);
      fade(secondLine, out);
    },
  };
}

/* -------------------------------------------------------------------------- */
/* 5. Tools and the agent loop                                                */
/* -------------------------------------------------------------------------- */

function toolsLoop(frame) {
  const root = s("g", { transform: "translate(0 16)" });
  frame.append(root);
  const up = 126;
  const down = 150;
  root.append(s("path", { class: "hw-line", d: `M112 ${up}H198` }), s("path", { class: "hw-line", d: `M282 ${up}H366` }));
  root.append(s("path", { class: "hw-line", d: `M366 ${down}H282` }), s("path", { class: "hw-line", d: `M198 ${down}H112` }));
  root.append(s("path", { class: "hw-line", d: "M352 84V192M352 84H366M352 192H366" }));
  text(root, 155, up - 16, "tool call");
  text(root, 155, down + 24, "result");
  const model = box(root, 66, 132, 92, 96, "Model", "", 102);
  const kiwi = box(root, 240, 132, 84, 96, "Kiwibro", "hw-node-tan", 102);
  const world = ["the web", "files", "programs"].map((name, index) => {
    const group = box(root, 412, 84 + index * 54, 92, 36, name);
    return group;
  });
  const ticket = s("g", { class: "hw-ticket" });
  ticket.append(s("rect", { x: -17, y: -11, width: 34, height: 22, rx: 4 }), s("path", { d: "M-10 -3h20M-10 4h12" }));
  const stamp = s("path", { class: "hw-check", d: "M-7 0l5 5 9-10" });
  ticket.append(stamp);
  const sheet = s("g", { class: "hw-sheet" });
  sheet.append(s("rect", { x: -17, y: -11, width: 34, height: 22, rx: 4 }), s("path", { d: "M-10 -4h20M-10 1h20M-10 6h12" }));
  const answer = s("g", { class: "hw-answer" });
  answer.append(s("rect", { x: 22, y: 30, width: 88, height: 36, rx: 10 }), s("path", { d: "M34 43h62M34 53h40" }));
  answer.append(s("text", { x: 66, y: 22, class: "hw-t-muted", "text-anchor": "middle" }, "answer"));
  root.append(ticket, sheet, answer);

  return {
    duration: 10,
    still: 3.4,
    draw(t) {
      const out = 1 - span(t, 9.4, 9.95);
      const toKiwi = span(t, 0.6, 1.8);
      const checked = span(t, 2.0, 2.5);
      const toWorld = span(t, 2.9, 4.0);
      at(ticket, mix(mix(128, 240, toKiwi), 384, toWorld), up);
      fade(ticket, during(t, 0.4, 4.4));
      fade(stamp, checked);
      on(kiwi, (t > 1.8 && t < 3.0) || (t > 5.6 && t < 6.5));
      on(world[1], t > 4.0 && t < 5.4);
      const back1 = span(t, 5.0, 5.9);
      const back2 = span(t, 6.2, 7.2);
      at(sheet, mix(mix(384, 240, back1), 128, back2), down);
      fade(sheet, during(t, 4.8, 7.5));
      on(model, t < 0.7 || (t > 7.2 && t < 9.4));
      const said = span(t, 7.6, 8.3);
      fade(answer, said * out);
      at(answer, 0, (1 - said) * 12);
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Ready parts: code picks a kit or recipe from the shelf                     */
/* -------------------------------------------------------------------------- */

function readyParts(root) {
  const COLS = 6;
  const PICK = 9;
  root.append(s("path", { class: "hw-line", d: "M100 82V152H138" }), s("path", { class: "hw-line", d: "M342 152H372" }));
  root.append(s("path", { class: "hw-line", d: "M420 102V88" }));
  text(root, 100, 32, "your message");
  const message = box(root, 100, 60, 164, 40, "make a dashboard");
  const tiles = Array.from({ length: 18 }, (_, index) => {
    const tile = s("rect", {
      class: "hw-part",
      x: 147 + (index % COLS) * 32,
      y: 118 + Math.floor(index / COLS) * 25,
      width: 26,
      height: 18,
      rx: 4,
    });
    root.append(tile);
    return tile;
  });
  text(root, 240, 216, "77 ready parts", "hw-t-tan");
  const chosen = text(root, 240, 236, "picked by code, not by the model");
  const model = box(root, 420, 152, 88, 96, "Model", "", 152);
  const card = s("rect", { class: "hw-part is-on", x: -13, y: -9, width: 26, height: 18, rx: 4 });
  root.append(card);
  const files = [0, 1, 2].map((index) => {
    const sheet = s("g", { class: "hw-sheet" });
    sheet.append(s("rect", { x: 380 + index * 30, y: 44, width: 22, height: 30, rx: 4 }));
    sheet.append(s("path", { d: `M${385 + index * 30} 53h12M${385 + index * 30} 59h12M${385 + index * 30} 65h7` }));
    root.append(sheet);
    return sheet;
  });
  const written = text(root, 420, 32, "working files");
  const home = { x: 147 + (PICK % COLS) * 32 + 13, y: 118 + Math.floor(PICK / COLS) * 25 + 9 };

  return {
    duration: 9,
    still: 6.4,
    draw(t) {
      const out = 1 - span(t, 8.4, 8.95);
      on(message, t > 0.3 && t < 1.4);
      const scan = Math.floor((t - 1.2) / 0.09);
      tiles.forEach((tile, index) => on(tile, (t > 1.2 && t < 2.8 && index === scan) || (t >= 2.8 && t < 3.3 && index === PICK)));
      fade(chosen, span(t, 2.8, 3.3) * out);
      const travel = span(t, 3.3, 4.4);
      at(card, mix(home.x, 400, travel), mix(home.y, 152, travel));
      fade(card, during(t, 3.2, 4.7, 0.15));
      on(model, t > 4.3 && t < 8.4);
      files.forEach((sheet, index) => fade(sheet, span(t, 4.9 + index * 0.45, 5.3 + index * 0.45) * out));
      fade(written, span(t, 5.0, 5.5) * out);
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Knowledge base: a strong match is handed over, anything less adds nothing   */
/* -------------------------------------------------------------------------- */

function knowledgeBase(root) {
  const PER = 11;
  const PICK = 15;
  root.append(s("path", { class: "hw-line", d: "M100 82V150H138" }), s("path", { class: "hw-line", d: "M346 150H372" }));
  text(root, 100, 32, "your question");
  const asked = box(root, 100, 60, 172, 40, "capital of Australia?");
  const other = box(root, 100, 60, 172, 40, "make the title bigger");
  const spines = [];
  for (let shelf = 0; shelf < 3; shelf += 1) {
    const floor = 128 + shelf * 30;
    root.append(s("path", { class: "hw-line", d: `M144 ${floor}H340` }));
    for (let index = 0; index < PER; index += 1) {
      const tall = 15 + ((index * 7 + shelf * 5) % 4) * 3;
      const spine = s("rect", { class: "hw-part", x: 149 + index * 17, y: floor - tall - 1, width: 13, height: tall, rx: 2.5 });
      root.append(spine);
      spines.push(spine);
    }
  }
  text(root, 242, 216, "24 topics, about 2,000 entries", "hw-t-tan");
  const found = text(root, 242, 236, "a strong match goes to the model");
  const silent = text(root, 242, 236, "no strong match: nothing is added");
  const model = box(root, 420, 150, 88, 96, "Model", "", 150);
  const card = s("rect", { class: "hw-part is-on", x: -6.5, y: -10, width: 13, height: 20, rx: 2.5 });
  root.append(card);
  const home = { x: 149 + (PICK % PER) * 17 + 6.5, y: 128 + Math.floor(PICK / PER) * 30 - 11 };
  const count = spines.length;

  return {
    duration: 11,
    still: 3,
    draw(t) {
      // First a question the shelf covers, then a request it has nothing to say about.
      const second = span(t, 5.1, 5.5) * (1 - span(t, 10.5, 10.95));
      fade(asked, 1 - second);
      fade(other, second);
      on(asked, t > 0.3 && t < 1.3);
      on(other, t > 5.8 && t < 6.8);
      // The first search stops at the entry it finds; the second runs to the end and finds none.
      const scan = t < 5 ? (t < 3.3 ? Math.min(PICK, Math.floor(((t - 1.2) / 1.4) * count)) : -1) : Math.floor(((t - 6.7) / 1.4) * count);
      spines.forEach((spine, index) => on(spine, index === scan));
      fade(found, span(t, 1.9, 2.4) * (1 - span(t, 4.8, 5.2)));
      const travel = span(t, 3.3, 4.3);
      at(card, mix(home.x, 398, travel), mix(home.y, 150, travel));
      fade(card, during(t, 3.2, 4.6, 0.15));
      fade(silent, span(t, 8.1, 8.6) * (1 - span(t, 10.2, 10.6)));
      on(model, (t > 4.2 && t < 5) || (t > 8.6 && t < 10.2));
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Shown in the chat                                                          */
/* -------------------------------------------------------------------------- */

function shownInChat(root) {
  const DOTS = 9;
  const cx = 303;
  const cy = 166;
  root.append(s("path", { class: "hw-line", d: "M106 140H150" }));
  const tools = box(root, 64, 140, 84, 46, "Tools");
  root.append(s("rect", { class: "hw-slab", x: 158, y: 40, width: 290, height: 200, rx: 10 }));
  text(root, 172, 64, "~/projects/site", "hw-t-muted", "start");
  text(root, 434, 64, "main", "hw-t-tan", "end");
  const shown = s("g");
  shown.append(s("rect", { class: "hw-mask", x: 172, y: 104, width: 262, height: 124, rx: 6 }));
  const seal = s("rect", { class: "hw-frame", x: 172, y: 104, width: 262, height: 124, rx: 6 });
  shown.append(seal);
  text(shown, 172, 92, "index.html", "hw-t", "start");
  const buttons = [392, 410, 428].map((x) => s("rect", { class: "hw-part", x: x - 7, y: 81, width: 14, height: 14, rx: 3 }));
  const dots = Array.from({ length: DOTS }, () => s("circle", { class: "hw-dot" }));
  shown.append(...buttons, ...dots);
  root.append(shown);
  const file = s("rect", { class: "hw-part is-on", x: -9, y: -12, width: 18, height: 24, rx: 3 });
  root.append(file);
  const captions = ["a file it writes appears in the chat", "a page runs live, in a sealed frame", "open it, show its folder, copy its path"].map((line) => text(root, cx, 262, line));

  return {
    duration: 10.5,
    still: 5.6,
    draw(t) {
      on(tools, t > 0.3 && t < 1.4);
      const travel = span(t, 1.1, 2.1);
      at(file, mix(106, 190, travel), mix(140, 132, travel));
      fade(file, during(t, 1, 2.4, 0.2));
      fade(shown, span(t, 2.1, 2.7) * (1 - span(t, 9.9, 10.3)));
      // A ring of dots turning in depth: the page inside the frame is running, not a picture of it.
      dots.forEach((dot, index) => {
        const angle = t * 1.1 + (index * 2 * Math.PI) / DOTS;
        const depth = Math.sin(angle);
        at(dot, cx + 86 * Math.cos(angle), cy + 26 * depth);
        dot.setAttribute("r", (3.4 + 1.8 * depth).toFixed(2));
        dot.setAttribute("opacity", (0.65 + 0.35 * depth).toFixed(2));
      });
      fade(captions[0], during(t, 2.3, 4.6));
      fade(captions[1], during(t, 4.8, 7.1));
      fade(captions[2], during(t, 7.3, 9.9));
      buttons.forEach((button, index) => on(button, t > 7.5 + index * 0.7 && t < 8.1 + index * 0.7));
    },
  };
}

/* -------------------------------------------------------------------------- */
/* 6. Checked answers                                                         */
/* -------------------------------------------------------------------------- */

function checkedAnswers(root) {
  const ys = [94, 140, 186];
  const cardRight = 186;
  const rowLeft = 294;
  text(root, 36, 62, "what was found", "hw-t-muted", "start");
  text(root, rowLeft, 62, "the draft", "hw-t-muted", "start");
  const cards = ys.map((y, index) => {
    const group = s("g", { class: "hw-card" });
    group.append(
      s("rect", { x: 36, y: y - 18, width: 150, height: 36, rx: 6 }),
      s("path", { d: `M48 ${y - 5}h${[112, 96, 104][index]}M48 ${y + 6}h${[70, 88, 58][index]}` }),
    );
    root.append(group);
    return group;
  });
  const links = ys.map(() => {
    const link = s("path", { class: "hw-proof" });
    root.append(link);
    return link;
  });
  const miss = s("path", { class: "hw-proof is-off" });
  root.append(miss);
  const rows = ys.map((y, index) => {
    const group = s("g", { class: "hw-chip" });
    group.append(
      s("rect", { x: rowLeft, y: y - 14, width: 150, height: 28, rx: 8 }),
      s("path", { d: `M${rowLeft + 12} ${y}h${[92, 78, 98][index]}` }),
    );
    const tick = s("path", { class: "hw-tick", d: `M${rowLeft + 124} ${y}l4 4 8-9` });
    group.append(tick);
    root.append(group);
    return { group, tick };
  });
  const verdict = text(root, rowLeft + 75, 224, "", "hw-t-orange");
  const reach = (link, y, k) => link.setAttribute("d", `M${rowLeft} ${y}H${mix(rowLeft, cardRight, k).toFixed(2)}`);

  return {
    duration: 11,
    still: 9.2,
    draw(t) {
      const out = 1 - span(t, 10.4, 10.95);
      rows.forEach(({ group }, index) => fade(group, span(t, 0.3 + index * 0.3, 0.8 + index * 0.3) * out));
      // Two sentences find their source at once.
      [1.6, 2.7].forEach((start, index) => {
        const k = span(t, start, start + 0.8);
        reach(links[index], ys[index], k);
        fade(links[index], (k > 0 ? 1 : 0) * out);
        fade(rows[index].tick, span(t, start + 0.8, start + 1.1));
        on(cards[index], t > start + 0.8 && t < 10.4);
      });
      // The third reaches for one and finds none: sent back once, written again, then it holds.
      const tried = span(t, 3.9, 4.7) * (1 - span(t, 4.9, 5.4));
      reach(miss, ys[2], tried * 0.7);
      fade(miss, tried > 0 ? 1 : 0);
      const off = t >= 4.9 && t < 6.9;
      rows[2].group.classList.toggle("is-off", off);
      if (t >= 6.4 && t < 7.4) fade(rows[2].group, Math.abs(t - 6.9) / 0.5);
      const held = span(t, 7.6, 8.4);
      reach(links[2], ys[2], held);
      fade(links[2], (held > 0 ? 1 : 0) * out);
      fade(rows[2].tick, span(t, 8.4, 8.7));
      on(cards[2], t > 8.4 && t < 10.4);
      verdict.setAttribute("class", t < 6.4 ? "hw-t-orange" : "hw-t-tan");
      verdict.textContent = t >= 4.9 && t < 6.4 ? "no source says this" : t >= 6.4 && t < 8.4 ? "written again" : "";
    },
  };
}

/* -------------------------------------------------------------------------- */
/* 7. Kiwi Sense                                                              */
/* -------------------------------------------------------------------------- */

function kiwiSense(root) {
  const y = 128;
  root.append(s("path", { class: "hw-line", d: `M106 ${y}H212` }), s("path", { class: "hw-line", d: `M268 ${y}H384` }));
  box(root, 60, y, 92, 64, "Model");
  box(root, 424, y, 80, 64, "You");
  const gate = s("g", { class: "hw-gate" });
  gate.append(s("circle", { cx: 240, cy: y, r: 28, class: "hw-gate-ring" }), s("circle", { cx: 240, cy: y, r: 9, class: "hw-gate-core" }));
  for (let index = 0; index < 8; index += 1) {
    gate.append(s("ellipse", { cx: 240, cy: y - 17, rx: 1.6, ry: 3.4, class: "hw-gate-seed", transform: `rotate(${index * 45} 240 ${y})` }));
  }
  root.append(gate);
  text(root, 240, y - 40, "Kiwi Sense", "hw-t");
  const verdict = text(root, 240, y + 52, "", "hw-t-orange");
  const again = text(root, 160, y + 62, "asked again, with a short note", "hw-t-tan");
  const chip = () => {
    const group = s("g", { class: "hw-chip" });
    group.append(s("rect", { x: -26, y: -9, width: 52, height: 18, rx: 9 }), s("path", { d: "M-16 0h32" }));
    root.append(group);
    return group;
  };
  const chips = [chip(), chip(), chip()];
  const note = s("rect", { class: "hw-note", x: -8, y: 11, width: 16, height: 9, rx: 2 });
  chips[2].append(note);
  const run = (t, start) => mix(120, 392, span(t, start, start + 2.4));

  return {
    duration: 11,
    still: 4.9,
    draw(t) {
      [0, 1].forEach((index) => {
        const start = 0.3 + index * 1.3;
        at(chips[index], run(t, start), y);
        fade(chips[index], during(t, start, start + 2.6, 0.2));
      });
      // The third sentence drifts: stopped at the gate, sent back, written again.
      const toGate = span(t, 3.0, 4.2);
      const back = span(t, 4.9, 6.0);
      const retry = span(t, 6.9, 9.3);
      const third = chips[2];
      const x = retry > 0 ? mix(120, 392, retry) : mix(mix(120, 212, toGate), 120, back);
      at(third, x, y);
      fade(third, during(t, 2.8, 9.6, 0.2));
      third.classList.toggle("is-off", t >= 4.2 && t < 6.4);
      fade(note, during(t, 5.8, 7.6, 0.3));
      const caught = t >= 4.2 && t < 5.4;
      const passing = [run(t, 0.3), run(t, 1.6)].some((where) => Math.abs(where - 240) < 26) || (retry > 0 && Math.abs(x - 240) < 26);
      gate.classList.toggle("is-ok", passing && !caught);
      gate.classList.toggle("is-off", caught);
      verdict.textContent = t >= 4.2 && t < 6.0 ? "off track" : "";
      fade(again, during(t, 5.6, 7.8));
    },
  };
}

/* -------------------------------------------------------------------------- */
/* 8. Memory and notes                                                        */
/* -------------------------------------------------------------------------- */

function memoryNotes(root) {
  text(root, 36, 46, "memory and notes, on this computer", "hw-t-muted", "start");
  const cards = [];
  const picked = [2, 9];
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      const index = row * 4 + col;
      const group = s("g", { class: `hw-card${index % 3 === 1 ? " hw-card-note" : ""}` });
      group.append(s("rect", { x: 0, y: 0, width: 44, height: 30, rx: 5 }), s("path", { d: `M8 11h${index % 2 ? 20 : 28}M8 19h${index % 2 ? 26 : 16}` }));
      at(group, 36 + col * 54, 62 + row * 42);
      root.append(group);
      cards.push({ group, x: 36 + col * 54, y: 62 + row * 42 });
    }
  }
  const frame = s("rect", { class: "hw-frame", x: 276, y: 52, width: 176, height: 104, rx: 12 });
  root.append(frame);
  const bubble = s("g", { class: "hw-bubble" });
  bubble.append(s("rect", { x: 288, y: 62, width: 152, height: 38, rx: 10 }), s("path", { d: "M300 76h44M300 88h82" }));
  const key = s("path", { class: "hw-key", d: "M352 76h38" });
  bubble.append(key);
  root.append(bubble);
  const noteLabel = text(root, 364, 174, "context note", "hw-t-tan");
  root.append(s("path", { class: "hw-line", d: "M364 186V214" }), s("path", { class: "hw-arrow", d: "M360 208l4 7 4-7" }));
  const model = box(root, 364, 240, 92, 44, "Model");
  const flying = picked.map(() => {
    const group = s("g", { class: "hw-card is-on" });
    group.append(s("rect", { x: 0, y: 0, width: 44, height: 30, rx: 5 }), s("path", { d: "M8 11h28M8 19h16" }));
    root.append(group);
    return group;
  });

  return {
    duration: 10,
    still: 5.2,
    draw(t) {
      const out = 1 - span(t, 9.4, 9.95);
      const asked = span(t, 0.3, 0.9);
      fade(bubble, asked * out);
      at(bubble, 0, (1 - asked) * 10);
      on(key, t > 1.2);
      const found = t > 1.8 && t < 9.4;
      cards.forEach((card, index) => on(card.group, found && picked.includes(index)));
      const fly = span(t, 2.6, 4.0);
      flying.forEach((group, which) => {
        const from = cards[picked[which]];
        at(group, mix(from.x, 300 + which * 56, fly), mix(from.y, 112, fly));
        fade(group, (t > 2.6 ? 1 : 0) * out);
      });
      fade(frame, span(t, 3.8, 4.4) * out);
      fade(noteLabel, span(t, 3.8, 4.4) * out);
      on(model, t > 5.6 && t < 9.4);
    },
  };
}

/* -------------------------------------------------------------------------- */
/* 9. LoRA adapters                                                           */
/* -------------------------------------------------------------------------- */

function adapters(root) {
  const STEP = 17;
  const TOP = 76;
  const grid = (x, y, rows, cols) => {
    const cells = [];
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const cell = s("rect", { class: "hw-cell", x: x + col * STEP, y: y + row * STEP, width: 14, height: 14, rx: 3 });
        root.append(cell);
        cells.push(cell);
      }
    }
    return cells;
  };
  const tasks = ["code", "writing", "translate"];
  const chips = tasks.map((name, index) => box(root, 148 + index * 92, 38, 84, 30, name));
  // One table of the model's numbers, and beside it the adapter: a tall thin table times a wide thin one.
  const frozen = grid(52, TOP, 8, 8);
  const tall = grid(222, TOP, 8, 2);
  const wide = grid(288, TOP + 3 * STEP, 2, 8);
  text(root, 204, TOP + 73, "+", "hw-t");
  text(root, 270, TOP + 73, "\u00d7", "hw-t");
  text(root, 118, TOP + 152, "a table in the model: frozen");
  text(root, 322, TOP + 152, "the adapter: two thin tables");
  // Each task has its own pair of thin tables. Their product is what gets added to the big one.
  const pairs = tasks.map((_, which) => {
    const down = Array.from({ length: 16 }, (_, index) => (((index >> 1) * (3 + which) + (index & 1) * 5 + which * 2) % 5) / 4);
    const across = Array.from({ length: 16 }, (_, index) => (((index % 8) * (2 + which) + (index >> 3) * 3 + which) % 5) / 4);
    const sum = Array.from({ length: 64 }, (_, index) => {
      const row = index >> 3;
      const col = index % 8;
      return (down[row * 2] * across[col] + down[row * 2 + 1] * across[8 + col]) / 2;
    });
    return { down, across, sum };
  });

  return {
    duration: 10.5,
    still: 2.4,
    draw(t) {
      const which = Math.min(2, Math.floor(t / 3.5));
      const local = t - which * 3.5;
      const leave = 1 - span(local, 2.8, 3.4);
      const shown = span(local, 0.3, 0.9) * leave;
      const added = span(local, 1.2, 2.2) * leave;
      const { down, across, sum } = pairs[which];
      chips.forEach((chip, index) => on(chip, index === which && local > 0.15 && local < 3.3));
      tall.forEach((cell, index) => cell.style.setProperty("--tint", (down[index] * shown).toFixed(3)));
      wide.forEach((cell, index) => cell.style.setProperty("--tint", (across[index] * shown).toFixed(3)));
      frozen.forEach((cell, index) => cell.style.setProperty("--tint", (sum[index] * added).toFixed(3)));
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Kiwi Mind: one model, eight skills, an optional helper                     */
/* -------------------------------------------------------------------------- */

function kiwiMind(root) {
  const slabs = [67, 91, 115, 139, 163];
  const skills = ["chat", "web", "terminal", "computer", "tools", "write", "see"];
  const lowerX = 58;
  const headX = 187;
  const upperX = 316;
  const railY = 122;

  // Left: the 24 shared layers, a stack of thin slabs.
  const lower = box(root, lowerX, 122, 96, 176, "", "hw-node-tan");
  for (const y of slabs) root.append(s("rect", { class: "hw-slab", x: lowerX - 34, y, width: 68, height: 14, rx: 4 }));
  text(root, lowerX, 232, "24 shared", "hw-t-big");
  text(root, lowerX, 252, "layers", "hw-t-big");

  // The path the message takes, with the route on it.
  root.append(s("path", { class: "hw-line", d: `M106 ${railY}H125` }), s("path", { class: "hw-line", d: `M249 ${railY}H268` }));
  root.append(s("path", { class: "hw-arrow", d: `M262 ${railY - 4}l6 4-6 4` }));
  const head = box(root, headX, railY, 124, 44, "", "hw-node");
  text(root, headX, railY + 5, "route", "hw-t-big");

  // Right: the 12 upper layers, with the code skill slotted in.
  const upper = box(root, upperX, 122, 96, 176, "", "hw-node-tan");
  slabs.forEach((y, index) => {
    if (index === 2) return;
    root.append(s("rect", { class: "hw-slab", x: upperX - 34, y, width: 68, height: 14, rx: 4 }));
  });
  root.append(s("rect", { class: "hw-slot", x: upperX - 34, y: 109, width: 68, height: 26, rx: 6 }));
  text(root, upperX, 127, "code", "hw-t-big-ink");
  text(root, upperX, 232, "12 upper", "hw-t-big");
  text(root, upperX, 252, "layers", "hw-t-big");

  // The seven other skills, quieter, beside the upper layers.
  skills.forEach((name, index) => {
    const y = 26 + index * 28;
    const group = s("g", { class: "hw-card" });
    group.append(s("rect", { x: 380, y, width: 88, height: 22, rx: 6 }));
    group.append(s("text", { x: 424, y: y + 15, class: "hw-t-big", "text-anchor": "middle" }, name));
    root.append(group);
  });

  // The dashed line from the route to the helper model, with its label on the line.
  root.append(s("path", { class: "hw-dash", d: `M${headX} ${railY + 22}V226` }));
  root.append(s("rect", { class: "hw-mask", x: headX - 58, y: 172, width: 116, height: 24 }));
  text(root, headX, 189, "heavy answers", "hw-t-big-muted");
  box(root, headX, 242, 124, 34, "", "hw-node");
  text(root, headX, 247, "helper model", "hw-t-big");

  // A dot carries the message in and out through the three blocks; it hides inside a box.
  const dot = s("circle", { r: 6, class: "hw-dot" });
  root.append(dot);
  const parts = [
    { x: lowerX, half: 48 },
    { x: headX, half: 56 },
    { x: upperX, half: 48 },
  ];

  return {
    duration: 6,
    still: 1.4,
    draw(t) {
      // Out to the upper layers and back again, one unbroken loop.
      const travel = t < 3 ? ease(clamp01(t / 3)) : ease(clamp01((6 - t) / 3));
      const x = mix(lowerX, upperX, travel);
      at(dot, x, railY);
      const inside = parts.map((part) => Math.abs(x - part.x) <= part.half);
      fade(dot, inside.some(Boolean) ? 0 : 1);
      on(lower, inside[0]);
      on(head, inside[1]);
      on(upper, inside[2]);
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Training data: tasks go to a teacher model, practice conversations come out */
/* -------------------------------------------------------------------------- */

// Counts from the pack's own record of its training (manifest.json of Kiwi Mind 3.0.0).
const LESSONS = { total: 6673, written: 3673, judged: 3000, trunk: 6130, held: 543 };
const LOSS = { base: 1.4563, trained: 0.7386, quantized: 0.9898, layered: 0.9514, steps: 205 };
const PACK = { gb: 2.6, layerMb: 41, ctx: 16384, trunkMinutes: 17.1, route: 93, act: 93, named: 76, valid: 92, pictures: 12000 };
const fmt = (value) => value.toLocaleString("en");
const about = (value) => fmt(Math.round(value / 100) * 100);

/** The point k (0 to 1) of the way along a path of straight pieces. */
function along(points, k) {
  const lengths = points.slice(1).map(([x, y], index) => Math.hypot(x - points[index][0], y - points[index][1]));
  let left = clamp01(k) * lengths.reduce((sum, length) => sum + length, 0);
  for (let index = 0; index < lengths.length - 1; index += 1) {
    if (left <= lengths[index]) break;
    left -= lengths[index];
    points = points.slice(1);
  }
  const share = Math.min(1, left / Math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]));
  return [mix(points[0][0], points[1][0], share), mix(points[0][1], points[1][1], share)];
}

function trainingData(root) {
  const total = LESSONS.total;
  const sources = [["oasst2", 66], ["Dolly", 112], ["MASSIVE", 158]];
  const teacherX = 226;
  const teacherY = 112;
  const pileX = 404;
  text(root, 62, 32, "real questions");
  const cards = sources.map(([name, y]) => {
    root.append(s("path", { class: "hw-line", d: `M110 ${y}H136V${teacherY}H164` }));
    const group = s("g", { class: "hw-card" });
    group.append(s("rect", { x: 14, y: y - 15, width: 96, height: 30, rx: 6 }));
    group.append(s("text", { x: 62, y: y + 5, class: "hw-t", "text-anchor": "middle" }, name));
    root.append(group);
    return group;
  });
  root.append(s("path", { class: "hw-line", d: `M288 ${teacherY}H340` }), s("path", { class: "hw-arrow", d: `M334 ${teacherY - 4}l6 4-6 4` }));
  const teacher = box(root, teacherX, teacherY, 124, 64, "", "hw-node");
  text(root, teacherX, teacherY + 5, "teacher model", "hw-t-big");

  // The pile of finished conversations grows upward, one sheet per sixth of the total.
  const sheets = [0, 1, 2, 3, 4, 5].map((index) => {
    const sheet = s("rect", { class: "hw-slab", x: pileX - 44, y: 142 - index * 15, width: 88, height: 11, rx: 3 });
    root.append(sheet);
    return sheet;
  });
  const count = text(root, pileX, 180, "0", "hw-t-big");
  text(root, pileX, 198, "lessons");
  const number = (value) => value.toLocaleString("en");
  text(root, 240, 252, `${number(LESSONS.trunk)} to learn from · ${LESSONS.held} held back`);
  const dot = s("circle", { r: 6, class: "hw-dot" });
  root.append(dot);

  return {
    duration: 8,
    still: 7.4,
    draw(t) {
      // Three rounds of two seconds: a task goes in, the teacher writes, conversations come out.
      const round = Math.min(2, Math.floor(t / 2));
      const local = t - round * 2;
      const y = sources[round][1];
      const outbound = span(local, 1.3, 1.9);
      const [x, yy] = outbound > 0
        ? along([[288, teacherY], [340, teacherY]], outbound)
        : along([[110, y], [136, y], [136, teacherY], [164, teacherY]], span(local, 0, 0.9));
      at(dot, x, yy);
      const writing = local >= 0.9 && local < 1.3;
      fade(dot, writing || local >= 1.9 ? 0 : 1);
      on(teacher, writing);
      cards.forEach((card, index) => on(card, index === round && local < 0.9));
      const done = (round + outbound) / 3;
      count.textContent = number(Math.round(total * done));
      sheets.forEach((sheet, index) => fade(sheet, done > (index + 0.5) / 6 ? 1 : 0.18));
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Training: the loss on held-back conversations, before and after            */
/* -------------------------------------------------------------------------- */

function training(root) {
  const floor = 204;
  const scale = 140 / Math.max(LOSS.base, LOSS.trained, LOSS.quantized, LOSS.layered, 0.01); // pixels per unit of loss
  const width = 76;
  const xs = [72, 184, 296, 408];
  const labels = ["as downloaded", "after training", "squeezed small", "with its skills"];
  text(root, 24, 30, "how wrong its next-word guesses are (lower is better)", "hw-t-muted", "start");
  root.append(s("path", { class: "hw-line", d: `M24 ${floor}H456` }));
  const bars = xs.map((x, index) => {
    const group = s("g", { class: `hw-bar ${["", "hw-bar-tan", "hw-bar-orange", "hw-bar-tan"][index]}`.trim() });
    const rect = s("rect", { x: x - width / 2, width, rx: 5 });
    const value = s("text", { x, class: "hw-t-big", "text-anchor": "middle" });
    group.append(rect, value);
    root.append(group);
    text(root, x, floor + 22, labels[index]);
    return { group, rect, value };
  });
  const steps = text(root, xs[1], floor + 42, "", "hw-t-green");
  const set = (bar, loss) => {
    const height = loss * scale;
    bar.rect.setAttribute("y", (floor - height).toFixed(1));
    bar.rect.setAttribute("height", height.toFixed(1));
    bar.value.setAttribute("y", (floor - height - 9).toFixed(1));
    bar.value.textContent = loss.toFixed(2);
  };
  set(bars[0], LOSS.base);

  return {
    duration: 9,
    still: 8.4,
    draw(t) {
      // Training brings the second bar down, squeezing lifts the third, the skill layers lower the fourth.
      const learning = span(t, 1.2, 4.4);
      set(bars[1], mix(LOSS.base, LOSS.trained, learning));
      fade(bars[1].group, span(t, 0.8, 1.2));
      on(bars[1].group, learning > 0 && learning < 1);
      steps.textContent = t >= 0.8 ? `step ${Math.round(LOSS.steps * learning)} of ${LOSS.steps}` : "";
      set(bars[2], mix(LOSS.trained, LOSS.quantized, span(t, 5.2, 6.2)));
      fade(bars[2].group, span(t, 4.8, 5.2));
      set(bars[3], mix(LOSS.quantized, LOSS.layered, span(t, 7.0, 8.0)));
      fade(bars[3].group, span(t, 6.6, 7.0));
    },
  };
}

/* -------------------------------------------------------------------------- */
/* The stops                                                                  */
/* -------------------------------------------------------------------------- */

// In `lines`, [[word]] marks a term the stop teaches.
const STOPS = [
  {
    title: "The big picture",
    question: "What happens when you press Enter?",
    lines:
      "Your message never reaches the model alone. Kiwibro first adds what it knows (the [[context]]), the model writes, and a small reader called [[Kiwi Sense]] checks the draft. If the model asks for a [[tool]], Kiwibro runs it and the model continues with the result.",
    fact: "Local inference, notes and memory run on this computer. Web tools make network requests when used.",
    tech: {
      model: [["Kiwi Mind", "A 4-billion-parameter transformer (Spark X2.6 4B, fine-tuned) with eight LoRA skills and a vision layer, run by llama.cpp."]],
      app: [
        ["Local server", "Flask on 127.0.0.1. The reply is streamed to the window as server-sent events (SSE)."],
        ["One database", "Chats, memory and notes are in a single SQLite file on this computer."],
        ["Agent loop", "App code plans and checks work, and the model handles decisions that need language. Independent reads can run together. File writes and stateful calls run in order."],
      ],
    },
    build: bigPicture,
  },
  {
    title: "Kiwi Mind",
    question: "Which model is answering?",
    lines:
      "[[Kiwi Mind]] uses Spark X2.6 4B to answer, use tools and read images. A three-word [[route]] picks one of eight [[skills]], estimates the difficulty and decides whether to speak or use a tool. Each skill switches on in the upper 12 layers, so the lower layers can reuse what they read. A trained Qwen3 0.6B extracts durable memories on the CPU. An earlier trained Qwen3 0.6B can answer in text when Spark is unavailable.",
    fact: `The Spark pack is ${PACK.gb} GB on disk. The memory expert and text fallback start only when needed.`,
    tech: {
      model: [
        ["Full fine-tune, then IQ2_M", "Every weight of the base model was trained on Kiwibro's conversations. The result is stored as a GGUF file at IQ2_M, about 3.2 bits per weight, and the skills were trained on that squeezed file, so they fit exactly what runs here."],
        ["Eight skills", "Chat, web, code, terminal, computer, tools, write and see. Each is a rank-16 activated LoRA adapter on layers 24–35. The part of the prompt before activation uses the shared trunk and can stay in the cache across skill changes."],
        ["Route", "A ninth adapter writes a route such as \"code light act\". The first step chooses the skill and weighs difficulty. Later steps reuse both. When the app already knows whether it needs a tool, the model goes straight to generation."],
        ["Tool calls", "When the route says act, the reply opens as a tool call and a grammar holds it to the tools on offer and to valid JSON."],
        ["Vision", "A CLIP ViT-L/14 image encoder and a two-layer projector turn a picture into 256 tokens that the model reads like words."],
      ],
      app: [
        ["Spark pack", "llama.cpp loads the IQ2_M trunk and switches the route and skill adapters automatically. The vision file supplies the image encoder and projector. The pack manifest lists the file checksums."],
        ["Optional answer model", "The helper you choose can write a heavy answer after Kiwi Mind has gathered evidence. It can also take over a turn when the app detects repeated failures."],
        ["Memory expert", "A Qwen3 0.6B fine-tuned on a Colab L4 runs at Q4_K_M on two low-priority CPU threads. The app checks its proposed facts against what you wrote. It yields to active chat and unloads after five idle minutes."],
        ["Memory evaluation", "On 60 synthetic transcripts the trained expert recovered 39 of 51 expected facts, up from 25 for stock Qwen3. Median warm extraction dropped from 1.46 to 0.45 seconds on the same CPU. Both models missed some facts."],
        ["Text fallback", "Kiwi Mind 2.0.1 uses its earlier Qwen3 0.6B trunk, control head and seven skill adapters through WebAssembly on Node.js. It starts on demand when the Spark pack or runtime is unavailable."],
      ],
    },
    build: kiwiMind,
  },
  {
    title: "Training data",
    question: "What did Kiwi Mind learn from?",
    lines:
      `A model learns from examples, the way you pick up a job by watching it done well. Kiwi Mind's examples are about ${about(LESSONS.written)} practice conversations in which an assistant uses Kiwibro's tools properly. A much larger [[teacher model]] wrote them: it was handed a task, sometimes a question a real person once asked, and wrote how a careful assistant would carry it out: which tool to call, what comes back, what to say. Handing a big model's habits to a small one like this is called [[distillation]]. The teacher also read ${fmt(LESSONS.judged)} real first messages and marked each one light or heavy. To learn to look, Kiwi Mind studied ${fmt(PACK.pictures)} photographs with captions and questions about them.`,
    fact: `One lesson in ${Math.round(LESSONS.total / LESSONS.held)} is never trained on. Those ${LESSONS.held} are the exam every number on the next stop comes from.`,
    tech: {
      model: [
        ["Teacher", "Qwen3.8-27B, an open model about seven times Kiwi Mind's size, run with vLLM. It samples at temperature 0.9 and its output is held to a JSON schema, so every conversation it writes can be read back by a program."],
        ["Four kinds of conversation", "A single request; a session of several turns; a repair, where a bad first reply is put right; and real first messages from three open collections: OpenAssistant oasst2, Databricks Dolly 15k and Amazon MASSIVE."],
        ["Written the way the model reads", "Each conversation is stored with the base model's own turn markers and tool-call tags, the exact text Kiwi Mind sees when you use it, down to Kiwibro's note with the local time. Every reply opens with its route."],
        ["What counts most", "Tokens do not weigh the same in training. A word of an answer counts 1, a token of a tool call 2, and the route and the first tokens of a reply 3, because that is where the model decides between calling a tool and talking."],
      ],
      app: [
        ["Checked against the real tools", "Every conversation is checked against Kiwibro's own tool list. One with a call that would not run is thrown away."],
        ["Two shares", `The conversations and the judged messages make ${fmt(LESSONS.total)} lessons. ${fmt(LESSONS.trunk)} train the model and then its skills; ${LESSONS.held} are held back for testing.`],
      ],
    },
    build: trainingData,
  },
  {
    title: "Training",
    question: "How was Kiwi Mind trained?",
    lines:
      "Kiwi Mind did not start from nothing. Its base is [[Spark X2.6 4B]], an open model that already knows language. Training shows it a practice conversation, lets it guess each next word, and nudges its numbers a little toward the right one, over and over. Here all four billion of its numbers were free to move, which is called a full fine-tune. The result is then squeezed to about 3 bits per number ([[IQ2_M]]) so that it fits a laptop, and that costs some accuracy. So last, with the squeezed model frozen, eight thin [[skills]] and the layer that writes the [[route]] are trained on top of it.",
    fact: `The release manifest records ${PACK.trunkMinutes} minutes for the trunk fine-tune. Quantization, adapter training, vision training and evaluation are separate stages.`,
    tech: {
      model: [
        ["Base", "Spark X2.6 4B: 36 layers, each 2,560 numbers wide, with a vocabulary of 131,072 tokens. Some layers look at the whole conversation, the others at a sliding window of it."],
        ["Full fine-tune", `${fmt(LESSONS.trunk)} lessons in ${LOSS.steps} steps (AdamW, learning rate 1e-5), with every weight free to move.`],
        ["What it gained", `Loss on the held-back conversations, lower is better: ${LOSS.base.toFixed(2)} as downloaded, ${LOSS.trained.toFixed(2)} after training, ${LOSS.quantized.toFixed(2)} once squeezed to IQ2_M, ${LOSS.layered.toFixed(2)} with the skill layers on.`],
        ["Squeezing", "The trained weights are converted to GGUF and quantized to IQ2_M with an importance matrix measured on the training conversations: the weights that matter most for them are stored most exactly."],
        ["Eight skills and the route", "With the squeezed trunk frozen, each one is an activated LoRA adapter of rank 16 on layers 24 to 35, trained on the very weights the file holds."],
        ["Held-out route accuracy", `The manifest reports ${PACK.route}% skill accuracy and ${PACK.act}% speak-or-act accuracy. In the tool-call evaluation, ${PACK.named}% named the expected tool and ${PACK.valid}% were valid calls. These measures describe the model outputs on the evaluation set.`],
        ["Vision", "The CLIP image encoder stays as it is. The projector behind it learns on 8,000 captioned photographs, then together with the see skill on 4,000 conversations about photographs."],
      ],
      app: [
        ["One script", "Train the trunk, squeeze it, train the layers, measure on the held-back set, export the pack."],
        ["Parity check", "The training scripts can compare next-token behavior in llama.cpp with the training runtime. The app checks file hashes when accepting the pack."],
      ],
    },
    build: training,
  },
  {
    title: "The model file",
    question: "How does a model fit on a laptop?",
    lines:
      "A model is billions of numbers called [[weights]]. [[Quantization]] stores each one with fewer bits, like rounding a smooth curve to steps: at 4 bits instead of 16 the file is four times smaller and the answers are only slightly less exact. [[GGUF]] is the file format that holds them.",
    fact: "Kiwibro runs GGUF files and shows, before you download one, whether it fits in this computer's memory.",
    tech: {
      model: [
        ["Kiwi Mind's weights", "Stored at IQ2_M: about 3 bits per weight on average, with more bits where rounding would hurt most."],
        ["GGUF", "Kiwi Mind and the helper models all come as GGUF, the file format of llama.cpp. Helpers are mostly at 4 bits (Q4_K_M)."],
      ],
      app: [["Fit check", "Before a download, the file's size is compared with this computer's free memory."]],
    },
    build: modelFile,
  },
  {
    title: "Context window",
    question: "How much can the model see at once?",
    lines:
      "A model reads text as [[tokens]] (pieces of words) and holds only a fixed number of them at once: its [[context window]]. When a long chat fills three quarters of it, Kiwibro folds the oldest turns into a short summary. That is [[compaction]], and it lets a chat go on without losing its thread.",
    fact: "The ring beside the send button shows how full the window is.",
    tech: {
      model: [["Context length", `Kiwi Mind holds ${fmt(PACK.ctx)} tokens at once.`]],
      app: [
        ["Compaction", "At 75% full, the oldest turns become a summary of up to 600 tokens. The last four messages stay word for word."],
        ["Session ledger", "Every step is stored in full. After compaction a short index stays in view: what was asked, files changed, commands run, pages read."],
        ["recall", "A tool that reads any earlier step again, word for word."],
      ],
    },
    build: contextWindow,
  },
  {
    title: "KV cache",
    question: "Why is the next answer faster?",
    lines:
      "Reading the prompt is the slow part. The model keeps its working notes on everything it has already read, the [[KV cache]], so when the beginning of the prompt is unchanged only the new part is read. Kiwibro keeps the beginning fixed: whatever changes from turn to turn, like the time or a memory, goes at the end.",
    fact: "Kiwibro's instructions stay byte for byte the same through a whole chat. Only compaction, which is rare, makes the model reread the chat.",
    tech: {
      model: [
        ["KV cache", "The attention keys and values of every token already read, kept between turns."],
        ["Skills that start late", "A skill switches on only where its turn's route ends, so everything read before it stays valid whichever skill comes next. A change of skill reads nothing again."],
      ],
      app: [
        ["Prefix caching", "The first message (instructions and tool list) is byte for byte the same every turn. The clock, memories and notes ride on the newest message."],
        ["Slots", "Two conversations and Kiwibro's own background requests each keep a cache of their own over one copy of the weights."],
      ],
    },
    build: kvCache,
  },
  {
    title: "Tools",
    question: "How can something that only writes text do things?",
    lines:
      "To act, the model writes a small request called a [[tool call]]: a tool's name and its inputs. Kiwibro checks it, asks you first when it is risky, runs it, and hands back the result. This back and forth is the [[agent loop]].",
    fact: "Small models fumble the format, so Kiwibro repairs near misses, like a call typed as plain text. What code can do for certain, such as a sum or saving a note you dictated, Kiwibro does itself.",
    tech: {
      model: [["Function calling", "The model writes a tool's name and inputs as JSON between tool-call tags. Kiwi Mind was trained on Kiwibro's own tool list."]],
      app: [
        ["Risk grades", "Every call is graded safe, low, medium, high or critical. Risky ones ask you first. Critical ones never run."],
        ["Browser tools", "Open and inspect a tab, follow links returned by the page, switch tabs or use back history. An explicitly requested URL can open before generation. Results name the current destination and go through the same tool and evidence checks."],
        ["Recovery and limits", "Malformed calls get a correction. A write cut off at the output limit is refused, then the next round asks for a smaller complete change. Repeated truncated writes stop the turn with the remaining work visible."],
        ["Sandbox", "Commands can run inside bubblewrap in a systemd scope with writable paths, network access and resource limits governed by app settings and tool policy."],
        ["Safe copy", "Work happens in a copy tracked by git. The real folder changes only after you have seen the diff and an undo command that was tried first."],
      ],
    },
    build: toolsLoop,
  },
  {
    title: "Ready parts",
    question: "Why does it not start from a blank page?",
    lines:
      "Writing a page, a script or a test file from nothing is slow for a small model, and that is where it slips. So Kiwibro keeps a shelf of 77 ready parts. A [[kit]] is a set of working starter files with a few blanks to fill. A [[recipe]] is a short list of proven steps for a job such as fixing a bug or checking a price. Code reads your message, picks the part that fits and hands it to the model along with your words.",
    fact: "The same idea covers small exact questions. A model reads word pieces, not letters, so the r's in strawberry are counted by code and the model is given the count.",
    tech: {
      model: [["Component tool", "Kiwi Mind was trained on the component tool and what it answers, so it knows how to fill a kit and follow a recipe."]],
      app: [
        ["Matching by phrase", "Each part lists the phrases that call for it. The words must come in order and close together, and a kit also needs a word like make or build. The longest phrase wins; a tie picks nothing."],
        ["47 kits", "Web pages, 3D scenes, music, dashboards, tables, charts, flowcharts, servers, terminal apps, command-line tools, scripts, tests and Ubuntu apps. Files pass the same safety checks as other writes."],
        ["30 recipes", "For code, the browser, the terminal, documents and data. The steps use Kiwibro's tools and arrive with the message."],
        ["3D and animation", "The scene kit handles supported actors, movement, sky and ground. Requests outside that set use a custom scene. Page checks look for load errors, layout and visible motion. The 3D kit exposes actor motion separately from camera movement."],
        ["Exact answers", "Standalone arithmetic and unit conversions use the calculator directly. Longer problems can include code-computed values in context. The physics reference supplies formulas and assumptions for supported topics."],
      ],
    },
    build: readyParts,
  },
  {
    title: "Shown in the chat",
    question: "Where does the thing it made go?",
    lines:
      "When Kiwibro writes a page, a drawing, a picture, a video or a sound file, it appears in the chat as it is: a page runs, an animation moves, a video plays. A page sits in a sealed frame. It can use the files beside it and the well-known script libraries, such as three.js for 3D, and it cannot reach your other files or this app. Above it stands the folder the chat works in, with its git branch, so you see where on your computer things land. A new chat starts in a temporary folder of its own, until you pick another; the picker lists the folders you used lately first.",
    fact: "Code has a copy button. A file has buttons to open it, show it in its folder and copy its path.",
    tech: {
      model: [["Nothing to learn", "The model writes the file as before. Showing it is the app's work and takes no room in the context window."]],
      app: [
        ["What is shown", "Pages, SVG drawings, pictures, video and sound that a turn wrote, or named in the output of a command it ran."],
        ["Sealed frame", "A sandboxed frame with an origin of its own. Its content security policy lets scripts run and load from the page's folder and four public script libraries, and from nowhere else."],
        ["Kept light", "A preview loads when it scrolls into view, nothing plays by itself, and a file over 5 MB is listed instead of shown."],
        ["Looked at first", "Before a reply about a page is final, Kiwibro opens the page in Chrome and has what is wrong fixed. A message that starts with /goal looks harder."],
        ["Safe copy", "In a safe copy the folder line says so, and opening on the desktop is switched off."],
      ],
    },
    build: shownInChat,
  },
  {
    title: "Knowledge base",
    question: "What if it does not know, and there is no internet?",
    lines:
      "Kiwibro carries a small reference of its own: about 2,000 short entries on 24 topics, from Linux commands and Python to maths, physics, history and the countries of the world. It is read-only, stays on your computer and works with or without the internet. Before the model answers a question, code looks the subject up. A strong match is handed to the model with your words. Anything less adds nothing, so the reference never pulls a chat off its subject.",
    fact: "A web search that finds nothing falls back on the same shelf.",
    tech: {
      model: [["Reading, not recalling", "A small model half-remembers a date, a formula or a command flag. With the entry in front of it, it reads the fact instead of guessing it."]],
      app: [
        ["Matching by code", "Rare words weigh most. Everything specific in the question has to be in the entry, and its subject in the entry's title. A word the reference has never seen means no match."],
        ["What goes into context", "A strong topic match provides a short passage. Physics entries can give the formula, units, approximation and source. Live facts and details of the current workspace come from tools."],
        ["Size", "Half a megabyte of plain text today, with a hard limit of 200 MB. Nothing writes to it while Kiwibro runs."],
      ],
    },
    build: knowledgeBase,
  },
  {
    title: "Checked answers",
    question: "How does it avoid making things up?",
    lines:
      "A small model fills a gap with a plausible guess. So Kiwibro looks before the model writes: the web for news and live facts, this computer for your files and hardware. It then compares the draft with what the search returned, which is called [[grounding]]: a name or number that no [[source]] holds is sent back once to be written again, and left out if it is still there.",
    fact: "When the look itself is the answer (how many files, which CPU, is this checkout at the same commit as its remote), Kiwibro writes it in the look's own figures. If nothing in a draft holds up, you see the sources as found.",
    tech: {
      model: [["Web skill", "The LoRA skill that was trained on search-then-cite conversations."]],
      app: [
        ["Retrieval before generation (RAG)", "Kiwibro searches first (Yahoo, Brave, Bing, DuckDuckGo, Wikipedia, and dated RSS feeds for news) and hands the model short passages to write from."],
        ["Grounding check", "A name, figure or link that no source holds is sent back once, then removed."],
        ["Answers by code", "When a lookup is the answer (a file count, the CPU), code writes it and the model is not asked."],
        ["Syntax check", "A file Kiwibro writes is parsed at once. A broken one goes back to the model with the line number."],
        ["Offline line", "Without a connection, an answer that needed the web says so in its first line."],
      ],
    },
    build: checkedAnswers,
  },
  {
    title: "Kiwi Sense",
    question: "What keeps a small model on track?",
    lines:
      "Beside the big model runs a tiny one that only reads. It turns each sentence into 64 numbers that capture its meaning (an [[embedding]]), and a small classifier (a [[probe]]) spots drift: posing as another assistant, claiming an action that never happened, announcing a step and not taking it. Kiwibro then stops the draft and asks again with a short note: [[steering]]. Five more tiny readers, the [[helpers]], read your message first: is it English, does it need the web, which dates it means, did the topic change.",
    fact: "Kiwi Sense is 25 KB of WebAssembly with 5.5 MB of data, and it works the same with every model. It reads English.",
    tech: {
      model: [
        ["Static embeddings", "29,528 word pieces with 64 numbers each, distilled from the embedding model potion-base-2M and stored in 8 bits in sense.bin."],
        ["Probe", "A six-class classifier on the sentence embedding that spots drift."],
        ["Five helpers", "Small classifiers inside sense.bin read language, intent, time phrases, topic changes and memory relevance. Their outputs help the app choose what context to include."],
        ["Four Colab-trained specialists", "Quantized classifiers suggest steps for browser navigation, 3D work, code diagnosis and review, or table analysis and document evidence. A low-confidence result adds no instructions."],
      ],
      app: [
        ["Steering", "When the probe fires, the draft is stopped and asked again with a short note."],
        ["Runtime", "The C reader compiles to the bundled sense.wasm binary. The browser runs that binary. The server uses it with wasmtime when available, or uses the Python implementation."],
        ["Specialist bundle", "The four classifiers fit in about 70 KB. They share one feature pass and add suggestions to the prompt when relevant tools are available. On a Ryzen 5 7530U, warm inference took 0.64 ms at the 95th percentile across 4,400 calls."],
        ["Held-out requests", "Correct hints on supported prompts were 17/20 for browser, 28/48 for 3D, 27/36 for code and 31/36 for data. Each classifier abstained on all 20 out-of-scope prompts in its final set."],
      ],
    },
    build: kiwiSense,
    extra: senseBench,
  },
  {
    title: "Memory and notes",
    question: "How does it remember you?",
    lines:
      "Facts about you ([[memory]]) and your [[notes]] live in a small database on this computer. Before each answer Kiwibro looks up the few that share a meaningful word with your message, or failing that mean something close to it. That is [[retrieval]]. They are attached to the end of your message as a context note.",
    fact: "Nothing from a chat marked Private is saved to memory or notes.",
    tech: {
      model: [["Relevance helper", "A tiny classifier drops a memory that only shares a word with your message."]],
      app: [
        ["Full-text search (SQLite FTS5)", "The words of your message are matched against an index of memories and notes."],
        ["Embedding fallback", "When no word matches, Kiwi Sense picks the closest memory by cosine similarity."],
        ["Direct capture", "Code saves clear statements about you and handles dictated notes in the notebook. Explicit note text stays out of automatic memory extraction."],
        ["Extraction", "After a reply, a background job can propose up to five memory operations. User statements supply the evidence, numeric details must appear in those statements and near-duplicates are merged."],
      ],
    },
    build: memoryNotes,
  },
  {
    title: "Adapters",
    question: "Can one model be good at different jobs?",
    lines:
      "A [[LoRA adapter]] is a small add-on file that nudges a model toward one kind of work, such as code or translation, without changing the model itself. Kiwibro reads the task from your message and switches the matching adapter on for that one request.",
    fact: "Adapters adjust themselves: when a reply goes wrong with one and comes out right without it, Kiwibro turns that adapter down, then off.",
    tech: {
      model: [
        ["LoRA (low-rank adaptation)", "A weight matrix is a big table of numbers. Instead of changing it, two thin tables are kept beside it; multiplied together they give a table of the same size, which is added on top while the adapter is on. The thin tables hold a small fraction of the model's numbers."],
        ["In Kiwi Mind", "Its eight skills and its route are LoRA adapters of rank 16."],
      ],
      app: [
        ["Per-request switching", "For a helper model the adapter is switched on for one request and off again."],
        ["Self-tuning", "An adapter that makes replies worse is turned down, then off."],
      ],
    },
    build: adapters,
  },
];

// The lights illustrate one layer's routing for successive tokens. The cycle
// is independent of the server's output and uses no conversation data.
function expertRouting(root, bank) {
  const exact = bank?.experts_after >= 1 && bank.experts_after <= 128;
  const count = exact ? bank.experts_after : 12;
  const used = exact ? bank.top_k_after : 3;
  const columns = Math.min(16, count);
  const width = Math.min(25, 400 / columns);
  const columnsStart = (W - columns * width) / 2;
  const top = 106;
  box(root, 96, 50, 90, 38, "Token");
  const router = box(root, 334, 50, 150, 38, "Layer router", "hw-node-tan");
  root.append(
    s("path", { class: "hw-line", d: "M141 50h118" }),
    s("path", { class: "hw-line", d: "M334 69V88H240V100" }),
    s("path", { class: "hw-arrow", d: "M236 94l4 6 4-6" }),
  );
  const pulse = s("circle", { r: 5, class: "hw-dot" });
  root.append(pulse);
  const experts = Array.from({ length: count }, (_, index) => {
    const tile = s("rect", { class: "hw-tile", x: columnsStart + (index % columns) * width + 2.5,
      y: top + Math.floor(index / columns) * 16, width: width - 5, height: 12, rx: 3 });
    root.append(tile);
    return tile;
  });
  const label = exact ? `${used} of ${count} experts selected in this layer` : "The router selects a subset of experts";
  text(root, W / 2, 258, label, "hw-t-big");
  text(root, W / 2, 276, "Illustrative token routing");
  let shown = -1;
  return {
    duration: 8,
    still: 2.6,
    draw(t) {
      const step = Math.floor(t / 2);
      const local = t % 2;
      const travel = span(local, 0.05, 0.5);
      at(pulse, mix(146, 254, travel), 50);
      fade(pulse, during(local, 0.03, 0.54, 0.12));
      on(router, local >= 0.5);
      const selection = local >= 0.5 ? step + 1 : 0;
      if (selection === shown) return;
      shown = selection;
      const start = (step * 5) % count;
      const stride = Math.floor(count / used);
      const selected = new Set(Array.from({ length: selection ? used : 0 }, (_, index) => (start + index * stride) % count));
      experts.forEach((expert, index) => on(expert, selected.has(index)));
    },
  };
}

function packSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${fmt(Math.round(bytes / 1e6))} MB`;
  return `${fmt(bytes)} bytes`;
}

function nativeStops(mind) {
  const bank = mind.expert_bank;
  const known = Boolean(mind.base);
  const gemma = /^google\/gemma-4-26b-a4b-it$/i.test(mind.base || "");
  const sparse = Boolean(bank || gemma);
  const model = gemma ? "Gemma 4 26B A4B" : mind.base || "Native Kiwi Mind";
  const source = mind.base || "";
  const caps = mind.capabilities || {};
  const checked = caps.tools === true;
  const context = checked && Number.isSafeInteger(mind.ctx) && mind.ctx > 0 ? fmt(mind.ctx) : "";
  const disk = checked ? packSize(mind.bytes) : "";
  const modelRows = [
    ...(known ? [["Base model", source]] : []),
    ...(mind.architecture ? [["GGUF architecture", mind.architecture]] : []),
    ...(bank ? [["Expert bank", `${bank.experts_after} stored per layer, ${bank.top_k_after} selected per token.`],
      ["Starting architecture", `${bank.experts_before} stored per layer, ${bank.top_k_before} selected per token.`]] : []),
    ...(bank?.width_after ? [["Expert intermediate width", `${bank.width_after} channels per expert, from ${bank.width_before}.`]] : []),
    ...(checked && Number.isSafeInteger(mind.training_steps) && mind.training_steps > 0 ?
      [["Adapter training", `${fmt(mind.training_steps)} cumulative optimizer steps recorded in the pack.`]] : []),
    ...(caps.tools ? [["Function calling", "The model's chat template sends structured calls to Kiwibro. The app checks the tool name, arguments and permissions."]] : []),
    ...(mind.vision ? [["Vision", "The matching vision projector lets the same model read attached images."]] : []),
    ...(caps.thinking ? [["Reasoning", caps.reasoning_effort ?
      "The runtime can change reasoning effort for a request." : "The runtime can turn thinking on or off for a request."]] : []),
  ];
  const configuration = {
    "The big picture": {
      tech: { model: known ? [["Kiwi Mind", `${model}, served through its native chat template.`]] : [] },
    },
    "Kiwi Mind": {
      question: mind.usable ? "What runs Kiwi Mind here?" : "What will run Kiwi Mind here?",
      lines: !checked ? "This app is configured for a native GGUF candidate. Its model and capabilities appear here once its files are verified."
        : sparse ? `${model} has a bank of [[experts]] in each routed layer. The learned router chooses which experts contribute to each token. The same model reads your message, makes tool requests and writes the answer.`
        : `${model} runs through its native chat template. The same model reads your message, makes tool requests and writes the answer.`,
      fact: disk ? `The verified pack occupies ${disk} on disk.` : "The Models panel shows the candidate's installation status.",
      tech: { model: gemma ? [...modelRows, ["Shared expert", "Gemma also uses a shared expert. The figure shows the routed bank in one layer."]] : modelRows, app: [
        ["Local runtime", "llama.cpp serves the candidate on an authenticated loopback address."],
        ["Pack verification", "The manifest lists the checksums of the weights and any vision projector."],
        ...(mind.reason ? [["Runtime status", mind.reason]] : []),
      ] },
      build: sparse ? (svg) => expertRouting(svg, bank) : bigPicture,
    },
    "The model file": {
      lines: "A model is billions of numbers called [[weights]]. [[Quantization]] stores each one with fewer bits, like rounding a smooth curve to steps. The figure illustrates rounding and the size of those numbers. [[GGUF]] is the file format that holds the packed weights.",
      fact: disk ? `The verified files occupy ${disk} on disk, including any projector in the pack.` : "The manifest records the size and SHA-256 of every file in the pack.",
      tech: { model: mind.quant ? [["Installed quantization", mind.quant]] : [],
        app: [["Load", "llama.cpp opens the GGUF pack locally. Memory use also includes the runtime and context cache."]] },
    },
    "Context window": {
      tech: { model: context ? [["Context length", `The pack is configured for ${context} tokens.`]] : [] },
    },
    "KV cache": {
      lines: "The runtime can reuse work for the beginning of a prompt when it matches a cached prefix. Kiwibro keeps stable instructions and the tool list at the beginning. Turn-specific context, such as matching memories and the clock, stays near the current message.",
      fact: "llama.cpp reports the prompt tokens served from the cache when its runtime supplies timings.",
      tech: { model: [["Attention state", "The KV cache stores keys and values for prompt tokens the runtime has read."]],
        app: [["Shared model", "Chat and utility requests share a single local copy of the model."]] },
    },
    "Tools": {
      tech: { model: [["Function calling", "The native chat template returns tool names and structured arguments."]] },
    },
    "Ready parts": {
      tech: { model: [["Component tool", "The model can use starter files and steps selected from Kiwibro's component library."]] },
    },
    "Knowledge base": {
      tech: { model: [["Reference text", "A relevant local passage gives the model facts, formulas and assumptions it can use when answering."]] },
    },
    "Checked answers": {
      tech: { model: [["Sources", "The candidate receives the passages and tool results Kiwibro collected for this turn."]] },
    },
    "Memory and notes": {
      fact: "Private chats keep their own local transcript and skip saving memories or notes.",
      tech: { model: [["Structured extraction", "The same local model proposes memory operations in a JSON schema."]] },
    },
  };
  return STOPS.filter((stop) => !["Training data", "Training", "Adapters"].includes(stop.title))
    .map((stop) => {
      const update = configuration[stop.title];
      return update ? { ...stop, ...update, tech: { ...stop.tech, ...update.tech } } : stop;
    });
}

const NATIVE_TERMS = {
  "Kiwi Mind": "The primary model Kiwibro runs on this computer.",
  experts: "Feed-forward networks inside a mixture-of-experts layer. The learned router selects a few for each token.",
};

// Plain definitions of the terms that a stop teaches, shown as a hover title on the term.
const TERM_NOTES = {
  "Kiwi Mind": "Kiwibro's own model: Spark X2.6 4B, trained on Kiwibro's conversations and squeezed to about 3 bits per weight.",
  route: "Three words the model writes before every reply, such as \"code light act\": the skill that answers, light or heavy, speak or use a tool.",
  skills: `Eight thin sets of extra weights, ${PACK.layerMb} MB each, one per kind of work. One is active at a time.`,
  "helper model": "Any other model you downloaded. Optional.",
  "teacher model": "A large model whose written examples a small model learns from.",
  distillation: "Training a small model on what a large one wrote, so it picks up the large one's behaviour.",
  "Spark X2.6 4B": "An open model with 4 billion weights, under the Apache 2.0 licence.",
  IQ2_M: "A way to store a model at about 3 bits per weight, with more bits where rounding would hurt most.",
  kit: "Starter files that already work, for example a page, a server or a test file.",
  recipe: "Four to eight steps written in terms of Kiwibro's own tools.",
};

// The optional "Under the hood" block: tech names per layer, model first.
function techBlock(tech) {
  const groups = [
    ["Model layer", tech.model, "model"],
    ["App layer", tech.app, "app"],
  ].filter(([, rows]) => rows && rows.length);
  return el("div", { class: "how-tech" }, [
    el("h4", { class: "how-tech-title", text: "Under the hood" }),
    el(
      "div",
      { class: "how-tech-groups" },
      groups.map(([label, rows, layer]) =>
        el("section", { class: `how-tech-group how-tech-${layer}` }, [
          el("h5", { class: "how-tech-layer", text: label }),
          el(
            "dl",
            { class: "how-tech-list" },
            rows.flatMap(([name, text]) => [
              el("dt", { class: "how-tech-name", text: name }),
              el("dd", { class: "how-tech-text", text }),
            ]),
          ),
        ]),
      ),
    ),
  ]);
}

function withTerms(line, terms) {
  return line.split(/(\[\[.+?\]\])/).map((part) => {
    if (!part.startsWith("[[")) return part;
    const word = part.slice(2, -2);
    return el("b", { class: "how-term", text: word, title: terms[word] || TERM_NOTES[word] });
  });
}

/* -------------------------------------------------------------------------- */
/* Kiwi Sense, live                                                           */
/* -------------------------------------------------------------------------- */

const SENSE_NAMES = {
  fine: "fine",
  denial: "says it cannot",
  other_identity: "poses as another assistant",
  action_claim: "claims an action",
  action_promise: "announces a step",
  meta_leak: "talks about its instructions",
};

const SENSE_SAMPLES = [
  "I am ChatGPT, a large language model trained by OpenAI.",
  "I've saved the file to your desktop.",
  "Paris is the capital of France.",
];

function senseBench() {
  const input = el("input", {
    class: "how-try-input",
    type: "text",
    value: SENSE_SAMPLES[0],
    maxlength: "300",
    spellcheck: "false",
    autocomplete: "off",
    "aria-label": "A sentence for Kiwi Sense to read",
  });
  const rows = el("div", { class: "how-try-rows" });
  const verdict = el("p", { class: "how-try-verdict", role: "status" });
  const samples = el(
    "div",
    { class: "how-try-samples" },
    SENSE_SAMPLES.map((sample) => {
      const button = el("button", { class: "text-btn", type: "button", text: sample.length > 34 ? `${sample.slice(0, 32)}…` : sample });
      button.addEventListener("click", () => {
        input.value = sample;
        read();
      });
      return button;
    }),
  );
  let reader = null;

  function show(scores, ms) {
    clear(rows);
    const labels = reader.labels.filter((label) => label in scores);
    const worst = labels.filter((label) => label !== "fine").sort((a, b) => scores[b] - scores[a])[0];
    const flagged = worst && scores[worst] >= reader.threshold;
    for (const label of labels) {
      const share = scores[label];
      const tone = label === "fine" ? "fine" : flagged && label === worst ? "off" : "quiet";
      rows.append(
        el("div", { class: "how-try-row", "data-tone": tone }, [
          el("span", { class: "how-try-name", text: SENSE_NAMES[label] || label }),
          el("span", { class: "how-try-bar" }, [el("span", { style: { width: `${(share * 100).toFixed(1)}%` } })]),
          el("span", { class: "how-try-share", text: `${Math.round(share * 100)}%` }),
        ]),
      );
    }
    const speed = ms < 0.1 ? "under 0.1 ms" : `${ms.toFixed(1)} ms`;
    verdict.dataset.tone = flagged ? "off" : "fine";
    verdict.textContent = flagged
      ? `Off track: ${SENSE_NAMES[worst]}. Kiwibro would stop this and ask again. Read in ${speed}.`
      : `Passes. Read in ${speed}.`;
  }

  async function read() {
    const sentence = input.value.trim();
    if (!sentence) {
      clear(rows);
      verdict.textContent = "";
      return;
    }
    try {
      reader = reader || (await loadSense());
    } catch {
      verdict.dataset.tone = "off";
      verdict.textContent = "Kiwi Sense could not be loaded in this browser.";
      return;
    }
    const started = performance.now();
    const scores = reader.probe(sentence);
    show(scores, performance.now() - started);
  }

  input.addEventListener("input", read);
  const node = el("div", { class: "how-try" }, [
    el("label", { class: "how-try-label" }, ["Try it: this box runs the real Kiwi Sense, here in the page.", input]),
    samples,
    rows,
    verdict,
  ]);
  return { node, start: read };
}

/* -------------------------------------------------------------------------- */
/* The panel                                                                  */
/* -------------------------------------------------------------------------- */

export function createHowView({ dialog, getMind = () => null }) {
  const body = dialog.querySelector("#how-body");
  const stillOnly = window.matchMedia("(prefers-reduced-motion: reduce)");
  let index = 0;
  let scene = null;
  let frame = 0;
  let origin = 0;
  let paused = false;
  let built = false;
  let stops = STOPS;
  let nativeKey = null;
  let steps = [];
  let figure = null;
  let copy = null;
  let playButton = null;
  let prevButton = null;
  let nextButton = null;

  function tick(now) {
    frame = 0;
    if (!scene || !dialog.open) return;
    scene.draw((Math.max(0, now - origin) / 1000) % scene.duration);
    if (!paused && !document.hidden && !stillOnly.matches) frame = window.requestAnimationFrame(tick);
  }

  function run() {
    window.cancelAnimationFrame(frame);
    frame = 0;
    if (!scene || !dialog.open) return;
    if (paused || stillOnly.matches) {
      scene.draw(scene.still);
      return;
    }
    frame = window.requestAnimationFrame(tick);
  }

  function renderPlay() {
    if (!playButton) return;
    const still = paused || stillOnly.matches;
    clear(playButton);
    playButton.append(icon(still ? "play" : "pause"));
    playButton.setAttribute("aria-label", still ? "Play the animation" : "Pause the animation");
    playButton.title = still ? "Play" : "Pause";
    playButton.hidden = stillOnly.matches;
  }

  function go(next) {
    index = Math.min(stops.length - 1, Math.max(0, next));
    const stop = stops[index];
    steps.forEach((button, which) => {
      button.setAttribute("aria-current", which === index ? "step" : "false");
      button.classList.toggle("is-seen", which < index);
    });
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "how-svg", role: "img", "aria-label": `${stop.title}: animated figure` });
    scene = stop.build(svg);
    clear(figure);
    figure.append(svg);
    clear(copy);
    copy.append(
      el("h3", { class: "how-question", text: stop.question }),
      el("p", { class: "how-lines" }, withTerms(stop.lines, nativeKey ? NATIVE_TERMS : TERM_NOTES)),
      el("p", { class: "how-fact" }, [el("span", { class: "how-fact-tag", text: "In Kiwibro" }), stop.fact]),
    );
    if (stop.tech) copy.append(techBlock(stop.tech));
    if (stop.extra) {
      const extra = stop.extra();
      copy.append(extra.node);
      extra.start();
    }
    prevButton.disabled = index === 0;
    nextButton.disabled = index === stops.length - 1;
    copy.parentElement.scrollTop = 0;
    origin = performance.now();
    run();
  }

  function build() {
    built = true;
    steps = stops.map((stop, which) => {
      const button = el("button", { class: "how-step", type: "button" }, [
        el("span", { class: "how-step-n", text: String(which + 1) }),
        el("span", { class: "how-step-name", text: stop.title }),
      ]);
      button.addEventListener("click", () => go(which));
      return button;
    });
    figure = el("div", { class: "how-figure" });
    copy = el("div", { class: "how-copy" });
    prevButton = el("button", { class: "btn btn-ghost", type: "button" }, [icon("arrow-left"), "Back"]);
    nextButton = el("button", { class: "btn btn-primary", type: "button" }, ["Next", icon("arrow-right")]);
    playButton = el("button", { class: "icon-btn how-play", type: "button" });
    prevButton.addEventListener("click", () => go(index - 1));
    nextButton.addEventListener("click", () => go(index + 1));
    playButton.addEventListener("click", () => {
      paused = !paused;
      if (!paused) origin = performance.now() - scene.still * 1000;
      renderPlay();
      run();
    });
    clear(body);
    body.append(
      el("nav", { class: "how-steps", "aria-label": "Topics" }, steps),
      el("div", { class: "how-stage" }, [
        el("div", { class: "how-scroll" }, [figure, copy]),
        el("div", { class: "how-nav" }, [prevButton, playButton, nextButton]),
      ]),
    );
    renderPlay();
  }

  dialog.addEventListener("keydown", (event) => {
    if (!built || event.target.closest("input, textarea")) return;
    if (event.key === "ArrowRight") go(index + 1);
    else if (event.key === "ArrowLeft") go(index - 1);
    else return;
    event.preventDefault();
  });
  dialog.addEventListener("close", () => {
    window.cancelAnimationFrame(frame);
    frame = 0;
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) run();
  });
  stillOnly.addEventListener("change", () => {
    renderPlay();
    run();
  });

  function syncModel() {
    const mind = getMind();
    if (!mind) {
      if (!built) {
        const text = mind === undefined ? "Checking the installed Kiwi Mind pack…" : "Kiwi Mind details are unavailable.";
        body.replaceChildren(el("p", { class: "how-lines", role: "status", text }));
      }
      return false;
    }
    const nextKey = mind?.engine === "native-gguf" ? JSON.stringify([
      mind.usable, mind.reason, mind.base, mind.architecture, mind.quant, mind.ctx, mind.bytes,
      mind.capabilities, mind.vision, mind.expert_bank, mind.training_steps,
    ]) : null;
    if (built && nextKey === nativeKey) return false;
    const previous = stops[index]?.title;
    const restoreTopicFocus = steps.includes(document.activeElement);
    nativeKey = nextKey;
    stops = nextKey ? nativeStops(mind) : STOPS;
    index = Math.max(0, stops.findIndex((stop) => stop.title === previous));
    build();
    if (restoreTopicFocus) steps[index].focus();
    return true;
  }

  return {
    open() {
      syncModel();
      openSheet(dialog);
      if (built) go(index);
    },
    refreshIfOpen() {
      if (dialog.open && syncModel()) go(index);
    },
  };
}
