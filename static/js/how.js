/**
 * How it works: eight small animated figures, one idea each.
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
/* 6. Kiwi Sense                                                              */
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
/* 7. Memory and notes                                                        */
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
/* 8. LoRA adapters                                                           */
/* -------------------------------------------------------------------------- */

function adapters(root) {
  const cols = 8;
  const rows = 6;
  const x0 = 164;
  const y0 = 90;
  const cells = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const cell = s("rect", { class: "hw-cell", x: x0 + col * 21, y: y0 + row * 21, width: 17, height: 17, rx: 3 });
      root.append(cell);
      cells.push(cell);
    }
  }
  text(root, x0 + 83, y0 + 148, "the model: one big file", "hw-t-muted");
  const tasks = ["code", "writing", "translate"];
  text(root, 74, 78, "your task", "hw-t-muted");
  const chips = tasks.map((name, index) => box(root, 74, 108 + index * 44, 92, 30, name));
  text(root, 410, 78, "adapters", "hw-t-muted");
  const strips = tasks.map((_, index) => {
    const strip = s("rect", { class: "hw-strip", width: 100, height: 12, rx: 4 });
    root.append(strip);
    return { strip, parkX: 360, parkY: 102 + index * 44 };
  });
  // Which cells each adapter nudges: a fixed scatter per adapter.
  const touched = [7, 11, 13].map((step, which) => cells.map((_, index) => (index * step + which * 3) % 5 === 0));

  return {
    duration: 10.5,
    still: 1.9,
    draw(t) {
      const which = Math.min(2, Math.floor(t / 3.5));
      const local = t - which * 3.5;
      const slide = span(local, 0.5, 1.3) * (1 - span(local, 2.7, 3.4));
      strips.forEach(({ strip, parkX, parkY }, index) => {
        const k = index === which ? slide : 0;
        strip.setAttribute("x", mix(parkX, x0 + 33, k).toFixed(2));
        strip.setAttribute("y", mix(parkY, y0 - 20, k).toFixed(2));
        on(strip, k > 0.05);
      });
      chips.forEach((chip, index) => on(chip, index === which && local > 0.15 && local < 3.3));
      cells.forEach((cell, index) => {
        cell.style.setProperty("--tint", (touched[which][index] ? slide : 0).toFixed(3));
      });
    },
  };
}

/* -------------------------------------------------------------------------- */
/* The eight stops                                                            */
/* -------------------------------------------------------------------------- */

// In `lines`, [[word]] marks a term the stop teaches.
const STOPS = [
  {
    title: "The big picture",
    question: "What happens when you press Enter?",
    lines:
      "Your message never reaches the model alone. Kiwibro first adds what it knows (the [[context]]), the model writes, and a small reader called [[Kiwi Sense]] checks the draft. If the model asks for a [[tool]], Kiwibro runs it and the model continues with the result.",
    fact: "The model and your data stay on this computer. Only web lookups go online.",
    build: bigPicture,
  },
  {
    title: "The model file",
    question: "How does a model fit on a laptop?",
    lines:
      "A model is billions of numbers called [[weights]]. [[Quantization]] stores each one with fewer bits, like rounding a smooth curve to steps: at 4 bits instead of 16 the file is four times smaller and the answers are only slightly less exact. [[GGUF]] is the file format that holds them.",
    fact: "Kiwibro runs GGUF files and shows, before you download one, whether it fits in this computer's memory.",
    build: modelFile,
  },
  {
    title: "Context window",
    question: "How much can the model see at once?",
    lines:
      "A model reads text as [[tokens]] (pieces of words) and holds only a fixed number of them at once: its [[context window]]. When a long chat fills three quarters of it, Kiwibro folds the oldest turns into a short summary. That is [[compaction]], and it lets a chat go on without losing its thread.",
    fact: "The ring beside the send button shows how full the window is.",
    build: contextWindow,
  },
  {
    title: "KV cache",
    question: "Why is the next answer faster?",
    lines:
      "Reading the prompt is the slow part. The model keeps its working notes on everything it has already read, the [[KV cache]], so when the beginning of the prompt is unchanged only the new part is read. Kiwibro never edits the beginning: whatever changes, like the time or a memory, goes at the end.",
    fact: "Kiwibro's instructions stay byte for byte the same through a whole chat, so the cache is never thrown away.",
    build: kvCache,
  },
  {
    title: "Tools",
    question: "How can something that only writes text do things?",
    lines:
      "To act, the model writes a small request called a [[tool call]]: a tool's name and its inputs. Kiwibro checks it, asks you first when it is risky, runs it, and hands back the result. This back and forth is the [[agent loop]].",
    fact: "Small models often fumble the format, so Kiwibro repairs near misses: a call typed as plain text, a slightly wrong tool name, a number sent as a word.",
    build: toolsLoop,
  },
  {
    title: "Kiwi Sense",
    question: "What keeps a small model on track?",
    lines:
      "Beside the big model runs a tiny one that only reads. It turns each sentence into 64 numbers that capture its meaning (an [[embedding]]), and a small classifier (a [[probe]]) spots drift: posing as another assistant, claiming an action that never happened, announcing a step and not taking it. Kiwibro then stops the draft and asks again with a short note: [[steering]].",
    fact: "Kiwi Sense is 9 KB of WebAssembly with 2.7 MB of data, and it works the same with every model. Its probe reads English.",
    build: kiwiSense,
    extra: senseBench,
  },
  {
    title: "Memory and notes",
    question: "How does it remember you?",
    lines:
      "Facts about you ([[memory]]) and your [[notes]] live in a small database on this computer. Before each answer Kiwibro looks up the few that share a meaningful word with your message, which is called [[retrieval]], and attaches them to the end of it as a context note.",
    fact: "Nothing from a chat marked Private is saved to memory or notes.",
    build: memoryNotes,
  },
  {
    title: "Adapters",
    question: "Can one model be good at different jobs?",
    lines:
      "A [[LoRA adapter]] is a small add-on file that nudges a model toward one kind of work, such as code or translation, without changing the model itself. Kiwibro reads the task from your message and switches the matching adapter on for that one request.",
    fact: "Adapters adjust themselves: when a reply goes wrong with one and comes out right without it, Kiwibro turns that adapter down, then off.",
    build: adapters,
  },
];

function withTerms(line) {
  return line.split(/(\[\[.+?\]\])/).map((part) =>
    part.startsWith("[[") ? el("b", { class: "how-term", text: part.slice(2, -2) }) : part,
  );
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

export function createHowView({ dialog }) {
  const body = dialog.querySelector("#how-body");
  const stillOnly = window.matchMedia("(prefers-reduced-motion: reduce)");
  let index = 0;
  let scene = null;
  let frame = 0;
  let origin = 0;
  let paused = false;
  let built = false;
  let steps = [];
  let figure = null;
  let copy = null;
  let playButton = null;
  let prevButton = null;
  let nextButton = null;

  function tick(now) {
    frame = 0;
    if (!scene || !dialog.open) return;
    scene.draw(((now - origin) / 1000) % scene.duration);
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
    index = Math.min(STOPS.length - 1, Math.max(0, next));
    const stop = STOPS[index];
    steps.forEach((button, which) => {
      button.setAttribute("aria-current", which === index ? "step" : "false");
    });
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "how-svg", role: "img", "aria-label": `${stop.title}: animated figure` });
    scene = stop.build(svg);
    clear(figure);
    figure.append(svg);
    clear(copy);
    copy.append(
      el("h3", { class: "how-question", text: stop.question }),
      el("p", { class: "how-lines" }, withTerms(stop.lines)),
      el("p", { class: "how-fact" }, [el("span", { class: "how-fact-tag", text: "In Kiwibro" }), stop.fact]),
    );
    if (stop.extra) {
      const extra = stop.extra();
      copy.append(extra.node);
      extra.start();
    }
    prevButton.disabled = index === 0;
    nextButton.disabled = index === STOPS.length - 1;
    copy.parentElement.scrollTop = 0;
    origin = performance.now();
    run();
  }

  function build() {
    built = true;
    steps = STOPS.map((stop, which) => {
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
    if (event.target.closest("input, textarea")) return;
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

  return {
    open() {
      if (!built) build();
      openSheet(dialog);
      go(index);
    },
    refreshIfOpen() {},
  };
}
