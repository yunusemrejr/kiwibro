// Draws the chart that layout.js worked out as one inline SVG.
const NS = "http://www.w3.org/2000/svg";
const stage = document.querySelector("#stage");

function make(tag, attrs, text) {
  const node = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function problem(message) {
  const line = document.createElement("p");
  line.className = "problem";
  line.setAttribute("role", "alert");
  line.textContent = message;
  stage.replaceChildren(line);
}

// A label on a small rectangle in the page colour; `end` hangs it to the left of x.
function tag(x, y, text, end = false) {
  const width = measurer.measureText(text).width + 14;
  const left = end ? x - width : x;
  return [make("rect", { class: "tag", x: left, y, width, height: 24, rx: 4 }), make("text", { class: "tag-text", x: left + 7, y: y + 17 }, text)];
}

function draw(flow, chart) {
  const { nodes, edges, width, height } = chart;
  const svg = make("svg", { class: "flow", width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": flow.title });
  const marker = make("marker", { id: "flow-arrow", viewBox: "0 0 10 10", refX: 10, refY: 5, markerWidth: 10, markerHeight: 10, markerUnits: "userSpaceOnUse", orient: "auto" });
  marker.append(make("path", { class: "arrow", d: "M0,0 L10,5 L0,10 z" }));
  const defs = make("defs", {});
  defs.append(marker);

  const paths = [];
  const labels = [];
  for (const edge of edges) {
    let d;
    let start;
    let end = false;
    if (edge.back) {
      // Out of the source's right side, around the chart, into the target's right side.
      const x1 = edge.from.x + edge.from.w / 2, y1 = edge.from.y;
      const x2 = edge.to.x + edge.to.w / 2, y2 = edge.to.y;
      d = `M${x1} ${y1} C${edge.bend} ${y1} ${edge.bend} ${y2} ${x2} ${y2}`;
      start = { x: x1 + 8, y: y1 - 30 };
    } else {
      const last = edge.chain.length - 1;
      const points = edge.chain.map((node, i) => ({
        x: node.x,
        y: node.y + (i === 0 ? node.h / 2 : i === last ? -node.h / 2 : 0),
      }));
      d = `M${points[0].x} ${points[0].y}`;
      for (let i = 1; i < points.length; i++) {
        const a = points[i - 1], b = points[i];
        const mid = (a.y + b.y) / 2;
        d += ` C${a.x} ${mid} ${b.x} ${mid} ${b.x} ${b.y}`;
      }
      // Branches of one choice share a start, so a link that heads left gets its label on the left.
      const left = points[1].x < points[0].x;
      start = { x: points[0].x + (left ? -8 : 8), y: points[0].y + 6 };
      end = left;
    }
    paths.push(make("path", { class: edge.back ? "edge back" : "edge", d, "marker-end": "url(#flow-arrow)" }));
    if (edge.link.label) labels.push(...tag(start.x, start.y, String(edge.link.label), end));
  }

  const shapes = nodes.map((node) => {
    const group = make("g", { class: `node ${node.kind}` });
    const { x, y, w, h } = node;
    if (node.kind === "choice") {
      group.append(make("polygon", { class: "shape", points: `${x},${y - h / 2} ${x + w / 2},${y} ${x},${y + h / 2} ${x - w / 2},${y}` }));
    } else {
      const pill = node.kind === "start" || node.kind === "end";
      group.append(make("rect", { class: "shape", x: x - w / 2, y: y - h / 2, width: w, height: h, rx: pill ? h / 2 : 10 }));
    }
    const first = y - ((node.lines.length - 1) * LINE) / 2 + 5;
    node.lines.forEach((line, i) => group.append(make("text", { x, y: first + i * LINE, "text-anchor": "middle" }, line)));
    return group;
  });

  svg.append(defs, ...paths, ...shapes, ...labels);
  stage.replaceChildren(svg);

  const text = new Map(flow.steps.map((step) => [step.id, step.text]));
  const list = document.querySelector("#steps");
  list.replaceChildren(...flow.links.map((link) => {
    const item = document.createElement("li");
    item.textContent = `${link.label ? `${link.label}: ` : ""}${text.get(link.from)} to ${text.get(link.to)}`;
    return item;
  }));
}

const flow = { title: FLOW.title, steps: FLOW.steps, links: FLOW.links ?? [] };
const mistake = checkData(flow);
if (mistake) problem(mistake);
else draw(flow, layout(flow));
