const NS = "http://www.w3.org/2000/svg";
const svg = document.querySelector("#chart");
let kind = "bar";

function make(tag, attrs, text = "") {
  const node = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  node.textContent = text;
  return node;
}

// A tick step of 1, 2, 2.5, 5 or 10 times a power of ten, so the axis has at most five readable intervals.
function axis(max) {
  const unit = 10 ** (Math.floor(Math.log10(max)) - 1);
  const step = [1, 2, 2.5, 5, 10].map((factor) => factor * unit).find((size) => max / size <= 5);
  return { step, peak: Math.ceil(max / step) * step };
}

// Redrawn at the real pixel width so the labels keep one readable size.
function draw() {
  const width = svg.clientWidth;
  const height = svg.clientHeight;
  const left = 38, right = 8, top = 12, bottom = 26;
  const points = CHART.points;
  const { step, peak } = axis(Math.max(...points.map((point) => point.value)));
  const slot = (width - left - right) / points.length;
  const y = (value) => height - bottom - (value / peak) * (height - bottom - top);
  const cx = (index) => left + slot * (index + 0.5);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  for (let tick = 0; tick <= peak / step + 1e-9; tick += 1) {
    const value = Number((step * tick).toPrecision(6));
    svg.append(make("line", { class: "grid", x1: left, x2: width - right, y1: y(value), y2: y(value) }));
    svg.append(make("text", { x: left - 6, y: y(value) + 4, "text-anchor": "end" }, value));
  }
  points.forEach((point, index) => {
    svg.append(make("text", { x: cx(index), y: height - 8, "text-anchor": "middle" }, point.label));
  });
  if (kind === "bar") {
    const barWidth = Math.min(slot * 0.62, 56);
    points.forEach((point, index) => {
      const bar = make("rect", { class: "bar", x: cx(index) - barWidth / 2, y: y(point.value), width: barWidth, height: y(0) - y(point.value), rx: 3 });
      bar.append(make("title", {}, `${point.label}: ${point.value} ${CHART.unit}`));
      svg.append(bar);
    });
  } else {
    svg.append(make("polyline", { class: "line", points: points.map((point, index) => `${cx(index)},${y(point.value)}`).join(" ") }));
    points.forEach((point, index) => {
      const dot = make("circle", { class: "dot", cx: cx(index), cy: y(point.value), r: 5 });
      dot.append(make("title", {}, `${point.label}: ${point.value} ${CHART.unit}`));
      svg.append(dot);
    });
  }
}

function fillTable() {
  const table = document.querySelector("#table");
  const rows = [["Label", CHART.unit]].concat(CHART.points.map((point) => [point.label, point.value]));
  rows.forEach((cells, index) => {
    const line = table.insertRow();
    cells.forEach((text, column) => {
      const cell = document.createElement(index === 0 || column === 0 ? "th" : "td");
      if (cell.localName === "th") cell.scope = index === 0 ? "col" : "row";
      cell.textContent = text;
      line.append(cell);
    });
  });
}

for (const button of document.querySelectorAll("[data-kind]")) {
  button.addEventListener("click", () => {
    kind = button.dataset.kind;
    document.querySelectorAll("[data-kind]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    draw();
  });
}
fillTable();
draw();
new ResizeObserver(draw).observe(svg);
