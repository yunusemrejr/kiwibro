const NS = "http://www.w3.org/2000/svg";

function make(tag, attrs = {}, text = "", ns) {
  const node = ns ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  node.textContent = text;
  return node;
}

function drawTiles() {
  const total = DATA.months.reduce((sum, month) => sum + month.value, 0);
  const best = DATA.months.reduce((top, month) => (month.value > top.value ? month : top));
  const open = DATA.rows.filter((row) => row.status === "Open").length;
  const tiles = [
    ["Total " + DATA.unit, total],
    ["Best month", best.label],
    ["Open orders", open],
  ];
  const holder = document.querySelector("#tiles");
  for (const [label, value] of tiles) {
    const list = make("dl", { class: "tile" });
    list.append(make("dt", {}, label), make("dd", {}, value));
    holder.append(list);
  }
}

// Redrawn at the real pixel width so the labels keep one readable size.
function drawChart() {
  const svg = document.querySelector("#chart");
  const width = svg.clientWidth;
  const height = svg.clientHeight;
  const left = 32, bottom = 22, top = 8;
  const peak = Math.ceil(Math.max(...DATA.months.map((month) => month.value)) / 20) * 20;
  const slot = (width - left) / DATA.months.length;
  const scale = (value) => height - bottom - (value / peak) * (height - bottom - top);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("aria-label", DATA.months.map((m) => `${m.label} ${m.value}`).join(", "));
  svg.replaceChildren();
  for (const tick of [0, peak / 2, peak]) {
    const y = scale(tick);
    svg.append(make("line", { class: "grid", x1: left, x2: width, y1: y, y2: y }, "", true));
    svg.append(make("text", { x: left - 6, y: y + 4, "text-anchor": "end" }, tick, true));
  }
  DATA.months.forEach((month, index) => {
    const barWidth = Math.min(slot * 0.6, 48);
    const x = left + slot * index + (slot - barWidth) / 2;
    svg.append(make("rect", { class: "bar", x, y: scale(month.value), width: barWidth, height: height - bottom - scale(month.value), rx: 3 }, "", true));
    svg.append(make("text", { x: x + barWidth / 2, y: height - 6, "text-anchor": "middle" }, month.label, true));
  });
}

function drawTable() {
  const table = document.querySelector("#table");
  const head = make("tr");
  for (const column of DATA.columns) head.append(make("th", { scope: "col", class: column.numeric ? "num" : "" }, column.label));
  const thead = make("thead");
  thead.append(head);
  table.append(thead);
  const body = make("tbody");
  for (const row of DATA.rows) {
    const line = make("tr");
    for (const column of DATA.columns) line.append(make("td", { class: column.numeric ? "num" : "" }, row[column.key]));
    body.append(line);
  }
  table.append(body);
}

drawTiles();
drawTable();
drawChart();
new ResizeObserver(drawChart).observe(document.querySelector("#chart"));
