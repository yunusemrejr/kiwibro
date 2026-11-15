// KIWIBRO: one entry per column; key is the field name in ROWS, decimals fixes the digits of a number
const COLUMNS = [
  { key: "name", label: "Name" },
  { key: "category", label: "Category" },
  { key: "stock", label: "Stock", numeric: true },
  { key: "price", label: "Price", numeric: true, decimals: 2 },
];
const PAGE_SIZE = 5;

const state = { sortKey: "", direction: 1, query: "", page: 0 };
const head = document.querySelector("#head");
const body = document.querySelector("#body");
const status = document.querySelector("#status");

function visibleRows() {
  const query = state.query.trim().toLowerCase();
  const rows = ROWS.filter((row) => Object.values(row).join(" ").toLowerCase().includes(query));
  const column = COLUMNS.find((item) => item.key === state.sortKey);
  if (!column) return rows;
  const compare = column.numeric
    ? (a, b) => a[column.key] - b[column.key]
    : (a, b) => String(a[column.key]).localeCompare(String(b[column.key]));
  return rows.sort((a, b) => compare(a, b) * state.direction);
}

function render() {
  const rows = visibleRows();
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  state.page = Math.min(state.page, pages - 1);
  body.replaceChildren(...rows.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE).map((row) => {
    const line = document.createElement("tr");
    for (const column of COLUMNS) {
      const cell = document.createElement("td");
      cell.textContent = column.decimals ? row[column.key].toFixed(column.decimals) : row[column.key];
      if (column.numeric) cell.className = "num";
      line.append(cell);
    }
    return line;
  }));
  for (const th of head.children) {
    th.setAttribute("aria-sort", th.dataset.key === state.sortKey ? (state.direction === 1 ? "ascending" : "descending") : "none");
  }
  document.querySelector("#empty").hidden = rows.length > 0;
  status.textContent = rows.length ? `Page ${state.page + 1} of ${pages}` : "No rows";
  document.querySelector("#prev").disabled = state.page === 0;
  document.querySelector("#next").disabled = state.page >= pages - 1;
}

for (const column of COLUMNS) {
  const th = document.createElement("th");
  const button = document.createElement("button");
  th.dataset.key = column.key;
  th.scope = "col";
  button.type = "button";
  button.textContent = column.label;
  button.className = column.numeric ? "sort num" : "sort";
  button.addEventListener("click", () => {
    state.direction = state.sortKey === column.key ? -state.direction : 1;
    state.sortKey = column.key;
    render();
  });
  th.append(button);
  head.append(th);
}

document.querySelector("#filter").addEventListener("input", (event) => {
  state.query = event.target.value;
  state.page = 0;
  render();
});
document.querySelector("#prev").addEventListener("click", () => { state.page -= 1; render(); });
document.querySelector("#next").addEventListener("click", () => { state.page += 1; render(); });
render();
