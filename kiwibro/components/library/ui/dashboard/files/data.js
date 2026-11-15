// KIWIBRO: sample numbers; replace with the project's own. Tiles, chart and table all read this one object.
const DATA = {
  unit: "orders",
  months: [
    { label: "Jan", value: 42 },
    { label: "Feb", value: 35 },
    { label: "Mar", value: 51 },
    { label: "Apr", value: 47 },
    { label: "May", value: 63 },
    { label: "Jun", value: 58 },
  ],
  columns: [
    { key: "id", label: "No." },
    { key: "item", label: "Item" },
    { key: "status", label: "Status" },
    { key: "total", label: "Total", numeric: true },
  ],
  rows: [
    { id: 1042, item: "Canvas bags", status: "Open", total: 86 },
    { id: 1041, item: "Clay mugs", status: "Shipped", total: 54 },
    { id: 1040, item: "Wool socks", status: "Shipped", total: 32 },
    { id: 1039, item: "Paper lamps", status: "Open", total: 120 },
    { id: 1038, item: "Tea tins", status: "Shipped", total: 41 },
  ],
};
