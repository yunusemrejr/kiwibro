const toasts = document.querySelector("#toasts");

// "error" toasts use the danger color and are announced at once.
function toast(message, kind = "info") {
  const item = document.createElement("p");
  item.className = `toast ${kind}`;
  item.textContent = message;
  if (kind === "error") item.setAttribute("role", "alert");
  toasts.append(item);
  setTimeout(() => item.remove(), 4000);
}

for (const opener of document.querySelectorAll("[data-open]")) {
  opener.addEventListener("click", () => document.getElementById(opener.dataset.open).showModal());
}
for (const trigger of document.querySelectorAll("[data-toast]")) {
  trigger.addEventListener("click", () => toast(trigger.dataset.toast));
}

for (const dialog of document.querySelectorAll("dialog")) {
  // The dialog box itself has no padding, so a click on it can only be a click on the backdrop.
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close("cancel");
  });
  dialog.addEventListener("close", () => {
    if (dialog.returnValue === "archive") toast("File archived.");
  });
}
