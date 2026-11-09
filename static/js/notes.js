/** Placeholder until the panel is built. */
import { openSheet } from "./ui.js";

export function createNotesView({ dialog }) {
  return { open: () => openSheet(dialog), refreshIfOpen: () => {} };
}
