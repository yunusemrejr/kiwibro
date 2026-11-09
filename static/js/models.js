/** Placeholder until the panel is built. */
import { openSheet } from "./ui.js";

export function createModelsView({ dialog }) {
  return { open: () => openSheet(dialog), refreshIfOpen: () => {} };
}
