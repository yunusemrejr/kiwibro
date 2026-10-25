/**
 * Chat export as Markdown. Pure functions only (no DOM, no fetch), so the
 * module can be imported from Node tests as well as from the browser.
 */

const STEERING_PREFIX = "[Steering note from the user, sent while you were working] ";
const TOOL_OUTPUT_LIMIT = 2000;

/** A safe, lowercase download name such as `php-zerodays.md`. */
export function exportFilename(title) {
  const slug = String(title ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60)
    .replace(/-+$/, "");
  return `${slug || "kiwibro-chat"}.md`;
}

/**
 * Render a conversation and its messages as one Markdown document.
 * Reasoning is never included. Assistant turns that only carried tool calls
 * (empty content) are skipped.
 */
export function conversationMarkdown(conversation, messages, { now = new Date() } = {}) {
  const title = singleLine(conversation?.title) || "Untitled chat";
  const model = singleLine(conversation?.model);
  const blocks = [
    `# ${title}\n\n_Exported from Kiwibro on ${localDay(now)}${model ? ` · model ${model}` : ""}_`,
  ];

  for (const message of Array.isArray(messages) ? messages : []) {
    const block = messageBlock(message);
    if (block) blocks.push(block);
  }

  return `${blocks.join("\n\n")}\n`;
}

function messageBlock(message) {
  const content = text(message?.content);
  switch (message?.role) {
    case "user": {
      if (content.startsWith(STEERING_PREFIX)) {
        return `**You (while it worked)**\n\n${trimBody(content.slice(STEERING_PREFIX.length))}`;
      }
      return `**You**\n\n${trimBody(content)}`;
    }
    case "assistant": {
      const body = trimBody(content);
      return body ? `**Kiwibro**\n\n${body}` : null;
    }
    case "tool":
      return toolBlock(message, content);
    default:
      return null;
  }
}

function toolBlock(message, content) {
  const name = singleLine(message.tool_name) || "tool";
  let output = Array.from(content.replace(/\n+$/, ""));
  if (output.length > TOOL_OUTPUT_LIMIT) {
    output = [...output.slice(0, TOOL_OUTPUT_LIMIT), "…"];
  }
  const body = output.join("");
  const fence = "`".repeat(Math.max(3, longestBacktickRun(body) + 1));
  return [
    `<details><summary>Tool: ${name.replace(/</g, "&lt;")}</summary>`,
    "",
    `${fence}text`,
    body,
    fence,
    "",
    "</details>",
  ].join("\n");
}

function text(value) {
  if (typeof value === "string") return value;
  return value == null ? "" : String(value);
}

// Drop leading blank lines and trailing whitespace, keep indentation of the first line.
function trimBody(value) {
  return value.replace(/^(?:[ \t]*\r?\n)+/, "").trimEnd();
}

function singleLine(value) {
  return text(value).replace(/\s+/g, " ").trim();
}

function longestBacktickRun(value) {
  let longest = 0;
  for (const run of value.match(/`+/g) || []) longest = Math.max(longest, run.length);
  return longest;
}

function localDay(date) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}
