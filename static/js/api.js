/**
 * Thin API client. In ?mock=1 it delegates to the in-browser fake backend;
 * otherwise every call hits the real Flask server on the same origin.
 */

import { MockBackend } from "./mock.js";

export const IS_MOCK = new URLSearchParams(window.location.search).get("mock") === "1";

const mock = IS_MOCK ? new MockBackend() : null;

export class ApiError extends Error {
  constructor(message, { status = 0, code = "" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request(path, { method = "GET", body, signal } = {}) {
  if (IS_MOCK) return mock.request(path, { method, body });

  let response;
  try {
    response = await fetch(path, {
      method,
      cache: "no-store",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ApiError("Can't reach the Kiwibro server.", { code: "network" });
  }

  let data = null;
  try {
    data = await response.json();
  } catch {
    /* some endpoints may be empty */
  }

  if (!response.ok) {
    throw new ApiError(data?.error?.message || `Request failed (${response.status})`, {
      status: response.status,
      code: data?.error?.code || "",
    });
  }
  return data;
}

function query(params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== undefined && value !== null && value !== "") search.set(key, value);
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

function chatError(response) {
  return response
    .json()
    .catch(() => ({}))
    .then((data) => {
      const message = data?.error?.message || `The chat request failed (${response.status}).`;
      return new ApiError(message, { status: response.status, code: data?.error?.code || "" });
    });
}

/**
 * Read an SSE response body, invoking onEvent(event, data) for every frame.
 * Handles `: comment` heartbeats and multi-line data fields.
 */
export async function consumeSSE(response, onEvent) {
  if (!response.body) throw new ApiError("This browser can't read the reply stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawEvent = false;

  const dispatch = (block) => {
    if (!block.trim()) return;
    let event = "message";
    const dataLines = [];
    for (const line of block.split(/\r?\n/)) {
      if (!line || line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") dataLines.push(value);
    }
    if (!dataLines.length) return;
    let data;
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch {
      return;
    }
    sawEvent = true;
    onEvent(event, data);
  };

  const drain = () => {
    const boundary = /\r?\n\r?\n/;
    let match = buffer.match(boundary);
    while (match) {
      const block = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      dispatch(block);
      match = buffer.match(boundary);
    }
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      drain();
    }
    buffer += decoder.decode();
    if (buffer.trim()) dispatch(buffer);
  } finally {
    reader.releaseLock();
  }
  return sawEvent;
}

export const api = {
  bootstrap: () => request("/api/bootstrap"),
  health: () => request("/health"),
  version: () => request("/api/version"),

  models: () => request("/api/models"),
  modelReady: () => request("/api/models/ready"),
  selectModel: (id) => request("/api/models/select", { method: "POST", body: { id } }),
  resizeContext: (id, contextLength) =>
    request("/api/models/context", { method: "POST", body: { id, context_length: contextLength } }),

  engine: () => request("/api/engine"),
  engineStop: () => request("/api/engine/stop", { method: "POST" }),

  conversations: () => request("/api/conversations"),
  searchConversations: (q, { signal } = {}) =>
    request(`/api/conversations/search?q=${encodeURIComponent(q)}`, { signal }),
  // warm: false reads a chat without prefilling its history for a next turn (export).
  conversation: (id, { warm = true } = {}) =>
    request(`/api/conversations/${encodeURIComponent(id)}${warm ? "" : "?warm=0"}`),
  patchConversation: (id, patch) =>
    request(`/api/conversations/${encodeURIComponent(id)}`, { method: "PATCH", body: patch }),
  deleteConversation: (id) =>
    request(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  forkConversation: (id, seq) =>
    request(`/api/conversations/${encodeURIComponent(id)}/fork`, {
      method: "POST",
      body: seq ? { seq } : {},
    }),
  compactConversation: (id) =>
    request(`/api/conversations/${encodeURIComponent(id)}/compact`, {
      method: "POST",
      body: {},
    }),
  clearConversation: (id) =>
    request(`/api/conversations/${encodeURIComponent(id)}/clear`, {
      method: "POST",
      body: {},
    }),
  activeTurns: () => request("/api/turns"),
  terminals: () => request("/api/terminals"),
  killTerminal: (id) =>
    request(`/api/terminals/${encodeURIComponent(id)}/kill`, {
      method: "POST",
      body: {},
    }),
  tasks: (conversationId) =>
    request(`/api/tasks${query({ conversation_id: conversationId })}`),

  chat(payload, { signal } = {}) {
    if (IS_MOCK) return mock.chat(payload, { signal });
    return fetch("/api/chat", {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
  },
  stopChat: (conversationId) =>
    request("/api/chat/stop", { method: "POST", body: { conversation_id: conversationId } }),
  steerChat: (conversationId, message) =>
    request("/api/chat/steer", {
      method: "POST",
      body: { conversation_id: conversationId, message },
    }),
  resolveApproval: (approvalId, decision) =>
    request(`/api/approvals/${encodeURIComponent(approvalId)}`, {
      method: "POST",
      body: { decision },
    }),

  memories: ({ q = "", kind = "" } = {}) =>
    request(`/api/memories${query({ q, kind })}`),
  memoryCount: async () => {
    const data = await request("/api/memories");
    return Number.isFinite(data?.total) ? data.total : (data?.memories?.length ?? 0);
  },
  createMemory: ({ content, kind }) =>
    request("/api/memories", { method: "POST", body: { content, kind } }),
  patchMemory: (id, patch) =>
    request(`/api/memories/${encodeURIComponent(id)}`, { method: "PATCH", body: patch }),
  deleteMemory: (id) =>
    request(`/api/memories/${encodeURIComponent(id)}`, { method: "DELETE" }),
  forgetMemories: () => request("/api/memories", { method: "DELETE" }),

  settings: () => request("/api/settings"),
  patchSettings: (patch) => request("/api/settings", { method: "PATCH", body: patch }),
  workspace: () => request("/api/workspace"),
};

export { chatError };
