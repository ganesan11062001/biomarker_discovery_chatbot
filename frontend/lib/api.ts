/**
 * lib/api.ts
 * Thin client for the FastAPI backend. All HTTP calls go through this module
 * so component code never hard-codes URLs or headers.
 */

import type {
  ChatMessage,
  SessionDetail,
  SessionSummary,
  StreamEvent,
  UploadedFile,
} from "@/types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

// ── REST helpers ──────────────────────────────────────────────────────────────

async function jsonRequest<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init.headers || {}),
    },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail.slice(0, 300)}`);
  }
  return (await res.json()) as T;
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export async function createSession(opts?: {
  diseaseProgram?: string;
  organism?:       string;
}): Promise<{ session_id: string }> {
  const params = new URLSearchParams();
  if (opts?.diseaseProgram) params.set("disease_program", opts.diseaseProgram);
  if (opts?.organism)       params.set("organism", opts.organism);
  return jsonRequest(`/chat/session?${params}`, { method: "POST" });
}

export async function listSessions(): Promise<SessionSummary[]> {
  return jsonRequest<SessionSummary[]>("/sessions");
}

export async function getSession(id: string): Promise<SessionDetail> {
  return jsonRequest<SessionDetail>(`/sessions/${id}`);
}

// ── Upload ────────────────────────────────────────────────────────────────────

export async function uploadFile(
  file: File,
  sessionId: string,
): Promise<UploadedFile> {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", sessionId);
  const res = await fetch(`${API_BASE}/upload/`, {
    method: "POST",
    body:   form,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch { /* ignore */ }
    throw new Error(`Upload failed (${res.status}): ${detail}`);
  }
  const body = await res.json();
  return {
    fileId:     body.file_id,
    filename:   body.filename,
    size:       body.size ?? 0,
    dataType:   body.data_type,
    software:   body.software,
    rowCount:   body.n_proteins,
    colCount:   body.n_samples,
    uploadedAt: new Date().toISOString(),
  };
}

// ── Files (download) ──────────────────────────────────────────────────────────

export function fileUrl(sessionId: string, path: string): string {
  return `${API_BASE}/results/${sessionId}/file?path=${encodeURIComponent(path)}`;
}

// ── Chat (SSE streaming) ──────────────────────────────────────────────────────

/**
 * Send a chat message and stream the response. The handler is called for
 * every parsed SSE event. Returns a function that aborts the stream.
 *
 * The backend should emit `text/event-stream` with named events matching
 * the `StreamEvent` union — see types/index.ts.
 *
 * Fallback: if the backend returns plain JSON (legacy synchronous path),
 * we adapt it to a single `message_complete` event so callers don't have
 * to branch.
 */
export function streamChat(
  sessionId: string,
  message: string,
  onEvent: (event: StreamEvent) => void,
): { abort: () => void } {
  const ac = new AbortController();

  (async () => {
    let res: Response;
    try {
      res = await fetch(`${API_BASE}/chat/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept:         "text/event-stream",
        },
        body: JSON.stringify({ session_id: sessionId, message }),
        signal: ac.signal,
      });
    } catch (err) {
      if (!ac.signal.aborted) {
        onEvent({
          type:      "error",
          sessionId,
          error:     err instanceof Error ? err.message : "Network error",
        });
      }
      return;
    }

    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      onEvent({ type: "error", sessionId,
                error: `API ${res.status}: ${detail.slice(0, 300)}` });
      return;
    }

    const contentType = res.headers.get("content-type") || "";

    // ── Legacy fallback: backend returned a single JSON response ────────────
    if (contentType.includes("application/json")) {
      try {
        const body: { session_id?: string; response: string; intent?: string } =
          await res.json();
        const finalSid = body.session_id || sessionId;
        const completed: ChatMessage = {
          id:        crypto.randomUUID(),
          role:      "assistant",
          content:   body.response,
          createdAt: new Date().toISOString(),
        };
        onEvent({ type: "message_complete", sessionId: finalSid,
                  message: completed });
        onEvent({ type: "done", sessionId: finalSid });
      } catch (err) {
        onEvent({ type: "error", sessionId,
                  error: err instanceof Error ? err.message : "Bad response" });
      }
      return;
    }

    // ── Real SSE stream ─────────────────────────────────────────────────────
    const reader = res.body?.getReader();
    if (!reader) {
      onEvent({ type: "error", sessionId,
                error: "No response body to stream." });
      return;
    }
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line. Process each complete one.
        let idx: number;
        while ((idx = buffer.indexOf("\n\n")) >= 0) {
          const frame = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const parsed = parseSseFrame(frame, sessionId);
          if (parsed) onEvent(parsed);
        }
      }
      onEvent({ type: "done", sessionId });
    } catch (err) {
      if (!ac.signal.aborted) {
        onEvent({ type: "error", sessionId,
                  error: err instanceof Error ? err.message : "Stream error" });
      }
    }
  })();

  return { abort: () => ac.abort() };
}

/** Parse a single SSE frame ("event: foo\ndata: {...}") into a StreamEvent. */
function parseSseFrame(
  frame: string,
  fallbackSession: string,
): StreamEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  const rawData = dataLines.join("\n");
  if (!rawData) return null;

  let payload: Record<string, unknown> = {};
  try {
    payload = JSON.parse(rawData);
  } catch {
    // Treat as plain text token
    return { type: "token", sessionId: fallbackSession, delta: rawData };
  }

  switch (event) {
    case "token":
      return { type: "token", sessionId: fallbackSession,
               delta: String(payload.delta ?? "") };
    case "skill":
      return { type: "skill", sessionId: fallbackSession,
               badge: payload.badge as never };
    case "artifact":
      return { type: "artifact", sessionId: fallbackSession,
               artifact: payload.artifact as never };
    case "message_complete":
      return { type: "message_complete", sessionId: fallbackSession,
               message: payload.message as ChatMessage };
    case "error":
      return { type: "error", sessionId: fallbackSession,
               error: String(payload.error ?? "Unknown error") };
    case "done":
      return { type: "done", sessionId: fallbackSession };
    default:
      return null;
  }
}
