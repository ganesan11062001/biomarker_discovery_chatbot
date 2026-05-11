/**
 * lib/store.ts
 * Zustand store — global app state shared across components & hooks.
 *
 * Stays intentionally small: chat history is keyed by session, artifacts
 * are flattened across the active session, and UI flags (artifact panel
 * open/closed, sidebar collapsed) live here so all components can react.
 *
 * Persistence: only the active sessionId + theme live in localStorage.
 * Conversation content reloads from the backend on mount.
 */

"use client";

import { create } from "zustand";

import type {
  Artifact,
  ChatMessage,
  SessionSummary,
  Theme,
  UploadedFile,
} from "@/types";

interface AppState {
  // ── Persisted ───────────────────────────────────────────────────────────────
  activeSessionId: string | null;
  theme:           Theme;

  // ── In-memory only ──────────────────────────────────────────────────────────
  sessions:        SessionSummary[];
  messages:        Record<string, ChatMessage[]>;   // sessionId → messages
  files:           Record<string, UploadedFile[]>;  // sessionId → uploaded files
  artifacts:       Record<string, Artifact[]>;      // sessionId → artifacts (newest last)
  sidebarOpen:     boolean;
  artifactPanelOpen: boolean;
  /** Most recently produced or selected artifact, used to focus the panel. */
  focusedArtifactId: string | null;

  // ── Actions ─────────────────────────────────────────────────────────────────
  setActiveSession:    (id: string | null) => void;
  setTheme:            (theme: Theme) => void;
  toggleTheme:         () => void;
  setSessions:         (sessions: SessionSummary[]) => void;
  upsertSession:       (session: SessionSummary) => void;

  /** Replace the entire message list for a session (used by GET /sessions/:id). */
  setMessages:         (sessionId: string, messages: ChatMessage[]) => void;
  appendMessage:       (sessionId: string, message: ChatMessage) => void;
  /** Update the partial content of a streaming assistant message by id. */
  updateMessage:       (sessionId: string, id: string,
                         patch: Partial<ChatMessage>) => void;

  setFiles:            (sessionId: string, files: UploadedFile[]) => void;
  appendFile:          (sessionId: string, file: UploadedFile) => void;
  removeFile:          (sessionId: string, fileId: string) => void;

  appendArtifact:      (sessionId: string, artifact: Artifact) => void;
  setArtifacts:        (sessionId: string, artifacts: Artifact[]) => void;

  setSidebarOpen:      (open: boolean) => void;
  setArtifactPanelOpen:(open: boolean) => void;
  focusArtifact:       (id: string | null) => void;
}

export const useAppStore = create<AppState>()((set) => ({
  // Initial state
  activeSessionId:   null,
  theme:             "light",
  sessions:          [],
  messages:          {},
  files:             {},
  artifacts:         {},
  sidebarOpen:       true,
  artifactPanelOpen: false,
  focusedArtifactId: null,

  // Actions
  setActiveSession: (id) => set({ activeSessionId: id }),
  setTheme:         (theme) => set({ theme }),
  toggleTheme:      () => set((s) => ({ theme: s.theme === "dark" ? "light" : "dark" })),

  setSessions: (sessions) => set({ sessions }),
  upsertSession: (session) =>
    set((s) => ({
      sessions: [
        session,
        ...s.sessions.filter((x) => x.id !== session.id),
      ],
    })),

  setMessages: (sessionId, messages) =>
    set((s) => ({ messages: { ...s.messages, [sessionId]: messages } })),

  appendMessage: (sessionId, message) =>
    set((s) => {
      const prev = s.messages[sessionId] ?? [];
      return { messages: { ...s.messages, [sessionId]: [...prev, message] } };
    }),

  updateMessage: (sessionId, id, patch) =>
    set((s) => {
      const prev = s.messages[sessionId] ?? [];
      return {
        messages: {
          ...s.messages,
          [sessionId]: prev.map((m) => (m.id === id ? { ...m, ...patch } : m)),
        },
      };
    }),

  setFiles:    (sessionId, files) =>
    set((s) => ({ files: { ...s.files, [sessionId]: files } })),
  appendFile:  (sessionId, file) =>
    set((s) => ({
      files: { ...s.files, [sessionId]: [...(s.files[sessionId] ?? []), file] },
    })),
  removeFile:  (sessionId, fileId) =>
    set((s) => ({
      files: {
        ...s.files,
        [sessionId]: (s.files[sessionId] ?? []).filter((f) => f.fileId !== fileId),
      },
    })),

  appendArtifact: (sessionId, artifact) =>
    set((s) => ({
      artifacts: {
        ...s.artifacts,
        [sessionId]: [...(s.artifacts[sessionId] ?? []), artifact],
      },
      artifactPanelOpen: true,
      focusedArtifactId: artifact.id,
    })),
  setArtifacts: (sessionId, artifacts) =>
    set((s) => ({ artifacts: { ...s.artifacts, [sessionId]: artifacts } })),

  setSidebarOpen:       (sidebarOpen) => set({ sidebarOpen }),
  setArtifactPanelOpen: (artifactPanelOpen) => set({ artifactPanelOpen }),
  focusArtifact:        (focusedArtifactId) => set({ focusedArtifactId }),
}));
