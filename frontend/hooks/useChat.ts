/**
 * hooks/useChat.ts
 * Public chat API used by the input box and the message list.
 *
 *   const { messages, sendMessage, streaming, abort } = useChat();
 *
 * Sending a message:
 *   1. Append the user message to the store immediately.
 *   2. Append an empty streaming assistant message (so UI shows a typing dot).
 *   3. Open the SSE stream; for every token event append to the streaming msg.
 *   4. On `message_complete`, replace the streaming content with the final
 *      version (which may include skill badges + artifacts).
 *   5. Re-extract inline artifacts from the final content into the panel.
 */
"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";

import { streamChat } from "@/lib/api";
import { extractArtifacts } from "@/lib/artifacts";
import { useAppStore } from "@/lib/store";
import type { ChatMessage, SkillBadge } from "@/types";

interface UseChatResult {
  messages:    ChatMessage[];
  streaming:   boolean;
  sendMessage: (text: string) => void;
  abort:       () => void;
}

export function useChat(): UseChatResult {
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const messagesBySession = useAppStore((s) => s.messages);
  const appendMessage   = useAppStore((s) => s.appendMessage);
  const updateMessage   = useAppStore((s) => s.updateMessage);
  const appendArtifact  = useAppStore((s) => s.appendArtifact);

  const streamingRef    = useRef<{ abort: () => void } | null>(null);
  const streamingMsgIdRef = useRef<string | null>(null);

  const messages = useMemo(
    () => (activeSessionId ? messagesBySession[activeSessionId] ?? [] : []),
    [activeSessionId, messagesBySession],
  );
  const streaming = Boolean(
    streamingMsgIdRef.current &&
    messages.some((m) => m.id === streamingMsgIdRef.current && m.streaming),
  );

  // Abort any in-flight stream when the active session changes or unmount
  useEffect(() => {
    return () => { streamingRef.current?.abort(); };
  }, [activeSessionId]);

  const sendMessage = useCallback((text: string) => {
    if (!activeSessionId || !text.trim()) return;
    // Guard against rapid double-submits while a stream is in flight.
    // Without this, dev-mode HMR / double-tap can fire the same long-running
    // workflow multiple times against the same session.
    if (streamingRef.current) {
      console.warn("sendMessage ignored — a stream is already in flight");
      return;
    }
    const sid = activeSessionId;
    const now = () => new Date().toISOString();

    // 1. User message
    const userMsg: ChatMessage = {
      id:        crypto.randomUUID(),
      role:      "user",
      content:   text,
      createdAt: now(),
    };
    appendMessage(sid, userMsg);

    // 2. Empty streaming assistant placeholder
    const placeholderId = crypto.randomUUID();
    streamingMsgIdRef.current = placeholderId;
    const placeholder: ChatMessage = {
      id:        placeholderId,
      role:      "assistant",
      content:   "",
      createdAt: now(),
      streaming: true,
      skills:    [],
    };
    appendMessage(sid, placeholder);

    // 3. Open SSE — guard above guarantees no stream is in flight here
    streamingRef.current = streamChat(sid, text, (event) => {
      switch (event.type) {
        case "token": {
          // Append delta to placeholder content
          const current = (useAppStore.getState().messages[sid] ?? [])
            .find((m) => m.id === placeholderId);
          updateMessage(sid, placeholderId, {
            content: (current?.content ?? "") + event.delta,
          });
          break;
        }
        case "skill": {
          // Merge the badge into the streaming message
          const current = (useAppStore.getState().messages[sid] ?? [])
            .find((m) => m.id === placeholderId);
          const existing: SkillBadge[] = current?.skills ?? [];
          // Update by name if already present (running → done)
          const idx = existing.findIndex((b) => b.name === event.badge.name &&
                                                  b.startedAt === event.badge.startedAt);
          const next: SkillBadge[] = idx >= 0
            ? existing.map((b, i) => (i === idx ? event.badge : b))
            : [...existing, event.badge];
          updateMessage(sid, placeholderId, { skills: next });
          break;
        }
        case "artifact":
          appendArtifact(sid, event.artifact);
          break;
        case "message_complete": {
          const finalContent = event.message.content || "";
          const artifacts = extractArtifacts(placeholderId, finalContent, now());
          updateMessage(sid, placeholderId, {
            content:   finalContent,
            streaming: false,
            hasPlots:  event.message.hasPlots,
            skills:    event.message.skills,
            artifacts,
          });
          for (const art of artifacts) appendArtifact(sid, art);
          break;
        }
        case "error":
          updateMessage(sid, placeholderId, {
            content:   `⚠ ${event.error}`,
            streaming: false,
          });
          break;
        case "done":
          updateMessage(sid, placeholderId, { streaming: false });
          streamingRef.current = null;
          streamingMsgIdRef.current = null;
          break;
      }
    });
  }, [activeSessionId, appendMessage, updateMessage, appendArtifact]);

  const abort = useCallback(() => {
    streamingRef.current?.abort();
    if (activeSessionId && streamingMsgIdRef.current) {
      updateMessage(activeSessionId, streamingMsgIdRef.current,
                    { streaming: false });
    }
    streamingRef.current = null;
    streamingMsgIdRef.current = null;
  }, [activeSessionId, updateMessage]);

  return { messages, sendMessage, streaming, abort };
}
