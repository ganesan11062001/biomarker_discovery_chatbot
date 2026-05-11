"use client";

import { FlaskConical, User } from "lucide-react";

import { Markdown } from "@/components/chat/Markdown";
import { SkillBadge } from "@/components/chat/SkillBadge";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/types";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 animate-fade-in",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-7 w-7 shrink-0 items-center justify-center rounded-md",
          isUser
            ? "bg-accent text-white"
            : "bg-surface-2 text-accent border border-border",
        )}
        aria-hidden
      >
        {isUser
          ? <User           className="h-3.5 w-3.5" />
          : <FlaskConical   className="h-3.5 w-3.5" />}
      </div>

      <div className={cn("min-w-0 max-w-[78%] flex-1",
                          isUser && "items-end flex flex-col")}>
        {/* Skill / tool badges (assistant only) */}
        {!isUser && (message.skills?.length ?? 0) > 0 && (
          <div className="mb-1.5 flex flex-wrap gap-1.5">
            {message.skills!.map((b, i) => (
              <SkillBadge key={`${b.name}-${b.startedAt}-${i}`} badge={b} />
            ))}
          </div>
        )}

        {/* Bubble */}
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-panel dark:shadow-panel-dark",
            isUser
              ? "bg-accent text-white rounded-tr-sm"
              : "bg-surface text-foreground border border-border rounded-tl-sm",
            message.streaming && "animate-pulse-dot",
          )}
        >
          {isUser
            ? <p className="whitespace-pre-wrap">{message.content}</p>
            : message.content
              ? <Markdown content={message.content} />
              : <TypingIndicator />}
        </div>

        {/* Timestamp */}
        <div className={cn("mt-1 text-[10px] text-muted",
                            isUser ? "text-right" : "text-left")}>
          {formatTime(message.createdAt)}
        </div>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <span className="inline-flex items-center gap-1.5 text-muted">
      <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-muted [animation-delay:-0.32s]"/>
      <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-muted [animation-delay:-0.16s]"/>
      <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-muted"/>
    </span>
  );
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "numeric", minute: "2-digit",
    });
  } catch {
    return "";
  }
}
