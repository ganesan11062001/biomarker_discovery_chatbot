"use client";

import { ExternalLink, FlaskConical, ImageIcon, User } from "lucide-react";

import { Markdown } from "@/components/chat/Markdown";
import { SkillBadge } from "@/components/chat/SkillBadge";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import type { ChatMessage, PlotArtifact } from "@/types";

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

        {/* Inline plot grid (assistant only) */}
        {!isUser && <InlinePlotGrid message={message} />}

        {/* Timestamp */}
        <div className={cn("mt-1 text-[10px] text-muted",
                            isUser ? "text-right" : "text-left")}>
          {formatTime(message.createdAt)}
        </div>
      </div>
    </div>
  );
}

/**
 * Render plot artifacts attached to this message as a small thumbnail grid
 * beneath the bubble. Clicking a thumbnail focuses it in the side panel and
 * opens the panel if it isn't already visible. Static PNG / SVG plots show
 * as `<img>` previews; HTML-only plots show an "Open interactive" placeholder.
 */
function InlinePlotGrid({ message }: { message: ChatMessage }) {
  const plots = (message.artifacts ?? []).filter(
    (a): a is PlotArtifact => a.kind === "plot",
  );
  const setPanelOpen = useAppStore((s) => s.setArtifactPanelOpen);
  const focusArtifact = useAppStore((s) => s.focusArtifact);

  if (plots.length === 0) return null;

  const open = (id: string) => {
    focusArtifact(id);
    setPanelOpen(true);
  };

  return (
    <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
      {plots.map((p) => (
        <button
          key={p.id}
          type="button"
          onClick={() => open(p.id)}
          className="group flex flex-col overflow-hidden rounded-lg border border-border
                     bg-surface text-left transition-colors hover:border-accent
                     focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
          aria-label={`Open ${p.title || "plot"} in side panel`}
        >
          <div className="flex h-24 w-full items-center justify-center bg-white dark:bg-surface-2">
            {p.imageUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={p.imageUrl}
                alt={p.title || "Plot"}
                className="max-h-full max-w-full object-contain"
                loading="lazy"
              />
            ) : (
              <div className="flex flex-col items-center gap-1 text-muted">
                <ExternalLink className="h-4 w-4" />
                <span className="text-[10px]">Interactive</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-1 border-t border-border px-2 py-1">
            <ImageIcon className="h-3 w-3 text-accent" />
            <span className="truncate text-[11px] text-foreground">
              {p.title || "Plot"}
            </span>
          </div>
        </button>
      ))}
    </div>
  );
}

function TypingIndicator() {
  return (
    <span
      className="inline-flex items-baseline gap-0.5 text-sm italic text-muted"
      role="status"
      aria-live="polite"
      aria-label="Thinking"
    >
      <span>Thinking</span>
      <span className="animate-pulse-dot [animation-delay:-0.32s]">.</span>
      <span className="animate-pulse-dot [animation-delay:-0.16s]">.</span>
      <span className="animate-pulse-dot">.</span>
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
