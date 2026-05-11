"use client";

import { Send, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { DropZone } from "@/components/upload/DropZone";
import { FileChip } from "@/components/upload/FileChip";
import { Button } from "@/components/ui/Button";
import { useChat } from "@/hooks/useChat";
import { useFileUpload } from "@/hooks/useFileUpload";
import { cn } from "@/lib/utils";

export function ChatInput() {
  const { sendMessage, streaming, abort } = useChat();
  const { files, removeFile } = useFileUpload();
  const [draft, setDraft] = useState("");
  const taRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-grow the textarea up to a max height
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "0px";
    ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`;
  }, [draft]);

  const submit = () => {
    if (streaming) {
      abort();
      return;
    }
    const text = draft.trim();
    if (!text) return;
    sendMessage(text);
    setDraft("");
  };

  return (
    <div className="border-t border-border bg-background/80 px-4 py-3 backdrop-blur">
      <div className="mx-auto w-full max-w-3xl space-y-2">
        {/* Attached files (above the textarea) */}
        {files.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {files.map((f) => (
              <FileChip key={f.fileId} file={f} onRemove={removeFile} />
            ))}
          </div>
        )}

        {/* Drop zone (small) */}
        {files.length === 0 && <DropZone variant="compact" />}

        {/* Input row */}
        <div className={cn(
          "flex items-end gap-2 rounded-2xl border border-border bg-surface px-3 py-2",
          "transition-shadow focus-within:border-accent focus-within:shadow-panel",
        )}>
          <textarea
            ref={taRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask anything about your proteomics data…"
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm leading-relaxed
                       placeholder:text-muted focus:outline-none"
          />
          <Button
            variant={streaming ? "danger" : "primary"}
            size="icon"
            onClick={submit}
            disabled={!streaming && !draft.trim()}
            aria-label={streaming ? "Stop generating" : "Send message"}
          >
            {streaming
              ? <Square className="h-3.5 w-3.5" />
              : <Send   className="h-3.5 w-3.5" />}
          </Button>
        </div>

        <p className="text-center text-[10px] text-muted">
          Enter to send · Shift+Enter for newline · Drop a file anywhere to attach
        </p>
      </div>
    </div>
  );
}
