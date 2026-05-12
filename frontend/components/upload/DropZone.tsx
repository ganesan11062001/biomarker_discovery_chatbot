"use client";

import { UploadCloud } from "lucide-react";
import { useRef, useState } from "react";

import { useChat } from "@/hooks/useChat";
import { useFileUpload } from "@/hooks/useFileUpload";
import { cn } from "@/lib/utils";

interface DropZoneProps {
  /** Compact prompt or hero prompt; affects size + copy. */
  variant?: "compact" | "hero";
}

const ACCEPTED = [".csv", ".tsv", ".xlsx", ".xls", ".parquet"];

/**
 * Auto-message dispatched to the chat as soon as a file finishes uploading.
 * Lands as a normal user turn → orchestrator routes it to run_full_pipeline
 * (per the rules in agents/learning_agent.py).
 */
const AUTO_ANALYSIS_PROMPT =
  "Run the full analysis pipeline on the uploaded dataset.";

export function DropZone({ variant = "compact" }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const { upload } = useFileUpload();
  const { sendMessage } = useChat();

  const handleFiles = async (files: FileList | File[]) => {
    // Upload files in parallel; on the first success, auto-trigger the
    // full pipeline. The sendMessage guard inside useChat prevents double
    // fires if the user attaches multiple files in one drop.
    const results = await Promise.all(
      Array.from(files).map((f) => upload(f)),
    );
    const firstOk = results.find((r) => r && !r.error);
    if (firstOk) {
      sendMessage(AUTO_ANALYSIS_PROMPT);
    }
  };

  const hero = variant === "hero";

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
      className={cn(
        "group cursor-pointer rounded-xl border-2 border-dashed transition-colors",
        dragOver
          ? "border-accent bg-accent/5"
          : "border-border bg-surface hover:border-accent/60",
        hero ? "px-6 py-10 text-center" : "px-3 py-2",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPTED.join(",")}
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
      {hero ? (
        <div className="flex flex-col items-center gap-2">
          <UploadCloud className={cn("h-8 w-8",
                                       dragOver ? "text-accent" : "text-muted")} />
          <div className="text-sm font-medium">
            Drag &amp; drop your proteomics workbook
          </div>
          <div className="text-xs text-muted">
            or click to browse — .csv, .tsv, .xlsx, .xls, .parquet · max 200 MB
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-2 text-xs">
          <UploadCloud className="h-3.5 w-3.5 text-muted" />
          <span className="text-muted">
            Drop or attach proteomics data (.csv / .xlsx / .parquet)
          </span>
        </div>
      )}
    </div>
  );
}
