"use client";

import { FileSpreadsheet, Loader2, X } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { cn, formatBytes, truncate } from "@/lib/utils";
import type { UploadedFile } from "@/types";

interface FileChipProps {
  file:     UploadedFile;
  onRemove: (fileId: string) => void;
}

export function FileChip({ file, onRemove }: FileChipProps) {
  return (
    <div
      className={cn(
        "group flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs",
        file.error
          ? "border-red-500/40 bg-red-500/5 text-red-500"
          : "border-border bg-surface text-foreground",
      )}
    >
      {file.uploading
        ? <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />
        : <FileSpreadsheet className="h-3.5 w-3.5 text-accent" />}

      <div className="flex min-w-0 flex-col leading-tight">
        <span className="truncate font-medium">{truncate(file.filename, 38)}</span>
        <span className="text-[10px] text-muted">
          {file.uploading
            ? "uploading…"
            : file.error
            ? truncate(file.error, 48)
            : [
                formatBytes(file.size),
                file.dataType,
                file.software,
                file.rowCount != null && `${file.rowCount} rows`,
                file.colCount != null && `${file.colCount} samples`,
              ].filter(Boolean).join(" · ")}
        </span>
      </div>

      {!file.uploading && (
        <Button
          variant="ghost" size="icon"
          aria-label="Remove file"
          onClick={() => onRemove(file.fileId)}
          className="ml-auto h-5 w-5 opacity-0 group-hover:opacity-100"
        >
          <X className="h-3 w-3" />
        </Button>
      )}
    </div>
  );
}
