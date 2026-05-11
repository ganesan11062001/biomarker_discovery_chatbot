"use client";

import { Download, ExternalLink } from "lucide-react";

import type { PlotArtifact } from "@/types";

// Anchor styled like our ghost Button. Anchors can't be wrapped in <button>,
// so we re-apply the same Tailwind classes directly.
const linkBtn =
  "inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] " +
  "text-muted hover:bg-surface-2 hover:text-foreground transition-colors";

export function PlotArtifactView({ artifact }: { artifact: PlotArtifact }) {
  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5">
        <span className="text-xs font-medium">
          {artifact.title || "Plot"}
        </span>
        <div className="flex items-center gap-1">
          {artifact.imageUrl && (
            <a href={artifact.imageUrl} download className={linkBtn}>
              <Download className="h-3.5 w-3.5" /> PNG
            </a>
          )}
          {artifact.htmlUrl && (
            <a href={artifact.htmlUrl} target="_blank" rel="noreferrer" className={linkBtn}>
              <ExternalLink className="h-3.5 w-3.5" /> Interactive
            </a>
          )}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3">
        {artifact.htmlUrl ? (
          <iframe
            src={artifact.htmlUrl}
            className="h-[60vh] w-full rounded-md border border-border bg-white"
            title={artifact.title}
          />
        ) : artifact.imageUrl ? (
          <img
            src={artifact.imageUrl}
            alt={artifact.title || "Plot"}
            className="mx-auto max-h-[70vh] w-auto rounded-md border border-border bg-white"
          />
        ) : (
          <p className="p-6 text-center text-xs text-muted">
            No plot data available.
          </p>
        )}
      </div>
    </div>
  );
}
