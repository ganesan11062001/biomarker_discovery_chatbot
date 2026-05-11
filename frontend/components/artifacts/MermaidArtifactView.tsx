"use client";

import { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

import { useTheme } from "@/hooks/useTheme";
import type { MermaidArtifact } from "@/types";

mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });

export function MermaidArtifactView({ artifact }: { artifact: MermaidArtifact }) {
  const { theme } = useTheme();
  const ref = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: theme === "dark" ? "dark" : "default",
      securityLevel: "loose",
    });
  }, [theme]);

  useEffect(() => {
    if (!ref.current) return;
    const id = `mermaid-${artifact.id.replace(/[^a-z0-9]/gi, "_")}`;
    mermaid.render(id, artifact.source)
      .then(({ svg }) => {
        if (ref.current) ref.current.innerHTML = svg;
        setError(null);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to render diagram");
      });
  }, [artifact.id, artifact.source, theme]);

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-3 py-1.5">
        <span className="text-xs font-medium">
          {artifact.title || "Diagram"}
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-4">
        {error
          ? <pre className="whitespace-pre-wrap text-xs text-red-500">{error}</pre>
          : <div ref={ref} className="mermaid-container" />}
      </div>
    </div>
  );
}
