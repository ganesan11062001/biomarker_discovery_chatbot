"use client";

import { Code2, FileText, GitBranch, ImageIcon, X } from "lucide-react";

import { CodeArtifactView }    from "@/components/artifacts/CodeArtifactView";
import { MermaidArtifactView } from "@/components/artifacts/MermaidArtifactView";
import { PlotArtifactView }    from "@/components/artifacts/PlotArtifactView";
import { TableArtifactView }   from "@/components/artifacts/TableArtifactView";
import { Button } from "@/components/ui/Button";
import { useArtifact } from "@/hooks/useArtifact";
import { cn, relativeTime, truncate } from "@/lib/utils";
import type { Artifact, ArtifactKind } from "@/types";

const KIND_ICON = {
  code:    Code2,
  plot:    ImageIcon,
  table:   FileText,
  mermaid: GitBranch,
} as const satisfies Record<ArtifactKind, React.ElementType>;

export function ArtifactPanel() {
  const { artifacts, focused, panelOpen, setPanelOpen, focus } = useArtifact();

  if (!panelOpen) return null;

  return (
    <aside className="flex w-[420px] shrink-0 flex-col border-l border-border bg-background">
      <div className="flex h-12 items-center justify-between border-b border-border px-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Artifacts</span>
          <span className="text-[10px] text-muted">{artifacts.length}</span>
        </div>
        <Button
          variant="ghost" size="icon"
          aria-label="Close artifact panel"
          onClick={() => setPanelOpen(false)}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {artifacts.length === 0 ? (
        <div className="flex flex-1 items-center justify-center px-6 text-center">
          <p className="text-xs text-muted">
            Code blocks, tables, plots, and diagrams from the assistant
            will appear here when generated.
          </p>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          {/* Artifact tabs (clickable list of recent artifacts) */}
          <div className="flex max-h-32 shrink-0 overflow-x-auto border-b border-border bg-surface">
            <ul className="flex w-full flex-col">
              {artifacts.slice().reverse().map((a) => {
                const Icon = KIND_ICON[a.kind];
                const active = focused?.id === a.id;
                return (
                  <li key={a.id}>
                    <button
                      onClick={() => focus(a.id)}
                      className={cn(
                        "flex w-full items-center gap-2 border-l-2 px-3 py-1.5 text-left transition-colors",
                        active
                          ? "border-accent bg-accent/5"
                          : "border-transparent hover:bg-surface-2",
                      )}
                    >
                      <Icon className="h-3.5 w-3.5 text-accent" />
                      <div className="flex min-w-0 flex-1 flex-col leading-tight">
                        <span className="truncate text-xs">
                          {truncate(a.title || a.kind, 32)}
                        </span>
                        <span className="text-[10px] text-muted">
                          {a.kind} · {relativeTime(a.createdAt)}
                        </span>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>

          {/* Focused artifact body */}
          <div className="min-h-0 flex-1 p-3">
            {focused
              ? <ArtifactBody artifact={focused} />
              : <p className="p-6 text-center text-xs text-muted">
                  Select an artifact to view it here.
                </p>}
          </div>
        </div>
      )}
    </aside>
  );
}

function ArtifactBody({ artifact }: { artifact: Artifact }) {
  switch (artifact.kind) {
    case "code":    return <CodeArtifactView    artifact={artifact} />;
    case "plot":    return <PlotArtifactView    artifact={artifact} />;
    case "table":   return <TableArtifactView   artifact={artifact} />;
    case "mermaid": return <MermaidArtifactView artifact={artifact} />;
  }
}
