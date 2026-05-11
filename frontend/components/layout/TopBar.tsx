"use client";

import { Moon, PanelRight, PanelRightClose, Sun } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { useTheme } from "@/hooks/useTheme";
import { useAppStore } from "@/lib/store";

export function TopBar() {
  const { theme, toggleTheme } = useTheme();
  const artifactPanelOpen    = useAppStore((s) => s.artifactPanelOpen);
  const setArtifactPanelOpen = useAppStore((s) => s.setArtifactPanelOpen);
  // Select a primitive (number), never a derived array — selectors that return
  // a fresh `[]` each call cause useSyncExternalStore to loop in Zustand v5.
  const artifactCount = useAppStore((s) =>
    s.activeSessionId ? (s.artifacts[s.activeSessionId]?.length ?? 0) : 0,
  );

  return (
    <header
      className="flex h-12 shrink-0 items-center justify-between border-b border-border
                 bg-background/80 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/60"
    >
      <div className="flex items-center gap-2">
        <h1 className="text-sm font-medium text-foreground">
          Biomarker Discovery
        </h1>
        <span className="hidden text-xs text-muted sm:inline">
          · multi-agent proteomics platform
        </span>
      </div>

      <div className="flex items-center gap-1">
        <Button
          variant="ghost" size="icon"
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          onClick={toggleTheme}
        >
          {theme === "dark"
            ? <Sun  className="h-4 w-4" />
            : <Moon className="h-4 w-4" />}
        </Button>
        <Button
          variant="ghost" size="icon"
          aria-label={artifactPanelOpen ? "Close artifact panel" : "Open artifact panel"}
          onClick={() => setArtifactPanelOpen(!artifactPanelOpen)}
          title={`Artifact panel (${artifactCount})`}
        >
          {artifactPanelOpen
            ? <PanelRightClose className="h-4 w-4" />
            : <PanelRight      className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  );
}
