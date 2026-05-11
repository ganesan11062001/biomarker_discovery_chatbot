/**
 * hooks/useArtifact.ts
 * Read-only selector + helpers for the artifact panel.
 */
"use client";

import { useCallback, useMemo } from "react";

import { useAppStore } from "@/lib/store";
import type { Artifact } from "@/types";

interface UseArtifactResult {
  artifacts:    Artifact[];
  focused:      Artifact | null;
  panelOpen:    boolean;
  setPanelOpen: (open: boolean) => void;
  focus:        (id: string | null) => void;
}

export function useArtifact(): UseArtifactResult {
  const sessionId        = useAppStore((s) => s.activeSessionId);
  const artifactsMap     = useAppStore((s) => s.artifacts);
  const panelOpen        = useAppStore((s) => s.artifactPanelOpen);
  const focusedId        = useAppStore((s) => s.focusedArtifactId);
  const setPanelOpen     = useAppStore((s) => s.setArtifactPanelOpen);
  const focus            = useAppStore((s) => s.focusArtifact);

  const artifacts = useMemo(
    () => (sessionId ? artifactsMap[sessionId] ?? [] : []),
    [sessionId, artifactsMap],
  );

  const focused = useMemo(() => {
    if (!focusedId) return artifacts[artifacts.length - 1] ?? null;
    return artifacts.find((a) => a.id === focusedId) ?? null;
  }, [focusedId, artifacts]);

  const focusCallback = useCallback((id: string | null) => focus(id), [focus]);

  return { artifacts, focused, panelOpen, setPanelOpen, focus: focusCallback };
}
