"use client";

import {
  BarChart3, CheckCircle2, Database, FileSearch, FlaskConical,
  GitBranch, Loader2, Microscope, Sparkles, XCircle,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";
import type { SkillBadge as SkillBadgeData, SkillName } from "@/types";

interface SkillBadgeProps {
  badge: SkillBadgeData;
}

const ICONS: Record<SkillName, LucideIcon> = {
  ingestion:               Database,
  biomarker:               Microscope,
  enrichment:              FlaskConical,
  visualization:           BarChart3,
  code_reviewer:           CheckCircle2,
  domain_expert:           Sparkles,
  query_data:              FileSearch,
  query_database:          FileSearch,
  load_preview_data:       FileSearch,
  complex_duckdb_query:    Database,
  simple_dataframe_query:  GitBranch,
};

export function SkillBadge({ badge }: SkillBadgeProps) {
  const Icon = ICONS[badge.name] ?? Sparkles;
  const running = badge.status === "running";
  const error   = badge.status === "error";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px]",
        "border transition-colors",
        running
          ? "border-accent/40 bg-accent/10 text-accent"
          : error
          ? "border-red-500/40 bg-red-500/10 text-red-500"
          : "border-border bg-surface-2 text-muted",
      )}
      title={`${badge.label} — ${badge.status}`}
    >
      {running
        ? <Loader2 className="h-3 w-3 animate-spin" />
        : error
        ? <XCircle className="h-3 w-3" />
        : <Icon className="h-3 w-3" />}
      <span className="font-medium">
        {running ? "Running: " : null}
        {badge.label}
      </span>
    </span>
  );
}
