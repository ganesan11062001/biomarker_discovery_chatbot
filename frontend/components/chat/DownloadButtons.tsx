"use client";

import { Download, FileSpreadsheet, FileText, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import {
  excelDownloadUrl,
  fetchAnalysisState,
  type AnalysisStateSnapshot,
} from "@/lib/api";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";

/**
 * Two download actions for the active session:
 *   1. Excel report — direct link to the backend's /results/:id/excel route.
 *      Browser handles the download via Content-Disposition: attachment.
 *   2. Summary markdown — fetches state, builds a markdown report client-side,
 *      and triggers a Blob download. No backend endpoint needed.
 *
 * Buttons are disabled with explanatory tooltips when no results exist yet
 * (no excel_path / no analysis_summary). State is refreshed on session
 * change and whenever the user clicks a button so the disabled state is
 * never stale.
 */
export function DownloadButtons() {
  const sessionId = useAppStore((s) => s.activeSessionId);
  const [snapshot, setSnapshot] = useState<AnalysisStateSnapshot | null>(null);
  const [downloadingSummary, setDownloadingSummary] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  // Re-fetch availability when the active session changes.
  useEffect(() => {
    if (!sessionId) {
      setSnapshot(null);
      return;
    }
    let cancelled = false;
    fetchAnalysisState(sessionId)
      .then((s) => { if (!cancelled) setSnapshot(s); })
      .catch(() => { if (!cancelled) setSnapshot(null); });
    return () => { cancelled = true; };
  }, [sessionId]);

  if (!sessionId) return null;

  const hasExcel   = !!snapshot?.excel_path;
  const hasSummary = !!snapshot?.analysis_summary || !!snapshot?.top_biomarkers?.length;

  const handleExcelClick = (e: React.MouseEvent) => {
    if (!hasExcel) {
      e.preventDefault();
      setError("No Excel report yet — run an analysis first.");
      window.setTimeout(() => setError(null), 3000);
    }
  };

  const handleSummaryClick = async () => {
    setError(null);
    setDownloadingSummary(true);
    try {
      // Always re-fetch on click so the summary reflects the latest analysis.
      const fresh = await fetchAnalysisState(sessionId);
      setSnapshot(fresh);
      const md = buildReportMarkdown(fresh);
      if (!md.trim()) {
        setError("No analysis summary yet — run an analysis first.");
        window.setTimeout(() => setError(null), 3000);
        return;
      }
      triggerBlobDownload(
        md,
        `biomarker_report_${sessionId.slice(0, 8)}.md`,
        "text/markdown",
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed.");
      window.setTimeout(() => setError(null), 4000);
    } finally {
      setDownloadingSummary(false);
    }
  };

  const baseBtn =
    "inline-flex h-8 items-center gap-1.5 rounded-md px-2.5 text-xs font-medium " +
    "transition-colors focus:outline-none focus-visible:ring-2 " +
    "focus-visible:ring-accent/60 focus-visible:ring-offset-2 " +
    "focus-visible:ring-offset-background";
  const enabledBtn  = "bg-surface-2 text-foreground hover:bg-border border border-border";
  const disabledBtn = "bg-surface-2/40 text-muted border border-border/60 cursor-not-allowed";

  return (
    <div className="relative flex items-center gap-1">
      {/* Excel — anchor that the backend serves as an attachment. */}
      <a
        href={hasExcel ? excelDownloadUrl(sessionId) : "#"}
        onClick={handleExcelClick}
        aria-disabled={!hasExcel}
        title={hasExcel
          ? "Download formatted Excel biomarker report"
          : "Run an analysis first"}
        className={cn(baseBtn, hasExcel ? enabledBtn : disabledBtn)}
      >
        <FileSpreadsheet className="h-3.5 w-3.5" />
        <span className="hidden sm:inline">Excel</span>
      </a>

      {/* Summary — built client-side from the state JSON. */}
      <button
        type="button"
        onClick={handleSummaryClick}
        disabled={!hasSummary || downloadingSummary}
        title={hasSummary
          ? "Download analysis summary as Markdown"
          : "Run an analysis first"}
        className={cn(baseBtn, (!hasSummary || downloadingSummary) ? disabledBtn : enabledBtn)}
      >
        {downloadingSummary
          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
          : <FileText className="h-3.5 w-3.5" />}
        <span className="hidden sm:inline">Summary</span>
      </button>

      {error && (
        <span className="absolute right-0 top-9 z-10 whitespace-nowrap rounded-md
                         border border-border bg-surface px-2 py-1 text-[11px] text-muted shadow-panel">
          {error}
        </span>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function triggerBlobDownload(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Build a markdown report from the analysis state. Includes dataset stats,
 * the comparison, QC summary, top-10 biomarkers table, the LLM analysis
 * summary, and (if available) the top enriched pathways.
 */
function buildReportMarkdown(s: AnalysisStateSnapshot): string {
  const lines: string[] = [];
  const now = new Date().toISOString().replace("T", " ").slice(0, 19);

  lines.push("# Biomarker Discovery Report");
  lines.push("");
  lines.push(`**Session:** \`${s.session_id}\`  `);
  lines.push(`**Generated:** ${now} UTC  `);
  if (s.status) lines.push(`**Status:** ${s.status}`);
  lines.push("");

  // Dataset
  if (s.n_proteins || s.n_samples || s.data_type) {
    lines.push("## Dataset");
    if (s.n_proteins  != null) lines.push(`- Proteins: **${s.n_proteins}**`);
    if (s.n_samples   != null) lines.push(`- Samples: **${s.n_samples}**`);
    if (s.data_type)           lines.push(`- Data type: \`${s.data_type}\``);
    if (s.omic_type)           lines.push(`- Omic type: \`${s.omic_type}\``);
    lines.push("");
  }

  // Comparison
  if (s.group1_label || s.group2_label) {
    lines.push("## Comparison");
    const n1 = s.group1_samples?.length ?? 0;
    const n2 = s.group2_samples?.length ?? 0;
    lines.push(
      `**${s.group1_label ?? "Group 1"}** (n=${n1}) vs ` +
      `**${s.group2_label ?? "Group 2"}** (n=${n2})  `,
    );
    if (s.analysis_mode) lines.push(`Mode: ${s.analysis_mode}`);
    lines.push("");
  }

  // QC
  const qc = s.qc_summary || {};
  if (Object.keys(qc).length > 0) {
    lines.push("## QC Summary");
    for (const [k, v] of Object.entries(qc)) {
      if (v == null || typeof v === "object") continue;
      lines.push(`- ${k}: ${String(v)}`);
    }
    lines.push("");
  }

  // Significance count
  if (s.n_significant != null) {
    lines.push(`## Significant biomarkers: **${s.n_significant}**`);
    lines.push("");
  }

  // Top biomarkers
  const top = (s.top_biomarkers || []).slice(0, 10);
  if (top.length > 0) {
    lines.push("## Top biomarkers");
    lines.push("");
    lines.push("| Rank | Protein | log2FC | adj p-value | Significance |");
    lines.push("|------|---------|--------|-------------|--------------|");
    top.forEach((b, i) => {
      const rank = b.rank ?? i + 1;
      const protein = b.protein ?? "?";
      const lfc =
        b.log2_fold_change ?? b.log2_ratio ?? b.rescue_score ?? "—";
      const adjp = b.adj_p_value ?? "—";
      const sig  = b.significance ?? "—";
      lines.push(`| ${rank} | ${protein} | ${fmt(lfc)} | ${fmt(adjp)} | ${sig} |`);
    });
    lines.push("");
  }

  // Analysis summary text
  if (s.analysis_summary && s.analysis_summary.trim()) {
    lines.push("## Analysis summary");
    lines.push("");
    lines.push(s.analysis_summary.trim());
    lines.push("");
  }

  // Enrichment (top 10 pathways)
  const pathways = (s.pathways || []).slice(0, 10);
  if (pathways.length > 0) {
    lines.push("## Top enriched pathways");
    lines.push("");
    lines.push("| Pathway | Library | adj p-value | Genes |");
    lines.push("|---------|---------|-------------|-------|");
    pathways.forEach((p) => {
      const name    = p.pathway ?? "?";
      const lib     = p.library ?? "—";
      const adjp    = p.p_adjust ?? p.adj_p ?? "—";
      const ngenes  = p.gene_count ?? p.overlap ?? "—";
      lines.push(`| ${name} | ${lib} | ${fmt(adjp)} | ${ngenes} |`);
    });
    lines.push("");
  }

  return lines.join("\n");
}

function fmt(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") {
    if (Math.abs(v) > 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4)) {
      return v.toExponential(2);
    }
    return Number.isInteger(v) ? String(v) : v.toFixed(3);
  }
  return String(v);
}
