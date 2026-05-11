/**
 * lib/artifacts.ts
 * Parse assistant markdown into structured artifacts (code, mermaid, table).
 *
 * Why: the backend returns a single markdown blob. The artifact panel
 * renders extracted blocks separately so the user can view, copy, or
 * download them. Backend-emitted artifacts (plots, structured tables)
 * arrive directly via SSE; this module only handles inline blocks.
 */

import type { Artifact, CodeArtifact, MermaidArtifact, TableArtifact } from "@/types";

const FENCE_RE = /```(\w+)?\n([\s\S]*?)```/g;

/**
 * Extract code / mermaid / pipe-table blocks from a markdown string.
 *
 * Each artifact gets a stable id derived from `baseId` so re-running the
 * extractor on the same message produces idempotent ids (important when
 * the assistant message is updated by streaming token deltas).
 */
export function extractArtifacts(
  baseId: string,
  markdown: string,
  createdAt: string,
): Artifact[] {
  if (!markdown) return [];
  const out: Artifact[] = [];
  let m: RegExpExecArray | null;
  let idx = 0;

  // ── Fenced code blocks (``` … ```) ──────────────────────────────────────────
  FENCE_RE.lastIndex = 0;
  while ((m = FENCE_RE.exec(markdown)) !== null) {
    const language = (m[1] || "text").toLowerCase();
    const code     = m[2].trimEnd();
    if (!code) continue;
    if (language === "mermaid") {
      const artifact: MermaidArtifact = {
        id:        `${baseId}-mermaid-${idx}`,
        kind:      "mermaid",
        source:    code,
        createdAt,
      };
      out.push(artifact);
    } else {
      // Show short snippets inline; only treat 3+ line / 200+ char blocks
      // as artifacts to avoid spamming the panel with trivial backticks.
      const longEnough = code.length >= 200 || code.split("\n").length >= 3;
      if (longEnough) {
        const artifact: CodeArtifact = {
          id:        `${baseId}-code-${idx}`,
          kind:      "code",
          language,
          code,
          title:     codeTitle(language, code),
          createdAt,
        };
        out.push(artifact);
      }
    }
    idx += 1;
  }

  // ── Pipe tables (| col | col | … |) ─────────────────────────────────────────
  for (const block of findPipeTables(markdown)) {
    const table = parsePipeTable(block.text);
    if (table && table.rows.length > 0) {
      const artifact: TableArtifact = {
        id:        `${baseId}-table-${idx}`,
        kind:      "table",
        columns:   table.columns,
        rows:      table.rows,
        title:     "Table",
        createdAt,
      };
      out.push(artifact);
      idx += 1;
    }
  }

  return out;
}

function codeTitle(language: string, code: string): string {
  const firstLine = code.split("\n", 1)[0].trim();
  // Use a short first-line summary if it's a comment / shebang
  if (firstLine.startsWith("#") || firstLine.startsWith("//") ||
      firstLine.startsWith("--") || firstLine.startsWith("/*")) {
    return firstLine.replace(/^[#/*\- ]+/, "").slice(0, 60);
  }
  return `${language.toUpperCase()} snippet`;
}

interface PipeTableBlock {
  text:  string;
  start: number;
  end:   number;
}

/** Locate contiguous pipe-table blocks in a markdown string. */
function findPipeTables(markdown: string): PipeTableBlock[] {
  const lines = markdown.split("\n");
  const blocks: PipeTableBlock[] = [];
  let i = 0;
  while (i < lines.length) {
    const header  = lines[i];
    const divider = lines[i + 1] ?? "";
    if (
      header.includes("|") &&
      /^\s*\|?[-: ]+\|[-: |]+\|?\s*$/.test(divider)
    ) {
      // Found a header + divider pair — collect rows until non-table line
      let j = i + 2;
      while (j < lines.length && lines[j].includes("|")) j += 1;
      const chunk = lines.slice(i, j).join("\n");
      blocks.push({ text: chunk, start: i, end: j });
      i = j;
    } else {
      i += 1;
    }
  }
  return blocks;
}

function parsePipeTable(text: string): { columns: string[];
                                          rows: Array<Record<string, string>> } | null {
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length < 2) return null;
  const splitRow = (row: string): string[] =>
    row
      .replace(/^\s*\|/, "")
      .replace(/\|\s*$/, "")
      .split("|")
      .map((c) => c.trim());

  const columns = splitRow(lines[0]);
  // Skip the divider row (lines[1])
  const rows = lines.slice(2).map((line) => {
    const cells = splitRow(line);
    const row: Record<string, string> = {};
    columns.forEach((col, i) => { row[col] = cells[i] ?? ""; });
    return row;
  });
  return { columns, rows };
}
