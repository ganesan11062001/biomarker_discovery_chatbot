"use client";

import { Download } from "lucide-react";

import { Button } from "@/components/ui/Button";
import type { TableArtifact } from "@/types";

export function TableArtifactView({ artifact }: { artifact: TableArtifact }) {
  const downloadCsv = () => {
    const escape = (v: unknown) => {
      const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = [
      artifact.columns.map(escape).join(","),
      ...artifact.rows.map((r) =>
        artifact.columns.map((c) => escape(r[c])).join(",")),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = `${artifact.id}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5">
        <span className="text-xs font-medium">
          {artifact.title || "Table"}
          <span className="ml-2 text-[10px] text-muted">
            {artifact.rows.length} rows × {artifact.columns.length} cols
          </span>
        </span>
        <Button variant="ghost" size="sm" onClick={downloadCsv}>
          <Download className="h-3.5 w-3.5" /> CSV
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-surface-2">
            <tr>
              {artifact.columns.map((c) => (
                <th key={c} className="border-b border-border px-3 py-1.5 text-left font-medium">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {artifact.rows.map((row, i) => (
              <tr key={i} className="even:bg-surface-2/50">
                {artifact.columns.map((c) => (
                  <td key={c} className="border-b border-border px-3 py-1">
                    {String(row[c] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
