"use client";

import { Check, Copy, Download } from "lucide-react";
import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";

import { Button } from "@/components/ui/Button";
import { useTheme } from "@/hooks/useTheme";
import type { CodeArtifact } from "@/types";

const EXT_BY_LANG: Record<string, string> = {
  python: "py", sql: "sql", javascript: "js", typescript: "ts",
  r: "R", json: "json", yaml: "yaml", bash: "sh", shell: "sh",
};

export function CodeArtifactView({ artifact }: { artifact: CodeArtifact }) {
  const { theme } = useTheme();
  const [copied, setCopied] = useState(false);

  const copy = () =>
    navigator.clipboard.writeText(artifact.code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }).catch(() => {});

  const download = () => {
    const ext  = EXT_BY_LANG[artifact.language] || "txt";
    const blob = new Blob([artifact.code], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = `${artifact.id}.${ext}`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-border bg-surface-2">
      <div className="flex items-center justify-between border-b border-border bg-surface px-3 py-1.5">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium uppercase tracking-wider text-muted">
            {artifact.language}
          </span>
          {artifact.title && (
            <span className="text-xs text-foreground">{artifact.title}</span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={copy}>
            {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? "Copied" : "Copy"}
          </Button>
          <Button variant="ghost" size="sm" onClick={download}>
            <Download className="h-3.5 w-3.5" />
            Download
          </Button>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <SyntaxHighlighter
          language={artifact.language}
          style={theme === "dark" ? oneDark : oneLight}
          customStyle={{
            margin: 0,
            padding: "0.75rem 1rem",
            background: "transparent",
            fontSize:  "0.78rem",
            lineHeight: "1.55",
          }}
          wrapLongLines
        >
          {artifact.code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
