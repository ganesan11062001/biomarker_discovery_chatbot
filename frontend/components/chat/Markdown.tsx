"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";

import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";

interface MarkdownProps {
  content: string;
}

/**
 * Markdown renderer with:
 *  - GFM (tables, strikethrough, task lists)
 *  - Syntax-highlighted code blocks via Prism (theme-aware)
 *  - Copy button on every code block
 *  - Tailwind-styled tables, headings, lists, blockquotes
 */
export function Markdown({ content }: MarkdownProps) {
  const { theme } = useTheme();

  return (
    <div
      className={cn(
        "prose prose-sm max-w-none",
        "prose-headings:font-semibold prose-headings:text-foreground",
        "prose-p:text-foreground prose-strong:text-foreground",
        "prose-a:text-accent prose-a:no-underline hover:prose-a:underline",
        "prose-code:rounded prose-code:bg-surface-2 prose-code:px-1 " +
          "prose-code:py-0.5 prose-code:font-mono prose-code:text-[0.85em] " +
          "prose-code:text-accent prose-code:before:content-[''] " +
          "prose-code:after:content-['']",
        "prose-pre:p-0 prose-pre:bg-transparent",
        "prose-table:text-xs prose-th:text-foreground",
        "prose-blockquote:border-accent prose-blockquote:text-muted",
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: function CodeBlock({ className, children, ...props }) {
            const text = String(children).replace(/\n$/, "");
            const inline = !className?.startsWith("language-");
            if (inline) {
              return <code className={className} {...props}>{children}</code>;
            }
            const language = className!.replace("language-", "");
            return <CodeFence text={text} language={language} dark={theme === "dark"} />;
          },
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto rounded-md border border-border">
              <table className="w-full text-left text-xs">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border-b border-border bg-surface-2 px-3 py-1.5 font-medium">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border-b border-border px-3 py-1.5">{children}</td>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function CodeFence({
  text, language, dark,
}: { text: string; language: string; dark: boolean }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }).catch(() => {});
  };

  return (
    <div className="my-3 overflow-hidden rounded-md border border-border bg-surface-2">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5">
        <span className="text-[10px] font-medium uppercase tracking-wider text-muted">
          {language || "code"}
        </span>
        <button
          onClick={copy}
          className="inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5
                     text-[11px] text-muted transition-colors hover:bg-border hover:text-foreground"
          aria-label="Copy code"
        >
          {copied
            ? <><Check className="h-3 w-3" /> Copied</>
            : <><Copy  className="h-3 w-3" /> Copy</>}
        </button>
      </div>
      <SyntaxHighlighter
        language={language || "text"}
        style={dark ? oneDark : oneLight}
        customStyle={{
          margin: 0,
          padding: "0.75rem 0.9rem",
          background: "transparent",
          fontSize:   "0.78rem",
          lineHeight: "1.55",
        }}
        wrapLongLines
      >
        {text}
      </SyntaxHighlighter>
    </div>
  );
}
