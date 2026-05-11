"use client";

import { useEffect, useRef } from "react";

import { MessageBubble } from "@/components/chat/MessageBubble";
import type { ChatMessage } from "@/types";

interface MessageListProps {
  messages: ChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Autoscroll to bottom on every message change unless the user has
  // scrolled up. We approximate "user is reading" by checking distance
  // from bottom; only autoscroll when within 200px of the end.
  useEffect(() => {
    const node = bottomRef.current;
    if (!node) return;
    const parent = node.parentElement;
    if (!parent) return;
    const distance = parent.scrollHeight - parent.scrollTop - parent.clientHeight;
    if (distance < 200) {
      node.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <Welcome />
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-3xl space-y-5 px-4 py-6">
      {messages.map((m) => <MessageBubble key={m.id} message={m} />)}
      <div ref={bottomRef} className="h-px" />
    </div>
  );
}

function Welcome() {
  const suggestions = [
    {
      title: "Compare two groups",
      body:  "Run differential expression analysis between Disease and Control samples.",
    },
    {
      title: "Pathway enrichment",
      body:  "Find enriched KEGG, GO, Reactome pathways on the top biomarkers.",
    },
    {
      title: "Top biomarkers by SpC",
      body:  "Show the highest-spectral-count proteins for a given sample.",
    },
    {
      title: "Inspect a sample",
      body:  "What strain and treatment is each MaxQuant sample mapped to?",
    },
  ];
  return (
    <div className="w-full max-w-2xl text-center">
      <div className="mx-auto mb-3 inline-flex items-center justify-center rounded-xl
                      bg-accent/10 px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-accent">
        Multi-agent · LangGraph · DuckDB
      </div>
      <h2 className="mb-3 text-2xl font-semibold tracking-tight">
        Biomarker Discovery, reimagined
      </h2>
      <p className="mb-8 text-sm text-muted">
        Upload your proteomics workbook and ask anything — differential expression,
        pathway enrichment, sample lookups, interactive plots.
      </p>
      <div className="grid grid-cols-1 gap-2 text-left sm:grid-cols-2">
        {suggestions.map((s) => (
          <div key={s.title}
               className="rounded-lg border border-border bg-surface p-3 transition-colors hover:border-accent">
            <div className="mb-1 text-sm font-medium">{s.title}</div>
            <div className="text-xs text-muted">{s.body}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
