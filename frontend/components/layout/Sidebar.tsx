"use client";

import { useEffect, useRef } from "react";
import { ChevronLeft, FlaskConical, MessageSquarePlus } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { createSession, listSessions } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import { cn, relativeTime, truncate } from "@/lib/utils";

export function Sidebar() {
  const sessions       = useAppStore((s) => s.sessions);
  const setSessions    = useAppStore((s) => s.setSessions);
  const upsertSession  = useAppStore((s) => s.upsertSession);
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const setActiveSession = useAppStore((s) => s.setActiveSession);
  const sidebarOpen    = useAppStore((s) => s.sidebarOpen);
  const setSidebarOpen = useAppStore((s) => s.setSidebarOpen);

  // Bootstrap: list existing sessions + auto-create one if none exists.
  // Guarded with a ref so this only runs once across renders / StrictMode
  // double-mounts, eliminating any possibility of re-entry loops.
  const bootstrappedRef = useRef(false);
  useEffect(() => {
    if (bootstrappedRef.current) return;
    bootstrappedRef.current = true;

    void (async () => {
      try {
        const existing = await listSessions();
        setSessions(existing);
        // If there's already an active session OR existing sessions on the
        // server, use them and skip auto-creating a new one.
        if (useAppStore.getState().activeSessionId) return;
        if (existing.length > 0) {
          setActiveSession(existing[0].id);
          return;
        }
        // No history — start a new conversation
        const created = await createSession();
        const sid = created.session_id;
        setActiveSession(sid);
        upsertSession({
          id: sid,
          title: "New conversation",
          createdAt: new Date().toISOString(),
          lastActiveAt: new Date().toISOString(),
          messageCount: 0,
        });
      } catch {
        // Backend unavailable — sidebar shows empty state
      }
    })();
  }, [setSessions, setActiveSession, upsertSession]);

  const handleNewChat = async () => {
    try {
      const r = await createSession();
      const sid = r.session_id;
      setActiveSession(sid);
      upsertSession({
        id: sid, title: "New conversation",
        createdAt: new Date().toISOString(),
        lastActiveAt: new Date().toISOString(),
        messageCount: 0,
      });
    } catch { /* swallow */ }
  };

  if (!sidebarOpen) {
    return (
      <aside className="flex w-12 shrink-0 flex-col items-center border-r border-border bg-surface py-3">
        <Button
          variant="ghost" size="icon"
          aria-label="Open sidebar"
          onClick={() => setSidebarOpen(true)}
        >
          <FlaskConical className="h-5 w-5 text-accent" />
        </Button>
      </aside>
    );
  }

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-border bg-surface">
      {/* ── Brand ─────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-5 w-5 text-accent" />
          <div className="flex flex-col leading-tight">
            <span className="text-sm font-semibold">BiomarkerAI</span>
            <span className="text-[10px] uppercase tracking-wider text-muted">
              Solid Biosciences
            </span>
          </div>
        </div>
        <Button
          variant="ghost" size="icon"
          aria-label="Collapse sidebar"
          onClick={() => setSidebarOpen(false)}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      </div>

      {/* ── New chat ──────────────────────────────────────────────────── */}
      <div className="p-3">
        <Button
          variant="primary"
          className="w-full"
          onClick={handleNewChat}
        >
          <MessageSquarePlus className="h-4 w-4" />
          New conversation
        </Button>
      </div>

      {/* ── Session list ──────────────────────────────────────────────── */}
      <nav className="flex-1 overflow-y-auto px-2 pb-3">
        {sessions.length === 0 ? (
          <p className="px-3 py-6 text-center text-xs text-muted">
            No previous conversations yet.
          </p>
        ) : (
          <ul className="space-y-0.5">
            {sessions.map((s) => (
              <li key={s.id}>
                <button
                  onClick={() => setActiveSession(s.id)}
                  className={cn(
                    "block w-full rounded-md px-3 py-2 text-left text-sm transition-colors",
                    s.id === activeSessionId
                      ? "bg-accent/10 text-accent"
                      : "text-foreground hover:bg-surface-2",
                  )}
                >
                  <div className="truncate">{truncate(s.title, 38)}</div>
                  <div className="mt-0.5 text-[10px] text-muted">
                    {relativeTime(s.lastActiveAt)} · {s.messageCount} msg
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </nav>
    </aside>
  );
}
