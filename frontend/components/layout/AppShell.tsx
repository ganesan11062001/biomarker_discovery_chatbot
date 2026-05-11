"use client";

import { ArtifactPanel } from "@/components/artifacts/ArtifactPanel";
import { ChatInput }     from "@/components/chat/ChatInput";
import { MessageList }   from "@/components/chat/MessageList";
import { Sidebar }       from "@/components/layout/Sidebar";
import { TopBar }        from "@/components/layout/TopBar";
import { useChat }       from "@/hooks/useChat";
import { useTheme }      from "@/hooks/useTheme";

export function AppShell() {
  // Wire the theme to <html class="dark"> on mount/change
  useTheme();

  const { messages } = useChat();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      <Sidebar />

      <main className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <div className="flex min-h-0 flex-1 overflow-y-auto">
          <MessageList messages={messages} />
        </div>
        <ChatInput />
      </main>

      <ArtifactPanel />
    </div>
  );
}
