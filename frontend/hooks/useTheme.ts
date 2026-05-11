/**
 * hooks/useTheme.ts
 * Syncs the persisted `theme` value to <html class="dark"|""> so Tailwind's
 * darkMode: "class" picks it up. Also reflects the system theme on first load
 * when the user has never explicitly toggled.
 */
"use client";

import { useEffect } from "react";

import { useAppStore } from "@/lib/store";
import type { Theme } from "@/types";

export function useTheme(): {
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
} {
  const theme       = useAppStore((s) => s.theme);
  const setTheme    = useAppStore((s) => s.setTheme);
  const toggleTheme = useAppStore((s) => s.toggleTheme);

  useEffect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    root.classList.toggle("dark", theme === "dark");
  }, [theme]);

  return { theme, setTheme, toggleTheme };
}
