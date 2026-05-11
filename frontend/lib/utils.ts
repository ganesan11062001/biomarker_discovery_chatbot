import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combine class names with Tailwind-aware conflict resolution.
 *   cn("p-2", isOpen && "p-4")  →  "p-4"   (last wins, conflicting tailwind utilities collapse)
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Format a byte count as a human-readable string. */
export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(units.length - 1,
                     Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(value >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

/** Truncate a string to `n` characters with a trailing ellipsis. */
export function truncate(s: string, n: number): string {
  if (!s) return "";
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

/** Relative time formatter for session list. */
export function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  const diff = Date.now() - then;
  if (Number.isNaN(diff))      return "";
  if (diff < 60_000)           return "just now";
  if (diff < 3_600_000)        return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000)       return `${Math.floor(diff / 3_600_000)}h ago`;
  if (diff < 7 * 86_400_000)   return `${Math.floor(diff / 86_400_000)}d ago`;
  return new Date(iso).toLocaleDateString();
}
