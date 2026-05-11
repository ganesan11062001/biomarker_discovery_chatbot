"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size    = "sm" | "md" | "icon";

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?:    Size;
}

const variantClasses: Record<Variant, string> = {
  primary:
    "bg-accent text-white hover:bg-accent-hover active:bg-accent-hover " +
    "shadow-sm disabled:opacity-50 disabled:cursor-not-allowed",
  secondary:
    "bg-surface-2 text-foreground hover:bg-border " +
    "border border-border disabled:opacity-50",
  ghost:
    "text-foreground hover:bg-surface-2 disabled:opacity-50",
  danger:
    "bg-red-500 text-white hover:bg-red-600 disabled:opacity-50",
};

const sizeClasses: Record<Size, string> = {
  sm:   "h-8 px-3 text-xs rounded-md gap-1.5",
  md:   "h-9 px-4 text-sm rounded-lg gap-2",
  icon: "h-8 w-8 rounded-md inline-flex items-center justify-center",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  function Button({ className, variant = "secondary", size = "md", ...rest }, ref) {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center font-medium",
          "transition-colors focus:outline-none focus-visible:ring-2 " +
            "focus-visible:ring-accent/60 focus-visible:ring-offset-2 " +
            "focus-visible:ring-offset-background",
          variantClasses[variant],
          sizeClasses[size],
          className,
        )}
        {...rest}
      />
    );
  },
);
