import type { Config } from "tailwindcss";

/**
 * Solid Biosciences biomarker chatbot — Tailwind theme.
 *
 * Palette:
 *   navy   — deep institutional blue (primary surface in dark mode)
 *   teal   — biotech / scientific accent (active states, links, focus)
 *   slate  — neutral text and borders (light mode)
 *
 * Dark mode is class-based: <html class="dark"> applies the dark variants.
 */
const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Surface tokens resolved by CSS variables in globals.css
        background:   "rgb(var(--background) / <alpha-value>)",
        foreground:   "rgb(var(--foreground) / <alpha-value>)",
        surface:      "rgb(var(--surface) / <alpha-value>)",
        "surface-2":  "rgb(var(--surface-2) / <alpha-value>)",
        border:       "rgb(var(--border) / <alpha-value>)",
        muted:        "rgb(var(--muted) / <alpha-value>)",
        accent:       "rgb(var(--accent) / <alpha-value>)",
        "accent-hover":"rgb(var(--accent-hover) / <alpha-value>)",

        // Brand scales — useful for fixed-tone elements (logos, branded chips)
        navy: {
          50:  "#f0f5fb",
          100: "#dbe6f1",
          200: "#b5cce3",
          300: "#85a8cc",
          400: "#5283b1",
          500: "#34669a",
          600: "#264f7e",
          700: "#1d3d63",
          800: "#152c48",
          900: "#0d1e33",
          950: "#06121f",
        },
        teal: {
          50:  "#effcfb",
          100: "#cdf6f3",
          200: "#9beceb",
          300: "#5fd9d8",
          400: "#2ebec1",
          500: "#1ba0a8",
          600: "#147f88",
          700: "#13656d",
          800: "#155159",
          900: "#15434a",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        "panel": "0 1px 2px rgb(0 0 0 / 0.04), 0 1px 3px rgb(0 0 0 / 0.08)",
        "panel-dark": "0 1px 2px rgb(0 0 0 / 0.4), 0 1px 3px rgb(0 0 0 / 0.6)",
      },
      animation: {
        "fade-in": "fadeIn 0.15s ease-out",
        "slide-up": "slideUp 0.2s ease-out",
        "pulse-dot": "pulseDot 1.4s ease-in-out infinite",
      },
      keyframes: {
        fadeIn:   { from: { opacity: "0" }, to: { opacity: "1" } },
        slideUp:  { from: { transform: "translateY(6px)", opacity: "0" },
                    to:   { transform: "translateY(0)",    opacity: "1" } },
        pulseDot: { "0%, 80%, 100%": { opacity: "0.2" },
                    "40%":           { opacity: "1"   } },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
export default config;
