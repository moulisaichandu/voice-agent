import type { Config } from "tailwindcss";

// Status palette values are fixed, not themed — see the dataviz skill's
// palette reference. Same four hex values in light and dark: all clear a 3:1
// contrast ratio against the dark surface, and warning/serious are
// deliberately sub-3:1 on light surfaces (mitigated by pairing every status
// color with an icon + label, never color alone — see components/ui/StatusBadge).
const config: Config = {
  darkMode: "media",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: "#2563eb",
          hover: "#1d4ed8",
        },
        status: {
          good: "#0ca30c",
          warning: "#fab219",
          serious: "#ec835a",
          critical: "#d03b3b",
        },
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
