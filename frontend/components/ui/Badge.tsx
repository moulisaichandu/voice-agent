// Plain categorical pill — campaign mode, active/inactive, and similar labels
// that are NOT part of the dial-readiness/call-outcome status system (that's
// StatusBadge, which carries the fixed, validated status palette + icons).
// Uses ordinary Tailwind tint/ink pairs, which are already contrast-safe.

type Tone = "neutral" | "accent" | "good" | "critical";

const TONE_CLASSES: Record<Tone, string> = {
  neutral: "bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300",
  accent: "bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  good: "bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300",
  critical: "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300",
};

export function Badge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: Tone;
}) {
  return (
    <span
      className={`inline-flex items-center whitespace-nowrap rounded-full px-2.5 py-0.5
        text-xs font-medium ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  );
}
