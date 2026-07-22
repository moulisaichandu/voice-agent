import type { ComponentType } from "react";
import {
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  MinusCircleIcon,
  PulseDot,
  XCircleIcon,
  type IconProps,
} from "./icons";

export type Tone = "good" | "warning" | "serious" | "critical" | "accent" | "muted";

// Solid-fill chip + a light/dark glyph, never the raw status hue as text or a
// bare small icon — warning (#fab219) and serious (#ec835a) fall below 3:1
// contrast as text/graphical-object color on a light surface (see the dataviz
// skill's palette notes). A dark-on-bright-fill chip sidesteps that: the
// relevant contrast pair becomes glyph-vs-fill, not glyph-vs-page, and that
// pair is high for every tone here.
const CHIP_TONE: Record<Exclude<Tone, "muted">, { bg: string; fg: string }> = {
  good: { bg: "bg-status-good", fg: "text-white" },
  warning: { bg: "bg-status-warning", fg: "text-neutral-900" },
  serious: { bg: "bg-status-serious", fg: "text-neutral-900" },
  critical: { bg: "bg-status-critical", fg: "text-white" },
  accent: { bg: "bg-accent", fg: "text-white" },
};

// Sensible default per tone, for callers (the readiness strip) that don't
// need a domain-specific glyph the way lead/call status do — just "this is
// the good/warning/critical one".
const DEFAULT_ICON: Record<Exclude<Tone, "muted" | "accent">, ComponentType<IconProps>> = {
  good: CheckCircleIcon,
  warning: ExclamationTriangleIcon,
  serious: ExclamationTriangleIcon,
  critical: XCircleIcon,
};

type StatusBadgeProps = {
  tone: Tone;
  label: string;
  icon?: ComponentType<IconProps>;
  /** Render a live pulse dot instead of a static icon — for "in progress". */
  pulse?: boolean;
};

/** The general-purpose primitive: any tone + label + icon. Used directly by
 * the readiness strip; LeadStatusBadge/CallStatusBadge below are thin,
 * opinionated wrappers over it for the two status vocabularies already in
 * the schema. Status is always icon + label together — CLAUDE.md's own
 * "never color alone" reasoning applies here as much as it does to a chart. */
export function StatusBadge({ tone, label, icon, pulse = false }: StatusBadgeProps) {
  const Icon = icon ?? (tone !== "muted" && tone !== "accent" ? DEFAULT_ICON[tone] : undefined);
  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-neutral-700 dark:text-neutral-300">
      {pulse ? (
        <PulseDot />
      ) : tone === "muted" ? (
        Icon && <Icon className="h-4 w-4 shrink-0 text-neutral-400 dark:text-neutral-500" />
      ) : (
        <span
          className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full ${CHIP_TONE[tone].bg}`}
        >
          {Icon && <Icon className={`h-2.5 w-2.5 ${CHIP_TONE[tone].fg}`} />}
        </span>
      )}
      <span>{label}</span>
    </span>
  );
}

// ── Lead status (app/db/models.py's LeadStatus) ───────────────────────────────

const LEAD_STATUS_META: Record<
  string,
  { label: string; tone: Tone; icon?: ComponentType<IconProps>; pulse?: boolean }
> = {
  pending: { label: "Pending", tone: "muted", icon: ClockIcon },
  queued: { label: "Queued", tone: "muted", icon: ClockIcon },
  calling: { label: "Calling", tone: "accent", pulse: true },
  done: { label: "Done", tone: "good", icon: CheckCircleIcon },
  failed: { label: "Failed", tone: "critical", icon: XCircleIcon },
  dnd: { label: "Do not call", tone: "muted", icon: MinusCircleIcon },
};

export function LeadStatusBadge({ status }: { status: string }) {
  const meta = LEAD_STATUS_META[status] ?? { label: status, tone: "muted" as const };
  return <StatusBadge tone={meta.tone} label={meta.label} icon={meta.icon} pulse={meta.pulse} />;
}

// ── Call status ────────────────────────────────────────────────────────────
// migrations/0001_init.sql's calls.status is unconstrained text (see
// db/models.py's CallStatus comment on why) — this covers the values this
// codebase actually writes, and falls back to a plain muted label for
// anything else, e.g. an ElevenLabs value never verified against here.

const CALL_STATUS_META: Record<string, { label: string; tone: Tone; icon?: ComponentType<IconProps> }> = {
  done: { label: "Done", tone: "good", icon: CheckCircleIcon },
  failed: { label: "Failed", tone: "critical", icon: XCircleIcon },
  answered: { label: "Answered", tone: "good", icon: CheckCircleIcon },
  no_answer: { label: "No answer", tone: "warning", icon: ExclamationTriangleIcon },
  unknown: { label: "Unknown", tone: "muted" },
};

export function CallStatusBadge({ status }: { status: string | null }) {
  if (!status) return <span className="text-sm text-neutral-400 dark:text-neutral-500">—</span>;
  const meta = CALL_STATUS_META[status] ?? { label: status, tone: "muted" as const };
  return <StatusBadge tone={meta.tone} label={meta.label} icon={meta.icon} />;
}
