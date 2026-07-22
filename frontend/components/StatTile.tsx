import type { ReactNode } from "react";

type Props = {
  label: string;
  value: number | string;
  hint?: string;
  /** Reserved for once there's enough history to plot — a single day of real
   * calls doesn't earn a chart yet (see the dataviz skill's form heuristic:
   * a single current value is a stat tile, not a chart). */
  sparkline?: ReactNode;
};

/** The number always renders in plain ink, never a status color — per the
 * "text wears text tokens" rule, color conveys identity on a MARK beside the
 * text, not by tinting the text itself. */
export function StatTile({ label, value, hint, sparkline }: Props) {
  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900/40">
      <div className="text-xs font-medium uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
        {label}
      </div>
      <div className="mt-1 text-3xl font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
        {value}
      </div>
      {hint && (
        <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">{hint}</div>
      )}
      {sparkline}
    </div>
  );
}
