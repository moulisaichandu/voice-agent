import type { ReactNode } from "react";

type Props = {
  label: string;
  hint?: string;
  error?: string;
  children: ReactNode;
  /** Lay the control out beside the label instead of below it — for a lone checkbox. */
  inline?: boolean;
};

const CONTROL_CLASS =
  "rounded-md border border-neutral-300 bg-white px-3 py-2 text-sm text-neutral-900 " +
  "placeholder:text-neutral-400 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent " +
  "dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-100";

/** Labeled wrapper around a form control. The control (input/select/textarea)
 * is passed as children so each call site keeps full control over its type
 * and behavior; this only standardizes the label/spacing/error chrome. */
export function Field({ label, hint, error, children, inline = false }: Props) {
  if (inline) {
    return (
      <label className="flex items-center gap-2 text-sm text-neutral-700 dark:text-neutral-300">
        {children}
        <span>{label}</span>
        {hint && <span className="text-neutral-500 dark:text-neutral-400">— {hint}</span>}
      </label>
    );
  }
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">{label}</span>
      {children}
      {hint && <span className="text-xs text-neutral-500 dark:text-neutral-400">{hint}</span>}
      {error && <span className="text-xs text-status-critical">{error}</span>}
    </label>
  );
}

export const fieldControlClass = CONTROL_CLASS;
