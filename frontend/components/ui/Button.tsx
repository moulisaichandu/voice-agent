import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "danger" | "ghost";

const VARIANT_CLASSES: Record<Variant, string> = {
  primary: "bg-accent text-white hover:bg-accent-hover disabled:hover:bg-accent",
  secondary:
    "border border-neutral-300 text-neutral-700 hover:bg-neutral-50 dark:border-neutral-700 dark:text-neutral-200 dark:hover:bg-neutral-900",
  danger: "bg-status-critical text-white hover:bg-red-700 disabled:hover:bg-status-critical",
  ghost:
    "text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-900",
};

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
};

export function Button({ variant = "primary", className = "", ...props }: Props) {
  return (
    <button
      className={`inline-flex items-center gap-1.5 rounded-md px-3.5 py-2 text-sm font-medium
        transition-colors disabled:cursor-not-allowed disabled:opacity-50
        ${VARIANT_CLASSES[variant]} ${className}`}
      {...props}
    />
  );
}
