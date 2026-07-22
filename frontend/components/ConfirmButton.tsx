"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Button } from "./ui/Button";

type Props = {
  onConfirm: () => void;
  children: ReactNode;
  /** What clicking through actually does — shown only in the confirm step,
   * e.g. "This will dial up to 10 dial-eligible leads from this campaign." */
  consequence: string;
  confirmLabel?: string;
  variant?: "primary" | "danger";
  disabled?: boolean;
};

/** Two-step "are you sure" for actions that place real phone calls or
 * permanently opt a lead out — cheap to build, expensive to skip. A placed
 * call or a do-not-call flag can't be undone by clicking again. */
export function ConfirmButton({
  onConfirm,
  children,
  consequence,
  confirmLabel = "Yes, do it",
  variant = "primary",
  disabled = false,
}: Props) {
  const [confirming, setConfirming] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  function startConfirming() {
    setConfirming(true);
    // Auto-cancel after a while, so a stray later click — after the operator
    // has walked away and forgotten this was mid-confirmation — can't fire it.
    timeoutRef.current = setTimeout(() => setConfirming(false), 8000);
  }

  function cancel() {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setConfirming(false);
  }

  function confirm() {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setConfirming(false);
    onConfirm();
  }

  if (!confirming) {
    return (
      <Button variant={variant} disabled={disabled} onClick={startConfirming}>
        {children}
      </Button>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-md border border-status-warning/40 bg-status-warning/10 p-3">
      <span className="text-sm text-neutral-700 dark:text-neutral-300">{consequence}</span>
      <div className="flex gap-2">
        <Button variant={variant} onClick={confirm}>
          {confirmLabel}
        </Button>
        <Button variant="ghost" onClick={cancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
