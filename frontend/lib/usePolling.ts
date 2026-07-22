"use client";

import { useEffect, useRef } from "react";

/** Calls `fn` every `intervalMs` while `enabled`, paused whenever the tab is
 * hidden — no point re-fetching a page nobody's looking at. Does NOT call fn
 * immediately on mount; the caller's own initial-load effect handles that,
 * this only owns the recurring part. `fn` is read from a ref so callers don't
 * need to memoize it themselves — passing a fresh closure each render is
 * fine and won't restart the interval. */
export function usePolling(fn: () => void, intervalMs: number, enabled: boolean) {
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!enabled) return;

    const id = setInterval(() => {
      if (document.visibilityState === "visible") {
        fnRef.current();
      }
    }, intervalMs);

    return () => clearInterval(id);
  }, [intervalMs, enabled]);
}
