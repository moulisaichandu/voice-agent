/** ISO timestamp -> "2:14 PM" if today, else "Jul 20, 2:14 PM". Never throws
 * on a null/invalid value — timestamps from the API are frequently null
 * (a call that hasn't ended, a lead never called) and this is used directly
 * in JSX without every call site guarding it. */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";

  const now = new Date();
  const isToday =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();

  const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (isToday) return time;
  const date = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return `${date}, ${time}`;
}

/** ISO timestamp -> "3m ago" / "2h ago" / "5d ago". Falls back to a plain
 * date once it's far enough back that "N ago" stops being useful. */
export function formatRelativeTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";

  const seconds = Math.round((Date.now() - d.getTime()) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
