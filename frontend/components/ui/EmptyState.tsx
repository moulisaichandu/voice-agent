import type { ReactNode } from "react";

export function EmptyState({
  title,
  action,
}: {
  title: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-3 py-10 text-center">
      <p className="text-sm text-neutral-500 dark:text-neutral-400">{title}</p>
      {action}
    </div>
  );
}
