import type { ReactNode } from "react";

type Props = {
  title?: string;
  description?: string;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
};

export function Card({ title, description, action, className = "", children }: Props) {
  return (
    <section
      className={`rounded-lg border border-neutral-200 bg-white p-5
        dark:border-neutral-800 dark:bg-neutral-900/40 ${className}`}
    >
      {(title || action) && (
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && (
              <h2 className="text-base font-semibold text-neutral-900 dark:text-neutral-50">
                {title}
              </h2>
            )}
            {description && (
              <p className="mt-0.5 text-sm text-neutral-500 dark:text-neutral-400">
                {description}
              </p>
            )}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}
