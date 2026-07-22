import type { ReactNode, ThHTMLAttributes, TdHTMLAttributes } from "react";

export function Table({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className={`w-full min-w-max text-sm ${className}`}>{children}</table>
    </div>
  );
}

export function Th({ children, className = "", ...rest }: ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      className={`whitespace-nowrap border-b border-neutral-200 px-3 py-2 text-left
        text-xs font-medium uppercase tracking-wide text-neutral-500
        dark:border-neutral-800 dark:text-neutral-400 ${className}`}
      {...rest}
    >
      {children}
    </th>
  );
}

export function Td({ children, className = "", ...rest }: TdHTMLAttributes<HTMLTableCellElement>) {
  return (
    <td
      className={`border-b border-neutral-100 px-3 py-2.5 align-middle text-neutral-700
        dark:border-neutral-900 dark:text-neutral-300 ${className}`}
      {...rest}
    >
      {children}
    </td>
  );
}

/** A <tr> whose only cell spans every column — for empty/loading states inside
 * a table body without the caller having to compute colSpan by hand elsewhere. */
export function TableMessageRow({
  colSpan,
  children,
}: {
  colSpan: number;
  children: ReactNode;
}) {
  return (
    <tr>
      <td colSpan={colSpan} className="px-3 py-8 text-center text-sm text-neutral-500 dark:text-neutral-400">
        {children}
      </td>
    </tr>
  );
}
