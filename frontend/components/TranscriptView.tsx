import type { TranscriptTurn } from "@/lib/api";

export function TranscriptView({ turns }: { turns: TranscriptTurn[] }) {
  return (
    <div className="border-l-2 border-accent/40 bg-neutral-50 p-4 dark:bg-neutral-900/60">
      <div className="flex flex-col gap-2">
        {turns.map((t, i) => (
          <p key={i} className="text-sm text-neutral-700 dark:text-neutral-300">
            <span className={t.role === "agent" ? "font-medium text-accent" : "font-medium"}>
              {t.role === "agent" ? "Agent" : "Lead"}:
            </span>{" "}
            {t.text}
          </p>
        ))}
        {turns.length === 0 && (
          <p className="text-sm text-neutral-500 dark:text-neutral-400">No transcript recorded.</p>
        )}
      </div>
    </div>
  );
}
