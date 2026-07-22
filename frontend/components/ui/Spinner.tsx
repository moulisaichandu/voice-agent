import { ArrowPathIcon } from "./icons";

export function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return <ArrowPathIcon className={`animate-spin text-neutral-400 ${className}`} />;
}
