import { PILOT_DISCLAIMER } from "@/lib/constants";

export function PilotDisclaimer({ className = "" }: { className?: string }) {
  return (
    <p
      role="note"
      className={`rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-100 ${className}`}
    >
      {PILOT_DISCLAIMER}
    </p>
  );
}
