export function DataUnavailable({ label }: { label?: string }) {
  return (
    <p className="text-sm text-zinc-600 dark:text-zinc-400" role="status">
      {label ?? "not available yet"}
    </p>
  );
}
