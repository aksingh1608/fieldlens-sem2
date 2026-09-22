import type { ResultsJson } from "@/lib/types/results";
import { DATA_PATHS } from "@/lib/constants";

/** Load results.json. Returns null when absent or unreadable (empty-state friendly). */
export async function loadResults(): Promise<ResultsJson | null> {
  try {
    const res = await fetch(DATA_PATHS.results, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as ResultsJson;
  } catch {
    return null;
  }
}

export function emptyResultsStub(): ResultsJson {
  return {
    status: "pending_runs",
    generated_at: null,
    dataset: null,
    runs: { run1: null, run2: null, run3: null },
    tile_compare: null,
    pr_curves: { run2: null, run3: null },
    robustness: null,
    training_curves: null,
    efficiency: null,
  };
}
