import type { RunDetailMetrics } from "@/lib/types/results";
import { formatNumber } from "@/lib/format";
import { DataUnavailable } from "../DataUnavailable";

const RUN_LABELS: Record<string, string> = {
  run1: "Run 1 (RGB, softmax)",
  run2: "Run 2 (RGB, sigmoid)",
  run3: "Run 3 (RGB+NIR, sigmoid)",
};

function metricValue(
  data: RunDetailMetrics | null,
  key: "pixel_miou" | "modified_miou" | "tile_alert_f1",
): number | null {
  if (!data) return null;
  const top = data[key];
  if (typeof top === "number") return top;
  const nested = data.metrics as Record<string, unknown> | null | undefined;
  const fromNested = nested?.[key === "modified_miou" ? "modified_miou" : key];
  return typeof fromNested === "number" ? fromNested : null;
}

export function RunMetricsTable({
  runs,
}: {
  runs: {
    run1: RunDetailMetrics | null;
    run2: RunDetailMetrics | null;
    run3: RunDetailMetrics | null;
  };
}) {
  const entries = (["run1", "run2", "run3"] as const).map((id) => ({
    id,
    data: runs[id] ?? null,
  }));

  return (
    <div className="overflow-x-auto rounded-md border border-zinc-200 dark:border-zinc-700">
      <table className="min-w-full text-left text-sm">
        <caption className="sr-only">Summary metrics per run</caption>
        <thead className="bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100">
          <tr>
            <th scope="col" className="px-4 py-3 font-medium">
              Run
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Pixel mIoU
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Modified mIoU
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Tile alert F1
            </th>
          </tr>
        </thead>
        <tbody>
          {entries.map(({ id, data }) => (
            <tr
              key={id}
              className="border-t border-zinc-200 dark:border-zinc-700"
            >
              <th scope="row" className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                {RUN_LABELS[id]}
              </th>
              <td className="px-4 py-3 text-zinc-700 dark:text-zinc-300">
                {data ? formatNumber(metricValue(data, "pixel_miou")) : <DataUnavailable />}
              </td>
              <td className="px-4 py-3 text-zinc-700 dark:text-zinc-300">
                {data ? formatNumber(metricValue(data, "modified_miou")) : <DataUnavailable />}
              </td>
              <td className="px-4 py-3 text-zinc-700 dark:text-zinc-300">
                {data ? formatNumber(metricValue(data, "tile_alert_f1")) : <DataUnavailable />}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
