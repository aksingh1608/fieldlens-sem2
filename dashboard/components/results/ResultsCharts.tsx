"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ResultsJson } from "@/lib/types/results";
import { DataUnavailable } from "../DataUnavailable";

function chartColors() {
  return {
    run1: "#0d9488",
    run2: "#2563eb",
    run3: "#ca8a04",
  };
}

export function TrainingCurvesSection({ data }: { data: ResultsJson }) {
  const series = data.training_curves;
  if (!series || series.length === 0) {
    return <DataUnavailable label="Training curves not available yet" />;
  }

  const byMetric = new Map<string, { epoch: number; [key: string]: number }[]>();
  for (const s of series) {
    if (!s.epochs || !s.values || s.epochs.length !== s.values.length) continue;
    const rows = s.epochs.map((epoch, i) => ({
      epoch,
      [`${s.run_id}_${s.metric}`]: s.values![i],
    }));
    const key = s.metric;
    const existing = byMetric.get(key) ?? [];
    for (const row of rows) {
      const match = existing.find((r) => r.epoch === row.epoch);
      if (match) Object.assign(match, row);
      else existing.push(row);
    }
    byMetric.set(key, existing);
  }

  const colors = chartColors();

  return (
    <div className="space-y-8">
      {[...byMetric.entries()].map(([metric, rows]) => (
        <div key={metric}>
          <h3 className="mb-2 text-sm font-medium text-zinc-900 dark:text-zinc-100">
            Training: {metric}
          </h3>
          <div className="h-64 w-full rounded-md border border-zinc-200 p-2 dark:border-zinc-700">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows.sort((a, b) => a.epoch - b.epoch)}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-zinc-200 dark:stroke-zinc-700" />
                <XAxis dataKey="epoch" name="Epoch" />
                <YAxis />
                <Tooltip />
                <Legend />
                {(["run1", "run2", "run3"] as const).map((run) => {
                  const dataKey = `${run}_${metric}`;
                  const hasKey = rows.some((r) => dataKey in r);
                  if (!hasKey) return null;
                  return (
                    <Line
                      key={dataKey}
                      type="monotone"
                      dataKey={dataKey}
                      name={run}
                      stroke={colors[run]}
                      dot={false}
                      strokeWidth={2}
                    />
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ))}
    </div>
  );
}

export function PrCurvesSection({ data }: { data: ResultsJson }) {
  const run2 = data.pr_curves?.run2;
  const run3 = data.pr_curves?.run3;

  if ((!run2 || run2.length === 0) && (!run3 || run3.length === 0)) {
    return <DataUnavailable label="PR curves not available yet" />;
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {(
        [
          { title: "Run 2 PR curves", series: run2 },
          { title: "Run 3 PR curves", series: run3 },
        ] as const
      ).map(({ title, series }) => {
        if (!series || series.length === 0) {
          return (
            <div key={title}>
              <h3 className="mb-2 text-sm font-medium">{title}</h3>
              <DataUnavailable />
            </div>
          );
        }
        const merged: Record<number, Record<string, number>> = {};
        for (const s of series) {
          if (!s.points) continue;
          for (const p of s.points) {
            if (!merged[p.x]) merged[p.x] = { recall: p.x };
            merged[p.x][s.class_id] = p.y;
          }
        }
        const chartData = Object.values(merged).sort((a, b) => a.recall - b.recall);
        const classIds = series.map((s) => s.class_id);

        return (
          <div key={title}>
            <h3 className="mb-2 text-sm font-medium text-zinc-900 dark:text-zinc-100">
              {title}
            </h3>
            <div className="h-64 rounded-md border border-zinc-200 p-2 dark:border-zinc-700">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="recall"
                    type="number"
                    domain={[0, 1]}
                    label={{ value: "Recall", position: "insideBottom", offset: -2 }}
                  />
                  <YAxis domain={[0, 1]} label={{ value: "Precision", angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Legend />
                  {classIds.map((id, i) => (
                    <Line
                      key={id}
                      type="monotone"
                      dataKey={id}
                      name={series.find((s) => s.class_id === id)?.class_label ?? id}
                      stroke={["#0d9488", "#2563eb", "#ca8a04", "#dc2626", "#64748b"][i % 5]}
                      dot={false}
                      strokeWidth={1.5}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function RobustnessTable({ data }: { data: ResultsJson }) {
  const rows = data.robustness;
  if (!rows || rows.length === 0) {
    return <DataUnavailable label="Robustness table not available yet" />;
  }

  return (
    <div className="overflow-x-auto rounded-md border border-zinc-200 dark:border-zinc-700">
      <table className="min-w-full text-left text-sm">
        <caption className="sr-only">Robustness by condition</caption>
        <thead className="bg-zinc-100 dark:bg-zinc-800">
          <tr>
            <th scope="col" className="px-4 py-3 font-medium">
              Condition
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Run 1 mIoU
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Run 2 mIoU
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Run 3 mIoU
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.condition} className="border-t border-zinc-200 dark:border-zinc-700">
              <th scope="row" className="px-4 py-3 font-medium">
                {row.condition}
              </th>
              <td className="px-4 py-3">{row.run1_miou ?? "not available yet"}</td>
              <td className="px-4 py-3">{row.run2_miou ?? "not available yet"}</td>
              <td className="px-4 py-3">{row.run3_miou ?? "not available yet"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EfficiencyTable({ data }: { data: ResultsJson }) {
  const rows = data.efficiency;
  if (!rows || rows.length === 0) {
    return <DataUnavailable label="Efficiency table not available yet" />;
  }

  return (
    <div className="overflow-x-auto rounded-md border border-zinc-200 dark:border-zinc-700">
      <table className="min-w-full text-left text-sm">
        <caption className="sr-only">Training and inference efficiency</caption>
        <thead className="bg-zinc-100 dark:bg-zinc-800">
          <tr>
            <th scope="col" className="px-4 py-3 font-medium">
              Run
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Params (M)
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Train s/epoch
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Infer ms/tile
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Peak VRAM (GB)
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.run_id} className="border-t border-zinc-200 dark:border-zinc-700">
              <th scope="row" className="px-4 py-3 font-medium">
                {row.run_id}
              </th>
              <td className="px-4 py-3">{row.params_m ?? "not available yet"}</td>
              <td className="px-4 py-3">{row.train_sec_per_epoch ?? "not available yet"}</td>
              <td className="px-4 py-3">{row.infer_ms_per_tile ?? "not available yet"}</td>
              <td className="px-4 py-3">{row.vram_gb_peak ?? "not available yet"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
