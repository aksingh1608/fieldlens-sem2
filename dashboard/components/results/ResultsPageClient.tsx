"use client";

import { useEffect, useState } from "react";
import type { ResultsJson } from "@/lib/types/results";
import { emptyResultsStub, loadResults } from "@/lib/data/loadResults";
import {
  EfficiencyTable,
  PrCurvesSection,
  RobustnessTable,
  TrainingCurvesSection,
} from "./ResultsCharts";
import { RunMetricsTable } from "./MetricsTable";
import { DataUnavailable } from "../DataUnavailable";

export function ResultsPageClient() {
  const [data, setData] = useState<ResultsJson | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadResults()
      .then((loaded) => {
        if (!loaded) {
          setData(emptyResultsStub());
          return;
        }
        setData(loaded);
      })
      .catch(() => setError("results.json could not be loaded."));
  }, []);

  if (error) {
    return (
      <p className="text-sm text-red-700 dark:text-red-400" role="alert">
        {error}
      </p>
    );
  }

  if (!data) {
    return <p className="text-sm text-zinc-600 dark:text-zinc-400">Loading results...</p>;
  }

  const dataset = data.dataset ?? null;
  const runs = {
    run1: (data.runs?.run1 as ResultsJson["runs"]["run1"]) ?? null,
    run2: (data.runs?.run2 as ResultsJson["runs"]["run2"]) ?? null,
    run3: (data.runs?.run3 as ResultsJson["runs"]["run3"]) ?? null,
  };

  return (
    <div className="space-y-10">
      <section>
        <h2 className="mb-2 text-lg font-semibold">Status</h2>
        <p className="text-sm text-zinc-700 dark:text-zinc-300">
          Export status:{" "}
          <span className="font-medium">{String(data.status).replace("_", " ")}</span>
          {data.generated_at ? ` (generated ${data.generated_at})` : null}
          {data.profile ? ` · profile ${data.profile}` : null}
        </p>
        <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
          <div>
            <dt className="font-medium text-zinc-900 dark:text-zinc-100">Dataset</dt>
            <dd className="text-zinc-700 dark:text-zinc-300">
              {dataset?.name ?? "not available yet"}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-zinc-900 dark:text-zinc-100">Subset tiles</dt>
            <dd className="text-zinc-700 dark:text-zinc-300">
              {dataset?.subset_tiles ?? "not available yet"}
            </dd>
          </div>
        </dl>
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold">Summary metrics</h2>
        <RunMetricsTable runs={runs} />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold">PR curves</h2>
        <PrCurvesSection data={data} />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold">Robustness</h2>
        <RobustnessTable data={data} />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold">Training curves</h2>
        <TrainingCurvesSection data={data} />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold">Efficiency</h2>
        <EfficiencyTable data={data} />
      </section>

      {data.status === "pending_runs" ? (
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Tables and charts stay empty or show not available yet until training export writes real
          numbers. The UI does not invent metrics.
        </p>
      ) : null}

      {!runs.run1 && !runs.run2 && !runs.run3 ? (
        <DataUnavailable label="Run-level metrics not available yet" />
      ) : null}
    </div>
  );
}
