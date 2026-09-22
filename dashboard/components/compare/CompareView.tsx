"use client";

import { useEffect, useMemo, useState } from "react";
import type { GalleryTile } from "@/lib/types/gallery";
import type { ResultsJson, TileCompareEntry } from "@/lib/types/results";
import { loadGalleryIndex } from "@/lib/data/loadGallery";
import { loadResults } from "@/lib/data/loadResults";
import { formatNumber } from "@/lib/format";
import { DataUnavailable } from "../DataUnavailable";

function iouForTile(entry: TileCompareEntry | undefined, run: "run1" | "run2" | "run3") {
  if (!entry) return null;
  return entry[`iou_${run}`];
}

export function CompareView() {
  const [tiles, setTiles] = useState<GalleryTile[]>([]);
  const [compare, setCompare] = useState<TileCompareEntry[] | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([loadGalleryIndex(), loadResults()])
      .then(([gallery, results]) => {
        setTiles(gallery.tiles);
        setCompare(results?.tile_compare ?? null);
        if (gallery.tiles.length > 0) setSelectedId(gallery.tiles[0].id);
      })
      .catch(() => setLoadError("Could not load gallery or results."));
  }, []);

  const tile = useMemo(
    () => tiles.find((t) => t.id === selectedId),
    [tiles, selectedId],
  );

  const tileMetrics = useMemo(
    () => compare?.find((c) => c.tile_id === selectedId),
    [compare, selectedId],
  );

  const panels = [
    { key: "gt", label: "Ground truth", url: tile?.gt_url ?? null },
    { key: "run1", label: "Run 1", url: tile?.predictions?.run1 ?? null },
    { key: "run2", label: "Run 2", url: tile?.predictions?.run2 ?? null },
    { key: "run3", label: "Run 3", url: tile?.predictions?.run3 ?? null },
  ] as const;

  return (
    <div className="space-y-6">
      {loadError ? (
        <p className="text-sm text-red-700 dark:text-red-400" role="alert">
          {loadError}
        </p>
      ) : null}

      <div>
        <label htmlFor="compare-tile" className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
          Tile
        </label>
        {tiles.length === 0 ? (
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            No tiles in gallery. Per-tile IoU will appear when results.json includes tile_compare.
          </p>
        ) : (
          <select
            id="compare-tile"
            value={selectedId ?? ""}
            onChange={(e) => setSelectedId(e.target.value)}
            className="mt-2 block w-full max-w-md rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-600 dark:bg-zinc-900"
          >
            {tiles.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label ?? t.id}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="overflow-x-auto rounded-md border border-zinc-200 dark:border-zinc-700">
        <table className="min-w-full text-left text-sm">
          <caption className="sr-only">Per-tile IoU for selected tile</caption>
          <thead className="bg-zinc-100 dark:bg-zinc-800">
            <tr>
              <th scope="col" className="px-4 py-3 font-medium">
                Run
              </th>
              <th scope="col" className="px-4 py-3 font-medium">
                IoU vs GT
              </th>
            </tr>
          </thead>
          <tbody>
            {(["run1", "run2", "run3"] as const).map((run) => (
              <tr key={run} className="border-t border-zinc-200 dark:border-zinc-700">
                <th scope="row" className="px-4 py-3 font-medium uppercase">
                  {run}
                </th>
                <td className="px-4 py-3">
                  {formatNumber(iouForTile(tileMetrics, run))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {panels.map((panel) => (
          <figure
            key={panel.key}
            className="overflow-hidden rounded-md border border-zinc-200 dark:border-zinc-700"
          >
            <figcaption className="border-b border-zinc-200 bg-zinc-50 px-3 py-2 text-sm font-medium dark:border-zinc-700 dark:bg-zinc-800">
              {panel.label}
            </figcaption>
            {panel.url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={panel.url}
                alt={`${panel.label} for tile ${tile?.id ?? "none"}`}
                className="h-40 w-full object-cover sm:h-48"
              />
            ) : (
              <div className="flex h-40 items-center justify-center bg-zinc-100 sm:h-48 dark:bg-zinc-900">
                <DataUnavailable />
              </div>
            )}
          </figure>
        ))}
      </div>
    </div>
  );
}
