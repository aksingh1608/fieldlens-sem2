"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { GalleryIndex, GalleryTile } from "@/lib/types/gallery";
import type { ClassItem } from "@/lib/types/classes";
import { loadGalleryIndex } from "@/lib/data/loadGallery";
import { loadClasses } from "@/lib/data/loadClasses";
import { ClassLegend } from "../ClassLegend";
import { DataUnavailable } from "../DataUnavailable";

type OverlaySource = "gt" | "run1" | "run2" | "run3";
type BaseLayer = "rgb" | "nir";

export function ExplorerView() {
  const [gallery, setGallery] = useState<GalleryIndex | null>(null);
  const [classItems, setClassItems] = useState<ClassItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [baseLayer, setBaseLayer] = useState<BaseLayer>("rgb");
  const [overlay, setOverlay] = useState<OverlaySource>("gt");
  const [opacity, setOpacity] = useState(0.55);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});

  useEffect(() => {
    Promise.all([loadGalleryIndex(), loadClasses()])
      .then(([galleryData, classesData]) => {
        setGallery(galleryData);
        if (galleryData.tiles.length > 0) setSelectedId(galleryData.tiles[0].id);
        if (classesData) {
          const items = [...classesData.anomalies];
          setClassItems(items);
          setEnabled(Object.fromEntries(items.map((c) => [c.id, true])));
        }
      })
      .catch(() => setError("Gallery or class list could not be loaded."));
  }, []);

  const tile: GalleryTile | undefined = useMemo(
    () => gallery?.tiles.find((t) => t.id === selectedId),
    [gallery, selectedId],
  );

  const onToggleClass = useCallback((classId: string) => {
    setEnabled((prev) => ({ ...prev, [classId]: !prev[classId] }));
  }, []);

  const overlayUrl = useMemo(() => {
    if (!tile) return null;
    if (overlay === "gt") return tile.gt_url;
    return tile.predictions?.[overlay] ?? null;
  }, [tile, overlay]);

  const baseUrl = useMemo(() => {
    if (!tile) return null;
    if (baseLayer === "nir") return tile.nir_url;
    return tile.rgb_url;
  }, [tile, baseLayer]);

  return (
    <div className="grid gap-6 lg:grid-cols-[240px_1fr]">
      <aside>
        <h2 className="mb-2 text-sm font-medium text-zinc-900 dark:text-zinc-100">
          Gallery tiles
        </h2>
        {error ? (
          <p className="text-sm text-red-700 dark:text-red-400" role="alert">
            {error}
          </p>
        ) : null}
        {!gallery ? (
          <p className="text-sm text-zinc-600 dark:text-zinc-400">Loading gallery...</p>
        ) : gallery.tiles.length === 0 ? (
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            No gallery tiles exported yet. Add entries to public/data/gallery/index.json after
            Phase 8 export.
          </p>
        ) : (
          <ul className="space-y-2">
            {gallery.tiles.map((t) => (
              <li key={t.id}>
                <button
                  type="button"
                  onClick={() => setSelectedId(t.id)}
                  className={`w-full rounded-md border px-3 py-2 text-left text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                    selectedId === t.id
                      ? "border-accent bg-accent/10 text-zinc-900 dark:text-zinc-50"
                      : "border-zinc-200 text-zinc-700 dark:border-zinc-700 dark:text-zinc-300"
                  }`}
                >
                  {t.id}
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="mt-6">
          <ClassLegend classes={classItems} enabled={enabled} onToggle={onToggleClass} />
        </div>
      </aside>

      <div className="space-y-4">
        {!tile ? (
          <DataUnavailable label="No tile selected" />
        ) : (
          <>
            <div className="flex flex-wrap gap-3 text-sm">
              <label className="flex items-center gap-2">
                <span>Base</span>
                <select
                  value={baseLayer}
                  onChange={(e) => setBaseLayer(e.target.value as BaseLayer)}
                  className="rounded-md border border-zinc-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900"
                >
                  <option value="rgb">RGB</option>
                  <option value="nir">NIR</option>
                </select>
              </label>
              <label className="flex items-center gap-2">
                <span>Overlay</span>
                <select
                  value={overlay}
                  onChange={(e) => setOverlay(e.target.value as OverlaySource)}
                  className="rounded-md border border-zinc-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900"
                >
                  <option value="gt">Ground truth</option>
                  <option value="run1">Run 1</option>
                  <option value="run2">Run 2</option>
                  <option value="run3">Run 3</option>
                </select>
              </label>
              <label className="flex items-center gap-2">
                <span>Opacity</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                />
              </label>
            </div>

            <div className="relative aspect-square max-w-xl overflow-hidden rounded-md border border-zinc-200 bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900">
              {baseUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={baseUrl}
                  alt={`${tile.id} ${baseLayer}`}
                  className="h-full w-full object-contain"
                />
              ) : (
                <DataUnavailable label="Base image not available yet" />
              )}
              {overlayUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={overlayUrl}
                  alt={`${tile.id} overlay ${overlay}`}
                  className="pointer-events-none absolute inset-0 h-full w-full object-contain"
                  style={{ opacity }}
                />
              ) : null}
            </div>

            <div className="rounded-md border border-zinc-200 p-4 text-sm dark:border-zinc-700">
              <h3 className="font-medium text-zinc-900 dark:text-zinc-100">Tile alerts</h3>
              {tile.alert ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-zinc-700 dark:text-zinc-300">
                  <li>
                    {tile.alert.class_label ?? tile.alert.class_id}
                    {tile.alert.coverage_percent != null
                      ? `: ${tile.alert.coverage_percent}% coverage`
                      : null}
                  </li>
                </ul>
              ) : (
                <p className="mt-2 text-zinc-600 dark:text-zinc-400">No alerts on this tile.</p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
