"use client";

import { useCallback, useEffect, useState } from "react";
import type { GalleryIndex } from "@/lib/types/gallery";
import { DATA_PATHS } from "@/lib/constants";
import { loadGalleryIndex } from "@/lib/data/loadGallery";
import { DataUnavailable } from "../DataUnavailable";

type ModelState = "checking" | "missing" | "ready" | "error";

export function TryInference() {
  const [modelState, setModelState] = useState<ModelState>("checking");
  const [gallery, setGallery] = useState<GalleryIndex | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [inferMs, setInferMs] = useState<number | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  useEffect(() => {
    fetch(DATA_PATHS.model, { method: "HEAD" })
      .then((res) => {
        setModelState(res.ok ? "ready" : "missing");
      })
      .catch(() => setModelState("missing"));

    loadGalleryIndex()
      .then((data) => {
        setGallery(data);
        if (data.tiles.length > 0) setSelectedId(data.tiles[0].id);
      })
      .catch(() => setGallery({ tiles: [] }));
  }, []);

  const runInference = useCallback(async () => {
    if (modelState !== "ready") return;
    setStatusMessage(null);
    setInferMs(null);
    try {
      const ort = await import("onnxruntime-web");
      const session = await ort.InferenceSession.create(DATA_PATHS.model, {
        executionProviders: ["wasm"],
      });
      const inputName = session.inputNames[0];
      if (!inputName) {
        setModelState("error");
        setStatusMessage("Model loaded but has no inputs.");
        return;
      }
      const tile = gallery?.tiles.find((t) => t.id === selectedId);
      if (!tile?.rgb_url) {
        setStatusMessage("Pick a gallery tile with an RGB URL or add upload support in a later phase.");
        return;
      }
      const t0 = performance.now();
      // Placeholder tensor: real preprocessing lands with export pipeline.
      const dummy = new ort.Tensor(
        "float32",
        new Float32Array(1 * 3 * 64 * 64),
        [1, 3, 64, 64],
      );
      await session.run({ [inputName]: dummy });
      const elapsed = performance.now() - t0;
      setInferMs(elapsed);
      setStatusMessage("Inference ran on this device using WebAssembly. Output visualization is not wired yet.");
    } catch {
      setModelState("error");
      setStatusMessage("Could not run ONNX model in the browser.");
    }
  }, [modelState, gallery, selectedId]);

  if (modelState === "checking") {
    return <p className="text-sm text-zinc-600 dark:text-zinc-400">Checking for exported model...</p>;
  }

  if (modelState === "missing") {
    return (
      <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">model not exported yet</p>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          Place fieldlens.onnx under public/models/ after Phase 10 export. This page will load it with
          onnxruntime-web when present.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-zinc-700 dark:text-zinc-300">
        Runs on your device. No image leaves the browser unless you enable analytics later.
      </p>

      <div>
        <label htmlFor="try-tile" className="text-sm font-medium">
          Gallery tile
        </label>
        {gallery && gallery.tiles.length > 0 ? (
          <select
            id="try-tile"
            value={selectedId ?? ""}
            onChange={(e) => setSelectedId(e.target.value)}
            className="mt-2 block w-full max-w-md rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-600 dark:bg-zinc-900"
          >
            {gallery.tiles.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label ?? t.id}
              </option>
            ))}
          </select>
        ) : (
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            No gallery tiles yet. Upload support can be added when AK approves hosting user images.
          </p>
        )}
      </div>

      <button
        type="button"
        onClick={() => void runInference()}
        className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        Run inference
      </button>

      {inferMs !== null ? (
        <p className="text-sm text-zinc-800 dark:text-zinc-200">
          Inference time: {inferMs.toFixed(1)} ms (includes session load on first run)
        </p>
      ) : null}

      {statusMessage ? (
        <p className="text-sm text-zinc-600 dark:text-zinc-400" role="status">
          {statusMessage}
        </p>
      ) : null}

      {modelState === "error" ? <DataUnavailable label="Model error" /> : null}
    </div>
  );
}
