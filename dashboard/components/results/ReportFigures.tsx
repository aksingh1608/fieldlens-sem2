"use client";

const FIGURES: { src: string; title: string; caption: string }[] = [
  {
    src: "/figures/sample_tiles.png",
    title: "Sample tiles",
    caption: "RGB tiles with anomaly label overlays from the pilot subset.",
  },
  {
    src: "/figures/training_curves.png",
    title: "Training curves",
    caption: "Loss and mIoU trends across Run 1–3 (pilot v1).",
  },
  {
    src: "/figures/pr_curves.png",
    title: "Precision–recall",
    caption: "Per-class PR curves for each run.",
  },
  {
    src: "/figures/per_class_iou_f1.png",
    title: "Per-class IoU / F1",
    caption: "Class-wise scores on the pilot test split.",
  },
  {
    src: "/figures/sample_predictions.png",
    title: "Sample predictions",
    caption: "Ground truth vs model overlays on held-out tiles.",
  },
  {
    src: "/figures/class_distribution.png",
    title: "Class distribution",
    caption: "How often each anomaly appears in the pilot splits.",
  },
];

export function ReportFigures() {
  return (
    <div className="space-y-8">
      <p className="text-sm text-zinc-600 dark:text-zinc-400">
        Static report PNGs from <code className="text-xs">notebooks/figures/</code> (also under{" "}
        <code className="text-xs">public/figures/</code>).
      </p>
      <div className="grid gap-8 lg:grid-cols-2">
        {FIGURES.map((fig) => (
          <figure key={fig.src} className="space-y-2">
            <figcaption className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
              {fig.title}
            </figcaption>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={fig.src}
              alt={fig.title}
              className="w-full rounded-md border border-zinc-200 bg-white dark:border-zinc-700"
            />
            <p className="text-xs text-zinc-600 dark:text-zinc-400">{fig.caption}</p>
          </figure>
        ))}
      </div>
    </div>
  );
}
