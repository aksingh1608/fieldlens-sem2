import { PageShell } from "@/components/PageShell";
import { ArchitectureDiagram } from "@/components/method/ArchitectureDiagram";

export const metadata = {
  title: "Method",
};

export default function MethodPage() {
  return (
    <PageShell
      title="Method"
      description="Model architecture, data split, Run 1 label rule, and metrics."
    >
      <div className="space-y-8">
        <section>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Architecture</h2>
          <div className="mt-3">
            <ArchitectureDiagram />
          </div>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Data split</h2>
          <p className="mt-2 text-zinc-700 dark:text-zinc-300">
            Agriculture-Vision 2021 supervised tiles, subset of 2,700 images (1,500 / 400 / 800)
            with whole field separation between splits. Profile pilot_v2 caps tiles per field
            (10 / 10 / 5) so more fields enter each split. Profile pilot (v1) used fewer fields
            and is kept only for comparison.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
            Run 1 single-label rule
          </h2>
          <p className="mt-2 text-zinc-700 dark:text-zinc-300">
            When multiple anomaly masks overlap on one pixel, Run 1 assigns the rarest class in the
            training subset (documented in docs/data.md after Phase 4). Runs 2 and 3 keep independent
            sigmoid channels per class.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Metrics</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-zinc-700 dark:text-zinc-300">
            <li>Pixel mIoU and Agriculture-Vision modified mIoU when eval export provides them.</li>
            <li>Per-class IoU, precision, and recall for multi-label runs.</li>
            <li>Tile alert F1 from exported tile-level predictions.</li>
            <li>Robustness slices and training or inference timing in results.json.</li>
          </ul>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Citations</h2>
          <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-zinc-700 dark:text-zinc-300">
            <li>
              Chiu et al., &quot;Agriculture-Vision: A Large Aerial Image Database for Agricultural
              Pattern Analysis,&quot; CVPR 2020.
            </li>
            <li>
              SegFormer / MiT: Xie et al., &quot;SegFormer: Simple and Efficient Design for Semantic
              Segmentation with Transformers,&quot; NeurIPS 2021.
            </li>
            <li>
              Li, X., Qiao, L., and Yang, C. (2025). AgriFusion: Multiscale RGB-NIR Fusion for
              Semantic Segmentation in Airborne Agricultural Imagery. AgriEngineering, 7(11), 388.
              https://doi.org/10.3390/agriengineering7110388. Used as a fusion component; not claimed
              as a FieldLens novel contribution.
            </li>
          </ul>
        </section>
      </div>
    </PageShell>
  );
}
