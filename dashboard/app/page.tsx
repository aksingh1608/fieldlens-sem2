import Link from "next/link";
import { PageShell } from "@/components/PageShell";
import { GITHUB_URL, RUNS } from "@/lib/constants";

export default function OverviewPage() {
  return (
    <PageShell
      title="Overview"
      description="FieldLens segments crop anomalies in drone tiles and reports tile-level alerts."
    >
      <section className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">What it does</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-zinc-700 dark:text-zinc-300">
            <li>Pixel masks for eight Agriculture-Vision anomaly classes (plus background in Run 1).</li>
            <li>Tile alerts with class name and coverage percent when export provides them.</li>
            <li>Three training runs to test multi-label heads and RGB+NIR fusion.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
            Research questions
          </h2>
          <ol className="mt-2 list-decimal space-y-2 pl-5 text-zinc-700 dark:text-zinc-300">
            <li>
              Overlapping labels: does sigmoid multi-label beat softmax when pixels can carry more
              than one anomaly?
            </li>
            <li>
              Does gated NIR fusion improve multi-label segmentation over RGB-only?
            </li>
          </ol>
        </div>

        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Three runs</h2>
          <div className="mt-3 overflow-x-auto rounded-md border border-zinc-200 dark:border-zinc-700">
            <table className="min-w-full text-left text-sm">
              <caption className="sr-only">Experiment matrix</caption>
              <thead className="bg-zinc-100 dark:bg-zinc-800">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Run
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Input
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Head
                  </th>
                </tr>
              </thead>
              <tbody>
                {RUNS.map((run) => (
                  <tr
                    key={run.id}
                    className="border-t border-zinc-200 dark:border-zinc-700"
                  >
                    <th scope="row" className="px-4 py-3 font-medium">
                      {run.name}
                    </th>
                    <td className="px-4 py-3 text-zinc-700 dark:text-zinc-300">{run.input}</td>
                    <td className="px-4 py-3 text-zinc-700 dark:text-zinc-300">{run.head}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Dataset</h2>
          <p className="mt-2 text-zinc-700 dark:text-zinc-300">
            Agriculture-Vision (2021 challenge release on AWS). Citation: Chiu et al., CVPR 2020.
            Subset size and split counts appear on the Results page when export fills results.json.
          </p>
        </div>

        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
            Scale and next steps
          </h2>
          <p className="mt-2 text-zinc-700 dark:text-zinc-300">
            This pilot uses 2,700 of the 94,986 labelled Agriculture-Vision tiles (1,500 train,
            400 val, 800 test), selected by field under profile pilot_v2. The main research uses
            the full dataset, which needs a larger GPU and longer training, and is in progress
            (profile full is a placeholder only).
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <Link
            href={GITHUB_URL}
            className="rounded-md border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-900 hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100"
          >
            GitHub repo
          </Link>
          <Link
            href="/results"
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Results and report data
          </Link>
          <Link
            href="/method"
            className="rounded-md border border-zinc-300 bg-white px-4 py-2 text-sm font-medium hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:border-zinc-600 dark:bg-zinc-900"
          >
            Method
          </Link>
        </div>
      </section>
    </PageShell>
  );
}
