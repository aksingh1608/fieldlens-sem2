import { PageShell } from "@/components/PageShell";
import { ResultsPageClient } from "@/components/results/ResultsPageClient";

export const metadata = {
  title: "Results",
};

export default function ResultsPage() {
  return (
    <PageShell
      title="Results"
      description="Metrics, curves, and efficiency from public/data/results.json."
    >
      <ResultsPageClient />
    </PageShell>
  );
}
