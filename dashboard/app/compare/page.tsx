import { PageShell } from "@/components/PageShell";
import { CompareView } from "@/components/compare/CompareView";

export const metadata = {
  title: "Compare",
};

export default function ComparePage() {
  return (
    <PageShell
      title="Compare"
      description="One tile with ground truth and three run predictions side by side."
    >
      <CompareView />
    </PageShell>
  );
}
