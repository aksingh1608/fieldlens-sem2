import { PageShell } from "@/components/PageShell";
import { ExplorerView } from "@/components/explorer/ExplorerView";

export const metadata = {
  title: "Explorer",
};

export default function ExplorerPage() {
  return (
    <PageShell
      title="Explorer"
      description="Inspect RGB and NIR tiles with ground truth or run overlays."
    >
      <ExplorerView />
    </PageShell>
  );
}
