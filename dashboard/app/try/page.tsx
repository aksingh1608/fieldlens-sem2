import { readFileSync, existsSync } from "fs";
import path from "path";
import { notFound } from "next/navigation";
import { PageShell } from "@/components/PageShell";
import { TryInference } from "@/components/try/TryInference";

export const metadata = {
  title: "Try",
};

function tryEnabled(): boolean {
  const sitePath = path.join(process.cwd(), "public", "data", "site.json");
  if (!existsSync(sitePath)) return false;
  try {
    const site = JSON.parse(readFileSync(sitePath, "utf8")) as { enable_try_page?: boolean };
    return Boolean(site.enable_try_page);
  } catch {
    return false;
  }
}

export default function TryPage() {
  if (!tryEnabled()) {
    notFound();
  }

  return (
    <PageShell
      title="Try in browser"
      description="Optional Phase 10 demo with onnxruntime-web when fieldlens.onnx is exported."
    >
      <TryInference />
    </PageShell>
  );
}
