import { readFileSync, existsSync } from "fs";
import path from "path";
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
    return (
      <PageShell
        title="Try in browser"
        description="Optional Phase 10 demo. Currently disabled for this deploy."
      >
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Set <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">enable_try_page</code> in{" "}
          <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">public/data/site.json</code> and
          place <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">fieldlens.onnx</code> under{" "}
          <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">public/models/</code> to enable.
        </p>
      </PageShell>
    );
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
