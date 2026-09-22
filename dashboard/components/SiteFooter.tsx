import Link from "next/link";
import { GITHUB_URL } from "@/lib/constants";

export function SiteFooter() {
  return (
    <footer className="mt-auto border-t border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900/50">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-6 text-sm text-zinc-600 dark:text-zinc-400 sm:flex-row sm:items-center sm:justify-between">
        <p>FieldLens research pilot dashboard</p>
        <nav aria-label="Footer" className="flex flex-wrap gap-4">
          <Link
            href="/terms"
            className="rounded-md text-zinc-700 underline-offset-2 hover:text-accent hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:text-zinc-300"
          >
            Terms
          </Link>
          <Link
            href="/privacy"
            className="rounded-md text-zinc-700 underline-offset-2 hover:text-accent hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:text-zinc-300"
          >
            Privacy
          </Link>
          <a
            href={GITHUB_URL}
            className="rounded-md text-zinc-700 underline-offset-2 hover:text-accent hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent dark:text-zinc-300"
            rel="noopener noreferrer"
            target="_blank"
          >
            GitHub
          </a>
        </nav>
      </div>
    </footer>
  );
}
