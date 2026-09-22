import { PageShell } from "@/components/PageShell";

export const metadata = {
  title: "Terms",
};

export default function TermsPage() {
  return (
    <PageShell title="Terms and Conditions" showDisclaimer={false}>
      <div className="space-y-4 text-sm text-zinc-700 dark:text-zinc-300">
        <p>
          FieldLens is a research pilot dashboard. Use it for informational review of exported
          experiment outputs only.
        </p>

        <h2 className="pt-4 text-lg font-semibold text-zinc-900 dark:text-zinc-50">Dataset redistribution</h2>
        <p>
          Agriculture-Vision tiles and labels remain subject to the dataset license and challenge
          terms. Do not redistribute raw tiles or masks through this site unless AK confirms
          permission in writing. The dashboard may show only the subset allowed for public display
          under those terms.
        </p>

        <h2 className="pt-4 text-lg font-semibold text-zinc-900 dark:text-zinc-50">No warranty</h2>
        <p>
          Models and metrics are experimental. They are not agronomic advice and not certified for
          operational farm decisions.
        </p>

        <h2 className="pt-4 text-lg font-semibold text-zinc-900 dark:text-zinc-50">Contact</h2>
        <p>
          Questions: open an issue on the GitHub repository linked in the footer.
        </p>
      </div>
    </PageShell>
  );
}
