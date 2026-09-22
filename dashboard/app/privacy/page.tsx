import { PageShell } from "@/components/PageShell";

export const metadata = {
  title: "Privacy",
};

export default function PrivacyPage() {
  return (
    <PageShell title="Privacy" showDisclaimer={false}>
      <div className="space-y-4 text-sm text-zinc-700 dark:text-zinc-300">
        <p>
          This static site does not collect personal data by default. Pages load JSON and images from
          the same origin under public/data and public/models.
        </p>
        <p>
          The /try page runs inference locally in your browser when an ONNX file is present. Images
          you select are not sent to a FieldLens server during inference.
        </p>
        <p>
          If analytics or error reporting is added later, this page will list the provider, data
          collected, and retention. Until then, no analytics cookies are set by FieldLens.
        </p>
      </div>
    </PageShell>
  );
}
