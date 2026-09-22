import { PilotDisclaimer } from "./PilotDisclaimer";

interface PageShellProps {
  title: string;
  description?: string;
  showDisclaimer?: boolean;
  children: React.ReactNode;
}

export function PageShell({
  title,
  description,
  showDisclaimer = true,
  children,
}: PageShellProps) {
  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-3xl">
          {title}
        </h1>
        {description ? (
          <p className="mt-2 max-w-3xl text-base text-zinc-600 dark:text-zinc-400">
            {description}
          </p>
        ) : null}
        {showDisclaimer ? <PilotDisclaimer className="mt-4" /> : null}
      </header>
      {children}
    </div>
  );
}
