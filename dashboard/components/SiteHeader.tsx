"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./theme/ThemeToggle";
import { loadSiteConfig } from "@/lib/data/loadClasses";

const BASE_NAV = [
  { href: "/", label: "Overview" },
  { href: "/explorer", label: "Explorer" },
  { href: "/compare", label: "Compare" },
  { href: "/results", label: "Results" },
  { href: "/method", label: "Method" },
];

function navClass(active: boolean) {
  return active
    ? "text-accent font-semibold underline decoration-2 underline-offset-4"
    : "text-zinc-700 hover:text-accent dark:text-zinc-300 dark:hover:text-accent";
}

export function SiteHeader() {
  const pathname = usePathname();
  const [nav, setNav] = useState(BASE_NAV);

  useEffect(() => {
    loadSiteConfig().then((site) => {
      if (site.enable_try_page) {
        setNav([...BASE_NAV, { href: "/try", label: "Try" }]);
      } else {
        setNav(BASE_NAV);
      }
    });
  }, []);

  return (
    <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="flex items-center gap-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <span
              className="inline-flex h-8 w-8 items-center justify-center rounded-md bg-accent text-sm font-bold text-white"
              aria-hidden
            >
              FL
            </span>
            <span className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
              FieldLens
            </span>
          </Link>
        </div>
        <nav aria-label="Main" className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
          {nav.map((item) => {
            const active =
              item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-md px-1 py-0.5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${navClass(active)}`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <ThemeToggle />
      </div>
    </header>
  );
}
