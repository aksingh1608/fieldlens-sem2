"use client";

import type { ClassItem } from "@/lib/types/classes";

interface ClassLegendProps {
  classes: ClassItem[];
  enabled: Record<string, boolean>;
  onToggle: (classId: string) => void;
}

export function ClassLegend({ classes, enabled, onToggle }: ClassLegendProps) {
  if (!classes.length) {
    return (
      <p className="text-sm text-zinc-600 dark:text-zinc-400">
        Class list not available yet. Export classes.json from training/configs/data.yaml.
      </p>
    );
  }

  return (
    <fieldset className="rounded-md border border-zinc-200 p-4 dark:border-zinc-700">
      <legend className="px-1 text-sm font-medium text-zinc-900 dark:text-zinc-100">
        Classes
      </legend>
      <ul className="mt-2 space-y-2">
        {classes.map((cls) => {
          const on = enabled[cls.id] ?? true;
          return (
            <li key={cls.id}>
              <label className="flex items-center gap-2 text-sm text-zinc-800 dark:text-zinc-200" style={{ cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={on}
                  onChange={() => onToggle(cls.id)}
                  className="h-4 w-4 rounded border-zinc-400 text-accent focus:ring-accent"
                />
                <span
                  className="inline-block h-3 w-3 shrink-0 rounded-sm border border-zinc-400"
                  style={{ backgroundColor: cls.color }}
                  aria-hidden
                />
                <span>{cls.label}</span>
                <span className="sr-only">{on ? "visible" : "hidden"}</span>
              </label>
            </li>
          );
        })}
      </ul>
    </fieldset>
  );
}
