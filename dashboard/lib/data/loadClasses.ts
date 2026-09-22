import type { ClassesJson, SiteJson } from "@/lib/types/classes";
import { DATA_PATHS } from "@/lib/constants";

export async function loadClasses(): Promise<ClassesJson | null> {
  try {
    const res = await fetch(DATA_PATHS.classes, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as ClassesJson;
  } catch {
    return null;
  }
}

export async function loadSiteConfig(): Promise<SiteJson> {
  try {
    const res = await fetch(DATA_PATHS.site, { cache: "no-store" });
    if (!res.ok) {
      return { enable_try_page: false };
    }
    const data = (await res.json()) as SiteJson;
    return {
      enable_try_page: Boolean(data.enable_try_page),
      profile: data.profile,
      pilot_disclaimer: data.pilot_disclaimer,
    };
  } catch {
    return { enable_try_page: false };
  }
}
