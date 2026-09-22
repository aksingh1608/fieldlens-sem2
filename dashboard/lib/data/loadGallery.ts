import type { GalleryIndex, GalleryTile } from "@/lib/types/gallery";

type RawGalleryTile = Partial<GalleryTile> & {
  tile_id?: string;
  path?: string;
  predictions?: GalleryTile["predictions"] | null;
};

function normalizeTile(raw: RawGalleryTile): GalleryTile | null {
  const id = raw.id ?? raw.tile_id;
  if (!id) return null;

  const basePath = raw.path
    ? `/data/${raw.path.replace(/^\/?data\//, "")}`
    : `/data/gallery/${id}`;

  const preds = raw.predictions ?? null;

  return {
    id,
    label: raw.label ?? id,
    rgb_url: raw.rgb_url ?? `${basePath}/rgb.webp`,
    nir_url: raw.nir_url ?? `${basePath}/nir.webp`,
    gt_url: raw.gt_url ?? `${basePath}/gt.webp`,
    predictions: {
      run1: preds?.run1 ?? null,
      run2: preds?.run2 ?? null,
      run3: preds?.run3 ?? null,
    },
    alert: raw.alert ?? null,
  };
}

export async function loadGalleryIndex(): Promise<GalleryIndex> {
  const res = await fetch("/data/gallery/index.json", { cache: "no-store" });
  if (!res.ok) {
    throw new Error("Failed to load gallery index");
  }
  const data = (await res.json()) as { tiles?: RawGalleryTile[] };
  const tiles = (data.tiles ?? [])
    .map(normalizeTile)
    .filter((t): t is GalleryTile => t !== null);
  return { tiles };
}
