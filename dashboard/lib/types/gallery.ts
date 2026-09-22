export interface GalleryTilePredictions {
  run1: string | null;
  run2: string | null;
  run3: string | null;
}

export interface GalleryTileAlert {
  class_id: string;
  class_label: string;
  coverage_percent: number | null;
}

export interface GalleryTile {
  id: string;
  label: string | null;
  rgb_url: string;
  nir_url: string | null;
  gt_url: string | null;
  predictions: GalleryTilePredictions;
  alert: GalleryTileAlert | null;
}

export interface GalleryIndex {
  tiles: GalleryTile[];
}
