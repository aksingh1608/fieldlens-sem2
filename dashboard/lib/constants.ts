export const PILOT_DISCLAIMER =
  "FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.";

export const GITHUB_URL = "https://github.com/aksingh1608/fieldlens-sem2";

export const DOMAIN_PLACEHOLDER = "https://fieldlens-sem2-i2gx.vercel.app";

export const RUNS = [
  { id: "run1" as const, name: "Run 1", input: "RGB", head: "Softmax (9 classes)" },
  { id: "run2" as const, name: "Run 2", input: "RGB", head: "Sigmoid multi-label (8 anomalies)" },
  { id: "run3" as const, name: "Run 3", input: "RGB + NIR gated fusion", head: "Sigmoid multi-label (8 anomalies)" },
];

export const DATA_PATHS = {
  results: "/data/results.json",
  galleryIndex: "/data/gallery/index.json",
  classes: "/data/classes.json",
  site: "/data/site.json",
  model: "/models/fieldlens.onnx",
} as const;

/** Forbidden 2020 challenge names. Must never appear in UI class lists. */
export const FORBIDDEN_LEGACY_CLASS_IDS = [
  "cloud_shadow",
  "standing_water",
  "weed_strip",
] as const;
