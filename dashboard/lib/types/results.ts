export type ResultsStatus = "pending_runs" | "partial" | "complete";

export interface CurvePoint {
  x: number;
  y: number;
}

export interface PrCurveSeries {
  class_id: string;
  class_label: string;
  points: CurvePoint[] | null;
}

export interface TrainingCurveSeries {
  run_id: "run1" | "run2" | "run3";
  metric: string;
  epochs: number[] | null;
  values: number[] | null;
}

export interface RobustnessRow {
  condition: string;
  run1_miou: number | null;
  run2_miou: number | null;
  run3_miou: number | null;
}

export interface EfficiencyRow {
  run_id: "run1" | "run2" | "run3";
  params_m: number | null;
  train_sec_per_epoch: number | null;
  infer_ms_per_tile: number | null;
  vram_gb_peak: number | null;
}

export interface RunSummaryMetrics {
  pixel_miou: number | null;
  tile_alert_f1: number | null;
  modified_miou: number | null;
  notes: string | null;
}

export interface PerClassMetrics {
  class_id: string;
  class_label: string;
  iou: number | null;
  precision: number | null;
  recall: number | null;
}

export interface RunDetailMetrics extends RunSummaryMetrics {
  per_class: PerClassMetrics[] | null;
  metrics?: Record<string, unknown> | null;
  robustness?: unknown;
  efficiency?: unknown;
  pr_curves?: unknown;
  training_log?: unknown;
}

export interface TileCompareEntry {
  tile_id: string;
  gt_available: boolean;
  iou_run1: number | null;
  iou_run2: number | null;
  iou_run3: number | null;
}

export interface ResultsJson {
  status: ResultsStatus | string;
  generated_at?: string | null;
  profile?: string | null;
  dataset?: {
    name?: string | null;
    subset_tiles?: number | null;
    train_tiles?: number | null;
    val_tiles?: number | null;
    test_tiles?: number | null;
  } | null;
  dataset_stats?: unknown;
  classes?: unknown;
  runs: {
    run1?: RunDetailMetrics | null;
    run2?: RunDetailMetrics | null;
    run3?: RunDetailMetrics | null;
    [key: string]: unknown;
  };
  tile_compare?: TileCompareEntry[] | null;
  pr_curves?: {
    run2?: PrCurveSeries[] | null;
    run3?: PrCurveSeries[] | null;
  };
  robustness?: RobustnessRow[] | null;
  training_curves?: TrainingCurveSeries[] | null;
  efficiency?: EfficiencyRow[] | null;
}
