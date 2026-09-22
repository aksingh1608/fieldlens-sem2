export interface ClassItem {
  id: string;
  label: string;
  color: string;
  color_rgb?: number[];
}

export interface ClassesJson {
  source: string;
  background: ClassItem;
  anomalies: ClassItem[];
  anomaly_ids: string[];
}

export interface SiteJson {
  enable_try_page: boolean;
  profile?: string;
  pilot_disclaimer?: string;
}
