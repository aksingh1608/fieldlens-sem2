# FieldLens

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

## What it is

FieldLens is a computer vision pilot for crop health from drone imagery. Given an RGB tile (and NIR when used), it predicts pixel masks for field anomalies and can raise a tile level alert with class name and coverage.

It supports an MSc robot vision assignment and a small CVPR style research pilot on multi label heads and RGB–NIR fusion.

## Problem

Farmers need early detection of planting and field problems across large areas. Manual scouting does not scale. Agriculture-Vision provides labelled aerial tiles, but overlapping anomaly labels make single label softmax a poor fit. Near infrared (NIR) may add signal for vegetation stress.

## Research questions

1. When labels overlap, does a multi label sigmoid head beat a single label softmax head?
2. Does gated RGB–NIR fusion improve multi label segmentation on top of that?

## Three runs

| Run | Input | Head |
|-----|-------|------|
| Run 1 | RGB | Softmax (background + 8 anomalies) |
| Run 2 | RGB | Sigmoid multi label (8 anomalies) |
| Run 3 | RGB + NIR gated fusion | Sigmoid multi label |

Encoder: SegFormer MiT-B0 for the pilot profiles.

## Dataset

Agriculture-Vision 2021 supervised split: 94,986 labelled 512×512 tiles with RGB and NIR, and eight anomaly classes (`double_plant`, `drydown`, `endrow`, `nutrient_deficiency`, `planter_skip`, `water`, `waterway`, `weed_cluster`).

This pilot uses **2,700 tiles** (1,500 train, 400 val, 800 test), selected by whole field separation so no field id appears in two splits.

**pilot_v2** (recommended) adds `max_tiles_per_field` (10 / 10 / 5) so more fields enter each split and class mixes stay closer. A plan under `pilot_v2` used 202 / 60 / 188 fields for train / val / test.

**pilot** (v1) kept the same tile counts but few fields (31 / 15 / 18). That skewed class mixes (for example drydown tile counts 80 / 1 / 566). Keep v1 outputs under `runs/pilot/` and `data/fieldlens/pilot/` as a finding, not as headline results. Prefer `pilot_v2` for new training.

The full official split (56,944 / 18,334 / 19,708) needs a larger GPU and longer schedule. That is profile `full` (placeholders, untested) and is future work.

## Results (pilot v1 only)

Filled only from files under `runs/pilot/*/eval/metrics.json` and `runs/pilot/*/log.csv`. These are pilot trends on the v1 field mix, **not** benchmarks. Re-run under `--profile pilot_v2` before treating numbers as the main result.

| Run | Test modified mIoU | Peak train VRAM (MB) | Mean epoch time (s) | Total train time (s, 10 epochs) |
|-----|-------------------:|---------------------:|--------------------:|--------------------------------:|
| Run 1 | 0.1444 | 1198.1 | 48.7 | 487.2 |
| Run 2 | 0.1356 | 1194.1 | 56.3 | 562.8 |
| Run 3 | 0.1287 | 1288.7 | 65.1 | 651.2 |

Sources: `runs/pilot/run*/eval/metrics.json` (`modified_miou`) and `runs/pilot/run*/log.csv` (full epochs with `epoch_sec` > 10). Pixel accuracy is added in the current `evaluate.py` for future exports; v1 metric files do not contain it.

pilot_v2 results: not trained yet. Leave this table empty until eval files exist under `runs/pilot_v2/`.

## Key figures

After you run the report notebook, figures land in `notebooks/figures/`:

- `sample_tiles.png`
- `drone_conditions.png`
- `class_distribution.png`
- `training_curves.png`
- `confusion_run1.png` / `run2` / `run3`
- `per_class_iou_f1.png`
- `pr_curves.png`
- `robustness.png`
- `efficiency_table.png`
- `sample_predictions.png`

Embed them in the report when present. Do not invent plots.

## Hardware

- Pilot target machine: Ubuntu laptop, RTX 3050 4 GB, Python 3.12
- Pilot v1 training used CUDA (logs show `device=cuda` and peak VRAM above)

## How to reproduce

Always pass `--profile` (`pilot_v2` recommended). Default remains `pilot` for backward compatibility.

```bash
cd /home/aksingh/FieldLens
source .venv/bin/activate
cd training

# 1) Plan + extract + stats
python scripts/download_subset.py --profile pilot_v2 --mode plan
python scripts/download_subset.py --profile pilot_v2 --mode extract
python scripts/compute_stats.py --profile pilot_v2

# 2) Train
python train.py --config configs/run1.yaml --profile pilot_v2
python train.py --config configs/run2.yaml --profile pilot_v2
python train.py --config configs/run3.yaml --profile pilot_v2

# 3) Evaluate (include robustness)
python evaluate.py --config configs/run1.yaml --profile pilot_v2 --checkpoint ../checkpoints/pilot_v2/run1/best.pt --robustness
python evaluate.py --config configs/run2.yaml --profile pilot_v2 --checkpoint ../checkpoints/pilot_v2/run2/best.pt --robustness
python evaluate.py --config configs/run3.yaml --profile pilot_v2 --checkpoint ../checkpoints/pilot_v2/run3/best.pt --robustness

# 4) Export dashboard JSON + gallery overlays (images only if terms allow)
python export.py --profile pilot_v2
python export.py --profile pilot_v2 --allow-images

# 5) Report figures
cd ../notebooks
FIELDLENS_PROFILE=pilot_v2 jupyter nbconvert --to notebook --execute report_figures.ipynb --output report_figures_executed.ipynb

# 6) Dashboard
cd ../dashboard && npm install && npm run dev
```

Outputs stay under `data/fieldlens/<profile>/`, `runs/<profile>/`, `checkpoints/<profile>/` so profiles never mix.

## Citations

- Chiu et al., Agriculture-Vision, CVPR 2020.
- Xie et al., SegFormer, NeurIPS 2021.
- Li, X., Qiao, L., and Yang, C. (2025). AgriFusion: Multiscale RGB-NIR Fusion for Semantic Segmentation in Airborne Agricultural Imagery. AgriEngineering, 7(11), 388. https://doi.org/10.3390/agriengineering7110388 (fusion component only).

## License

License unset. Dataset use is governed by Agriculture-Vision Workshop terms. Do not publish gallery tiles without permission.
