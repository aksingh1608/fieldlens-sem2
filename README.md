# FieldLens

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

## 1. What it is

FieldLens is a drone crop health vision system. It predicts pixel masks for field anomalies on aerial tiles and can raise a tile level alert with class name and coverage.

## 2. Research questions and runs

1. When labels overlap, does a multi label sigmoid head beat a single label softmax head?
2. Does gated RGB and NIR fusion improve multi label segmentation?

| Run | Input | Head |
|-----|-------|------|
| Run 1 | RGB | Softmax (background + 8 anomalies) |
| Run 2 | RGB | Sigmoid multi label (8 anomalies) |
| Run 3 | RGB + NIR gated fusion | Sigmoid multi label |

## 3. Dataset

Agriculture-Vision 2021 supervised split: 94,986 labelled 512 by 512 tiles with RGB and NIR, and eight anomaly classes (`double_plant`, `drydown`, `endrow`, `nutrient_deficiency`, `planter_skip`, `water`, `waterway`, `weed_cluster`).

This pilot uses **2,700 tiles** (1,500 train, 400 val, 800 test) selected by whole field separation. Profile **pilot_v2** caps tiles per field (10 / 10 / 5) so more fields enter each split (plan: 202 / 60 / 188 fields).

Profile **pilot** (v1) kept the same tile counts but only 31 / 15 / 18 fields, so class mixes differed across splits. Keep v1 under `runs/pilot/` as a finding, not as headline results.

The full official split needs a larger GPU and longer training. That is profile `full` (placeholders, untested) and is future work for the main research.

## 4. Method

SegFormer MiT-B0 encoder with an All-MLP decoder. Run 1 uses softmax. Runs 2 and 3 use sigmoid multi label heads. Run 3 adds a NIR stem and gated fusion (AgriFusion component). See the Method page architecture diagram and `training/fieldlens/models.py`.

## 5. Results (pilot v1 files only)

From `runs/pilot/*/eval/metrics.json` and `runs/pilot/*/log.csv`. Not benchmarks.

| Run | Test modified mIoU | Peak train VRAM (MB) | Mean epoch (s) | Total 10 epochs (s) |
|-----|-------------------:|---------------------:|---------------:|--------------------:|
| Run 1 | 0.1444 | 1198.1 | 48.7 | 487.2 |
| Run 2 | 0.1356 | 1194.1 | 56.3 | 562.8 |
| Run 3 | 0.1287 | 1288.7 | 65.1 | 651.2 |

pilot_v2 results: fill this table only after `runs/pilot_v2/*/eval/metrics.json` exists.

## 6. Key figures

After the report script or notebook runs, figures are in `notebooks/figures/` (sample tiles, drone conditions, class distribution, loss and accuracy curves, confusion, IoU/F1 bars, PR curves, robustness, efficiency, sample predictions, overlap examples). Embed those PNGs in the written report. Do not invent plots.

## 7. Dashboard

Static Next.js site under `dashboard/`. Screenshots: `docs/screenshots/` after the Playwright script. Live link: DOMAIN_TBD.

## 8. Hardware

Ubuntu laptop, RTX 3050 4 GB, Python 3.12. Training times above are from pilot v1 logs on that class of machine.

## 9. Reproduce

```bash
cd /home/aksingh/FieldLens
source .venv/bin/activate

# Setup check
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
PY

cd training
python scripts/download_subset.py --profile pilot_v2 --mode extract
python scripts/compute_stats.py --profile pilot_v2

python train.py --config configs/run1.yaml --profile pilot_v2
python train.py --config configs/run2.yaml --profile pilot_v2
python train.py --config configs/run3.yaml --profile pilot_v2

python evaluate.py --config configs/run1.yaml --profile pilot_v2 --checkpoint ../checkpoints/pilot_v2/run1/best.pt --robustness
python evaluate.py --config configs/run2.yaml --profile pilot_v2 --checkpoint ../checkpoints/pilot_v2/run2/best.pt --robustness
python evaluate.py --config configs/run3.yaml --profile pilot_v2 --checkpoint ../checkpoints/pilot_v2/run3/best.pt --robustness

python export.py --profile pilot_v2
python export.py --profile pilot_v2 --allow-images

python scripts/make_report_figures.py --profile pilot_v2
# or: cd ../notebooks && FIELDLENS_PROFILE=pilot_v2 jupyter nbconvert --to notebook --execute report_figures.ipynb --inplace

cd ../dashboard && npm install && npm run dev
# screenshots (dev server running):
# npm install -D playwright && npx playwright install chromium
# node scripts/screenshot.mjs http://localhost:3000
```

## 10. Limitations and future work

Class imbalance and rare classes hurt IoU. Motion blur and other drone conditions are tested with severity sweeps. Overlap pixels are rare in small pilots. Full scale training, Jetson class deployment notes, and a public domain (DOMAIN_TBD) remain open.

## 11. Citations

- Chiu et al., Agriculture-Vision, CVPR 2020.
- Xie et al., SegFormer, NeurIPS 2021.
- Li, X., Qiao, L., and Yang, C. (2025). AgriFusion: Multiscale RGB-NIR Fusion for Semantic Segmentation in Airborne Agricultural Imagery. AgriEngineering, 7(11), 388. https://doi.org/10.3390/agriengineering7110388

## License

License unset. Dataset use follows Agriculture-Vision Workshop terms.
