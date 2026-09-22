# FieldLens

**Pilot disclaimer:** FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

FieldLens predicts crop-anomaly masks on aerial farm tiles (RGB + NIR) and can raise a tile-level alert with class name and coverage.

---

## Quick map

| Want to… | Go here |
|----------|---------|
| See the live UI | https://fieldlens-sem2-i2gx.vercel.app (or `dashboard/` → `npm run dev`) |
| See the code on GitHub | https://github.com/aksingh1608/fieldlens-sem2 |
| See report graphs | [`notebooks/figures/`](notebooks/figures/) |
| See page screenshots | [`docs/screenshots/`](docs/screenshots/) |
| Local image pack (gallery + graphs + screenshots) | `data/images/` (gitignored) |
| Train / eval / export | Section [Reproduce](#9-reproduce) |

---

## 1. What it is

Drone crop-health computer vision for **Agriculture-Vision 2021** tiles:

- **Input:** 512×512 RGB (and NIR for Run 3)
- **Output:** pixel masks for 8 anomaly classes + optional tile alert
- **Classes:** `double_plant`, `drydown`, `endrow`, `nutrient_deficiency`, `planter_skip`, `water`, `waterway`, `weed_cluster`

### Sample tiles (RGB + labels)

![Sample tiles](notebooks/figures/sample_tiles.png)

### Drone imaging conditions (augmentation examples)

![Drone conditions](notebooks/figures/drone_conditions.png)

---

## 2. Research questions and runs

1. When labels overlap, does a **multi-label sigmoid** head beat a **single-label softmax** head?
2. Does **gated RGB + NIR fusion** improve multi-label segmentation?

| Run | Input | Head | Idea |
|-----|-------|------|------|
| **Run 1** | RGB | Softmax (bg + 8 classes) | Single label per pixel (rarest class on overlap) |
| **Run 2** | RGB | Sigmoid multi-label | Independent channel per anomaly |
| **Run 3** | RGB + NIR | Sigmoid + gated fusion | Same as Run 2, plus NIR path |

---

## 3. Dataset

Official set: **94,986** labelled 512×512 tiles (RGB + NIR).

This pilot uses **2,700 tiles** (1,500 train / 400 val / 800 test) with whole-field separation.

| Profile | Tiles | Fields (plan) | Notes |
|---------|------:|---------------|-------|
| **pilot** (v1) | 2,700 | 31 / 15 / 18 | First completed runs; keep as a finding |
| **pilot_v2** | 2,700 | ~202 / 60 / 188 | Caps tiles per field for more field variety |
| **full** | official split | all | Needs bigger GPU; future work |

### Class distribution (pilot subset)

![Class distribution](notebooks/figures/class_distribution.png)

---

## 4. Method

- **Encoder:** SegFormer MiT-B0  
- **Decoder:** All-MLP  
- **Run 1:** softmax head  
- **Runs 2–3:** sigmoid multi-label heads  
- **Run 3:** NIR stem + gated fusion (AgriFusion-style component)

Details: dashboard **Method** page and `training/fieldlens/models.py`.

### Method page (dashboard)

![Method page](docs/screenshots/method_full.png)

---

## 5. Results (pilot v1 only)

Numbers from `runs/pilot/*/eval/metrics.json` and training logs. **Not** full-benchmark scores.

| Run | Pixel acc. | Modified mIoU | Tile-alert F1 (mean) | Peak train VRAM | Mean epoch | 10 epochs |
|-----|-----------:|--------------:|---------------------:|----------------:|-----------:|----------:|
| Run 1 | 0.578 | **0.144** | 0.277 | 1198 MB | 48.7 s | 487 s |
| Run 2 | 0.590 | **0.136** | 0.261 | 1194 MB | 56.3 s | 563 s |
| Run 3 | 0.435 | **0.129** | 0.267 | 1289 MB | 65.1 s | 651 s |

`pilot_v2` numbers: fill only after `runs/pilot_v2/*/eval/metrics.json` exists.

### Training curves

![Training curves](notebooks/figures/training_curves.png)

![Train loss](notebooks/figures/loss_curves.png)

![Val loss](notebooks/figures/val_loss_curves.png)

![Val mIoU](notebooks/figures/val_miou_curves.png)

![Val modified mIoU](notebooks/figures/miou_curves.png)

### Per-class IoU / F1

![Per-class IoU and F1](notebooks/figures/per_class_iou_f1.png)

### Precision–recall curves

![PR curves](notebooks/figures/pr_curves.png)

### Confusion matrices

| Run 1 | Run 2 | Run 3 |
|-------|-------|-------|
| ![Confusion run1](notebooks/figures/confusion_run1.png) | ![Confusion run2](notebooks/figures/confusion_run2.png) | ![Confusion run3](notebooks/figures/confusion_run3.png) |

### Efficiency snapshot

![Efficiency table](notebooks/figures/efficiency_table.png)

### Sample predictions vs ground truth

![Sample predictions](notebooks/figures/sample_predictions.png)

### Results page (dashboard)

![Results page](docs/screenshots/results_full.png)

---

## 6. Dashboard

Next.js app in `dashboard/`. Pages:

| Page | Purpose |
|------|---------|
| Overview | Project summary |
| Explorer | RGB / NIR tiles + GT or run overlays |
| Compare | One tile: GT vs Run 1 / 2 / 3 |
| Results | Metrics, PR curves, training curves |
| Method | Architecture, split, metrics notes |

### Overview

![Overview](docs/screenshots/overview_full.png)

### Explorer

![Explorer](docs/screenshots/explorer_full.png)

### Compare

![Compare](docs/screenshots/compare_full.png)

Local image pack (includes gallery webps): `data/images/`  
Regen figures: `python training/scripts/make_report_figures.py --profile pilot`  
Regen screenshots: with `npm run dev` running, `node dashboard/scripts/screenshot.mjs http://localhost:3000`

---

## 7. Hardware

Ubuntu laptop, **RTX 3050 4 GB**, Python 3.12. Times above are from pilot v1 on that class of machine.

---

## 8. Project layout

```
FieldLens/
├── training/          # train, evaluate, export, models
├── dashboard/         # Next.js pilot UI
├── notebooks/figures/ # report graphs (committed)
├── docs/screenshots/  # dashboard page captures (committed)
├── data/images/       # local pack: figures + gallery + shots (gitignored)
├── configs/           # run + profile YAMLs (under training/configs)
├── checkpoints/       # best.pt per profile/run (gitignored)
└── runs/              # logs + eval JSON (gitignored)
```

---

## 9. Reproduce

```bash
cd /home/aksingh/FieldLens
source .venv/bin/activate

# GPU check
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
python export.py --profile pilot_v2 --allow-images   # gallery + compare overlays

python scripts/make_report_figures.py --profile pilot_v2

cd ../dashboard && npm install && npm run dev
# optional screenshots:
# node scripts/screenshot.mjs http://localhost:3000
```

**Already completed for pilot v1:** train → eval → export (with images) → figures → dashboard screenshots.

---

## 10. Limitations and future work

- Class imbalance and rare classes hurt IoU.
- Robustness sweeps need `--robustness` on eval (empty on current pilot export).
- Overlap pixels are rare in small pilots.
- Full-scale training, Jetson-class notes, and hardening the public deploy remain open. Live UI: https://fieldlens-sem2-i2gx.vercel.app

---

## 11. Citations

- Chiu et al., Agriculture-Vision, CVPR 2020.
- Xie et al., SegFormer, NeurIPS 2021.
- Li, X., Qiao, L., and Yang, C. (2025). AgriFusion: Multiscale RGB-NIR Fusion for Semantic Segmentation in Airborne Agricultural Imagery. *AgriEngineering*, 7(11), 388. https://doi.org/10.3390/agriengineering7110388

---

## License

License unset. Dataset use follows Agriculture-Vision Workshop terms.
