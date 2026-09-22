# FieldLens agenda

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

Status values: `not started` | `in progress` | `done` | `blocked (AK)`  
Owner values: `dev` | `AK`

Last updated: pilot_v2 profile, legacy move, git init, export overlays, report notebook

---

## FIX PASS (dashboard + profiles)

| Task | Status | Owner |
|------|--------|-------|
| Remove 2020 class names; classes from data.yaml -> classes.json | done | dev |
| Results page empty/absent results.json safe | done | dev |
| Hide /try behind site.json enable_try_page (default false) | done | dev |
| AgriFusion citation (Li, Qiao, Yang, AgriEngineering 2025) | done | dev |
| configs/profiles/pilot.yaml and full.yaml + --profile on scripts | done | dev |
| Class sync + profile + empty-results tests | done | dev |
| Re-extract/train under pilot profile (1500/400/800) | done | AK |
| pilot_v2 max_tiles_per_field + plan (202/60/188 fields) | done | dev |
| Extract/train/eval/export under pilot_v2 | not started | AK |

---

## PHASE 0: Repo skeleton and agenda

| Task | Status | Owner |
|------|--------|-------|
| Create folder layout with placeholders | done | dev |
| Write `docs/agenda.md` checklist | done | dev |
| Write first version of root `README.md` | done | dev |
| Write `.gitignore` | done | dev |
| AK git init/commit | blocked (AK) | AK |

---

## PHASE 1: Environment

| Task | Status | Owner |
|------|--------|-------|
| Run `nvidia-smi`, read CUDA driver version | done | dev |
| Pick matching PyTorch install command | done | dev |
| Create venv and install packages | blocked (AK) | AK |
| Short torch/CUDA/VRAM check | blocked (AK) | dev (after AK install) |
| Generate `training/requirements.txt` from real installs | blocked (AK) | dev (after AK install) |
| Write `docs/setup.md` | done | dev |

---

## PHASE 2: Inspect the dataset

| Task | Status | Owner |
|------|--------|-------|
| Write `training/scripts/inspect_bucket.py` | done | dev |
| Confirm layout (unsigned S3 list) | done | dev |
| Check free disk space | done | dev (~226 GB free on `/`) |
| Write `docs/data.md` from confirmed layout | done | dev |

---

## PHASE 3: Download the subset and build the split

| Task | Status | Owner |
|------|--------|-------|
| Write `download_subset.py` + `configs/data.yaml` | done | dev |
| Write `compute_stats.py` | done | dev |
| Smoke-test download on 5 tiles | blocked (AK) | dev/AK (needs archive) |
| Download archive + extract subset | blocked (AK) | AK |
| Run stats script | blocked (AK) | AK |
| Update `docs/data.md` with stats summary | blocked (AK) | dev (after paste) |

---

## PHASE 4: Dataset and data loader

| Task | Status | Owner |
|------|--------|-------|
| Write `dataset.py` | done | dev |
| Write `transforms_drone.py` | done | dev |
| DataLoader defaults + `eval.yaml` severity params | done | dev |
| Dataset/metric unit tests | done | dev (metrics 5/5; dataset/model need torch install) |
| Preview script | done | dev (run after subset exists) |
| Document singlelabel rule in `docs/data.md` | done | dev |

---

## PHASE 5: Models

| Task | Status | Owner |
|------|--------|-------|
| MiT-B0 encoder + All-MLP decoder + heads + gated fusion | done | dev |
| Parameter count helper | done | dev |
| Forward-pass tests | blocked (AK) | dev (needs torch/transformers) |

---

## PHASE 6: Losses and training

| Task | Status | Owner |
|------|--------|-------|
| Losses + `train.py` + run configs | done | dev |
| Smoke / benchmark / full train | blocked (AK) | AK |
| Write `docs/training.md` | done | dev |

---

## PHASE 7: Evaluation

| Task | Status | Owner |
|------|--------|-------|
| Confirm official Agriculture-Vision mIoU rule | done | dev (diff from prompt draft; official used) |
| Write `evaluate.py` | done | dev |
| Hand-worked modified mIoU unit tests | done | dev (5 passed) |
| Full eval and robustness | blocked (AK) | AK |
| Write `docs/evaluation.md` | done | dev |

---

## PHASE 8: Export for the dashboard

| Task | Status | Owner |
|------|--------|-------|
| Read terms re: public tiles | done | dev |
| AK decision on publishing gallery images | blocked (AK) | AK |
| Write `export.py` | done | dev |
| Run full export | blocked (AK) | AK |

---

## PHASE 9: Dashboard

| Task | Status | Owner |
|------|--------|-------|
| Next.js scaffold + pages + favicon + stubs | done | dev |
| `npm install` / deploy | blocked (AK) | AK |
| `dashboard/README.md` + `docs/deploy.md` | done | dev |

---

## PHASE 10: In-browser model

| Task | Status | Owner |
|------|--------|-------|
| ONNX export script + `/try` page scaffold | done | dev |
| Run ONNX export after best checkpoint exists | blocked (AK) | AK |

---

## PHASE 11: Docs and report support

| Task | Status | Owner |
|------|--------|-------|
| Root README with pending results table | done | dev |
| `notebooks/report_figures.ipynb` | done | dev |
| `docs/report_mapping.md` | done | dev |
| Final agenda pass | done | dev |
| Fill results from real `results.json` | blocked (AK) | dev (after runs) |
