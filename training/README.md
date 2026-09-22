# FieldLens training

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

Python side of FieldLens: dataset loading, models, losses, metrics, training, evaluation, and export.

## Layout

- `configs/` shared data.yaml, run YAMLs, eval.yaml, and `profiles/pilot.yaml` plus `profiles/full.yaml`
- `fieldlens/` package (dataset, models, losses, metrics, transforms, paths, profile)
- `scripts/` bucket inspect, subset download, stats, previews, ONNX export
- `train.py`, `evaluate.py`, `export.py` entry points
- `tests/` unit tests
- `requirements.txt` unpinned names; pin from the real environment after install

## Setup

Use a repo-local virtualenv. Do not install packages globally.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Profiles

Every long script takes `--profile` (default `pilot`). Scale fields live only in the profile YAMLs. Nothing else branches on scale.

- Pilot: 1500 / 400 / 800, MiT-B0, 10 epochs, RTX 3050 4 GB
- Full: official split sizes; encoder and schedule are placeholders and untested

Data: `data/fieldlens/<profile>/`
Runs: `runs/<profile>/<run_name>/`
Checkpoints: `checkpoints/<profile>/<run_name>/`

## Paths

Repo-relative paths resolve against the FieldLens repo root (the parent of `training/`).
Absolute paths in a config are used as given. Set `FIELDLENS_ROOT` to keep data and
outputs somewhere other than the checkout:

```bash
export FIELDLENS_ROOT=/mnt/fieldlens
```

## Data

```bash
python scripts/download_subset.py --profile pilot --mode plan
python scripts/download_subset.py --profile pilot --mode smoke5
python scripts/download_subset.py --profile pilot --mode extract
python scripts/compute_stats.py --profile pilot
```

## Train

```bash
python train.py --config configs/run1.yaml --profile pilot --smoke
python train.py --config configs/run1.yaml --profile pilot
python train.py --config configs/run1.yaml --profile pilot --benchmark
python train.py --config configs/run1.yaml --profile pilot --resume
```

- run1: softmax head, 9 classes, cross-entropy, RGB
- run2: sigmoid head, 8 classes, BCE + Dice, RGB
- run3: run2 plus a NIR stem and per-scale gated fusion, RGB+NIR

## Evaluate

```bash
python evaluate.py --config configs/run1.yaml --profile pilot --checkpoint ../checkpoints/pilot/run1/best.pt
python evaluate.py --config configs/run1.yaml --profile pilot --checkpoint ../checkpoints/pilot/run1/best.pt --robustness
```

## Export

```bash
python export.py --profile pilot
python export.py --profile pilot --allow-images
```

Export also writes `dashboard/public/data/classes.json` from `configs/data.yaml`.
`--allow-images` writes dataset-derived tiles into `dashboard/public/data/`. Publishing
those is redistribution; only pass it after the terms question has been settled.
