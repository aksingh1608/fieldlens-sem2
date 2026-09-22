# Training

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

## Profiles

Pass `--profile pilot_v2` (recommended) or `--profile pilot` (v1 comparison) or `--profile full` (untested).

- pilot_v2: `configs/profiles/pilot_v2.yaml` (1500/400/800, max_tiles_per_field 10/10/5, MiT-B0, 10 epochs)
- pilot: `configs/profiles/pilot.yaml` (same sizes, few fields; keep for comparison)
- Full: `configs/profiles/full.yaml` (official split sizes; placeholders; untested)

Checkpoints: `checkpoints/<profile>/<run_name>/`
Logs: `runs/<profile>/<run_name>/`

## Commands

From repo root, with `.venv` active:

```bash
cd /home/aksingh/FieldLens
source .venv/bin/activate
cd training
```

Smoke (2 epochs, 20 tiles):

```bash
python train.py --config configs/run1.yaml --profile pilot --smoke
```

Benchmark (100 steps, then exit):

```bash
python train.py --config configs/run1.yaml --profile pilot --benchmark
python train.py --config configs/run2.yaml --profile pilot --benchmark
python train.py --config configs/run3.yaml --profile pilot --benchmark
```

Full pilot runs:

```bash
python train.py --config configs/run1.yaml --profile pilot
python train.py --config configs/run2.yaml --profile pilot
python train.py --config configs/run3.yaml --profile pilot
```

Resume:

```bash
python train.py --config configs/run1.yaml --profile pilot --resume
```

## Optimiser

- AdamW
- Encoder LR 6e-5, decoder/heads/fusion LR 6e-4
- Weight decay 0.01
- Polynomial decay with short linear warmup
- fp16 AMP + GradScaler
- Optional gradient accumulation (`optim.accum_steps`)
- Default 20 epochs
- Checkpoints: `checkpoints/<run_name>/last.pt`, `best.pt` (best val modified mIoU), per-epoch files

## log.csv columns

| Column | Meaning |
|--------|---------|
| epoch | 0-based epoch index |
| train_loss | mean training loss |
| val_loss | mean validation loss |
| val_miou | validation modified mIoU (official Agriculture-Vision rule) |
| lr_encoder | encoder learning rate this epoch |
| lr_other | decoder/heads/fusion learning rate |
| epoch_sec | wall time for the epoch |
| peak_vram_mb | peak CUDA memory this epoch (0 on CPU) |

TensorBoard logs: `runs/<run_name>/tb/`.

## OOM

If CUDA OOM, the trainer prints a message suggesting a smaller batch plus higher `accum_steps`, then exits.
