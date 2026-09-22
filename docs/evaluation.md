# Evaluation

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

## Official modified mIoU (what we implement)

Source: [SHI-Labs/Agriculture-Vision](https://github.com/SHI-Labs/Agriculture-Vision) Evaluation section.

For each valid pixel, prediction `x` and ground-truth label set `Y` (`Y = {background}` if no anomalies, else all present anomaly ids):

1. If `x` is in `Y`: add 1 to `M[y, y]` for **each** `y` in `Y`.
2. Otherwise: add 1 to `M[x, y]` for **each** `y` in `Y`.

IoU for class `c` = `M[c,c] / (row_sum[c] + col_sum[c] - M[c,c])`. mIoU averages over all classes including background.

### Difference from the FieldLens build-prompt draft

The build prompt said:

- correct: `M[x][x] += 1` only
- incorrect: `M[y][x] += 1` for each `y` in `Y` (gt rows, pred cols)

Official SHI-Labs says:

- correct: `M[y,y] += 1` for **every** ground-truth label in `Y`
- incorrect: `M[x,y] += 1` for each `y` in `Y`

We follow the **official** rule. Hand tests live in `training/tests/test_metrics.py`.

## How each run picks one class per pixel

- Run 1: argmax over 9 softmax channels.
- Runs 2 and 3: highest-scoring anomaly if sigmoid score >= threshold (default 0.5), else background.

## Multi-label metrics

Per-class IoU, precision, recall, F1 on the 8 binary masks (Run 1 converted to one-hot). The same metrics are also computed only on pixels with 2+ labels (overlap slice) to test research question 1.

## Tile-level

A tile is positive for a class if that class covers at least `tile_positive_fraction` of valid pixels (default 1%, `configs/eval.yaml`). Same rule for GT and prediction.

Tile score for PR curves: **`covered_fraction`** by default (fraction of valid pixels predicted positive for that class). Optional `mean_probability` for sigmoid runs. Documented in eval outputs as `tile_score_mode`.

## Robustness

Severities 0-4 apply motion blur, downscale-upscale, brightness shift, and Gaussian noise (same params on RGB and NIR). Plot modified mIoU vs severity.

## Efficiency

Parameter count, size MB, mean inference seconds per tile on GPU and CPU (warmup then 50 tiles, CUDA synchronize), peak VRAM.

## Commands

Pass `--profile pilot` (default) or `--profile full`. Checkpoints live under `checkpoints/<profile>/<run>/`.

```bash
cd /home/aksingh/FieldLens
source .venv/bin/activate
cd training
python evaluate.py --config configs/run1.yaml --profile pilot --checkpoint ../checkpoints/pilot/run1/best.pt --robustness
python evaluate.py --config configs/run2.yaml --profile pilot --checkpoint ../checkpoints/pilot/run2/best.pt --robustness
python evaluate.py --config configs/run3.yaml --profile pilot --checkpoint ../checkpoints/pilot/run3/best.pt --robustness
```
