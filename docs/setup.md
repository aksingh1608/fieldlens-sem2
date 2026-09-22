# Setup

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

## Hardware (this machine)

| Item | Value |
|------|-------|
| OS | Ubuntu (linux) |
| Python | 3.12.10 |
| NVIDIA driver (from `/proc/driver/nvidia/version`) | 595.84 |
| `nvidia-smi` | Failed in this environment: could not communicate with the NVIDIA driver |
| `/dev/nvidia*` | Not present here (GPU device nodes missing) |

Interpretation: the driver package is installed, but the GPU is not usable in the current session (no device nodes). On your RTX 3050 laptop with a working driver, install the CUDA build of PyTorch below. If CUDA is still unavailable after install, fall back to the CPU wheel and tell the project maintainer.

Driver 595.x supports current CUDA 12.x PyTorch wheels. Selected install: PyTorch stable with CUDA 12.4 index (official selector).

## Virtualenv

Always use the repo `.venv`. Never install packages globally.

## Packages

torch, torchvision, transformers, numpy, pillow, opencv-python-headless, scikit-learn, matplotlib, pyyaml, tqdm, boto3, tensorboard, onnx, onnxruntime, pytest

`training/requirements.txt` is filled after install from `pip freeze` (real versions only).

## After install: short check

```bash
cd /home/aksingh/FieldLens
source .venv/bin/activate
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print("vram_free_mb", round(free / 1024**2, 1))
    print("vram_total_mb", round(total / 1024**2, 1))
else:
    print("gpu", None)
PY
```

Paste the output back into the project notes.
