#!/usr/bin/env python3
"""Export best checkpoint to ONNX and compare vs PyTorch on N tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fieldlens.dataset import FieldLensDataset  # noqa: E402
from fieldlens.models import build_model  # noqa: E402
from fieldlens.paths import REPO_ROOT, repo_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "dashboard" / "public" / "models" / "fieldlens.onnx"))
    parser.add_argument("--tiles", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = cfg["data"]
    ds = FieldLensDataset(
        subset_root=repo_path(data["subset_root"]),
        split_csv=repo_path(data["split_csv"]),
        our_split="test",
        stats_json=repo_path(data["stats_json"]),
        channels=data["channels"],
        augment=False,
    )
    ch = 4 if data["channels"] == "rgb_nir" else 3
    dummy = torch.randn(1, ch, 512, 512)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Wrote {out_path}")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    for i in range(min(args.tiles, len(ds))):
        x = ds[i]["image"].unsqueeze(0)
        with torch.no_grad():
            pt = model(x).numpy()
        ort_out = sess.run(None, {"image": x.numpy()})[0]
        max_diff = float(np.max(np.abs(pt - ort_out)))
        print(f"tile={ds[i]['tile_id']} max_abs_diff={max_diff}")
        if max_diff > args.atol:
            raise SystemExit(f"ONNX mismatch above atol={args.atol}")
    print("ONNX check OK")


if __name__ == "__main__":
    main()
