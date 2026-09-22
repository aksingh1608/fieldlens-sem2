"""SegFormer-style models for FieldLens runs 1-3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel


@dataclass
class ModelInfo:
    parameters: int
    size_mb: float


def count_parameters(model: nn.Module) -> ModelInfo:
    n = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    return ModelInfo(parameters=n, size_mb=size_mb)


class AllMLPDecoder(nn.Module):
    """All-MLP decoder as in SegFormer: project, upsample to stride-4, concat, fuse, predict."""

    def __init__(self, in_channels: list[int], embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        target = features[0].shape[-2:]  # stride-4 spatial size
        ups: list[torch.Tensor] = []
        for proj, feat in zip(self.projs, features):
            x = proj(feat)
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
            ups.append(x)
        x = torch.cat(ups, dim=1)
        x = self.fuse(x)
        x = self.pred(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class NirStem(nn.Module):
    """Small CNN stem: 1-channel NIR -> 4 feature maps matching MiT-B0 strides/channels."""

    def __init__(self, out_channels: list[int]) -> None:
        super().__init__()
        # Progressive downsampling: /4, /8, /16, /32
        c1, c2, c3, c4 = out_channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, c1 // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 2, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

    def forward(self, nir: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(nir)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class GatedFusion(nn.Module):
    """Per-scale gate: g = sigmoid(conv1x1(cat(Frgb, Fnir))); F = g*Frgb + (1-g)*Fnir."""

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.gates = nn.ModuleList([nn.Conv2d(2 * c, c, kernel_size=1) for c in channels])

    def forward(
        self, rgb_feats: list[torch.Tensor], nir_feats: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        fused: list[torch.Tensor] = []
        gate_maps: list[torch.Tensor] = []
        for gate, fr, fn in zip(self.gates, rgb_feats, nir_feats):
            if fr.shape[-2:] != fn.shape[-2:]:
                fn = F.interpolate(fn, size=fr.shape[-2:], mode="bilinear", align_corners=False)
            g = torch.sigmoid(gate(torch.cat([fr, fn], dim=1)))
            fused.append(g * fr + (1.0 - g) * fn)
            gate_maps.append(g)
        return fused, gate_maps


class FieldLensSegFormer(nn.Module):
    def __init__(
        self,
        head: str = "softmax",
        num_classes: int | None = None,
        use_fusion: bool = False,
        encoder_name: str = "nvidia/mit-b0",
        decoder_embed_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.head = head
        self.use_fusion = use_fusion
        if num_classes is None:
            num_classes = 9 if head == "softmax" else 8
        self.num_classes = num_classes

        if pretrained:
            self.encoder = SegformerModel.from_pretrained(encoder_name)
        else:
            # Randomly initialized MiT-B0-shaped encoder: for tests and offline
            # shape checks, never for a real run.
            self.encoder = SegformerModel(SegformerConfig())
        cfg: SegformerConfig = self.encoder.config
        # hidden_sizes: channels at each encoder stage
        self.in_channels = list(cfg.hidden_sizes)
        self.encoder_name = encoder_name

        if use_fusion:
            self.nir_stem = NirStem(self.in_channels)
            self.fusion = GatedFusion(self.in_channels)
        else:
            self.nir_stem = None
            self.fusion = None

        self.decoder = AllMLPDecoder(self.in_channels, decoder_embed_dim, num_classes)
        self.last_gates: list[torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W) for RGB runs, or (B,4,H,W) for Run 3 (RGB+NIR).
        """
        if self.use_fusion:
            if x.shape[1] != 4:
                raise ValueError(f"Run 3 expects 4 channels, got {x.shape[1]}")
            rgb = x[:, :3]
            nir = x[:, 3:4]
        else:
            rgb = x[:, :3]
            nir = None

        out = self.encoder(pixel_values=rgb, output_hidden_states=True)
        # hidden_states: embedding + 4 stages; take last 4 stage outputs
        hidden = out.hidden_states
        rgb_feats = list(hidden[-4:])

        if self.use_fusion:
            assert self.nir_stem is not None and self.fusion is not None and nir is not None
            nir_feats = self.nir_stem(nir)
            feats, gates = self.fusion(rgb_feats, nir_feats)
            self.last_gates = gates
        else:
            feats = rgb_feats
            self.last_gates = None

        h, w = x.shape[-2:]
        return self.decoder(feats, out_size=(h, w))


def build_model(run_cfg: dict) -> FieldLensSegFormer:
    model_cfg = run_cfg["model"]
    return FieldLensSegFormer(
        head=model_cfg["head"],
        num_classes=model_cfg.get("num_classes"),
        use_fusion=bool(model_cfg.get("fusion", False)),
        encoder_name=model_cfg.get("encoder", "nvidia/mit-b0"),
        decoder_embed_dim=int(model_cfg.get("decoder_embed_dim", 256)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )
