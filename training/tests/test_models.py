"""Model forward-pass tests (requires torch + transformers)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from fieldlens.models import FieldLensSegFormer, count_parameters  # noqa: E402


@pytest.mark.parametrize(
    "head,fusion,channels,out_ch",
    [
        ("softmax", False, 3, 9),
        ("sigmoid", False, 3, 8),
        ("sigmoid", True, 4, 8),
    ],
)
def test_forward_shapes(head, fusion, channels, out_ch):
    # pretrained=False keeps the shape check offline; weights are irrelevant here.
    model = FieldLensSegFormer(
        head=head, use_fusion=fusion, num_classes=out_ch, pretrained=False
    )
    model.eval()
    x = torch.randn(2, channels, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, out_ch, 256, 256)
    info = count_parameters(model)
    assert info.parameters > 0
    assert info.size_mb > 0
    if fusion:
        assert model.last_gates is not None
        assert len(model.last_gates) == 4
    else:
        assert model.last_gates is None


def test_fusion_model_rejects_three_channel_input():
    model = FieldLensSegFormer(head="sigmoid", use_fusion=True, num_classes=8, pretrained=False)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 64, 64))


def test_build_model_reads_the_config():
    from fieldlens.models import build_model

    model = build_model(
        {
            "model": {
                "head": "sigmoid",
                "num_classes": 8,
                "fusion": True,
                "decoder_embed_dim": 64,
                "pretrained": False,
            }
        }
    )
    assert model.use_fusion is True
    assert model.decoder.pred.out_channels == 8
    assert model.decoder.pred.in_channels == 64
