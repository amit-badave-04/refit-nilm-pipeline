"""NILMFormer port: attention equivalence with the original formulation, shapes, calendar features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from refit_nilm.models.nilmformer import DiagonallyMaskedSelfAttention, NILMFormer
from refit_nilm.train_seq2seq import calendar_features


def _reference_attention(att: DiagonallyMaskedSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """The original repository's einsum formulation (eval mode, so dropout is inactive)."""
    b, n, _ = x.shape
    h, e = att.n_heads, att.head_dim
    q, k, v = (w(x).view(b, n, h, e) for w in (att.wq, att.wk, att.wv))
    mask = torch.diag(torch.ones(n, dtype=torch.bool)).repeat(b, 1, 1, 1)
    scores = torch.einsum("blhe,bshe->bhls", q, k)
    attn = torch.softmax(e**-0.5 * scores.masked_fill_(mask, torch.finfo(scores.dtype).min), dim=-1)
    out = torch.einsum("bhls,bshd->blhd", attn, v)
    return att.wo(out.reshape(b, n, -1))


def test_attention_matches_original_formulation():
    torch.manual_seed(0)
    att = DiagonallyMaskedSelfAttention(dim=96, n_heads=8, dropout=0.2).eval()
    x = torch.randn(3, 17, 96)
    with torch.no_grad():
        assert torch.allclose(att(x), _reference_attention(att, x), atol=1e-5)


def test_forward_shape_and_size():
    model = NILMFormer().eval()
    x = torch.randn(4, 9, 129)
    with torch.no_grad():
        out = model(x)["power"]
    assert out.shape == (4, 129)
    n_params = sum(p.numel() for p in model.parameters())
    assert 300_000 < n_params < 450_000  # ~0.38 M in the paper's configuration


def test_calendar_features_use_uk_local_time():
    # 2014-07-01 12:30 UTC is 13:30 BST; separators (-1) are zero
    t = np.array([pd.Timestamp("2014-07-01 12:30", tz="UTC").value // 10**9, -1])
    f = calendar_features(t)
    assert f.shape == (8, 2)
    assert np.isclose(f[2, 0], np.sin(2 * np.pi * 13 / 24)) and np.isclose(f[3, 0], np.cos(2 * np.pi * 13 / 24))
    assert np.all(f[:, 1] == 0)
