"""NILMFormer port: attention equivalence, the whole pass pinned to verified values, shapes, calendar features."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from refit_nilm.models.nilmformer import DiagonallyMaskedSelfAttention, NILMFormer
from refit_nilm.train_seq2seq import calendar_features

from _golden import GOLDEN, compute


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


def test_whole_forward_and_backward_pass_matches_the_verified_port():
    """Pinned to values recorded when the port matched the original code (scripts/make_nilmformer_golden.py)."""
    golden = json.loads(GOLDEN.read_text())
    now = compute()
    np.testing.assert_allclose(now["output"], golden["output"], rtol=1e-4, atol=1e-6)
    assert now["grad_norms"].keys() == golden["grad_norms"].keys()
    for name, value in golden["grad_norms"].items():
        assert now["grad_norms"][name] == pytest.approx(value, rel=1e-4, abs=1e-7), name
    np.testing.assert_allclose(now["input_grad_norm_per_channel"], golden["input_grad_norm_per_channel"], rtol=1e-4, atol=1e-7)


def test_calendar_features_use_uk_local_time_on_every_channel():
    # 2014-07-01 12:30 UTC is 13:30 BST on a Tuesday in July; separators (-1) are zero
    t = np.array([pd.Timestamp("2014-07-01 12:30", tz="UTC").value // 10**9, -1])
    f = calendar_features(t)
    assert f.shape == (8, 2)
    for row, (value, period) in enumerate([(30, 60), (13, 24), (1, 7), (7, 12)]):
        angle = 2 * np.pi * value / period
        assert np.isclose(f[2 * row, 0], np.sin(angle)) and np.isclose(f[2 * row + 1, 0], np.cos(angle)), row
    assert np.all(f[:, 1] == 0)
