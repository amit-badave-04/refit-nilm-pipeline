# Copyright © 2025 EDF. Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in
# writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# Ported from https://github.com/adrienpetralia/NILMFormer (commit e73a975, files
# src/nilmformer/{model,congif}.py and layers/{transformer,embedding}.py).
# Changes from the original: the four files are merged into one module; the optional xformers
# attention path is removed, and the attention is computed with torch's scaled_dot_product_attention
# using the same diagonal mask (equivalence with the original einsum formulation is pinned by
# tests/test_nilmformer.py); the forward pass returns a dict like the other models in this package.
"""NILMFormer (Petralia, Charpentier, Kadhi & Palpanas, KDD 2025), a transformer for NILM.

Sequence-to-sequence: a window of L aggregate minutes plus 8 calendar channels (sin/cos of minute,
hour, day of week and month) -> the appliance power at every one of the L minutes. Two ideas
target the non-stationarity of household load: the aggregate is standardised per window
(instance normalisation), and the window's mean and standard deviation enter the encoder as an
extra token from which the output scale is re-learned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class NILMFormerConfig:
    c_in: int = 1
    c_embedding: int = 8
    c_out: int = 1
    kernel_size: int = 3
    kernel_size_head: int = 3
    dilations: tuple[int, ...] = field(default=(1, 2, 4, 8))
    conv_bias: bool = True
    n_encoder_layers: int = 3
    d_model: int = 96
    dp_rate: float = 0.2
    pffn_ratio: int = 4
    n_head: int = 8
    norm_eps: float = 1e-5


class ResUnit(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, bias: bool):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, dilation=dilation, bias=bias, padding="same"),
            nn.GELU(),
            nn.BatchNorm1d(c_out),
        )
        # as in the original: a 1-channel input is broadcast onto the residual, not projected
        self.match_residual = c_in > 1 and c_in != c_out
        if self.match_residual:
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.match_residual:
            return self.conv(x) + self.layers(x)
        return x + self.layers(x)


class DilatedBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilations: tuple[int, ...], bias: bool):
        super().__init__()
        units = [ResUnit(c_in if i == 0 else c_out, c_out, kernel_size, d, bias) for i, d in enumerate(dilations)]
        self.network = nn.Sequential(*units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DiagonallyMaskedSelfAttention(nn.Module):
    """Multi-head self-attention in which no token attends to itself."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads, self.head_dim, self.dropout = n_heads, dim // n_heads, dropout
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = (w(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        allowed = ~torch.eye(n, dtype=torch.bool, device=x.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0)
        return self.out_dropout(self.wo(out.transpose(1, 2).reshape(b, n, -1)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: NILMFormerConfig):
        super().__init__()
        self.attention_layer = DiagonallyMaskedSelfAttention(cfg.d_model, cfg.n_head, cfg.dp_rate)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.dropout = nn.Dropout(cfg.dp_rate)
        hidden = cfg.d_model * cfg.pffn_ratio
        self.pffn = nn.Sequential(nn.Linear(cfg.d_model, hidden), nn.GELU(), nn.Dropout(cfg.dp_rate), nn.Linear(hidden, cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # as in the original, each residual is added to the normalised input
        x = self.norm1(x)
        x = x + self.attention_layer(x)
        x = self.norm2(x)
        return x + self.dropout(self.pffn(x))


class NILMFormer(nn.Module):
    def __init__(self, cfg: NILMFormerConfig = NILMFormerConfig()):
        super().__init__()
        assert cfg.d_model % 4 == 0, "d_model must be divisible by 4"
        d = cfg.d_model
        self.embed_block = DilatedBlock(cfg.c_in, 3 * d // 4, cfg.kernel_size, cfg.dilations, cfg.conv_bias)
        self.proj_embedding = nn.Conv1d(cfg.c_embedding, d // 4, kernel_size=1)
        self.proj_stats1 = nn.Linear(2, d)
        self.proj_stats2 = nn.Linear(d, 2)
        self.encoder = nn.Sequential(*[EncoderLayer(cfg) for _ in range(cfg.n_encoder_layers)], nn.LayerNorm(d))
        self.head = nn.Conv1d(d, cfg.c_out, kernel_size=cfg.kernel_size_head, padding=cfg.kernel_size_head // 2, padding_mode="replicate")
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, 1 + 8, L) -> {"power": (B, L)} in the same scaled units as the input load."""
        load, calendar = x[:, :1, :], x[:, 1:, :]
        mean = load.mean(dim=-1, keepdim=True).detach()
        std = torch.sqrt(load.var(dim=-1, keepdim=True, unbiased=False) + 1e-6).detach()
        z = self.embed_block((load - mean) / std)
        z = torch.cat([z, self.proj_embedding(calendar)], dim=1).permute(0, 2, 1)   # (B, L, d)
        stats_token = self.proj_stats1(torch.cat([mean, std], dim=1).permute(0, 2, 1))  # (B, 1, d)
        z = self.encoder(torch.cat([z, stats_token], dim=1))[:, :-1, :]
        z = self.head(z.permute(0, 2, 1))                                              # (B, 1, L)
        stats_out = self.proj_stats2(stats_token)                                      # (B, 1, 2)
        z = z * stats_out[:, :, 1:2] + stats_out[:, :, 0:1]
        return {"power": z.squeeze(1)}
