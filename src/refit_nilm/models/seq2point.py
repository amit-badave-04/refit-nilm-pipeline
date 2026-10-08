"""Seq2Point (Zhang et al., AAAI 2018) and a state-gated multi-task variant.

Seq2Point: a window of W aggregate samples -> the appliance power at the window's midpoint.
Architecture as in the authors' reference code: five 1-D convolutions
(30x10, 30x8, 40x6, 50x5, 50x5; ReLU; 'same' padding), Dense(1024, ReLU), Dense(1).
A true temporal Conv1D is used (one public TF2 port reshapes the input so its kernels are
effectively 1 sample wide in time).

Gated Seq2Point: the same convolutional trunk feeds two heads, a power regressor and an on/off
classifier; the output power is ``p_on * power`` (subtask gating as in Shin et al., AAAI 2019;
multi-task state + power heads as in UNet-NILM, Faustine et al. 2020). The classification head
gives the model an explicit way to output exactly "off", which a pure regressor on a sparse
target struggles to do.
"""
from __future__ import annotations

import torch
from torch import nn

CONV_SPEC = [(30, 10), (30, 8), (40, 6), (50, 5), (50, 5)]


class Seq2PointTrunk(nn.Module):
    def __init__(self, window: int):
        super().__init__()
        layers, c_in = [], 1
        for c_out, k in CONV_SPEC:
            left = (k - 1) // 2  # TensorFlow 'same': the extra pad for even kernels goes right
            layers += [nn.ConstantPad1d((left, k - 1 - left), 0.0), nn.Conv1d(c_in, c_out, k), nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.flat_dim = c_in * window

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, W)
        return self.conv(x.unsqueeze(1)).flatten(1)


class Seq2Point(nn.Module):
    def __init__(self, window: int, hidden: int = 1024):
        super().__init__()
        self.window = window
        self.trunk = Seq2PointTrunk(window)
        self.head = nn.Sequential(nn.Linear(self.trunk.flat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"power": self.head(self.trunk(x)).squeeze(-1)}


class GatedSeq2Point(nn.Module):
    def __init__(self, window: int, hidden: int = 1024):
        super().__init__()
        self.window = window
        self.trunk = Seq2PointTrunk(window)
        self.power_head = nn.Sequential(nn.Linear(self.trunk.flat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.state_head = nn.Sequential(nn.Linear(self.trunk.flat_dim, hidden // 4), nn.ReLU(), nn.Linear(hidden // 4, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        power = self.power_head(z).squeeze(-1)
        logit = self.state_head(z).squeeze(-1)
        return {"power_raw": power, "logit": logit, "p_on": torch.sigmoid(logit)}


def build(name: str, window: int) -> nn.Module:
    return {"seq2point": Seq2Point, "gated_seq2point": GatedSeq2Point}[name](window)
