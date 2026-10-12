"""The seeded NILMFormer, fixed input and summary statistics pinned by tests/data/nilmformer_golden.json."""
from __future__ import annotations

from pathlib import Path

import torch

from refit_nilm.models.nilmformer import NILMFormer

GOLDEN = Path(__file__).resolve().parent / "data" / "nilmformer_golden.json"


def golden_input() -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    x = torch.randn(2, 9, 33, generator=g)
    x[:, 0] = x[:, 0].abs() * 0.05  # load channel, at the scale the model sees in training
    return x


def golden_model() -> NILMFormer:
    torch.manual_seed(0)
    return NILMFormer().eval()


def compute() -> dict:
    """Eval-mode outputs, and the gradient of the summed output for every parameter tensor and the input.

    The input gradient is what pins the original's detached window statistics: detaching them
    changes no parameter gradient, only the gradient with respect to the load itself.
    """
    model, x = golden_model(), golden_input().requires_grad_(True)
    out = model(x)["power"]
    out.sum().backward()
    return {
        "output": out.detach().flatten().tolist(),
        "grad_norms": {k: float(p.grad.norm()) for k, p in model.named_parameters()},
        "input_grad_norm_per_channel": x.grad.norm(dim=-1).flatten().tolist(),
    }
