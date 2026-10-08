"""Training and inference for the windowed (seq2point-style) models.

* Inputs are standardised with training-split statistics only: x = (agg - mean) / std.
* The target is scaled as y / wm_std (non-negative, so the gated output p_on * power stays
  meaningful). For a linear output layer this is equivalent to the z-score used by the
  reference implementation, up to the learned bias.
* Seq2Point loss: MSE. Gated loss: MSE on the gated output + lambda * BCE on the on/off state
  (state label: y >= 20 W), as in subtask-gated networks.
* Early stopping on the validation-house loss (MSE of the output, reported as NDE), patience 5.
* Windows are gathered on the GPU from one flat tensor; nothing is materialised.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .datasets import Corpus
from .models.seq2point import build


@dataclass
class TrainConfig:
    model: str = "seq2point"
    window: int = 237
    seed: int = 42
    lr: float = 1e-3
    batch_size: int = 1000
    max_epochs: int = 30
    patience: int = 5
    min_epochs: int = 3
    state_weight: float = 1.0      # lambda for the gated model's BCE term
    train_samples_per_epoch: int | None = None  # None = all training windows
    val_stride: int = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class WindowSource:
    """Holds the standardised corpus on the device and gathers batches by window start."""

    def __init__(self, corpus: Corpus, stats: dict, device: str):
        self.window = corpus.window
        self.device = device
        x = (corpus.agg - stats["agg_mean"]) / stats["agg_std"]
        self.x = torch.as_tensor(x, dtype=torch.float32, device=device)
        y = np.nan_to_num(corpus.wm, nan=0.0) / stats["wm_std"]
        self.y = torch.as_tensor(y, dtype=torch.float32, device=device)
        self.on = torch.as_tensor(np.nan_to_num(corpus.wm, nan=0.0) >= C.WM_ON_THRESHOLD_W, device=device)
        self.offsets = torch.arange(self.window, device=device)
        self.half = self.window // 2

    def batch(self, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = starts[:, None] + self.offsets[None, :]
        t = starts + self.half
        return self.x[idx], self.y[t], self.on[t].float()


def output_power(out: dict[str, torch.Tensor]) -> torch.Tensor:
    if "p_on" in out:
        return out["p_on"] * out["power_raw"]
    return out["power"]


def loss_fn(out: dict[str, torch.Tensor], y: torch.Tensor, on: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    loss = F.mse_loss(output_power(out), y)
    if "logit" in out:
        loss = loss + cfg.state_weight * F.binary_cross_entropy_with_logits(out["logit"], on)
    return loss


@torch.no_grad()
def predict(model: torch.nn.Module, src: WindowSource, starts: np.ndarray, stats: dict, batch_size: int = 8192) -> dict[str, np.ndarray]:
    """Power in watts (clipped at 0) and, for gated models, the on-probability."""
    model.eval()
    powers, probs = [], []
    st = torch.as_tensor(starts, device=src.device)
    for i in range(0, len(st), batch_size):
        xb, _, _ = src.batch(st[i : i + batch_size])
        out = model(xb)
        powers.append(output_power(out).clamp_min(0) * stats["wm_std"])
        if "p_on" in out:
            probs.append(out["p_on"])
    res = {"power": torch.cat(powers).cpu().numpy()}
    if probs:
        res["p_on"] = torch.cat(probs).cpu().numpy()
    return res


def nde_np(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.sum((yhat - y) ** 2) / np.sum(y**2)))


def fit(corpus: Corpus, stats: dict, cfg: TrainConfig, device: str = "cuda", log_every: int = 1) -> tuple[torch.nn.Module, list[dict]]:
    set_seed(cfg.seed)
    src = WindowSource(corpus, stats, device)
    model = build(cfg.model, cfg.window).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    train_starts = torch.as_tensor(corpus.starts["train"], device=device)
    val_starts = corpus.starts["val"][:: cfg.val_stride]
    y_val = corpus.wm[val_starts + corpus.window // 2]
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    history, best, best_state, bad = [], np.inf, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        t0 = time.time()
        perm = train_starts[torch.randperm(len(train_starts), device=device, generator=gen)]
        if cfg.train_samples_per_epoch:
            perm = perm[: cfg.train_samples_per_epoch]
        total, n = torch.zeros((), device=device), 0
        for i in range(0, len(perm), cfg.batch_size):
            xb, yb, onb = src.batch(perm[i : i + cfg.batch_size])
            loss = loss_fn(model(xb), yb, onb, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * len(xb)
            n += len(xb)
        total = float(total)
        pred = predict(model, src, val_starts, stats)["power"]
        val_nde = nde_np(y_val, pred)
        val_mae = float(np.mean(np.abs(pred - y_val)))
        history.append({"epoch": epoch, "train_loss": total / n, "val_nde": val_nde, "val_mae_w": val_mae, "seconds": time.time() - t0})
        if epoch % log_every == 0:
            print(f"epoch {epoch:2d}  train_loss {total / n:.4f}  val_NDE {val_nde:.4f}  val_MAE {val_mae:.2f} W  ({time.time() - t0:.0f}s)", flush=True)
        if val_nde < best - 1e-4:
            best, bad = val_nde, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if epoch >= cfg.min_epochs and bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, history


def save_run(path: Path, model: torch.nn.Module, cfg: TrainConfig, stats: dict, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "stats": stats}, path)
    with open(path.with_suffix(".json"), "w") as fh:
        json.dump({"config": asdict(cfg), "stats": stats, "history": history}, fh, indent=1)


def load_run(path: Path, device: str = "cuda") -> tuple[torch.nn.Module, TrainConfig, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ck["config"])
    model = build(cfg.model, cfg.window).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, ck["stats"]
