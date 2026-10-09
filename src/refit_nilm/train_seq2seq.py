"""Training and inference for sequence-to-sequence models (NILMFormer) on the same data contract.

Differences from ``train.py`` that follow the model's published recipe (Petralia et al. 2025):
* Scaling: aggregate and target are divided by one constant, the largest training aggregate
  ("MaxScaling" / "SameAsPower"); the model standardises each window internally.
* Inputs: the aggregate plus 8 calendar channels (sin/cos of minute, hour, day of week, month),
  computed in UK local time because laundry follows the household's clock, not UTC.
* Training: each epoch draws as many random windows as would tile the training minutes once
  (the original tiles them without overlap); masked MSE over every minute of the window whose
  target is usable; Adam, lr 1e-4, batch 64, ReduceLROnPlateau (patience 5), early stopping
  (patience 10), at most 50 epochs.
Inference (a declared deviation from the original's non-overlapping tiling): every target minute
is predicted by the window centred on it, so each minute gets symmetric context, exactly as
Seq2Point is evaluated. Early stopping uses validation NDE, the criterion used for Seq2Point.
Training runs with PyTorch's deterministic algorithms, so a run repeats bit for bit on the same
GPU (without them, the attention kernels drift by ~1e-6 within 60 steps).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# cuBLAS needs a fixed workspace for deterministic GEMMs; it is read when the first CUDA handle is
# created, so it must be set before any GPU work in the process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from . import config as C
from .datasets import Corpus
from .models.nilmformer import NILMFormer
from .train import nde_np, set_seed


@dataclass
class Seq2SeqConfig:
    model: str = "nilmformer"
    window: int = 129
    seed: int = 42
    lr: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 10
    lr_patience: int = 5
    val_stride: int = 1
    windows_per_epoch_factor: int = 1  # 1 = the published budget (training minutes tiled once per epoch)


def calendar_features(epoch_s: np.ndarray) -> np.ndarray:
    """(8, n) float32: sin/cos of minute, hour, day of week and month in UK local time.

    Separator positions (epoch -1) get zeros.
    """
    valid = epoch_s >= 0
    t = pd.to_datetime(np.where(valid, epoch_s, 0), unit="s", utc=True).tz_convert(C.LOCAL_TZ)
    parts = [(t.minute, 60.0), (t.hour, 24.0), (t.dayofweek, 7.0), (t.month, 12.0)]
    feats = []
    for v, period in parts:
        ang = 2 * np.pi * np.asarray(v, dtype=float) / period
        feats += [np.sin(ang), np.cos(ang)]
    out = np.stack(feats).astype(np.float32)
    out[:, ~valid] = 0.0
    return out


class SeqWindowSource:
    """Scaled aggregate, calendar channels and masked targets on the device; gathers windows."""

    def __init__(self, corpus: Corpus, scale: float, device: str):
        self.window, self.half, self.device, self.scale = corpus.window, corpus.window // 2, device, scale
        self.x = torch.as_tensor(corpus.agg / scale, dtype=torch.float32, device=device)
        self.cal = torch.as_tensor(calendar_features(corpus.time), device=device)
        wm = corpus.wm
        self.y = torch.as_tensor(np.nan_to_num(wm, nan=0.0) / scale, dtype=torch.float32, device=device)
        self.ok = torch.as_tensor(~np.isnan(wm), device=device)
        self.offsets = torch.arange(self.window, device=device)

    def batch(self, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = starts[:, None] + self.offsets[None, :]                        # (B, L)
        x = torch.cat([self.x[idx][:, None, :], self.cal[:, idx].permute(1, 0, 2)], dim=1)
        return x, self.y[idx], self.ok[idx]


def build(cfg: Seq2SeqConfig) -> torch.nn.Module:
    if cfg.model != "nilmformer":
        raise ValueError(f"unknown seq2seq model {cfg.model}")
    return NILMFormer()


def training_scale(corpus: Corpus) -> float:
    return float(corpus.agg[corpus.targets("train")].max())


@torch.no_grad()
def predict(model: torch.nn.Module, src: SeqWindowSource, starts: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Power in watts (clipped at 0) at each window's centre minute."""
    model.eval()
    out = []
    st = torch.as_tensor(starts, device=src.device)
    for i in range(0, len(st), batch_size):
        xb, _, _ = src.batch(st[i : i + batch_size])
        out.append(model(xb)["power"][:, src.half].clamp_min(0) * src.scale)
    return torch.cat(out).cpu().numpy()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the minutes whose target is usable; 0 if none is."""
    return (((pred - target) ** 2) * ok).sum() / ok.sum().clamp_min(1)


def _epoch(model, opt, src: SeqWindowSource, train_starts: torch.Tensor, n_draw: int, cfg: Seq2SeqConfig, gen) -> float:
    model.train()
    pick = train_starts[torch.randint(len(train_starts), (n_draw,), device=src.device, generator=gen)]
    total, n = torch.zeros((), device=src.device), 0
    for i in range(0, n_draw, cfg.batch_size):
        xb, yb, okb = src.batch(pick[i : i + cfg.batch_size])
        loss = masked_mse(model(xb)["power"], yb, okb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.detach() * len(xb)
        n += len(xb)
    return float(total) / n


def fit(corpus: Corpus, cfg: Seq2SeqConfig, device: str = "cuda") -> tuple[torch.nn.Module, list[dict], float]:
    """Train with deterministic algorithms, then restore the caller's setting (on/off and warn-only).

    Like ``train.fit``, this seeds the global random generators and sets the cuDNN flags.
    """
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _fit(corpus, cfg, device)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _fit(corpus: Corpus, cfg: Seq2SeqConfig, device: str) -> tuple[torch.nn.Module, list[dict], float]:
    set_seed(cfg.seed)
    scale = training_scale(corpus)
    src = SeqWindowSource(corpus, scale, device)
    model = build(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=cfg.lr_patience)
    train_starts = torch.as_tensor(corpus.starts["train"], device=device)
    n_draw = cfg.windows_per_epoch_factor * int(np.ceil(len(train_starts) / cfg.window))
    val_starts = corpus.starts["val"][:: cfg.val_stride]
    y_val = corpus.wm[val_starts + corpus.window // 2]
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    history, best, best_state, bad = [], np.inf, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        train_loss = _epoch(model, opt, src, train_starts, n_draw, cfg, gen)
        if not np.isfinite(train_loss):
            raise RuntimeError(f"training loss became {train_loss} at epoch {epoch}")
        pred = predict(model, src, val_starts)
        val_nde = nde_np(y_val, pred)
        sched.step(val_nde)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_nde": val_nde,
                        "val_mae_w": float(np.mean(np.abs(pred - y_val))), "lr": opt.param_groups[0]["lr"],
                        "seconds": time.time() - t0})
        print(f"epoch {epoch:2d}  train_loss {train_loss:.5f}  val_NDE {val_nde:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if val_nde < best - 1e-4:
            best, bad = val_nde, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("validation NDE never improved (non-finite predictions?); there is no checkpoint to restore")
    model.load_state_dict(best_state)
    return model, history, scale


def save_run(path: Path, model: torch.nn.Module, cfg: Seq2SeqConfig, scale: float, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "scale": scale}, path)
    with open(path.with_suffix(".json"), "w") as fh:
        json.dump({"config": asdict(cfg), "scale": scale, "history": history}, fh, indent=1)


def load_run(path: Path, device: str = "cuda") -> tuple[torch.nn.Module, Seq2SeqConfig, float]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Seq2SeqConfig(**ck["config"])
    model = build(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, float(ck["scale"])
