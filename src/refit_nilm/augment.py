"""Training-time augmentation with real appliance activations.

Idea (Kelly & Knottenbelt 2015, "synthetic aggregate"; Rafiq et al. 2021): add real recorded
activations to training windows so the model sees more combinations than the training homes
happen to contain.

Here the augmentation is aimed at the failure measured on the validation house: false
washing-machine power while *other* water-heating or motor appliances run. So the main ingredient
is **distractors** — activations of dishwashers, tumble dryers, washer-dryers and kettles added to
the input with the target left unchanged — plus, optionally, a small rate of extra
washing-machine activations (added to input *and* target).

Leakage rule: activations are taken only from minutes labelled ``train`` (the training period of
the training homes). Augmentation is applied at training time only, in watts, before
standardisation (the model input is linear in watts, so adding ``segment / agg_std`` to the
standardised window is the same thing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import io
from .cycles import CycleRule, activation_mask

DISTRACTOR_TYPES = ("dishwasher", "tumble dryer", "washer dryer", "kettle")
DISTRACTOR_RULE = CycleRule(on_threshold_w=50, min_off_min=3, min_on_min=2)  # kettles last 2-4 min
MAX_SEGMENT_MIN = 240


@dataclass(frozen=True)
class AugmentConfig:
    p_distractor: float = 0.5
    p_wm: float = 0.0


def _segments(power: np.ndarray, allowed: np.ndarray, rule: CycleRule) -> list[np.ndarray]:
    p = np.nan_to_num(power, nan=0.0)
    mask = activation_mask(p, rule)
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    out = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if allowed[s:e].all():
            out.append(p[s : min(e, s + MAX_SEGMENT_MIN)].astype(np.float32))
    return out


def build_banks(labels: dict[int, pd.Series], houses: list[int]) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    """Distractor activations by appliance type, and WM activations, from training minutes only."""
    amap = io.parse_appliance_map(C.DATA_DIR / "CLEAN_READ_ME_081116.txt")
    wm_ch = io.washing_machine_channels(amap)
    distractors: dict[str, list[np.ndarray]] = {t: [] for t in DISTRACTOR_TYPES}
    wm: list[np.ndarray] = []
    for h in houses:
        m = io.load_clean_minutes(h)
        allowed = (labels[h].reindex(m.index) == "train").to_numpy() & (m["n_readings"] > 0).to_numpy()
        for ch, desc in amap[h].items():
            d = desc.lower()
            kind = next((t for t in DISTRACTOR_TYPES if d.startswith(t)), None)
            if kind is not None:
                distractors[kind] += _segments(m[f"Appliance{ch}"].to_numpy(), allowed, DISTRACTOR_RULE)
        wm += _segments(m[f"Appliance{wm_ch[h][0]}"].to_numpy(), allowed, CycleRule())
    return distractors, wm


class _Bank:
    def __init__(self, segments: list[np.ndarray], device: str):
        lens = np.array([len(s) for s in segments], dtype=np.int64)
        pad = np.zeros((len(segments), int(lens.max())), dtype=np.float32)
        for i, s in enumerate(segments):
            pad[i, : len(s)] = s
        self.values = torch.as_tensor(pad, device=device)
        self.lens = torch.as_tensor(lens, device=device)

    def sample(self, n: int, window: int, gen: torch.Generator) -> torch.Tensor:
        """n windows of length ``window`` each overlapping one random segment at a random offset."""
        dev = self.values.device
        k = torch.randint(len(self.lens), (n,), device=dev, generator=gen)
        L = self.lens[k]
        span = window + L - 1
        o = (torch.rand(n, device=dev, generator=gen) * span).long() - (L - 1)  # segment start rel. to window
        idx = torch.arange(window, device=dev)[None, :] - o[:, None]
        valid = (idx >= 0) & (idx < L[:, None])
        vals = self.values[k[:, None], idx.clamp(0, self.values.shape[1] - 1)]
        return vals * valid


class Augmenter:
    def __init__(self, distractors: dict[str, list[np.ndarray]], wm: list[np.ndarray], stats: dict, cfg: AugmentConfig, window: int, device: str, seed: int):
        all_d = [s for segs in distractors.values() for s in segs]
        self.d = _Bank(all_d, device) if all_d else None
        self.w = _Bank(wm, device) if wm else None
        self.cfg, self.window = cfg, window
        self.agg_std, self.wm_std = stats["agg_std"], stats["wm_std"]
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed + 7919)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, on: torch.Tensor):
        B = x.shape[0]
        if self.d is not None and self.cfg.p_distractor > 0:
            sel = torch.rand(B, device=x.device, generator=self.gen) < self.cfg.p_distractor
            n = int(sel.sum())
            if n:
                x[sel] += self.d.sample(n, self.window, self.gen) / self.agg_std
        if self.w is not None and self.cfg.p_wm > 0:
            sel = torch.rand(B, device=x.device, generator=self.gen) < self.cfg.p_wm
            n = int(sel.sum())
            if n:
                add = self.w.sample(n, self.window, self.gen)
                x[sel] += add / self.agg_std
                y[sel] += add[:, self.window // 2] / self.wm_std
                on = (y * self.wm_std >= C.WM_ON_THRESHOLD_W).float()
        return x, y, on
