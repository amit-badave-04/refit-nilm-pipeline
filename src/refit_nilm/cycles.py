"""Washing-machine cycle (activation) detection on 1-minute power series.

Rule (Kelly & Knottenbelt 2015, Table 4, ported to 1-minute samples as in Precioso &
Gómez-Ullate 2023): a minute is ON when power >= 20 W; OFF gaps shorter than 3 minutes inside
an activation are bridged (pauses during a wash); activations shorter than 30 minutes are
discarded. Missing minutes (NaN) are treated as OFF for detection but counted per cycle so
cycles that touch an outage can be excluded from statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass(frozen=True)
class CycleRule:
    on_threshold_w: float = C.WM_ON_THRESHOLD_W
    min_off_min: int = C.WM_MIN_OFF_MIN
    min_on_min: int = C.WM_MIN_ON_MIN
    max_power_w: float = C.WM_MAX_POWER_W
    max_duration_min: int = 240  # longer activations are flagged as probable artefacts


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) indices of True runs."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def activation_mask(power: np.ndarray, rule: CycleRule = CycleRule()) -> np.ndarray:
    """Boolean ON mask after bridging short OFF gaps and dropping short activations."""
    p = np.asarray(power, dtype=float)
    on = np.nan_to_num(p, nan=0.0) >= rule.on_threshold_w
    starts, ends = _runs(on)
    if len(starts) == 0:
        return on
    # bridge OFF gaps shorter than min_off between consecutive ON runs
    merged_s, merged_e = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged_e[-1] < rule.min_off_min:
            merged_e[-1] = e
        else:
            merged_s.append(s)
            merged_e.append(e)
    out = np.zeros_like(on)
    for s, e in zip(merged_s, merged_e):
        if e - s >= rule.min_on_min:
            out[s:e] = True
    return out


def detect_cycles(power: pd.Series, rule: CycleRule = CycleRule()) -> pd.DataFrame:
    """Table of cycles for a 1-minute power series indexed by timestamp."""
    p = power.to_numpy(dtype=float)
    mask = activation_mask(p, rule)
    starts, ends = _runs(mask)
    idx = power.index
    rows = []
    for s, e in zip(starts, ends):
        seg = p[s:e]
        rows.append(
            {
                "start": idx[s],
                "end": idx[e - 1],
                "duration_min": int(e - s),
                "peak_w": float(np.nanmax(seg)),
                "mean_w": float(np.nanmean(seg)),
                "energy_kwh": float(np.nansum(seg) / 60.0 / 1000.0),
                "minutes_over_1500w": int(np.nansum(seg > 1500)),
                "heating_kwh": float(np.nansum(np.where(seg > 1500, seg, 0.0)) / 60.0 / 1000.0),
                "missing_minutes": int(np.isnan(seg).sum()),
                "over_max_power": bool(np.nanmax(seg) > rule.max_power_w),
                "too_long": bool((e - s) > rule.max_duration_min),
            }
        )
    cols = [
        "start", "end", "duration_min", "peak_w", "mean_w", "energy_kwh", "minutes_over_1500w",
        "heating_kwh", "missing_minutes", "over_max_power", "too_long",
    ]
    return pd.DataFrame(rows, columns=cols)


def kmeans_midpoint_threshold(power: np.ndarray, iters: int = 50) -> float:
    """Precioso & Gómez-Ullate 'Middle-Point' threshold: 2-means on power, midpoint of centroids."""
    x = np.asarray(power, dtype=float)
    x = x[~np.isnan(x)]
    lo, hi = np.percentile(x, 5), np.percentile(x, 99.5)
    for _ in range(iters):
        mid = (lo + hi) / 2
        a, b = x[x < mid], x[x >= mid]
        if len(a) == 0 or len(b) == 0:
            break
        new_lo, new_hi = a.mean(), b.mean()
        if np.isclose(new_lo, lo) and np.isclose(new_hi, hi):
            break
        lo, hi = new_lo, new_hi
    return float((lo + hi) / 2)


def intrinsic_error(power: np.ndarray, on_mask: np.ndarray) -> float:
    """MAE between the true series and its reconstruction from binary ON/OFF labels
    (ON minutes replaced by the mean ON power, OFF minutes by the mean OFF power)."""
    p = np.asarray(power, dtype=float)
    ok = ~np.isnan(p)
    p, m = p[ok], on_mask[ok]
    if m.all() or (~m).all():
        return float(np.mean(np.abs(p - p.mean())))
    recon = np.where(m, p[m].mean(), p[~m].mean())
    return float(np.mean(np.abs(p - recon)))
