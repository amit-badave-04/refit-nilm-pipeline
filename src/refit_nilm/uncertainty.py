"""Resampling uncertainty and event-level sensitivity for disaggregation metrics.

* Within a home: whole UTC days are resampled with replacement (a block bootstrap with one-day
  blocks), so the dependence between minutes of the same day (a wash spans an hour or two) is kept.
  As a robustness check, a stationary bootstrap (Politis & Romano 1994) draws runs of consecutive
  days of random length (mean 7), which also keeps day-to-day dependence. Every metric is
  recomputed from per-day sufficient statistics, so 1,000 replicates take well under a second.
* Across homes: a cluster bootstrap that resamples whole homes and keeps all of each home's data
  (Field & Welsh 2007). Six homes allow only 462 distinct resamples, and few-cluster inference is
  known to be unreliable (Cameron, Gelbach & Miller 2008), so this interval is a rough guide.
* Cycles are detected and matched once on the full series, exactly as ``metrics.cycle_scores``
  does, and each cycle is assigned to the day it starts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cycles import CycleRule, detect_cycles
from .metrics import daily_energy_errors, match_cycles

STAT_COLS = ["sum_y2", "sum_err2", "y_kwh", "yhat_kwh", "energy_day", "n_true", "n_pred", "n_matched"]


def daily_stats(y: pd.Series, yhat: pd.Series, rule: CycleRule = CycleRule(), iou_min: float = 0.5) -> pd.DataFrame:
    """Per-day sufficient statistics for NDE, daily energy error and cycle F1."""
    ok = y.notna() & yhat.notna()
    yv, pv = y.where(ok), yhat.where(ok)
    day = y.index.floor("D")
    sq = pd.DataFrame({"sum_y2": yv**2, "sum_err2": (pv - yv) ** 2}).groupby(day).sum()
    energy = daily_energy_errors(yv, pv)
    tc, pc = detect_cycles(yv, rule), detect_cycles(pv.where(yv.notna()), rule)
    m = match_cycles(tc, pc, iou_min)
    count = lambda starts: pd.Series(1, index=pd.DatetimeIndex(starts).floor("D")).groupby(level=0).sum()  # noqa: E731
    out = sq[ok.groupby(day).sum() > 0].copy()  # days without a scored minute are not resampling units
    out["y_kwh"] = energy["y_kwh"].reindex(out.index)
    out["yhat_kwh"] = energy["yhat_kwh"].reindex(out.index)
    out["energy_day"] = out["y_kwh"].notna()
    out["n_true"] = count(tc["start"]).reindex(out.index).fillna(0) if len(tc) else 0
    out["n_pred"] = count(pc["start"]).reindex(out.index).fillna(0) if len(pc) else 0
    matched_starts = tc.loc[m["true_idx"], "start"] if len(m) else []
    out["n_matched"] = count(matched_starts).reindex(out.index).fillna(0) if len(m) else 0
    return out[STAT_COLS]


def metrics_from_days(d: pd.DataFrame) -> dict:
    e = d[d["energy_day"].astype(bool)]
    t, p, m = d["n_true"].sum(), d["n_pred"].sum(), d["n_matched"].sum()
    return {
        "nde": float(np.sqrt(d["sum_err2"].sum() / d["sum_y2"].sum())) if d["sum_y2"].sum() > 0 else np.nan,
        "epd_wh": float((e["yhat_kwh"] - e["y_kwh"]).abs().mean() * 1000) if len(e) else np.nan,
        "cycle_f1": float(2 * m / (t + p)) if t + p > 0 else np.nan,
    }


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: runs of consecutive positions (circular), geometric lengths."""
    idx = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        run = (rng.integers(n) + np.arange(rng.geometric(1 / mean_block))) % n
        take = min(len(run), n - i)
        idx[i : i + take] = run[:take]
        i += take
    return idx


def bootstrap_home(d: pd.DataFrame, n_boot: int = 1000, seed: int = 0, mean_block: float | None = None) -> pd.DataFrame:
    """Replicates of the per-home metrics: independent days, or a stationary bootstrap of mean block length."""
    rng = np.random.default_rng(seed)
    arr = d.sort_index().reset_index(drop=True)
    n = len(arr)
    draw = (lambda: rng.integers(0, n, n)) if mean_block is None else (lambda: stationary_indices(n, mean_block, rng))
    return pd.DataFrame([metrics_from_days(arr.iloc[draw()]) for _ in range(n_boot)])


def cluster_bootstrap(per_home: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Replicates of the across-home mean: resample whole homes (rows of ``per_home``) with replacement."""
    rng = np.random.default_rng(seed)
    vals = per_home.to_numpy(dtype=float)
    reps = vals[rng.integers(0, len(vals), (n_boot, len(vals)))].mean(axis=1)
    return pd.DataFrame(reps, columns=per_home.columns)


def percentile_ci(reps: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    a = (1 - level) / 2
    return pd.DataFrame({"low": reps.quantile(a), "high": reps.quantile(1 - a)})


def match_by_start(true_cycles: pd.DataFrame, pred_cycles: pd.DataFrame, tol_min: float) -> int:
    """One-to-one matches whose start times differ by at most ``tol_min`` (closest pairs first)."""
    if len(true_cycles) == 0 or len(pred_cycles) == 0:
        return 0
    ts = pd.DatetimeIndex(true_cycles["start"]).asi8 / 60e9
    ps = pd.DatetimeIndex(pred_cycles["start"]).asi8 / 60e9
    diff = np.abs(ts[:, None] - ps[None, :])
    i, j = np.nonzero(diff <= tol_min)
    order = np.argsort(diff[i, j], kind="stable")
    used_t, used_p, n = set(), set(), 0
    for k in order:
        if i[k] not in used_t and j[k] not in used_p:
            used_t.add(i[k])
            used_p.add(j[k])
            n += 1
    return n


def f1_from_counts(n_true: int, n_pred: int, n_matched: int) -> float:
    return 2 * n_matched / (n_true + n_pred) if n_true + n_pred else float("nan")
