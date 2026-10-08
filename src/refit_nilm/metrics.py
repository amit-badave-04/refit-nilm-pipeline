"""Disaggregation metrics.

Regression: MAE, MAE_ON (Langevin et al. 2022), SAE (Zhang et al. 2018), per-day SAE, NDE,
EpD (mean absolute daily energy error).
State: precision / recall / F1 at a power threshold applied identically to truth and
prediction; AUPRC when a model outputs an ON probability.
Cycle level (project-specific; no NILM standard exists): cycles detected with the same rule
on truth and prediction are matched by temporal IoU >= 0.5.
Energy split: overlap / missing / extra energy (after Rafiq et al. 2021).
All functions ignore positions where the truth is NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config as C
from .cycles import CycleRule, detect_cycles


def _valid(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = ~(np.isnan(y) | np.isnan(yhat))
    return y[ok], yhat[ok]


def mae(y, yhat) -> float:
    y, yhat = _valid(y, yhat)
    return float(np.mean(np.abs(yhat - y)))


def mae_on(y, yhat, threshold: float = C.WM_ON_THRESHOLD_W) -> float:
    y, yhat = _valid(y, yhat)
    on = y >= threshold
    return float(np.mean(np.abs(yhat[on] - y[on]))) if on.any() else float("nan")


def sae(y, yhat) -> float:
    y, yhat = _valid(y, yhat)
    total = y.sum()
    return float(abs(yhat.sum() - total) / total) if total > 0 else float("nan")


def nde(y, yhat) -> float:
    y, yhat = _valid(y, yhat)
    denom = np.sum(y**2)
    return float(np.sqrt(np.sum((yhat - y) ** 2) / denom)) if denom > 0 else float("nan")


def daily_energy_errors(y: pd.Series, yhat: pd.Series, min_coverage: float = 0.9) -> pd.DataFrame:
    """Per-day true vs predicted energy (kWh) on days with >= ``min_coverage`` valid minutes."""
    df = pd.DataFrame({"y": y, "yhat": yhat})
    ok = df["y"].notna() & df["yhat"].notna()
    day = df.index.floor("D")
    g = df[ok].groupby(day[ok.to_numpy()])
    cov = ok.groupby(day).mean()
    out = pd.DataFrame({"y_kwh": g["y"].sum() / 60_000, "yhat_kwh": g["yhat"].sum() / 60_000})
    out["coverage"] = cov.reindex(out.index)
    return out[out["coverage"] >= min_coverage]


def epd_wh(y: pd.Series, yhat: pd.Series) -> float:
    d = daily_energy_errors(y, yhat)
    return float((d["yhat_kwh"] - d["y_kwh"]).abs().mean() * 1000) if len(d) else float("nan")


def sae_daily(y: pd.Series, yhat: pd.Series) -> float:
    """Mean over days of |E_hat - E| / E, on days where the appliance used >= 0.05 kWh."""
    d = daily_energy_errors(y, yhat)
    d = d[d["y_kwh"] >= 0.05]
    return float(((d["yhat_kwh"] - d["y_kwh"]).abs() / d["y_kwh"]).mean()) if len(d) else float("nan")


def state_scores(y, yhat, threshold: float = C.WM_ON_THRESHOLD_W) -> dict:
    y, yhat = _valid(y, yhat)
    t, p = y >= threshold, yhat >= threshold
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def auprc(y, p_on, threshold: float = C.WM_ON_THRESHOLD_W) -> float:
    y, p_on = _valid(y, p_on)
    t = y >= threshold
    return float(average_precision_score(t, p_on)) if t.any() else float("nan")


def energy_split(y, yhat) -> dict:
    """Share of true energy recovered (overlap), missed, and predicted in excess."""
    y, yhat = _valid(y, yhat)
    total = y.sum()
    return {
        "overlap_pct": float(100 * np.minimum(y, yhat).sum() / total),
        "missing_pct": float(100 * np.clip(y - yhat, 0, None).sum() / total),
        "extra_pct": float(100 * np.clip(yhat - y, 0, None).sum() / total),
    }


def _minutes(delta) -> float:
    return pd.Timedelta(delta).total_seconds() / 60.0


def match_cycles(true_cycles: pd.DataFrame, pred_cycles: pd.DataFrame, iou_min: float = 0.5) -> pd.DataFrame:
    """Greedy one-to-one matching of cycles by temporal IoU (minutes, inclusive ends)."""
    rows, used = [], set()
    for i, tc in true_cycles.iterrows():
        best, best_iou = None, 0.0
        for j, pc in pred_cycles.iterrows():
            if j in used:
                continue
            inter = _minutes(min(tc.end, pc.end) - max(tc.start, pc.start)) + 1
            if inter <= 0:
                continue
            union = _minutes(max(tc.end, pc.end) - min(tc.start, pc.start)) + 1
            iou = inter / union
            if iou > best_iou:
                best, best_iou = j, iou
        if best is not None and best_iou >= iou_min:
            used.add(best)
            pc = pred_cycles.loc[best]
            rows.append(
                {
                    "true_idx": i, "pred_idx": best, "iou": best_iou,
                    "true_kwh": tc.energy_kwh, "pred_kwh": pc.energy_kwh,
                    "start_error_min": _minutes(pc.start - tc.start),
                }
            )
    return pd.DataFrame(rows, columns=["true_idx", "pred_idx", "iou", "true_kwh", "pred_kwh", "start_error_min"])


def cycle_scores(y: pd.Series, yhat: pd.Series, rule: CycleRule = CycleRule()) -> dict:
    tc = detect_cycles(y, rule)
    pc = detect_cycles(yhat.where(y.notna()), rule)
    m = match_cycles(tc, pc)
    n_t, n_p, n_m = len(tc), len(pc), len(m)
    prec = n_m / n_p if n_p else 0.0
    rec = n_m / n_t if n_t else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    err = (m["pred_kwh"] - m["true_kwh"]) if n_m else pd.Series(dtype=float)
    return {
        "true_cycles": n_t, "pred_cycles": n_p, "matched": n_m,
        "cycle_precision": prec, "cycle_recall": rec, "cycle_f1": f1,
        "matched_energy_mape_pct": float(100 * (err.abs() / m["true_kwh"]).mean()) if n_m else float("nan"),
        "median_start_error_min": float(m["start_error_min"].abs().median()) if n_m else float("nan"),
    }


def all_metrics(y: pd.Series, yhat: pd.Series, p_on: pd.Series | None = None) -> dict:
    out = {
        "mae_w": mae(y, yhat), "mae_on_w": mae_on(y, yhat), "sae": sae(y, yhat),
        "sae_daily": sae_daily(y, yhat), "nde": nde(y, yhat), "epd_wh": epd_wh(y, yhat),
    }
    out.update(state_scores(y, yhat))
    out.update(energy_split(y, yhat))
    out.update(cycle_scores(y, yhat))
    if p_on is not None:
        out["auprc"] = auprc(y, p_on)
    return out


def g_loss(err_unseen: float, err_seen: float) -> float:
    """Generalisation loss for an error metric (Klemenjak et al. 2019), in percent."""
    return float(100 * (err_unseen / err_seen - 1))


def noise_to_aggregate_ratio(aggregate: pd.Series, submeters: pd.DataFrame) -> float:
    """NAR (Klemenjak et al. 2020): share of aggregate energy not explained by sub-meters."""
    ok = aggregate.notna()
    agg = aggregate[ok]
    sub = submeters[ok].sum(axis=1, min_count=1).fillna(0)
    return float(np.abs(agg - sub).sum() / agg.sum())
