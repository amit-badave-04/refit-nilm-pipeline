"""Hand-crafted window features for the classical (LightGBM) baseline and simple profiles.

All features are computed from the aggregate only, over centred windows (like seq2point, the
model sees the minutes before *and* after the target minute: this is offline disaggregation of
recorded data, not real-time detection). Time-of-day and day-of-week use local UK time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

SCALES = (5, 15, 31, 61, 121, 237)


def window_features(agg: pd.Series) -> pd.DataFrame:
    """Per-minute features from the aggregate (NaN-tolerant, centred rolling windows)."""
    a = agg.astype("float32")
    d = a.diff()
    f = {"agg": a, "diff": d, "abs_diff": d.abs()}
    for w in SCALES:
        r = a.rolling(w, center=True, min_periods=max(1, w // 2))
        f[f"mean_{w}"] = r.mean()
        f[f"std_{w}"] = r.std()
        f[f"min_{w}"] = r.min()
        f[f"max_{w}"] = r.max()
        up = (d > 200).astype("float32").rolling(w, center=True, min_periods=1).sum()
        big = (d.abs() > 1000).astype("float32").rolling(w, center=True, min_periods=1).sum()
        f[f"edges200_{w}"] = up
        f[f"edges1k_{w}"] = big
        f[f"above1500_{w}"] = (a > 1500).astype("float32").rolling(w, center=True, min_periods=1).mean()
    f["agg_minus_min61"] = a - f["min_61"]
    local = agg.index.tz_convert(C.LOCAL_TZ)
    hour = local.hour + local.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["weekend"] = (local.dayofweek >= 5).astype("float32")
    return pd.DataFrame(f, index=agg.index).astype("float32")


def time_profile(y: pd.Series) -> pd.Series:
    """Mean washing-machine power by local (weekday, hour): a behaviour-only baseline."""
    local = y.index.tz_convert(C.LOCAL_TZ)
    return y.groupby([local.dayofweek, local.hour]).mean()


def apply_profile(profile: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert(C.LOCAL_TZ)
    keys = pd.MultiIndex.from_arrays([local.dayofweek, local.hour])
    return profile.reindex(keys).fillna(0).to_numpy()
