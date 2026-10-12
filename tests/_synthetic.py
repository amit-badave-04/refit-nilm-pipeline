"""Small synthetic house frames shared by the training tests (one wash a day, a few unusable minutes)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def house_frame(house: int, days: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + house)
    idx = pd.date_range("2014-03-01", periods=days * 1440, freq="1min", tz="UTC")
    wm = np.zeros(len(idx))
    for d in range(days):
        s = d * 1440 + 500 + int(rng.integers(0, 200))
        wm[s : s + 15] = 2000.0
        wm[s + 15 : s + 70] = 150.0
    agg = 300 + 40 * rng.random(len(idx)) + wm
    df = pd.DataFrame({"agg": agg, "wm": wm}, index=idx)
    df["agg_missing"] = False
    df["issues"] = False
    df["target_ok"] = True
    df.iloc[100:110, df.columns.get_loc("target_ok")] = False  # a few unusable target minutes
    df["house"] = house
    return df
