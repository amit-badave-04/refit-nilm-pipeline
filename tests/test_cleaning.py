import numpy as np
import pandas as pd
import pytest

from refit_nilm import cleaning
from refit_nilm import config as C


def _epoch(s: str) -> int:
    return pd.Timestamp(s).value // 10**9


def _raw(unix, agg, iam1=None):
    n = len(unix)
    df = pd.DataFrame({"Time": ["x"] * n, "Unix": np.asarray(unix, dtype="int64"), "Aggregate": np.asarray(agg, float)})
    for i, col in enumerate(C.CHANNELS[1:], start=1):
        df[col] = np.asarray(iam1, float) if (i == 1 and iam1 is not None) else 0.0
    return df


def test_local_epoch_to_utc_shifts_summer_not_winter():
    local = np.array([_epoch("2014-07-01 12:00:00"), _epoch("2014-12-01 12:00:00")])
    utc, info = cleaning.local_epoch_to_utc(local)
    assert utc[0] == local[0] - 3600  # BST -> UTC
    assert utc[1] == local[1]  # GMT == UTC
    assert info["rows_shifted_by_1h"] == 1


def test_local_epoch_to_utc_resolves_repeated_hour_by_file_order():
    # 01:30 BST, then the clock goes back, then 01:10 GMT (a backward step in file order)
    local = np.array([_epoch("2013-10-27 01:30:00"), _epoch("2013-10-27 01:59:58"), _epoch("2013-10-27 01:00:01")])
    utc, info = cleaning.local_epoch_to_utc(local)
    assert utc[0] == local[0] - 3600
    assert utc[1] == local[1] - 3600
    assert utc[2] == local[2]
    assert np.all(np.diff(utc) > 0)
    assert info["rows_dropped_ambiguous_or_nonexistent"] == 0


def test_local_epoch_to_utc_drops_undecidable_sorted_repeated_hour():
    local = np.array([_epoch("2014-10-26 00:30:00"), _epoch("2014-10-26 01:15:00"), _epoch("2014-10-26 02:30:00")])
    utc, info = cleaning.local_epoch_to_utc(local)
    assert utc[1] == -1
    assert info["rows_dropped_ambiguous_or_nonexistent"] == 1


def test_drop_duplicates_rule():
    df = _raw([10, 10, 10, 20], [100, 100, 150, 200]).assign(utc=[10, 10, 10, 20])
    out, info = cleaning.drop_duplicates(df)
    assert info["exact_duplicates_removed"] == 1
    assert info["conflicting_same_time_removed"] == 1
    assert out.loc[out.utc == 10, "Aggregate"].item() == 150  # last poll wins


def test_mask_impossible_sets_nan_not_zero():
    df = _raw([1, 2, 3], [100, 30_000, 200], iam1=[65535, 50, 5000])
    out, info = cleaning.mask_impossible(df)
    assert np.isnan(out.loc[1, "Aggregate"])
    assert np.isnan(out.loc[0, "Appliance1"]) and np.isnan(out.loc[2, "Appliance1"])
    assert out.loc[1, "Appliance1"] == 50
    assert info["iam_cells_masked"] == 2 and info["aggregate_cells_masked"] == 1


def test_ffill_event_rows_respects_time_limit():
    df = _raw([0, 8, 16, 400], [100, np.nan, np.nan, np.nan]).assign(utc=[0, 8, 16, 400])
    out, _ = cleaning.ffill_event_rows(df, limit_s=120)
    assert out["Aggregate"].tolist()[:3] == [100, 100, 100]
    assert np.isnan(out["Aggregate"].iloc[3])


def test_to_minutes_and_gap_filling():
    utc = np.r_[np.arange(0, 120, 10), np.arange(300, 360, 10), np.arange(1200, 1260, 10)]
    df = _raw(utc, np.full(len(utc), 100.0)).assign(utc=utc)
    m = cleaning.to_minutes(df)
    assert len(m) == 21 and m["n_readings"].iloc[0] == 6
    filled, long_gaps, info = cleaning.fill_short_gaps(m, max_gap_min=3)
    assert info["short_gaps_filled"] == 1  # minutes 2-4 (3 minutes)
    assert info["long_gaps_flagged"] == 1  # minutes 6-19 (14 minutes)
    assert filled["imputed"].sum() == 3
    assert long_gaps["minutes"].iloc[0] == 14


def test_clock_change_hours_detects_wall_clock_file():
    # one row every 10 s across the spring-forward day, skipping 01:00-02:00 (wall-clock file)
    start = _epoch("2015-03-28 21:00:00")
    u = np.arange(start, start + 8 * 3600, 10)
    u = u[(u < _epoch("2015-03-29 01:00:00")) | (u >= _epoch("2015-03-29 02:00:00"))]
    tab = cleaning.clock_change_hours(pd.Series(u), "2015-03-01", "2015-04-30")
    row = tab[tab["change"].str.startswith("spring")].iloc[0]
    assert row["rows_01_02"] == 0 and row["rows_per_hour_before"] == 360
