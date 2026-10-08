import numpy as np
import pandas as pd
import pytest

from refit_nilm import metrics
from refit_nilm.cycles import CycleRule, activation_mask, detect_cycles, intrinsic_error, kmeans_midpoint_threshold


def _series(values):
    idx = pd.date_range("2014-01-01", periods=len(values), freq="1min", tz="UTC")
    return pd.Series(np.asarray(values, float), index=idx)


def _wash(minutes_on=60, pause_at=None, pause_len=2, level=500.0, pre=10, post=10):
    on = np.full(minutes_on, level)
    if pause_at is not None:
        on[pause_at : pause_at + pause_len] = 0.0
    return np.r_[np.zeros(pre), on, np.zeros(post)]


def test_short_pause_is_bridged():
    s = _series(_wash(60, pause_at=30, pause_len=2))
    cyc = detect_cycles(s)
    assert len(cyc) == 1 and cyc["duration_min"].iloc[0] == 60


def test_long_pause_splits_cycle():
    s = _series(_wash(80, pause_at=40, pause_len=5))
    assert len(detect_cycles(s)) == 2


def test_short_activation_is_discarded():
    s = _series(_wash(20))
    assert len(detect_cycles(s)) == 0


def test_cycle_energy_and_heating():
    v = _wash(60)
    v[15:30] = 2000.0  # 15 minutes of heating inside the wash
    cyc = detect_cycles(_series(v)).iloc[0]
    expected = (45 * 500 + 15 * 2000) / 60 / 1000
    assert cyc.energy_kwh == pytest.approx(expected)
    assert cyc.minutes_over_1500w == 15
    assert cyc.heating_kwh == pytest.approx(15 * 2000 / 60 / 1000)


def test_nan_counts_as_off_but_is_reported():
    v = _wash(60)
    v[40] = np.nan
    cyc = detect_cycles(_series(v)).iloc[0]
    assert cyc.missing_minutes == 1


def test_midpoint_threshold_between_modes():
    x = np.r_[np.zeros(1000), np.full(200, 2000.0)]
    assert 500 < kmeans_midpoint_threshold(x) < 1500


def test_intrinsic_error_zero_for_two_level_signal():
    x = np.r_[np.zeros(50), np.full(50, 100.0)]
    assert intrinsic_error(x, x > 50) == pytest.approx(0.0)


def test_regression_metrics_known_values():
    y = np.array([0, 0, 100, 100.0])
    yhat = np.array([10, 0, 90, 100.0])
    assert metrics.mae(y, yhat) == pytest.approx(5.0)
    assert metrics.mae_on(y, yhat) == pytest.approx(5.0)
    assert metrics.sae(y, yhat) == pytest.approx(0.0)
    assert metrics.nde(y, yhat) == pytest.approx(np.sqrt(200 / 20000))


def test_metrics_ignore_nan_truth():
    y = np.array([np.nan, 100.0])
    yhat = np.array([1e6, 100.0])
    assert metrics.mae(y, yhat) == 0.0


def test_state_scores():
    y = np.array([0, 50, 50, 0.0])
    yhat = np.array([50, 50, 0, 0.0])
    s = metrics.state_scores(y, yhat, threshold=20)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5


def test_energy_split_sums():
    y = np.array([100, 100, 0.0])
    yhat = np.array([50, 150, 20.0])
    e = metrics.energy_split(y, yhat)
    assert e["overlap_pct"] == pytest.approx(75.0)
    assert e["missing_pct"] == pytest.approx(25.0)
    assert e["extra_pct"] == pytest.approx(35.0)


def test_cycle_matching_perfect_and_shifted():
    y = _series(np.r_[_wash(60), _wash(60)])
    perfect = metrics.cycle_scores(y, y)
    assert perfect["cycle_f1"] == 1.0 and perfect["matched_energy_mape_pct"] == pytest.approx(0.0)
    shifted = metrics.cycle_scores(y, y.shift(45).fillna(0))
    assert shifted["cycle_f1"] < 1.0


def test_g_loss():
    assert metrics.g_loss(30.0, 20.0) == pytest.approx(50.0)
