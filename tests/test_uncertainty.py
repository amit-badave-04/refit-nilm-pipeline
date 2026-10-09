"""Bootstrap statistics must reproduce the point metrics exactly, and event matching by start time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from refit_nilm import metrics
from refit_nilm import uncertainty as U


def _synthetic(days: int = 6, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=days * 1440, freq="1min", tz="UTC")
    y = np.zeros(len(idx))
    for d in range(days):
        s = d * 1440 + 600 + int(rng.integers(0, 120))
        y[s : s + 15] = 2000
        y[s + 15 : s + 80] = 150
    yhat = np.clip(y * 0.7 + rng.normal(0, 10, len(y)), 0, None)
    yhat[3 * 1440 + 200 : 3 * 1440 + 240] = 400  # one false cycle
    y_s, p_s = pd.Series(y, idx), pd.Series(yhat, idx)
    y_s.iloc[4 * 1440 + 30 : 4 * 1440 + 90] = np.nan  # an outage
    return y_s, p_s


def test_daily_statistics_reproduce_the_point_metrics():
    y, yhat = _synthetic()
    point = metrics.all_metrics(y, yhat)
    from_days = U.metrics_from_days(U.daily_stats(y, yhat))
    assert from_days["nde"] == pytest.approx(point["nde"], rel=1e-9)
    assert from_days["epd_wh"] == pytest.approx(point["epd_wh"], rel=1e-9)
    assert from_days["cycle_f1"] == pytest.approx(point["cycle_f1"], rel=1e-9)


def test_daily_statistics_match_point_metrics_when_the_prediction_has_a_gap():
    y, yhat = _synthetic(days=8, seed=2)
    first_wash = int(np.flatnonzero(y.to_numpy() > 0)[0])
    yhat.iloc[first_wash - 5 : first_wash + 85] = np.nan  # prediction missing across a whole true cycle
    point = metrics.all_metrics(y, yhat)
    from_days = U.metrics_from_days(U.daily_stats(y, yhat))
    for k in ("nde", "epd_wh", "cycle_f1"):
        assert from_days[k] == pytest.approx(point[k], rel=1e-9), k


def test_bootstrap_interval_contains_the_point_estimate():
    y, yhat = _synthetic(days=20)
    d = U.daily_stats(y, yhat)
    ci = U.percentile_ci(U.bootstrap_home(d, n_boot=300))
    point = U.metrics_from_days(d)
    for k in ("nde", "epd_wh"):
        assert ci.loc[k, "low"] <= point[k] <= ci.loc[k, "high"]


def test_stationary_bootstrap_draws_runs_of_consecutive_days():
    rng = np.random.default_rng(0)
    idx = U.stationary_indices(200, mean_block=7, rng=rng)
    assert len(idx) == 200 and idx.min() >= 0 and idx.max() < 200
    steps = np.diff(idx)
    assert np.mean((steps == 1) | (steps == -199)) > 0.7   # mostly consecutive (circular), as blocks should be


def test_cluster_bootstrap_resamples_whole_homes():
    per_home = pd.DataFrame({"nde": [0.4, 0.7, 1.4]})
    reps = U.cluster_bootstrap(per_home, n_boot=500, seed=1)
    # every replicate is a mean of three of the three values, so it lies within their range
    assert reps["nde"].between(0.4, 1.4).all() and reps["nde"].round(9).nunique() <= 10  # C(5, 3) multisets


def test_match_by_start_is_one_to_one():
    t = pd.DataFrame({"start": pd.to_datetime(["2014-01-01 10:00", "2014-01-01 10:20"], utc=True)})
    p = pd.DataFrame({"start": pd.to_datetime(["2014-01-01 10:05", "2014-01-01 10:06", "2014-01-01 13:00"], utc=True)})
    assert U.match_by_start(t, p, tol_min=10) == 1   # both predictions are near the first true cycle only
    assert U.match_by_start(t, p, tol_min=15) == 2   # 10:06 can now take the 10:20 cycle
    assert U.f1_from_counts(2, 3, 2) == pytest.approx(0.8)
