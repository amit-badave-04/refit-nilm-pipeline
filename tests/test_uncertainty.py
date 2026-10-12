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


def test_daily_statistics_match_point_metrics_when_the_prediction_misses_a_whole_day():
    y, yhat = _synthetic(days=8, seed=2)
    yhat.iloc[2 * 1440 : 3 * 1440] = np.nan  # no prediction at all on day 3, which has a true wash
    point = metrics.all_metrics(y, yhat)
    d = U.daily_stats(y, yhat)
    assert d["n_true"].sum() == point["true_cycles"] and d["n_pred"].sum() == point["pred_cycles"]
    for k in ("nde", "epd_wh", "cycle_f1"):
        assert U.metrics_from_days(d)[k] == pytest.approx(point[k], rel=1e-9), k


def test_bootstrap_interval_contains_the_point_estimate_and_really_resamples():
    y, yhat = _synthetic(days=20)
    d = U.daily_stats(y, yhat)
    reps = U.bootstrap_home(d, n_boot=300)
    ci = U.percentile_ci(reps)
    point = U.metrics_from_days(d)
    for k in ("nde", "epd_wh"):
        assert reps[k].nunique() > 50  # replicates differ: days are actually being resampled
        assert ci.loc[k, "low"] < point[k] < ci.loc[k, "high"]


def test_percentile_interval_returns_the_requested_quantiles():
    reps = pd.DataFrame({"m": np.arange(1001, dtype=float)})  # 0..1000
    ci = U.percentile_ci(reps, level=0.95)
    assert ci.loc["m", "low"] == pytest.approx(25.0) and ci.loc["m", "high"] == pytest.approx(975.0)
    ci90 = U.percentile_ci(reps, level=0.90)
    assert ci90.loc["m", "low"] == pytest.approx(50.0) and ci90.loc["m", "high"] == pytest.approx(950.0)


def test_stationary_bootstrap_draws_runs_of_consecutive_days_with_the_requested_mean_length():
    rng = np.random.default_rng(0)
    idx = U.stationary_indices(20_000, mean_block=7, rng=rng)
    assert len(idx) == 20_000 and idx.min() >= 0 and idx.max() < 20_000
    breaks = np.flatnonzero(np.diff(idx) != 1)  # a new block starts wherever the next index is not consecutive
    mean_run = len(idx) / (len(breaks) + 1)
    assert 6.0 < mean_run < 8.0  # geometric block lengths with mean 7 (a few joins are invisible)


def test_seven_day_blocks_widen_the_interval_when_days_are_dependent():
    rng = np.random.default_rng(3)
    n = 400
    level = np.repeat(rng.normal(0, 1, n // 20), 20)  # 20-day spells: strong day-to-day dependence
    d = pd.DataFrame({"sum_y2": 1.0, "sum_err2": np.exp(level), "y_kwh": 1.0, "yhat_kwh": 1.0 + np.exp(level),
                      "energy_day": True, "n_true": 1, "n_pred": 1, "n_matched": 1},
                     index=pd.date_range("2014-01-01", periods=n, freq="D", tz="UTC"))
    width = lambda reps: float(reps["epd_wh"].quantile(0.975) - reps["epd_wh"].quantile(0.025))  # noqa: E731
    one_day = width(U.bootstrap_home(d, n_boot=400, seed=1))
    week = width(U.bootstrap_home(d, n_boot=400, seed=1, mean_block=7))
    assert week > 1.3 * one_day  # the block option is really used, and it does what it is for


def test_cluster_bootstrap_resamples_whole_homes():
    per_home = pd.DataFrame({"nde": [0.4, 0.7, 1.4]})
    per_home["twice"] = 2 * per_home["nde"]
    reps = U.cluster_bootstrap(per_home, n_boot=500, seed=1)
    # every replicate is a mean of three of the three values, so it lies within their range
    assert reps["nde"].between(0.4, 1.4).all() and reps["nde"].round(9).nunique() <= 10  # C(5, 3) multisets
    assert reps["nde"].nunique() > 3  # it does resample
    # columns are resampled together (whole homes), so the tie between them holds in every replicate
    np.testing.assert_allclose(reps["twice"], 2 * reps["nde"])


def test_match_by_start_is_one_to_one():
    t = pd.DataFrame({"start": pd.to_datetime(["2014-01-01 10:00", "2014-01-01 10:20"], utc=True)})
    p = pd.DataFrame({"start": pd.to_datetime(["2014-01-01 10:05", "2014-01-01 10:06", "2014-01-01 13:00"], utc=True)})
    assert U.match_by_start(t, p, tol_min=10) == 1   # both predictions are near the first true cycle only
    assert U.match_by_start(t, p, tol_min=15) == 2   # 10:06 can now take the 10:20 cycle
    assert U.f1_from_counts(2, 3, 2) == pytest.approx(0.8)
