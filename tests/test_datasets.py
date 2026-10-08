import numpy as np
import pandas as pd
import pytest
import torch

from refit_nilm import datasets as D
from refit_nilm.models.seq2point import build


def _frame(house: int, n: int = 3000, start: str = "2014-01-01", missing=()):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    agg = pd.Series(300 + 50 * np.sin(np.arange(n) / 30), index=idx)
    wm = pd.Series(np.where((np.arange(n) % 500) < 60, 500.0, 0.0), index=idx)
    for a, b in missing:
        agg.iloc[a:b] = np.nan
    minutes = pd.DataFrame({"Aggregate": agg, "Appliance1": wm, "n_readings": agg.notna().astype(int) * 7, "issues": 0}, index=idx)
    return D.house_frame(house, "Appliance1", minutes)


def test_short_gap_ffill_only_fills_short_runs():
    s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0])
    out = D.short_gap_ffill(s, limit=3)
    assert out.tolist()[:4] == [1.0, 1.0, 1.0, 4.0]
    assert out.iloc[4:8].isna().all()


def test_flat_run_mask():
    x = np.r_[np.arange(10.0), np.full(70, 5.0), np.arange(5.0)]
    m = D.flat_run_mask(x, min_len=60)
    assert m.sum() == 70


def test_windows_never_cross_splits_or_houses():
    frames = {2: _frame(2), 18: _frame(18), 8: _frame(8)}
    spec = D.SplitSpec(train_houses=[2], val_house=18, test_houses=[8], seen_test_frac=0.3, embargo_days=0)
    W = 61
    labels = D.assign_splits(frames, spec, W)
    corpus = D.build_corpus(frames, labels, W)
    audit = D.leakage_audit(corpus)
    assert (audit["windows_spanning_two_labels"] == 0).all()
    assert (audit["windows_spanning_two_houses"] == 0).all()
    assert set(audit["split"]) == {"train", "seen_test", "val", "test"}
    gaps = D.temporal_gap_check(corpus)
    assert gaps["ok"].all()


def test_missing_fraction_cap_removes_windows():
    W = 61
    full = D.build_corpus({8: _frame(8)}, {8: pd.Series("test", index=_frame(8).index)}, W)
    holed_frame = _frame(8, missing=[(1000, 1020)])  # 20 missing minutes > 10 % of 61
    holed = D.build_corpus({8: holed_frame}, {8: pd.Series("test", index=holed_frame.index)}, W)
    assert len(holed.starts["test"]) < len(full.starts["test"]) - 20


def test_standardisation_uses_training_targets_only():
    frames = {2: _frame(2), 8: _frame(8)}
    frames[8]["agg"] = frames[8]["agg"] + 10_000  # a test house with a very different level
    spec = D.SplitSpec(train_houses=[2], val_house=99, test_houses=[8], seen_test_frac=0.2, embargo_days=0)
    corpus = D.build_corpus(frames, D.assign_splits(frames, spec, 61), 61)
    stats = D.standardisation(corpus)
    assert stats["agg_mean"] < 1000


@pytest.mark.parametrize("name", ["seq2point", "gated_seq2point"])
def test_models_output_shapes(name):
    m = build(name, 61)
    out = m(torch.randn(4, 61))
    key = "power" if name == "seq2point" else "p_on"
    assert out[key].shape == (4,)
