"""Seq2Point training loop on a tiny synthetic corpus (CPU): a normal run and both failure paths."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from refit_nilm import datasets as D
from refit_nilm import train as T

from _synthetic import house_frame

WINDOW = 17


@pytest.fixture(scope="module")
def corpus_and_stats():
    frames = {h: house_frame(h) for h in (1, 2, 3)}
    spec = D.SplitSpec(train_houses=[1], val_house=2, test_houses=[3])
    cp = D.build_corpus(frames, D.assign_splits(frames, spec, WINDOW), WINDOW)
    return cp, D.standardisation(cp)


def _cfg(**kw) -> T.TrainConfig:
    base = {"window": WINDOW, "batch_size": 64, "max_epochs": 2, "min_epochs": 1, "patience": 1,
            "train_samples_per_epoch": 512, "val_stride": 5}
    return T.TrainConfig(**{**base, **kw})


def test_fit_trains_and_predicts_non_negative_watts(corpus_and_stats):
    cp, st = corpus_and_stats
    model, hist = T.fit(cp, st, _cfg(), device="cpu")
    assert 1 <= len(hist) <= 2 and np.isfinite(hist[0]["val_nde"])
    out = T.predict(model, T.WindowSource(cp, st, "cpu"), cp.starts["val"][:100], st)["power"]
    assert out.shape == (100,) and np.all(out >= 0)


def test_fit_stops_with_a_clear_error_on_non_finite_loss(corpus_and_stats, monkeypatch):
    class _NaN(torch.nn.Module):
        def __init__(self, window):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return {"power": x[:, 0] * self.w * float("nan")}

    cp, st = corpus_and_stats
    monkeypatch.setattr(T, "build", lambda name, window: _NaN(window))
    with pytest.raises(RuntimeError, match="training loss became nan"):
        T.fit(cp, st, _cfg(), device="cpu")


def test_fit_reports_when_validation_never_improves(corpus_and_stats, monkeypatch):
    class _NaNOnValidation(torch.nn.Module):
        def __init__(self, window):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            out = x[:, 0] * self.w
            return {"power": out if self.training else out * float("nan")}

    cp, st = corpus_and_stats
    monkeypatch.setattr(T, "build", lambda name, window: _NaNOnValidation(window))
    with pytest.raises(RuntimeError, match="validation NDE never improved"):
        T.fit(cp, st, _cfg(), device="cpu")
