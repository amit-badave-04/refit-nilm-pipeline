"""Sequence-to-sequence training and inference on a tiny synthetic corpus (CPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from refit_nilm import datasets as D
from refit_nilm import train_seq2seq as S

WINDOW = 17


def _frame(house: int, days: int = 3, seed: int = 0) -> pd.DataFrame:
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


@pytest.fixture(scope="module")
def corpus() -> D.Corpus:
    frames = {h: _frame(h) for h in (1, 2, 3)}
    spec = D.SplitSpec(train_houses=[1], val_house=2, test_houses=[3])
    return D.build_corpus(frames, D.assign_splits(frames, spec, WINDOW), WINDOW)


class _Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"power": x[:, 0, :]}


def _small_cfg(**kw) -> S.Seq2SeqConfig:
    return S.Seq2SeqConfig(window=WINDOW, batch_size=16, max_epochs=2, patience=2, val_stride=7, **kw)


def test_window_source_gathers_aligned_inputs_targets_and_masks(corpus):
    scale = S.training_scale(corpus)
    src = S.SeqWindowSource(corpus, scale, "cpu")
    starts = torch.as_tensor(corpus.starts["train"][:5])
    x, y, ok = src.batch(starts)
    assert x.shape == (5, 9, WINDOW) and y.shape == (5, WINDOW) and ok.shape == (5, WINDOW)
    s0 = int(starts[0])
    np.testing.assert_allclose(x[0, 0].numpy() * scale, corpus.agg[s0 : s0 + WINDOW], rtol=1e-5)
    assert np.array_equal(ok[0].numpy(), ~np.isnan(corpus.wm[s0 : s0 + WINDOW]))


def test_predict_returns_the_centre_minute_in_watts(corpus):
    scale = S.training_scale(corpus)
    src = S.SeqWindowSource(corpus, scale, "cpu")
    starts = corpus.starts["val"][:50]
    pred = S.predict(_Identity(), src, starts, batch_size=16)
    np.testing.assert_allclose(pred, corpus.agg[starts + WINDOW // 2], rtol=1e-5)


def test_fit_runs_is_deterministic_and_round_trips(corpus, tmp_path):
    model_a, hist, scale = S.fit(corpus, _small_cfg(), device="cpu")
    model_b, _, _ = S.fit(corpus, _small_cfg(), device="cpu")
    assert 1 <= len(hist) <= 2 and scale == S.training_scale(corpus)
    for (ka, va), (kb, vb) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), ka
    path = tmp_path / "run.pt"
    S.save_run(path, model_a, _small_cfg(), scale, hist)
    loaded, cfg, sc = S.load_run(path, device="cpu")
    src = S.SeqWindowSource(corpus, scale, "cpu")
    starts = corpus.starts["val"][:40]
    np.testing.assert_array_equal(S.predict(model_a.eval(), src, starts), S.predict(loaded, src, starts))
    assert cfg == _small_cfg() and sc == scale
    assert not torch.are_deterministic_algorithms_enabled()  # fit restores the caller's setting


def test_fit_stops_with_a_clear_error_on_non_finite_loss(corpus, monkeypatch):
    class _NaN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return {"power": x[:, 0, :] * self.w * float("nan")}

    monkeypatch.setattr(S, "build", lambda cfg: _NaN())
    with pytest.raises(RuntimeError, match="training loss became nan"):
        S.fit(corpus, _small_cfg(), device="cpu")


def test_fit_reports_when_validation_never_improves(corpus, monkeypatch):
    class _NaNOnValidation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            out = x[:, 0, :] * self.w
            return {"power": out if self.training else out * float("nan")}

    monkeypatch.setattr(S, "build", lambda cfg: _NaNOnValidation())
    with pytest.raises(RuntimeError, match="validation NDE never improved"):
        S.fit(corpus, _small_cfg(), device="cpu")


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown seq2seq model"):
        S.build(S.Seq2SeqConfig(model="transformer-xl"))
