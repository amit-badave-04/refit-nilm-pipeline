"""Sequence-to-sequence training and inference on a tiny synthetic corpus (CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from refit_nilm import datasets as D
from refit_nilm import train_seq2seq as S

from _synthetic import house_frame

WINDOW = 17


@pytest.fixture(scope="module")
def corpus() -> D.Corpus:
    frames = {h: house_frame(h) for h in (1, 2, 3)}
    spec = D.SplitSpec(train_houses=[1], val_house=2, test_houses=[3])
    return D.build_corpus(frames, D.assign_splits(frames, spec, WINDOW), WINDOW)


class _Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"power": x[:, 0, :]}


def _small_cfg(**kw) -> S.Seq2SeqConfig:
    base = {"window": WINDOW, "batch_size": 16, "max_epochs": 2, "patience": 2, "val_stride": 7}
    return S.Seq2SeqConfig(**{**base, **kw})


def test_window_source_gathers_aligned_inputs_targets_masks_and_calendar(corpus):
    scale = S.training_scale(corpus)
    src = S.SeqWindowSource(corpus, scale, "cpu")
    wash = int(np.flatnonzero(np.nan_to_num(corpus.wm) > 1000)[0])
    starts = [92, wash - 5]  # one window across the unusable minutes 100-109, one across a wash
    x, y, ok = src.batch(torch.as_tensor(starts))
    assert x.shape == (2, 9, WINDOW) and y.shape == (2, WINDOW) and ok.shape == (2, WINDOW)
    for i, s0 in enumerate(starts):
        span = slice(s0, s0 + WINDOW)
        np.testing.assert_allclose(x[i, 0].numpy() * scale, corpus.agg[span], rtol=1e-5)
        np.testing.assert_allclose(y[i].numpy() * scale, np.nan_to_num(corpus.wm[span]), rtol=1e-5, atol=1e-3)
        assert np.array_equal(ok[i].numpy(), ~np.isnan(corpus.wm[span]))
        np.testing.assert_allclose(x[i, 1:].numpy(), S.calendar_features(corpus.time[span]), atol=1e-6)
    assert not ok[0].all() and ok[1].all() and y[1].max() > 0  # both cases are really exercised


def test_masked_mse_ignores_unusable_minutes():
    target = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    ok = torch.tensor([[True, True, False, True]])
    pred = target.clone()
    pred[0, 2] = 1e6  # wrong only where the target is unusable
    assert float(S.masked_mse(pred, target, ok)) == 0.0
    pred[0, 0] = 3.0  # one usable minute off by 2 -> 4 / 3 usable minutes
    assert float(S.masked_mse(pred, target, ok)) == pytest.approx(4 / 3)
    assert float(S.masked_mse(pred, target, torch.zeros_like(ok))) == 0.0


def test_training_scale_is_the_largest_training_aggregate(corpus):
    assert S.training_scale(corpus) == pytest.approx(float(corpus.agg[corpus.targets("train")].max()))


def test_predict_returns_the_centre_minute_in_watts_clipped_at_zero(corpus):
    scale = S.training_scale(corpus)
    src = S.SeqWindowSource(corpus, scale, "cpu")
    starts = corpus.starts["val"][:50]
    np.testing.assert_allclose(S.predict(_Identity(), src, starts, batch_size=16), corpus.agg[starts + WINDOW // 2], rtol=1e-5)

    class _Negative(torch.nn.Module):
        def forward(self, x):
            return {"power": -x[:, 0, :]}

    assert np.all(S.predict(_Negative(), src, starts) == 0.0)


def test_fit_is_deterministic_and_round_trips(corpus, tmp_path):
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


def test_fit_stops_early_and_restores_the_best_epoch(corpus, monkeypatch):
    snapshots, calls = [], {"n": 0}
    original_epoch, original_predict = S._epoch, S.predict

    def recording_epoch(model, *args, **kwargs):
        loss = original_epoch(model, *args, **kwargs)
        snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return loss

    def scripted_predict(model, src, starts, batch_size=4096):
        calls["n"] += 1
        truth = np.nan_to_num(corpus.wm[starts + WINDOW // 2])
        return truth if calls["n"] == 1 else np.zeros_like(truth)  # best at epoch 1, worse after

    monkeypatch.setattr(S, "_epoch", recording_epoch)
    monkeypatch.setattr(S, "predict", scripted_predict)
    model, hist, _ = S.fit(corpus, _small_cfg(max_epochs=5, patience=1), device="cpu")
    assert len(hist) == 2  # stopped one epoch after the last improvement, not at max_epochs
    state = model.state_dict()
    assert all(torch.equal(state[k], snapshots[0][k]) for k in state)
    assert any(not torch.equal(snapshots[0][k], snapshots[1][k]) for k in state)
    monkeypatch.setattr(S, "predict", original_predict)


def test_fit_enables_deterministic_mode_and_restores_the_callers_setting(corpus, monkeypatch):
    seen = []

    class _Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            seen.append(torch.are_deterministic_algorithms_enabled())
            return {"power": x[:, 0, :] * self.w}

    monkeypatch.setattr(S, "build", lambda cfg: _Recorder())
    torch.use_deterministic_algorithms(False)
    S.fit(corpus, _small_cfg(max_epochs=1), device="cpu")
    assert seen and all(seen) and not torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        S.fit(corpus, _small_cfg(max_epochs=1), device="cpu")
        assert torch.are_deterministic_algorithms_enabled() and torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(False)


def test_fit_stops_with_a_clear_error_on_non_finite_loss_and_restores_the_setting(corpus, monkeypatch):
    class _NaN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return {"power": x[:, 0, :] * self.w * float("nan")}

    monkeypatch.setattr(S, "build", lambda cfg: _NaN())
    with pytest.raises(RuntimeError, match="training loss became nan"):
        S.fit(corpus, _small_cfg(), device="cpu")
    assert not torch.are_deterministic_algorithms_enabled()


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
