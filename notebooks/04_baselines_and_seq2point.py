# %% [markdown]
# # 04 · Baselines and the reference model (Seq2Point)
#
# **Requirement 3.** Train simple baselines and the reference deep model, and choose the window
# length, using the validation house only. The test homes are not touched in this notebook.
#
# | | |
# |---|---|
# | **Models** | M0a always-off · M0b weekday × hour usage profile · M1 LightGBM on window features · M2 Seq2Point (W = 81 and W = 237) |
# | **Output** | checkpoints in `artifacts/models/` (not versioned; rebuilt by this notebook), `artifacts/tables/nb04_validation.csv`, `artifacts/metrics/nb04_summary.json`, learning curves |
# | **Run time** | about 40 minutes on one laptop GPU (RTX-class); LightGBM on CPU |
#
# **Why these models.**
# * **Floors (M0).** An always-off predictor shows how flattering MAE is on a sparse target
#   (the WM is off ~97 % of the time); a usage profile shows what behaviour alone, without any
#   power signal, can explain.
# * **LightGBM (M1).** A strong non-deep baseline on hand-crafted window statistics. If it came
#   close to the deep model, the extra complexity would not be justified.
# * **Seq2Point (M2).** The reference architecture for REFIT washing machines (Zhang et al.
#   2018; D'Incecco et al. 2020): a CNN maps a window of aggregate power to the appliance power at
#   the window's centre. It is chosen because it is the most widely replicated NILM model on
#   REFIT, it is small enough to train several seeds in minutes, and its window-to-point design
#   suits a load with a long, distinctive temporal shape. Transformer models (e.g. NILMFormer,
#   KDD 2025) report lower 1-minute washer errors, but they are a heavier build and are listed as
#   the next step.

# %%
import json
import time
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from refit_nilm import datasets as D
from refit_nilm import features as F
from refit_nilm import metrics, plots
from refit_nilm import config as C
from refit_nilm import train as T

warnings.filterwarnings("ignore", category=FutureWarning)
plots.setup()
pd.set_option("display.width", 160, "display.max_columns", 30)
P = plots.PALETTE
C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
SUMMARY: dict = {}

WINDOWS = [81, 237]
frames, corpora, stats = D.prepare(WINDOWS)
VAL = C.VAL_HOUSE

# %% [markdown]
# ## 1. Validation scoring
#
# Every model is scored on House 18 with the same metric set (definitions in
# `src/refit_nilm/metrics.py`):
#
# * **MAE** (W) — average error per minute; flattering on sparse targets.
# * **MAE_ON** (W) — error only while the machine is really on (≥ 20 W).
# * **SAE** — relative error in total energy over the period; **EpD** (Wh/day) — mean absolute
#   daily energy error. These matter most for an energy report.
# * **NDE** — squared error normalised by the target's energy; 1.0 equals predicting zero.
# * **F1** — on/off agreement minute by minute at 20 W.
# * **cycle F1** — cycles detected on prediction and truth with the notebook-02 rule and
#   matched by temporal overlap (IoU ≥ 0.5). Project-specific, stated as such.
# * **energy split** — share of true energy recovered (overlap), missed, and predicted in excess.
#
# No single metric is enough here: a model can lower MAE by predicting less, and lower missed
# energy by predicting more. Selection uses **NDE** (the training loss, energy-normalised) with
# MAE_ON, F1 and cycle F1 as checks.

# %%
def score(name: str, corpus: D.Corpus, split: str, power: np.ndarray, p_on: np.ndarray | None = None) -> dict:
    starts = corpus.starts[split]
    rows = []
    for h in D.split_houses(corpus, split):
        y = D.to_series(corpus, starts, corpus.wm[starts + corpus.window // 2], h)
        yh = D.to_series(corpus, starts, power, h)
        po = D.to_series(corpus, starts, p_on, h) if p_on is not None else None
        rows.append({"model": name, "house": h, **metrics.all_metrics(y, yh, po)})
    return rows


val_rows = []
cv = corpora[237]
y_val = cv.wm[cv.targets("val")]
val_rows += score("M0a always-off", cv, "val", np.zeros_like(y_val))

# %% [markdown]
# ## 2. M0b: weekday × hour usage profile
#
# The mean WM power for each local (weekday, hour) over the training period of the training
# houses. It uses no aggregate signal at all: it is what "people do laundry on Saturday
# mornings" is worth.

# %%
tr_t = cv.targets("train")
y_tr = pd.Series(cv.wm[tr_t], index=pd.to_datetime(cv.time[tr_t], unit="s", utc=True))
profile = F.time_profile(y_tr)
val_idx = pd.to_datetime(cv.time[cv.targets("val")], unit="s", utc=True)
val_rows += score("M0b weekday×hour profile", cv, "val", F.apply_profile(profile, val_idx))
profile.to_csv(C.TABLE_DIR / "nb04_m0b_profile.csv")

# %% [markdown]
# ## 3. M1: LightGBM on window features
#
# Features (all from the aggregate, centred windows of 5–237 minutes): rolling mean, standard
# deviation, minimum and maximum; counts of rising edges above 200 W and of steps above 1 kW;
# share of minutes above 1.5 kW (heating-like load); the aggregate minus its 61-minute minimum
# (a crude "active load above baseline"); and local time of day / weekend.
#
# Trained on a random 1.5 M of the 3.75 M training minutes (L2 loss, early stopping on House 18).

# %%
t0 = time.time()
feat = {}
for h, f in frames.items():
    a = f["agg"].where(~f["agg_missing"])
    feat[h] = F.window_features(a)
print(f"features built in {time.time() - t0:.0f}s, {feat[2].shape[1]} columns")


def design(corpus: D.Corpus, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    t = corpus.targets(split)
    parts, ys = [], []
    for h in D.split_houses(corpus, split):
        th = t[corpus.house[t] == h]
        idx = pd.to_datetime(corpus.time[th], unit="s", utc=True)
        parts.append(feat[h].loc[idx])
        ys.append(corpus.wm[th])
    return pd.concat(parts), np.concatenate(ys)


X_tr, Y_tr = design(cv, "train")
rng = np.random.default_rng(42)
sub = rng.choice(len(X_tr), size=min(1_500_000, len(X_tr)), replace=False)
X_va, Y_va = design(cv, "val")
t0 = time.time()
lgbm = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
)
lgbm.fit(X_tr.iloc[sub], Y_tr[sub], eval_set=[(X_va, Y_va)], eval_metric="l2", callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"LightGBM: {lgbm.best_iteration_} trees, {time.time() - t0:.0f}s")
lgbm.booster_.save_model(str(C.MODEL_DIR / "m1_lightgbm.txt"))
pred_lgbm = np.clip(lgbm.predict(X_va), 0, None)
val_rows += score("M1 LightGBM", cv, "val", pred_lgbm)
imp = pd.Series(lgbm.booster_.feature_importance("gain"), index=X_tr.columns).sort_values(ascending=False)
(100 * imp / imp.sum()).head(10).round(1).to_frame("gain %")

# %% [markdown]
# ## 4. M2: Seq2Point, two window lengths (seed 42)
#
# Training settings follow the reference implementation: Adam (lr 1e-3), batch 1,000, MSE loss,
# at most 30 epochs (one epoch = every training window once, in random order), early stopping on
# the validation loss with patience 5.

# %%
histories = {}
for W in WINDOWS:
    cfg = T.TrainConfig(model="seq2point", window=W, seed=42)
    t0 = time.time()
    model, hist = T.fit(corpora[W], stats[W], cfg, device=DEVICE)
    T.save_run(C.MODEL_DIR / f"m2_seq2point_w{W}_s42.pt", model, cfg, stats[W], hist)
    histories[(W, 42)] = hist
    src = T.WindowSource(corpora[W], stats[W], DEVICE)
    pred = T.predict(model, src, corpora[W].starts["val"], stats[W])["power"]
    val_rows += score(f"M2 Seq2Point W={W} s42", corpora[W], "val", pred)
    print(f"W={W}: {len(hist)} epochs, best val NDE {min(h['val_nde'] for h in hist):.4f}, {time.time() - t0:.0f}s")
    del model, src
    torch.cuda.empty_cache()

# %%
val = pd.DataFrame(val_rows)
cols = ["model", "mae_w", "mae_on_w", "sae", "epd_wh", "nde", "f1", "cycle_f1", "overlap_pct", "extra_pct"]
val[cols].round(3)

# %% [markdown]
# **Window choice.** The window with the lower validation NDE is kept for the remaining seeds and
# for the improved model; the other metrics are checked for agreement.

# %%
w_scores = {W: float(val.loc[val.model == f"M2 Seq2Point W={W} s42", "nde"].iloc[0]) for W in WINDOWS}
BEST_W = min(w_scores, key=w_scores.get)
SUMMARY["window_selection"] = {"val_nde": w_scores, "chosen_window": BEST_W}
print("chosen window:", BEST_W, w_scores)

# %% [markdown]
# ## 5. Seq2Point, two more seeds at the chosen window
#
# Deep models trained on the same data can differ noticeably from seed to seed, especially on a
# sparse target. Three seeds (42, 10, 20) give a mean and a spread, so later comparisons are not
# read off a single lucky run.

# %%
for seed in [s for s in C.SEEDS if s != 42]:
    cfg = T.TrainConfig(model="seq2point", window=BEST_W, seed=seed)
    model, hist = T.fit(corpora[BEST_W], stats[BEST_W], cfg, device=DEVICE)
    T.save_run(C.MODEL_DIR / f"m2_seq2point_w{BEST_W}_s{seed}.pt", model, cfg, stats[BEST_W], hist)
    histories[(BEST_W, seed)] = hist
    src = T.WindowSource(corpora[BEST_W], stats[BEST_W], DEVICE)
    pred = T.predict(model, src, corpora[BEST_W].starts["val"], stats[BEST_W])["power"]
    val_rows += score(f"M2 Seq2Point W={BEST_W} s{seed}", corpora[BEST_W], "val", pred)
    del model, src
    torch.cuda.empty_cache()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for (W, seed), hist in histories.items():
    h = pd.DataFrame(hist)
    axes[0].plot(h.epoch, h.train_loss, label=f"W={W} seed {seed}")
    axes[1].plot(h.epoch, h.val_nde, label=f"W={W} seed {seed}")
axes[0].set_title("training loss (scaled MSE)")
axes[1].set_title("validation NDE, House 18 (1.0 = predicting zero)")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend()
plots.save(fig, "nb04_learning_curves")
plt.show()

# %%
val = pd.DataFrame(val_rows)
val.to_csv(C.TABLE_DIR / "nb04_validation.csv", index=False)
SUMMARY["validation"] = val[cols].round(4).to_dict(orient="records")
SUMMARY["lightgbm_trees"] = int(lgbm.best_iteration_)
with open(C.METRIC_DIR / "nb04_summary.json", "w") as fh:
    json.dump(SUMMARY, fh, indent=2, default=str)
val[cols].round(3)

# %% [markdown]
# ## Summary
#
# **Validation house 18** (WM on in 1 % of minutes, washer-dryer present). Deep-model figures
# for W = 81 are the mean of three seeds.
#
# | model | MAE (W) | NDE | SAE | F1 | cycle F1 | true energy recovered | extra energy |
# |---|---|---|---|---|---|---|---|
# | M0a always-off | **3.9** | 1.00 | 1.00 | 0 | 0 | 0 % | 0 % |
# | M0b weekday × hour profile | 19.7 | 1.03 | 3.15 | 0.02 | 0.00 | 5 % | 410 % |
# | M1 LightGBM | 16.0 | 0.87 | 2.65 | 0.13 | 0.34 | 28 % | 337 % |
# | M2 Seq2Point, W = 81 | 5.3 | **0.67** | 0.59 | **0.43** | **0.60** | 61 % | 98 % |
#
# **What this shows.**
# * **MAE alone would pick the model that never predicts anything.** On a target that is off
#   99 % of the time, "always off" has the lowest MAE. That is why model selection uses NDE and
#   the event metrics.
# * **Behaviour alone has no skill.** The weekday × hour profile spreads power over typical laundry
#   hours and is worse than predicting zero.
# * **Hand-crafted features find "a big load", not "the washing machine".** LightGBM's validation
#   error started rising after 21 trees: what it learns about the training homes does not transfer.
#   Its recall is high (0.80) but precision is 0.07, so most minutes it calls "washing" are other
#   appliances.
# * **Seq2Point learns the signature.** It finds 71 % of the cycles in an unseen home (cycle recall; cycle F1 0.60) and recovers
#   61 % of the true energy. Its dominant error is **false positives**: it adds almost as much
#   energy as it recovers (98 %), mostly while other 2 kW appliances run.
# * **Window.** W = 81 (the 1-minute equivalent of the reference 80-minute window) beats W = 237
#   on validation NDE (0.72 vs 0.80, seed 42) and on every event metric. The longer window gives
#   the dense layer three times as many inputs to overfit with. W = 81 is kept.
#
# **Seed spread.** Validation NDE ranges 0.62–0.72 across seeds, so differences smaller than
# about 0.05 between models should not be read as real.
#
# **Next:** notebook 05 tests whether gating the output by an on/off classifier removes the false
# positives without losing the cycles.
