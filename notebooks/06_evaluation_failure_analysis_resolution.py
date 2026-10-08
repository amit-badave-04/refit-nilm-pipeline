# %% [markdown]
# # 06 · Evaluation on unseen homes, failure analysis, and the effect of 15/30-minute data
#
# **Requirement 3 (evaluation) and Requirement 4 (resolution question).** All choices (window,
# loss weight, augmentation rate, epochs, thresholds, which model is primary) were frozen on the
# validation house in notebooks 04–05b. This notebook scores every model once on the test homes,
# explains where and why the models fail, and measures what is lost with 15- or 30-minute data.
#
# | | |
# |---|---|
# | **Input** | checkpoints from notebooks 04, 05 and 05b; the corpus of notebook 03 |
# | **Output** | `artifacts/tables/nb06_*`, `artifacts/figures/nb06_*`, `artifacts/metrics/nb06_summary.json` |
# | **Run time** | about 15 minutes (includes training the 15/30-minute models) |
#
# **Models.** M0a always-off · M0b weekday × hour profile · M1 LightGBM · M2 Seq2Point ·
# M3 gated Seq2Point · M4 Seq2Point trained with distractor augmentation · and the three-seed
# ensembles of M2 and M4. Single deep models are reported as mean ± sd over seeds 42, 10, 20.

# %%
import json
import re
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from refit_nilm import cycles, io, metrics, plots
from refit_nilm import config as C
from refit_nilm import datasets as D
from refit_nilm import features as F
from refit_nilm import train as T

warnings.filterwarnings("ignore", category=FutureWarning)
plots.setup()
pd.set_option("display.width", 170, "display.max_columns", 30)
P = plots.PALETTE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
nb04 = json.load(open(C.METRIC_DIR / "nb04_summary.json"))
nb05 = json.load(open(C.METRIC_DIR / "nb05_summary.json"))
nb05b = json.load(open(C.METRIC_DIR / "nb05b_summary.json"))
W = int(nb04["window_selection"]["chosen_window"])
LAMBDA = float(nb05["lambda_selection"]["chosen_lambda"])
M4_TAG = "dw" if "WM" in nb05b["chosen_config"] else "d"
PRIMARY = "M4 augmented ensemble" if nb05b["accepted"] else "M2 Seq2Point ensemble"
frames, corpora, stats = D.prepare([W])
cp, st = corpora[W], stats[W]
UNSEEN = [C.TEST_HOUSE] + C.EXTRA_UNSEEN_HOUSES
SPLITS = ["test", "seen_test", "val"]
amap = io.parse_appliance_map(C.DATA_DIR / "CLEAN_READ_ME_081116.txt")
SUMMARY: dict = {"window": W, "lambda": LAMBDA, "m4_config": nb05b["chosen_config"], "m4_accepted_on_validation": nb05b["accepted"], "primary_model": PRIMARY}
print(SUMMARY)

# %% [markdown]
# **Primary model.** Chosen on the validation house before this notebook was run: the M4 seed
# ensemble if augmentation passed its pre-registered acceptance rule in notebook 05b, otherwise
# the M2 seed ensemble.

# %% [markdown]
# ## 1. Predictions for every model on every split

# %%
PRED: dict[str, dict[str, dict]] = {}


def times(split: str) -> pd.DatetimeIndex:
    return pd.to_datetime(cp.time[cp.targets(split)], unit="s", utc=True)


PRED["M0a always-off"] = {sp: {"power": np.zeros(len(cp.starts[sp]))} for sp in SPLITS}
prof = pd.read_csv(C.TABLE_DIR / "nb04_m0b_profile.csv", index_col=[0, 1]).iloc[:, 0]
PRED["M0b weekday×hour profile"] = {sp: {"power": F.apply_profile(prof, times(sp))} for sp in SPLITS}

booster = lgb.Booster(model_file=str(C.MODEL_DIR / "m1_lightgbm.txt"))
feat = {h: F.window_features(f["agg"].where(~f["agg_missing"])) for h, f in frames.items()}
PRED["M1 LightGBM"] = {}
for sp in SPLITS:
    t = cp.targets(sp)
    idx = pd.to_datetime(cp.time[t], unit="s", utc=True)
    X = pd.concat([feat[h].loc[idx[cp.house[t] == h]] for h in D.split_houses(cp, sp)])
    PRED["M1 LightGBM"][sp] = {"power": np.clip(booster.predict(X), 0, None)}

RUNS = {f"M2 Seq2Point s{s}": C.MODEL_DIR / f"m2_seq2point_w{W}_s{s}.pt" for s in C.SEEDS}
RUNS.update({f"M3 gated s{s}": C.MODEL_DIR / f"m3_gated_w{W}_l{LAMBDA}_s{s}.pt" for s in C.SEEDS})
RUNS.update({f"M4 augmented s{s}": C.MODEL_DIR / f"m4_aug{M4_TAG}_w{W}_s{s}.pt" for s in C.SEEDS})
src = T.WindowSource(cp, st, DEVICE)
for name, path in RUNS.items():
    model, cfg, stt = T.load_run(path, DEVICE)
    PRED[name] = {sp: T.predict(model, src, cp.starts[sp], stt) for sp in SPLITS}
    del model
torch.cuda.empty_cache()
for fam, pref in [("M2 Seq2Point ensemble", "M2 Seq2Point s"), ("M4 augmented ensemble", "M4 augmented s")]:
    PRED[fam] = {sp: {"power": np.mean([PRED[f"{pref}{s}"][sp]["power"] for s in C.SEEDS], axis=0)} for sp in SPLITS}


def predict_family(prefix: str, corpus: D.Corpus, split: str = "test") -> np.ndarray:
    """Seed-averaged prediction of a model family on another corpus."""
    outs = []
    for s in C.SEEDS:
        model, _, stt = T.load_run(RUNS[f"{prefix} s{s}"], DEVICE)
        outs.append(T.predict(model, T.WindowSource(corpus, stt, DEVICE), corpus.starts[split], stt)["power"])
        del model
    return np.mean(outs, axis=0)


print(list(PRED))

# %% [markdown]
# ## 2. Scores per home

# %%
def family(name: str) -> str:
    return re.sub(r" s\d+$", "", name)


rows = []
for name, by_split in PRED.items():
    for sp in SPLITS:
        starts = cp.starts[sp]
        for h in D.split_houses(cp, sp):
            y = D.to_series(cp, starts, cp.wm[starts + W // 2], h)
            yh = D.to_series(cp, starts, by_split[sp]["power"], h)
            po = D.to_series(cp, starts, by_split[sp]["p_on"], h) if "p_on" in by_split[sp] else None
            rows.append({"model": name, "family": family(name), "split": sp, "house": h, **metrics.all_metrics(y, yh, po)})
scores = pd.DataFrame(rows)
scores.to_csv(C.TABLE_DIR / "nb06_scores_all.csv", index=False)
KEY = ["mae_w", "mae_on_w", "sae", "sae_daily", "epd_wh", "nde", "f1", "precision", "recall", "cycle_f1", "cycle_recall", "cycle_precision", "matched_energy_mape_pct", "overlap_pct", "missing_pct", "extra_pct"]
FAM_ORDER = ["M0a always-off", "M0b weekday×hour profile", "M1 LightGBM", "M2 Seq2Point", "M3 gated", "M4 augmented", "M2 Seq2Point ensemble", "M4 augmented ensemble"]

# %% [markdown]
# ### 2a. Primary result: House 8 (the reference unseen test house)

# %%
h8 = scores[(scores.split == "test") & (scores.house == C.TEST_HOUSE)]
h8_mean = h8.groupby("family")[KEY].mean().reindex(FAM_ORDER)
h8_std = h8.groupby("family")[KEY].std().reindex(FAM_ORDER)
h8_mean.to_csv(C.TABLE_DIR / "nb06_house8_mean.csv")
h8_std.to_csv(C.TABLE_DIR / "nb06_house8_std.csv")
SHOW = ["mae_w", "mae_on_w", "sae", "epd_wh", "nde", "f1", "cycle_f1", "cycle_recall", "overlap_pct", "extra_pct"]
fmt = h8_mean[SHOW].copy().astype(object)
for c in SHOW:
    fmt[c] = [f"{m:.3f} ± {s:.3f}" if not np.isnan(s) else f"{m:.3f}" for m, s in zip(h8_mean[c], h8_std[c])]
fmt

# %% [markdown]
# ### 2b. All six unseen homes, and seen homes
#
# Error / accuracy on unseen houses = mean over House 8 and the five extra unseen homes
# (Klemenjak et al. 2019). Seen = the later 20 % of each training home. Generalisation loss:
# 100 × (unseen / seen − 1) for error metrics, 100 × (1 − unseen / seen) for F1.

# %%
unseen = scores[scores.split == "test"].groupby(["family", "house"])[KEY].mean().reset_index()
seen = scores[scores.split == "seen_test"].groupby(["family", "house"])[KEY].mean().reset_index()
per_home = unseen.pivot(index="house", columns="family", values="nde").reindex(columns=FAM_ORDER).round(3)
per_home.to_csv(C.TABLE_DIR / "nb06_unseen_nde_per_home.csv")
per_home_cyc = unseen.pivot(index="house", columns="family", values="cycle_f1").reindex(columns=FAM_ORDER).round(3)
per_home_cyc.to_csv(C.TABLE_DIR / "nb06_unseen_cyclef1_per_home.csv")
euh = unseen.groupby("family")[KEY].mean().reindex(FAM_ORDER)
seen_m = seen.groupby("family")[KEY].mean().reindex(FAM_ORDER)
euh.to_csv(C.TABLE_DIR / "nb06_unseen_mean.csv")
seen_m.to_csv(C.TABLE_DIR / "nb06_seen_mean.csv")
gen = pd.DataFrame(
    {
        "NDE seen": seen_m["nde"], "NDE unseen": euh["nde"], "G-loss NDE %": 100 * (euh["nde"] / seen_m["nde"] - 1),
        "F1 seen": seen_m["f1"], "F1 unseen": euh["f1"], "G-loss F1 %": 100 * (1 - euh["f1"] / seen_m["f1"]),
        "cycle F1 seen": seen_m["cycle_f1"], "cycle F1 unseen": euh["cycle_f1"],
        "EpD unseen (Wh/day)": euh["epd_wh"], "SAE unseen": euh["sae"],
    }
).round(3)
gen.to_csv(C.TABLE_DIR / "nb06_seen_vs_unseen.csv")
print("NDE per unseen home (1.0 = predicting zero):")
display(per_home)
print("Cycle F1 per unseen home:")
display(per_home_cyc)
gen

# %% [markdown]
# ### 2c. Post-processing with the activation filter (unseen homes)
#
# Notebook 05b showed on the validation house that zeroing predicted power outside predicted
# activations of ≥ 30 minutes trades recovered energy for fewer false alarms. Here is the same
# trade-off on the unseen homes, for both ensembles.

# %%
def activation_filter(p: pd.Series) -> pd.Series:
    mask = cycles.activation_mask(p.fillna(0).to_numpy())
    return p.where(mask, 0.0).where(p.notna())


filt_rows = []
for fam in ("M2 Seq2Point ensemble", "M4 augmented ensemble"):
    for hh in UNSEEN:
        y = D.to_series(cp, cp.starts["test"], cp.wm[cp.targets("test")], hh)
        p = D.to_series(cp, cp.starts["test"], PRED[fam]["test"]["power"], hh)
        for variant, q in (("raw", p), ("activation filter", activation_filter(p))):
            r = metrics.all_metrics(y, q)
            filt_rows.append({"model": fam, "variant": variant, "house": hh, **{k: r[k] for k in KEY}})
filt = pd.DataFrame(filt_rows)
filt.to_csv(C.TABLE_DIR / "nb06_activation_filter_unseen.csv", index=False)
filt.groupby(["model", "variant"])[["mae_w", "nde", "sae", "epd_wh", "f1", "precision", "cycle_f1", "overlap_pct", "extra_pct"]].mean().round(3)

# %% [markdown]
# ## 3. Actual vs predicted on House 8
#
# The three days are chosen programmatically as the three consecutive fully-covered days with
# the most true washing-machine cycles.

# %%
h = C.TEST_HOUSE
starts = cp.starts["test"]
y8 = D.to_series(cp, starts, cp.wm[starts + W // 2], h)
agg8 = D.to_series(cp, starts, cp.agg[starts + W // 2], h)
p_m2 = D.to_series(cp, starts, PRED["M2 Seq2Point ensemble"]["test"]["power"], h)
p_m4 = D.to_series(cp, starts, PRED["M4 augmented ensemble"]["test"]["power"], h)
p_pr = p_m4 if PRIMARY.startswith("M4") else p_m2
true_cyc = cycles.detect_cycles(y8)
day_cyc = true_cyc.groupby(true_cyc["start"].dt.floor("D")).size()
cov = y8.notna().groupby(y8.index.floor("D")).mean()
roll = day_cyc.reindex(cov.index, fill_value=0).where(cov > 0.95, -99).rolling(3).sum()
d_end = roll.idxmax()
d0 = d_end - pd.Timedelta("2D")
sl = slice(d0, d_end + pd.Timedelta("1D"))
fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
axes[0].plot(agg8[sl].index, agg8[sl], color=P["aggregate"], lw=0.6)
axes[0].set_ylabel("aggregate (W)")
axes[0].set_title(f"House 8 (unseen), {d0.date()} to {d_end.date()}: input aggregate (top) and washing machine (bottom)")
axes[1].fill_between(y8[sl].index, 0, y8[sl].fillna(0), color=P["wm"], alpha=0.55, lw=0, label="actual (plug monitor)")
axes[1].plot(p_m2[sl].index, p_m2[sl], color=P["alt2"], lw=0.9, label="M2 Seq2Point (ensemble)")
axes[1].plot(p_m4[sl].index, p_m4[sl], color=P["pred"], lw=0.9, label="M4 augmented (ensemble)")
axes[1].set_ylabel("washing machine (W)")
axes[1].legend(loc="upper right", ncol=3)
plots.save(fig, "nb06_actual_vs_predicted_days")
plt.show()

# %% [markdown]
# ### 3a. Zoom on single cycles, and daily energy

# %%
pred_cyc = cycles.detect_cycles(p_pr.where(y8.notna()))
match = metrics.match_cycles(true_cyc, pred_cyc)
examples = match.sort_values("iou", ascending=False)
picks = [examples.iloc[0], examples.iloc[len(examples) // 2], examples.iloc[-1]] if len(examples) >= 3 else [r for _, r in examples.iterrows()]
fig, axes = plt.subplots(1, len(picks), figsize=(13, 3.4), sharey=True)
for ax, r in zip(np.atleast_1d(axes), picks):
    c = true_cyc.loc[r.true_idx]
    sl2 = slice(c.start - pd.Timedelta("30min"), c.end + pd.Timedelta("30min"))
    ax.fill_between(y8[sl2].index, 0, y8[sl2].fillna(0), color=P["wm"], alpha=0.55, lw=0, label="actual")
    ax.plot(p_m2[sl2].index, p_m2[sl2], color=P["alt2"], lw=1, label="M2 ens.")
    ax.plot(p_m4[sl2].index, p_m4[sl2], color=P["pred"], lw=1, label="M4 ens.")
    ax.set_title(f"IoU {r.iou:.2f}: true {r.true_kwh:.2f} kWh, predicted {r.pred_kwh:.2f} kWh")
    ax.tick_params(axis="x", rotation=30)
np.atleast_1d(axes)[0].set_ylabel("W")
np.atleast_1d(axes)[0].legend()
fig.suptitle(f"Best, median and worst matched cycles of the primary model ({PRIMARY}), House 8", fontweight="bold")
plots.save(fig, "nb06_cycle_zoom")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
daily_r = {}
for ax, (name, p) in zip(axes, [("M2 Seq2Point ensemble", p_m2), ("M4 augmented ensemble", p_m4)]):
    d = metrics.daily_energy_errors(y8, p)
    ax.scatter(d["y_kwh"], d["yhat_kwh"], s=8, alpha=0.5, color=P["pred"])
    lim = max(d["y_kwh"].max(), d["yhat_kwh"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", lw=0.8)
    daily_r[name] = float(np.corrcoef(d["y_kwh"], d["yhat_kwh"])[0, 1])
    ax.set_title(f"{name}: daily WM energy (r = {daily_r[name]:.2f}, {len(d)} days)")
    ax.set_xlabel("actual kWh/day")
axes[0].set_ylabel("predicted kWh/day")
plots.save(fig, "nb06_daily_energy_scatter")
plt.show()

# %% [markdown]
# ## 4. Failure analysis (House 8)
#
# Three ways a disaggregator fails on cycles: it **misses** a real cycle, it invents a **false**
# cycle, or it finds the cycle but gets the **energy** wrong. For every false cycle I look at the
# house's other plug monitors over the same minutes and record which appliance drew the most
# energy; if none drew more than 50 Wh, the cause is attributed to unmetered load.

# %%
raw8 = io.load_clean_minutes(h)
names8 = {f"Appliance{c}": d.split(",")[0] for c, d in amap[h].items()}
others = [c for c in names8 if c != f"Appliance{io.washing_machine_channels(amap)[h][0]}"]
tax = []
for mname, p in [("M2 Seq2Point ensemble", p_m2), ("M4 augmented ensemble", p_m4)]:
    pc = cycles.detect_cycles(p.where(y8.notna()))
    mt = metrics.match_cycles(true_cyc, pc)
    missed = true_cyc.drop(index=mt["true_idx"])
    false = pc.drop(index=mt["pred_idx"])
    culprits = []
    for _, fc in false.iterrows():
        e = raw8.loc[fc.start : fc.end, others].sum() / 60_000
        culprits.append(names8[e.idxmax()] if e.max() > 0.05 else "unmetered load")
    cc = pd.Series(culprits, dtype=object).value_counts()
    tax.append(
        {
            "model": mname, "true_cycles": len(true_cyc), "predicted_cycles": len(pc), "matched": len(mt),
            "missed": len(missed), "false": len(false),
            "missed_median_kwh": round(missed["energy_kwh"].median(), 3) if len(missed) else np.nan,
            "matched_true_median_kwh": round(true_cyc.loc[mt["true_idx"], "energy_kwh"].median(), 3) if len(mt) else np.nan,
            "missed_with_heating_pct": round(100 * (missed["minutes_over_1500w"] >= 5).mean(), 1) if len(missed) else np.nan,
            "matched_with_heating_pct": round(100 * (true_cyc.loc[mt["true_idx"], "minutes_over_1500w"] >= 5).mean(), 1) if len(mt) else np.nan,
            "matched_energy_bias_pct": round(100 * (mt["pred_kwh"].sum() / mt["true_kwh"].sum() - 1), 1) if len(mt) else np.nan,
            "false_cycle_culprits": "; ".join(f"{k}: {v}" for k, v in cc.head(6).items()),
        }
    )
taxonomy = pd.DataFrame(tax).set_index("model")
taxonomy.to_csv(C.TABLE_DIR / "nb06_failure_taxonomy_house8.csv")
taxonomy.T

# %% [markdown]
# ### 4a. The worst false-positive day for the primary model

# %%
fpw = (p_pr - y8.fillna(0)).clip(lower=0)
worst = fpw.groupby(fpw.index.floor("D")).sum().idxmax()
sl3 = slice(worst, worst + pd.Timedelta("1D"))
fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True)
axes[0].plot(agg8[sl3].index, agg8[sl3], color=P["aggregate"], lw=0.6, label="aggregate")
top = raw8.loc[sl3, others].sum().sort_values(ascending=False).index[:2]
for c, col in zip(top, [P["alt"], P["alt2"]]):
    axes[0].plot(raw8.loc[sl3].index, raw8.loc[sl3, c], lw=0.8, color=col, label=f"{names8[c]} (plug monitor)")
axes[0].legend(loc="upper right")
axes[0].set_ylabel("W")
axes[0].set_title(f"House 8, {worst.date()}: the day with the most false washing-machine energy ({PRIMARY})")
axes[1].fill_between(y8[sl3].index, 0, y8[sl3].fillna(0), color=P["wm"], alpha=0.55, lw=0, label="actual WM")
axes[1].plot(p_pr[sl3].index, p_pr[sl3], color=P["pred"], lw=0.9, label="predicted WM")
axes[1].legend(loc="upper right")
axes[1].set_ylabel("W")
plots.save(fig, "nb06_worst_false_positive_day")
plt.show()

# %% [markdown]
# ### 4b. Effect of the `Issues` flag (plug monitors summing above the aggregate)
#
# Those minutes were excluded from training and scoring. Here the House 8 test set is rebuilt
# with them included and scored with the primary model.

# %%
prefix = "M4 augmented" if PRIMARY.startswith("M4") else "M2 Seq2Point"
f8 = frames[h]
f8_all = f8.assign(target_ok=~f8["agg_missing"] & f8["wm"].notna())
cp_iss = D.build_corpus({h: f8_all}, {h: pd.Series("test", index=f8_all.index)}, W)
pi = predict_family(prefix, cp_iss)
s_i = cp_iss.starts["test"]
yi = D.to_series(cp_iss, s_i, cp_iss.wm[s_i + W // 2], h)
issue_tab = pd.DataFrame(
    [
        {"targets": "issues minutes excluded (standard)", "minutes": int(y8.notna().sum()), **{k: metrics.all_metrics(y8, p_pr, with_cycles=False)[k] for k in ["mae_w", "nde", "sae", "f1"]}},
        {"targets": "issues minutes included", "minutes": int(yi.notna().sum()), **{k: metrics.all_metrics(yi, D.to_series(cp_iss, s_i, pi, h), with_cycles=False)[k] for k in ["mae_w", "nde", "sae", "f1"]}},
    ]
).round(4)
issue_tab

# %% [markdown]
# ### 4c. House 1: official cleaned data vs my own cleaning of the raw files
#
# This links Requirement 1 to Requirement 3. House 1 is an unseen home; here it is scored twice
# with the same models, once on the official cleaned release and once on the series produced by
# my raw-data pipeline in notebook 01. If my cleaning is sound, the scores should be close.

# %%
mine1 = pd.read_parquet(C.PARQUET_DIR / "raw_house1_clean_1min.parquet")
f1_mine = D.house_frame(1, "Appliance5", mine1)
cp_m = D.build_corpus({1: f1_mine}, {1: pd.Series("test", index=f1_mine.index)}, W)
h1_rows = []
for fam, pre in (("M2 Seq2Point ensemble", "M2 Seq2Point"), ("M4 augmented ensemble", "M4 augmented")):
    pm_ = predict_family(pre, cp_m)
    s_m = cp_m.starts["test"]
    ym = D.to_series(cp_m, s_m, cp_m.wm[s_m + W // 2], 1)
    r_mine = metrics.all_metrics(ym, D.to_series(cp_m, s_m, pm_, 1))
    r_off = scores[(scores.model == fam) & (scores.split == "test") & (scores.house == 1)].iloc[0]
    for label, r in (("official cleaned", r_off), ("my cleaning of raw", r_mine)):
        h1_rows.append({"model": fam, "data": label, **{k: float(r[k]) for k in ["mae_w", "nde", "sae", "epd_wh", "f1", "cycle_f1"]}})
h1_tab = pd.DataFrame(h1_rows).round(3)
h1_tab.to_csv(C.TABLE_DIR / "nb06_house1_official_vs_mine.csv", index=False)
h1_tab

# %% [markdown]
# ## 5. What changes with 15- or 30-minute data?
#
# Most utility smart meters report energy per 15 or 30 minutes (UK half-hourly settlement,
# Indian AMI block-load profiles), not per minute. Two questions:
#
# 1. **Trained at the coarse resolution.** Both 1-minute series are averaged to 15 and 30
#    minutes (an interval is usable only if ≥ 80 % of its minutes were usable), and Seq2Point is
#    retrained with a ~4-hour window (17 and 9 samples): same homes, same validation house, seed
#    42, batch 256 (there are 15–30× fewer samples).
# 2. **Fair comparison.** The 1-minute models' predictions are averaged to the same intervals and
#    scored with the same metrics, so every row is evaluated at the same granularity.
#
# One architecture (Seq2Point) is used so that the comparison isolates resolution. Cycle metrics
# are not meaningful when a 60-minute cycle spans 2–4 samples, so the comparison uses MAE, NDE,
# SAE, daily energy error (EpD) and interval-level F1.

# %%
res_rows, res_frames = [], {}
houses_r = C.TRAIN_HOUSES + [C.VAL_HOUSE] + UNSEEN
MKEYS = ["mae_w", "nde", "sae", "epd_wh", "f1", "precision", "recall"]
one_min = {"M2 Seq2Point s42 (1-min)": "M2 Seq2Point s42", f"{PRIMARY} (1-min)": PRIMARY}
for res in (15, 30):
    fr = {hh: D.resample_frame(frames[hh], res) for hh in houses_r}
    res_frames[res] = fr
    Wr = int(np.ceil(240 / res)) // 2 * 2 + 1
    cpr = D.build_corpus(fr, D.assign_splits(fr, D.SplitSpec(test_houses=UNSEEN), Wr), Wr)
    str_ = D.standardisation(cpr)
    cfg = T.TrainConfig(model="seq2point", window=Wr, seed=42, batch_size=256, max_epochs=60, patience=8)
    model, hist = T.fit(cpr, str_, cfg, device=DEVICE, log_every=100)
    out = T.predict(model, T.WindowSource(cpr, str_, DEVICE), cpr.starts["test"], str_)
    del model
    torch.cuda.empty_cache()
    st_r = cpr.starts["test"]
    for hh in UNSEEN:
        y = D.to_series(cpr, st_r, cpr.wm[st_r + Wr // 2], hh, freq=f"{res}min")
        yh = D.to_series(cpr, st_r, out["power"], hh, freq=f"{res}min")
        res_rows.append({"evaluated at": f"{res} min", "model": f"M2 Seq2Point s42 trained on {res}-min data", "house": hh, **{k: metrics.all_metrics(y, yh, with_cycles=False)[k] for k in MKEYS}})
        yc = fr[hh]["wm"].where(fr[hh]["target_ok"])
        for label, key in one_min.items():
            p1 = D.to_series(cp, cp.starts["test"], PRED[key]["test"]["power"], hh)
            pc = p1.resample(f"{res}min").mean().reindex(yc.index)
            res_rows.append({"evaluated at": f"{res} min", "model": f"{label}, averaged to {res} min", "house": hh, **{k: metrics.all_metrics(yc, pc, with_cycles=False)[k] for k in MKEYS}})
for label, key in one_min.items():
    for hh in UNSEEN:
        r = scores[(scores.model == key) & (scores.split == "test") & (scores.house == hh)].iloc[0]
        res_rows.append({"evaluated at": "1 min", "model": label, "house": hh, **{k: r[k] for k in MKEYS}})
res_tab = pd.DataFrame(res_rows)
res_tab.to_csv(C.TABLE_DIR / "nb06_resolution_all.csv", index=False)
res_sum = res_tab.groupby(["evaluated at", "model"])[MKEYS].mean().round(3)
res_h8 = res_tab[res_tab.house == C.TEST_HOUSE].set_index(["evaluated at", "model"])[MKEYS].round(3)
res_sum.to_csv(C.TABLE_DIR / "nb06_resolution_mean_unseen.csv")
res_h8.to_csv(C.TABLE_DIR / "nb06_resolution_house8.csv")
print("Mean over the six unseen homes:")
display(res_sum)
print("House 8:")
res_h8

# %%
c = true_cyc.sort_values("energy_kwh").iloc[len(true_cyc) // 2]
sl4 = slice(c.start - pd.Timedelta("90min"), c.end + pd.Timedelta("90min"))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for ax, res in zip(axes, (1, 15, 30)):
    src_f = frames[h] if res == 1 else res_frames[res][h]
    a, wv = src_f["agg"][sl4], src_f["wm"][sl4]
    ax.step(a.index, a, where="post", color=P["aggregate"], lw=0.8, label="aggregate")
    ax.fill_between(wv.index, 0, wv.fillna(0), step="post", color=P["wm"], alpha=0.6, lw=0, label="washing machine")
    ax.set_title(f"{res}-minute resolution")
    ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("W")
axes[0].legend()
fig.suptitle(f"One House 8 washing-machine cycle ({c.energy_kwh:.2f} kWh) seen at 1, 15 and 30 minutes", fontweight="bold")
plots.save(fig, "nb06_cycle_at_three_resolutions")
plt.show()

# %% [markdown]
# ## 6. Save

# %%
SUMMARY.update(
    {
        "house8_mean": h8_mean[KEY].round(4).to_dict(orient="index"),
        "house8_std": h8_std[KEY].round(4).to_dict(orient="index"),
        "unseen_mean": euh[KEY].round(4).to_dict(orient="index"),
        "seen_mean": seen_m[KEY].round(4).to_dict(orient="index"),
        "generalisation": gen.round(3).to_dict(orient="index"),
        "activation_filter_unseen": {" | ".join(k): v for k, v in filt.groupby(["model", "variant"])[["nde", "sae", "epd_wh", "f1", "precision", "cycle_f1", "overlap_pct", "extra_pct"]].mean().round(3).to_dict(orient="index").items()},
        "failure_taxonomy_house8": taxonomy.astype(str).to_dict(orient="index"),
        "issues_effect": issue_tab.to_dict(orient="records"),
        "house1_official_vs_mine": h1_tab.to_dict(orient="records"),
        "resolution_mean_unseen": {" | ".join(k): v for k, v in res_sum.to_dict(orient="index").items()},
        "resolution_house8": {" | ".join(k): v for k, v in res_h8.to_dict(orient="index").items()},
        "daily_energy_r_house8": daily_r,
        "plot_days": [str(d0.date()), str(d_end.date())],
        "worst_fp_day": str(worst.date()),
    }
)
with open(C.METRIC_DIR / "nb06_summary.json", "w") as fh:
    json.dump(SUMMARY, fh, indent=2, default=str)
print("saved")

# %% [markdown]
# ## Summary
#
# _Filled in after execution._
