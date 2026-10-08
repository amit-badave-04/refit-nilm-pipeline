# %% [markdown]
# # 06 · Evaluation on unseen homes, failure analysis, and the effect of 15/30-minute data
#
# **Requirement 3 (evaluation) and Requirement 4 (resolution question).** All choices (window,
# λ, epochs, thresholds) were frozen on the validation house in notebooks 04–05. This notebook
# scores every model once on the test homes, explains where and why the models fail, and
# measures what is lost when only 15- or 30-minute smart-meter data are available.
#
# | | |
# |---|---|
# | **Input** | checkpoints from notebooks 04–05, the corpus of notebook 03 |
# | **Output** | `artifacts/tables/nb06_*`, `artifacts/figures/nb06_*`, `artifacts/metrics/nb06_summary.json` |
# | **Run time** | about 15 minutes (includes training the 15/30-minute models) |

# %%
import json
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
W = int(nb04["window_selection"]["chosen_window"])
LAMBDA = float(nb05["lambda_selection"]["chosen_lambda"])
frames, corpora, stats = D.prepare([W])
cp, st = corpora[W], stats[W]
UNSEEN = [C.TEST_HOUSE] + C.EXTRA_UNSEEN_HOUSES
SPLITS = ["test", "seen_test", "val"]
amap = io.parse_appliance_map(C.DATA_DIR / "CLEAN_READ_ME_081116.txt")
SUMMARY: dict = {"window": W, "lambda": LAMBDA}
print("window", W, "lambda", LAMBDA)

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

runs = {f"M2 Seq2Point s{s}": C.MODEL_DIR / f"m2_seq2point_w{W}_s{s}.pt" for s in C.SEEDS}
runs.update({f"M3 gated s{s}": C.MODEL_DIR / f"m3_gated_w{W}_l{LAMBDA}_s{s}.pt" for s in C.SEEDS})
src = T.WindowSource(cp, st, DEVICE)
for name, path in runs.items():
    model, cfg, stt = T.load_run(path, DEVICE)
    PRED[name] = {sp: T.predict(model, src, cp.starts[sp], stt) for sp in SPLITS}
    del model
torch.cuda.empty_cache()
print(list(PRED))

# %% [markdown]
# ## 2. Scores per home

# %%
def family(name: str) -> str:
    return name.rsplit(" s", 1)[0] if name.startswith(("M2", "M3")) else name


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
FAM_ORDER = ["M0a always-off", "M0b weekday×hour profile", "M1 LightGBM", "M2 Seq2Point", "M3 gated"]

# %% [markdown]
# ### 2a. Primary result: House 8 (the reference unseen test house)
#
# Deep models: mean ± standard deviation over three seeds.

# %%
h8 = scores[(scores.split == "test") & (scores.house == C.TEST_HOUSE)]
h8_mean = h8.groupby("family")[KEY].mean().reindex(FAM_ORDER)
h8_std = h8.groupby("family")[KEY].std().reindex(FAM_ORDER)
h8_mean.to_csv(C.TABLE_DIR / "nb06_house8_mean.csv")
h8_std.to_csv(C.TABLE_DIR / "nb06_house8_std.csv")
show = ["mae_w", "mae_on_w", "sae", "epd_wh", "nde", "f1", "cycle_f1", "overlap_pct", "extra_pct"]
fmt = h8_mean[show].round(3).astype(str)
for c in show:
    fmt[c] = [f"{m:.3f} ± {s:.3f}" if not np.isnan(s) else f"{m:.3f}" for m, s in zip(h8_mean[c], h8_std[c])]
fmt

# %% [markdown]
# ### 2b. All unseen homes (House 8 + five extra homes) and seen homes
#
# Error on Unseen Houses (EUH) and Accuracy on Unseen Houses (AUH) are the means over the six
# unseen homes (Klemenjak et al. 2019). The seen-house figures use the later 20 % of the
# training homes. Generalisation loss: for error metrics 100 × (unseen / seen − 1); for F1
# 100 × (1 − unseen / seen).

# %%
unseen = scores[scores.split == "test"].groupby(["family", "house"])[KEY].mean().reset_index()
seen = scores[scores.split == "seen_test"].groupby(["family", "house"])[KEY].mean().reset_index()
per_home = unseen.pivot(index="house", columns="family", values="nde").reindex(columns=FAM_ORDER).round(3)
per_home.to_csv(C.TABLE_DIR / "nb06_unseen_nde_per_home.csv")
euh = unseen.groupby("family")[KEY].mean().reindex(FAM_ORDER)
seen_m = seen.groupby("family")[KEY].mean().reindex(FAM_ORDER)
gen = pd.DataFrame(
    {
        "MAE seen": seen_m["mae_w"], "MAE unseen": euh["mae_w"],
        "G-loss MAE %": 100 * (euh["mae_w"] / seen_m["mae_w"] - 1),
        "NDE seen": seen_m["nde"], "NDE unseen": euh["nde"],
        "G-loss NDE %": 100 * (euh["nde"] / seen_m["nde"] - 1),
        "F1 seen": seen_m["f1"], "F1 unseen": euh["f1"],
        "G-loss F1 %": 100 * (1 - euh["f1"] / seen_m["f1"]),
        "cycle F1 unseen": euh["cycle_f1"], "EpD unseen (Wh/day)": euh["epd_wh"],
    }
).round(3)
gen.to_csv(C.TABLE_DIR / "nb06_seen_vs_unseen.csv")
print("NDE per unseen home (1.0 = predicting zero):")
display(per_home)
gen

# %% [markdown]
# ## 3. Actual vs predicted on House 8
#
# Plots use seed 42 of each deep model. The three days are chosen programmatically as the
# three consecutive fully-covered days with the most true washing-machine cycles.

# %%
h = C.TEST_HOUSE
starts = cp.starts["test"]
y8 = D.to_series(cp, starts, cp.wm[starts + W // 2], h)
agg8 = D.to_series(cp, starts, cp.agg[starts + W // 2], h)
p_m2 = D.to_series(cp, starts, PRED["M2 Seq2Point s42"]["test"]["power"], h)
p_m3 = D.to_series(cp, starts, PRED["M3 gated s42"]["test"]["power"], h)
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
axes[1].plot(p_m2[sl].index, p_m2[sl], color=P["alt2"], lw=0.9, label="M2 Seq2Point")
axes[1].plot(p_m3[sl].index, p_m3[sl], color=P["pred"], lw=0.9, label="M3 gated Seq2Point")
axes[1].set_ylabel("washing machine (W)")
axes[1].legend(loc="upper right", ncol=3)
plots.save(fig, "nb06_actual_vs_predicted_days")
plt.show()

# %% [markdown]
# ### 3a. Zoom on single cycles, and daily energy

# %%
pred_cyc = cycles.detect_cycles(p_m3.where(y8.notna()))
match = metrics.match_cycles(true_cyc, pred_cyc)
examples = match.sort_values("iou", ascending=False)
picks = [examples.iloc[0], examples.iloc[len(examples) // 2], examples.iloc[-1]] if len(examples) >= 3 else [r for _, r in examples.iterrows()]
fig, axes = plt.subplots(1, len(picks), figsize=(13, 3.4), sharey=True)
for ax, r in zip(np.atleast_1d(axes), picks):
    c = true_cyc.loc[r.true_idx]
    sl2 = slice(c.start - pd.Timedelta("30min"), c.end + pd.Timedelta("30min"))
    ax.fill_between(y8[sl2].index, 0, y8[sl2].fillna(0), color=P["wm"], alpha=0.55, lw=0, label="actual")
    ax.plot(p_m2[sl2].index, p_m2[sl2], color=P["alt2"], lw=1, label="M2")
    ax.plot(p_m3[sl2].index, p_m3[sl2], color=P["pred"], lw=1, label="M3")
    ax.set_title(f"IoU {r.iou:.2f}: true {r.true_kwh:.2f} kWh, M3 {r.pred_kwh:.2f} kWh")
    ax.tick_params(axis="x", rotation=30)
np.atleast_1d(axes)[0].set_ylabel("W")
np.atleast_1d(axes)[0].legend()
plots.save(fig, "nb06_cycle_zoom")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
for ax, (name, p) in zip(axes, [("M2 Seq2Point", p_m2), ("M3 gated Seq2Point", p_m3)]):
    d = metrics.daily_energy_errors(y8, p)
    ax.scatter(d["y_kwh"], d["yhat_kwh"], s=8, alpha=0.5, color=P["pred"])
    lim = max(d["y_kwh"].max(), d["yhat_kwh"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", lw=0.8)
    r = np.corrcoef(d["y_kwh"], d["yhat_kwh"])[0, 1]
    ax.set_title(f"{name}: daily WM energy, House 8 (r = {r:.2f}, {len(d)} days)")
    ax.set_xlabel("actual kWh/day")
axes[0].set_ylabel("predicted kWh/day")
plots.save(fig, "nb06_daily_energy_scatter")
plt.show()

# %% [markdown]
# ## 4. Failure analysis (House 8, seed 42)
#
# Three ways a disaggregator fails on cycles: it **misses** a real cycle, it invents a **false**
# cycle, or it finds the cycle but gets the **energy** wrong. For every false cycle I look at the
# house's other plug monitors over the same minutes and record which appliance was drawing the
# most energy; if none drew more than 50 Wh, the cause is attributed to unmetered load.

# %%
raw8 = io.load_clean_minutes(h)
names8 = {f"Appliance{c}": d.split(",")[0] for c, d in amap[h].items()}
others = [c for c in names8 if c != f"Appliance{io.washing_machine_channels(amap)[h][0]}"]
tax = []
for mname, p in [("M2 Seq2Point", p_m2), ("M3 gated", p_m3)]:
    pc = cycles.detect_cycles(p.where(y8.notna()))
    mt = metrics.match_cycles(true_cyc, pc)
    missed = true_cyc.drop(index=mt["true_idx"])
    false = pc.drop(index=mt["pred_idx"])
    culprits = []
    for _, fc in false.iterrows():
        seg = raw8.loc[fc.start : fc.end, others]
        e = seg.sum() / 60_000
        culprits.append(names8[e.idxmax()] if e.max() > 0.05 else "unmetered load")
    culprit_counts = pd.Series(culprits, dtype=object).value_counts()
    tax.append(
        {
            "model": mname, "true_cycles": len(true_cyc), "predicted_cycles": len(pc), "matched": len(mt),
            "missed": len(missed), "false": len(false),
            "missed_median_kwh": round(missed["energy_kwh"].median(), 3) if len(missed) else np.nan,
            "matched_true_median_kwh": round(true_cyc.loc[mt["true_idx"], "energy_kwh"].median(), 3) if len(mt) else np.nan,
            "missed_with_heating_pct": round(100 * (missed["minutes_over_1500w"] >= 5).mean(), 1) if len(missed) else np.nan,
            "matched_energy_bias_pct": round(100 * (mt["pred_kwh"].sum() / mt["true_kwh"].sum() - 1), 1) if len(mt) else np.nan,
            "false_cycle_culprits": "; ".join(f"{k}: {v}" for k, v in culprit_counts.head(5).items()),
        }
    )
taxonomy = pd.DataFrame(tax).set_index("model")
taxonomy.to_csv(C.TABLE_DIR / "nb06_failure_taxonomy_house8.csv")
taxonomy.T

# %% [markdown]
# ### 4a. The worst false-positive day

# %%
fp = (p_m3 - y8.fillna(0)).clip(lower=0)
worst = fp.groupby(fp.index.floor("D")).sum().idxmax()
sl3 = slice(worst, worst + pd.Timedelta("1D"))
fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True)
axes[0].plot(agg8[sl3].index, agg8[sl3], color=P["aggregate"], lw=0.6, label="aggregate")
top = raw8.loc[sl3, others].sum().sort_values(ascending=False).index[:2]
for c, col in zip(top, [P["alt"], P["alt2"]]):
    axes[0].plot(raw8.loc[sl3].index, raw8.loc[sl3, c], lw=0.8, color=col, label=f"{names8[c]} (plug monitor)")
axes[0].legend(loc="upper right")
axes[0].set_ylabel("W")
axes[0].set_title(f"House 8, {worst.date()}: the day with the most false washing-machine energy (M3, seed 42)")
axes[1].fill_between(y8[sl3].index, 0, y8[sl3].fillna(0), color=P["wm"], alpha=0.55, lw=0, label="actual WM")
axes[1].plot(p_m3[sl3].index, p_m3[sl3], color=P["pred"], lw=0.9, label="M3 predicted WM")
axes[1].legend(loc="upper right")
axes[1].set_ylabel("W")
plots.save(fig, "nb06_worst_false_positive_day")
plt.show()

# %% [markdown]
# ### 4b. Effect of the `Issues` flag (minutes where the plug monitors exceed the aggregate)
#
# Those minutes were excluded from training and scoring. Here the House 8 test set is rebuilt
# with them included, to show how much the choice matters.

# %%
f8 = frames[h].copy()
f8_all = f8.assign(target_ok=~f8["agg_missing"] & f8["wm"].notna())
cp_iss = D.build_corpus({h: f8_all}, {h: pd.Series("test", index=f8_all.index)}, W)
model, cfg, stt = T.load_run(runs["M3 gated s42"], DEVICE)
pi = T.predict(model, T.WindowSource(cp_iss, stt, DEVICE), cp_iss.starts["test"], stt)
del model
s_i = cp_iss.starts["test"]
yi = D.to_series(cp_iss, s_i, cp_iss.wm[s_i + W // 2], h)
issue_tab = pd.DataFrame(
    [
        {"targets": "issues minutes excluded (standard)", "minutes": int(y8.notna().sum()), **{k: metrics.all_metrics(y8, p_m3, with_cycles=False)[k] for k in ["mae_w", "nde", "sae", "f1"]}},
        {"targets": "issues minutes included", "minutes": int(yi.notna().sum()), **{k: metrics.all_metrics(yi, D.to_series(cp_iss, s_i, pi["power"], h), with_cycles=False)[k] for k in ["mae_w", "nde", "sae", "f1"]}},
    ]
).round(4)
issue_tab

# %% [markdown]
# ### 4c. House 1: official cleaned data vs my own cleaning of the raw files
#
# This links Requirement 1 to Requirement 3. House 1 is an unseen home; here it is scored twice
# with the same models, once on the official cleaned release and once on the series produced by
# my raw-data pipeline in notebook 01. If my cleaning is sound, the scores should be close; where
# they differ, it is because my pipeline marks faulty readings as missing instead of zero.

# %%
mine1 = pd.read_parquet(C.PARQUET_DIR / "raw_house1_clean_1min.parquet")
f1_mine = D.house_frame(1, "Appliance5", mine1)
cp_m = D.build_corpus({1: f1_mine}, {1: pd.Series("test", index=f1_mine.index)}, W)
h1_rows = []
for key in ("M2 Seq2Point s42", "M3 gated s42"):
    model, cfg, stt = T.load_run(runs[key], DEVICE)
    out = T.predict(model, T.WindowSource(cp_m, stt, DEVICE), cp_m.starts["test"], stt)
    del model
    s_m = cp_m.starts["test"]
    ym = D.to_series(cp_m, s_m, cp_m.wm[s_m + W // 2], 1)
    pm = D.to_series(cp_m, s_m, out["power"], 1)
    r_off = scores[(scores.model == key) & (scores.split == "test") & (scores.house == 1)].iloc[0]
    r_mine = metrics.all_metrics(ym, pm)
    for label, r, n in (("official cleaned", r_off, None), ("my cleaning of raw", r_mine, int(ym.notna().sum()))):
        h1_rows.append({"model": key, "data": label, **{k: float(r[k]) for k in ["mae_w", "nde", "sae", "epd_wh", "f1", "cycle_f1"]}})
h1_tab = pd.DataFrame(h1_rows).round(3)
h1_tab.to_csv(C.TABLE_DIR / "nb06_house1_official_vs_mine.csv", index=False)
h1_tab

# %% [markdown]
# ## 5. What changes with 15- or 30-minute data?
#
# Most utility smart meters (UK SMETS2 half-hourly settlement, Indian AMI block-load profiles)
# report energy per 15 or 30 minutes, not per minute. Two questions:
#
# 1. **Trained at the coarse resolution.** Both 1-minute series are averaged to 15 and 30
#    minutes (an interval is usable only if ≥ 80 % of its minutes were usable), and Seq2Point and
#    the gated model are retrained with a ~4-hour window (17 and 9 samples), same training
#    houses, same validation house, seed 42, batch 256 (fewer samples per epoch).
# 2. **Fair comparison.** The 1-minute models' predictions are averaged to the same intervals and
#    scored with the same metrics, so every row below is evaluated at the same granularity.
#
# Cycle metrics are not meaningful when a 60-minute cycle spans 2–4 samples, so the comparison
# uses MAE, NDE, SAE, daily energy error (EpD) and interval-level F1.

# %%
res_rows, res_frames = [], {}
houses_r = C.TRAIN_HOUSES + [C.VAL_HOUSE] + UNSEEN


def coarse_truth_and_pred(res: int, pred_1min: pd.Series, y_1min_frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    fr = D.resample_frame(y_1min_frame, res)
    y = fr["wm"].where(fr["target_ok"])
    p = pred_1min.resample(f"{res}min").mean().reindex(y.index)
    return y, p


for res in (15, 30):
    fr = {hh: D.resample_frame(frames[hh], res) for hh in houses_r}
    res_frames[res] = fr
    Wr = int(np.ceil(240 / res)) // 2 * 2 + 1
    spec = D.SplitSpec(test_houses=UNSEEN)
    cpr = D.build_corpus(fr, D.assign_splits(fr, spec, Wr), Wr)
    str_ = D.standardisation(cpr)
    srcr = T.WindowSource(cpr, str_, DEVICE)
    for mname, kind in [("M2 Seq2Point", "seq2point"), ("M3 gated", "gated_seq2point")]:
        cfg = T.TrainConfig(model=kind, window=Wr, seed=42, state_weight=LAMBDA, batch_size=256, max_epochs=60, patience=8)
        model, hist = T.fit(cpr, str_, cfg, device=DEVICE, log_every=100)
        out = T.predict(model, srcr, cpr.starts["test"], str_)
        st_r = cpr.starts["test"]
        for hh in UNSEEN:
            y = D.to_series(cpr, st_r, cpr.wm[st_r + Wr // 2], hh, freq=f"{res}min")
            yh = D.to_series(cpr, st_r, out["power"], hh, freq=f"{res}min")
            res_rows.append({"resolution": f"{res} min", "input": f"{res}-min input", "model": mname, "house": hh, **metrics.all_metrics(y, yh, with_cycles=False)})
        del model, srcr
        torch.cuda.empty_cache()
        srcr = T.WindowSource(cpr, str_, DEVICE)
    # 1-minute models, averaged to the same intervals
    for mname, key in [("M2 Seq2Point", "M2 Seq2Point s42"), ("M3 gated", "M3 gated s42")]:
        for hh in UNSEEN:
            p1 = D.to_series(cp, cp.starts["test"], PRED[key]["test"]["power"], hh)
            y, p = coarse_truth_and_pred(res, p1, frames[hh])
            res_rows.append({"resolution": f"{res} min", "input": "1-min input, averaged", "model": mname, "house": hh, **metrics.all_metrics(y, p, with_cycles=False)})
# 1-minute reference rows (seed 42)
for mname, key in [("M2 Seq2Point", "M2 Seq2Point s42"), ("M3 gated", "M3 gated s42")]:
    for hh in UNSEEN:
        r = scores[(scores.model == key) & (scores.split == "test") & (scores.house == hh)].iloc[0]
        res_rows.append({"resolution": "1 min", "input": "1-min input", "model": mname, "house": hh, **{k: r[k] for k in ["mae_w", "mae_on_w", "sae", "sae_daily", "nde", "epd_wh", "precision", "recall", "f1", "overlap_pct", "missing_pct", "extra_pct"]}})
res_tab = pd.DataFrame(res_rows)
res_tab.to_csv(C.TABLE_DIR / "nb06_resolution_all.csv", index=False)
res_sum = res_tab.groupby(["resolution", "input", "model"])[["mae_w", "nde", "sae", "epd_wh", "f1"]].mean().round(3)
res_h8 = res_tab[res_tab.house == C.TEST_HOUSE].set_index(["resolution", "input", "model"])[["mae_w", "nde", "sae", "epd_wh", "f1"]].round(3)
res_sum.to_csv(C.TABLE_DIR / "nb06_resolution_mean_unseen.csv")
print("Mean over the six unseen homes:")
display(res_sum)
print("House 8:")
res_h8

# %%
c = true_cyc.sort_values("energy_kwh").iloc[len(true_cyc) // 2]
sl4 = slice(c.start - pd.Timedelta("90min"), c.end + pd.Timedelta("90min"))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for ax, res in zip(axes, (1, 15, 30)):
    if res == 1:
        a, wv = frames[h]["agg"][sl4], frames[h]["wm"][sl4]
    else:
        a, wv = res_frames[res][h]["agg"][sl4], res_frames[res][h]["wm"][sl4]
    ax.step(a.index, a, where="post", color=P["aggregate"], lw=0.8, label="aggregate")
    ax.fill_between(wv.index, 0, wv.fillna(0), step="post", color=P["wm"], alpha=0.6, lw=0, label="washing machine")
    ax.set_title(f"{res}-minute resolution")
    ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("W")
axes[0].legend()
fig.suptitle(f"One House 8 washing-machine cycle ({c.energy_kwh:.2f} kWh) as seen at 1, 15 and 30 minutes", fontweight="bold")
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
        "failure_taxonomy_house8": taxonomy.astype(str).to_dict(orient="index"),
        "issues_effect": issue_tab.to_dict(orient="records"),
        "house1_official_vs_mine": h1_tab.to_dict(orient="records"),
        "resolution_mean_unseen": {" | ".join(k): v for k, v in res_sum.to_dict(orient="index").items()},
        "resolution_house8": {" | ".join(k): v for k, v in res_h8.to_dict(orient="index").items()},
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
