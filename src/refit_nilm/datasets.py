"""Modelling data contract: per-house frames, masks, splits and leakage-safe window indices.

A *target minute* t is usable for training or evaluation when
  * the aggregate and the washing-machine readings at t are valid (reading present after the
    ≤ 3-minute gap fill, not inside a forward-filled flat run, no ``issues`` flag), and
  * the input window [t - W//2, t + W//2] lies inside a single split segment of a single house
    and has at most ``max_missing_frac`` missing aggregate minutes (filled for the input only).

Windows are never materialised: the model gathers them on the fly from one concatenated 1-D
array, using integer start indices. Houses are separated by a block of missing minutes so no
window can span two houses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import io

FLAT_RUN_MIN = 60


def flat_run_mask(x: np.ndarray, min_len: int = FLAT_RUN_MIN) -> np.ndarray:
    """True inside runs of >= ``min_len`` identical, non-NaN values (forward-filled outages)."""
    v = np.asarray(x, dtype=float)
    valid = ~np.isnan(v)
    chg = np.r_[True, ~(v[1:] == v[:-1])]
    rid = np.cumsum(chg)
    counts = np.bincount(rid)
    return (counts[rid] >= min_len) & valid


def short_gap_ffill(x: pd.Series, limit: int = C.SHORT_GAP_FILL_MIN) -> pd.Series:
    """Forward-fill only runs of NaN no longer than ``limit`` minutes."""
    isna = x.isna()
    run_id = (~isna).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    return x.ffill().where(~isna | (run_len <= limit))


def house_frame(house: int, wm_channel: str, minutes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate, washing-machine target and validity masks for one house (1-minute, UTC)."""
    m = minutes if minutes is not None else io.load_clean_minutes(house)
    have = m["n_readings"] > 0 if "n_readings" in m else m["Aggregate"].notna()
    agg = short_gap_ffill(m["Aggregate"].where(have)).clip(0, C.AGG_CLIP_W)
    wm = short_gap_ffill(m[wm_channel].where(have)).clip(0, C.APPLIANCE_CLIP_W)
    flat = flat_run_mask(agg.to_numpy())
    issues = m["issues"].astype(bool) if "issues" in m else pd.Series(False, index=m.index)
    df = pd.DataFrame({"agg": agg, "wm": wm}, index=m.index)
    df["agg_missing"] = df["agg"].isna() | flat
    df["issues"] = issues.to_numpy()
    df["target_ok"] = ~df["agg_missing"] & df["wm"].notna() & ~df["issues"]
    df["house"] = house
    return df


@dataclass
class SplitSpec:
    train_houses: list[int] = field(default_factory=lambda: list(C.TRAIN_HOUSES))
    val_house: int = C.VAL_HOUSE
    test_houses: list[int] = field(default_factory=lambda: [C.TEST_HOUSE])
    seen_test_frac: float = 0.2      # last 20 % of each training house = seen-house temporal test
    embargo_days: int = 1            # extra gap (plus one window) between train and seen-test


def assign_splits(frames: dict[int, pd.DataFrame], spec: SplitSpec, window: int) -> dict[int, pd.Series]:
    """Label every minute of every house: train / seen_test / val / test / embargo."""
    labels = {}
    for h, df in frames.items():
        lab = pd.Series("unused", index=df.index)
        if h in spec.train_houses:
            t0, t1 = df.index.min(), df.index.max()
            cut = t0 + (t1 - t0) * (1 - spec.seen_test_frac)
            gap = pd.Timedelta(f"{spec.embargo_days}D") + pd.Timedelta(f"{window}min")
            lab[df.index < cut] = "train"
            lab[(df.index >= cut) & (df.index < cut + gap)] = "embargo"
            lab[df.index >= cut + gap] = "seen_test"
        elif h == spec.val_house:
            lab[:] = "val"
        elif h in spec.test_houses:
            lab[:] = "test"
        labels[h] = lab
    return labels


@dataclass
class Corpus:
    """All houses concatenated into flat arrays with window-start indices per split."""
    agg: np.ndarray            # float32, aggregate (missing minutes filled for input only)
    wm: np.ndarray             # float32, washing-machine target (NaN where unusable)
    house: np.ndarray          # int16 house id per position (-1 in separators)
    time: np.ndarray           # int64 UTC epoch seconds per position (-1 in separators)
    split: np.ndarray          # str label per position
    window: int
    starts: dict[str, np.ndarray]  # window start indices per split (target = start + window//2)

    def targets(self, split: str) -> np.ndarray:
        return self.starts[split] + self.window // 2


def build_corpus(
    frames: dict[int, pd.DataFrame],
    labels: dict[int, pd.Series],
    window: int,
    max_missing_frac: float = 0.10,
) -> Corpus:
    sep = window  # separator length between houses: all-missing, so windows cannot cross it
    aggs, wms, hs, ts, sps, miss, ok = [], [], [], [], [], [], []
    for h, df in frames.items():
        a = df["agg"].to_numpy(dtype=float)
        miss_h = df["agg_missing"].to_numpy()
        a_in = pd.Series(np.where(miss_h, np.nan, a)).ffill().bfill().fillna(0).to_numpy()
        aggs += [a_in, np.zeros(sep)]
        wms += [np.where(df["target_ok"], df["wm"].to_numpy(dtype=float), np.nan), np.full(sep, np.nan)]
        hs += [np.full(len(df), h, dtype=np.int16), np.full(sep, -1, dtype=np.int16)]
        ts += [df.index.asi8 // 10**9, np.full(sep, -1, dtype=np.int64)]
        sps += [labels[h].to_numpy(dtype=object), np.full(sep, "sep", dtype=object)]
        miss += [miss_h, np.ones(sep, dtype=bool)]
        ok += [df["target_ok"].to_numpy(), np.zeros(sep, dtype=bool)]
    agg = np.concatenate(aggs).astype(np.float32)
    wm = np.concatenate(wms).astype(np.float32)
    house = np.concatenate(hs)
    time = np.concatenate(ts)
    split = np.concatenate(sps)
    missing = np.concatenate(miss)
    target_ok = np.concatenate(ok)

    n = len(agg)
    half = window // 2
    cmiss = np.r_[0, np.cumsum(missing)]
    # a window [s, s + window) is homogeneous when the split label does not change inside it
    codes, uniq = pd.factorize(pd.Series(split))
    change = np.r_[0, np.cumsum(codes[1:] != codes[:-1])]
    s = np.arange(0, n - window + 1)
    homogeneous = change[s] == change[s + window - 1]
    miss_frac = (cmiss[s + window] - cmiss[s]) / window
    t = s + half
    valid = homogeneous & (miss_frac <= max_missing_frac) & target_ok[t]
    starts = {lab: s[valid & (split[t] == lab)] for lab in ("train", "seen_test", "val", "test")}
    return Corpus(agg=agg, wm=wm, house=house, time=time, split=split, window=window, starts=starts)


def leakage_audit(corpus: Corpus) -> pd.DataFrame:
    """Prove that no window mixes splits or houses and that splits do not share minutes."""
    rows = []
    W = corpus.window
    for lab, st in corpus.starts.items():
        if len(st) == 0:
            continue
        first, last = st, st + W - 1
        rows.append(
            {
                "split": lab,
                "windows": len(st),
                "houses": sorted(int(h) for h in np.unique(corpus.house[st + W // 2])),
                "windows_spanning_two_labels": int((corpus.split[first] != corpus.split[last]).sum()),
                "windows_spanning_two_houses": int((corpus.house[first] != corpus.house[last]).sum()),
                "first_target_utc": pd.to_datetime(corpus.time[st + W // 2].min(), unit="s"),
                "last_target_utc": pd.to_datetime(corpus.time[st + W // 2].max(), unit="s"),
            }
        )
    return pd.DataFrame(rows)


def temporal_gap_check(corpus: Corpus) -> pd.DataFrame:
    """Per training house: minutes between the last train window and the first seen-test window.

    The gap must exceed one window length, so no seen-test target is ever inside a training
    input window (and vice versa).
    """
    W = corpus.window
    tr, st = corpus.starts["train"], corpus.starts["seen_test"]
    rows = []
    for h in np.unique(corpus.house[tr + W // 2]):
        last_train_input = corpus.time[tr[corpus.house[tr] == h] + W - 1].max()
        first_test_input = corpus.time[st[corpus.house[st] == h]].min()
        gap_min = (first_test_input - last_train_input) / 60
        rows.append({"house": int(h), "gap_minutes": float(gap_min), "window": W, "ok": bool(gap_min > 0)})
    return pd.DataFrame(rows)


def standardisation(corpus: Corpus, split: str = "train") -> dict:
    """Mean/std of aggregate and target over the *training* targets only."""
    t = corpus.targets(split)
    a, y = corpus.agg[t], corpus.wm[t]
    return {
        "agg_mean": float(np.mean(a)), "agg_std": float(np.std(a)),
        "wm_mean": float(np.nanmean(y)), "wm_std": float(np.nanstd(y)),
    }


def prepare(windows: list[int], houses: list[int] | None = None) -> tuple[dict, dict, dict]:
    """Frames, corpora and training statistics for the standard split (used by notebooks 03-06)."""
    wm = io.washing_machine_channels(io.parse_appliance_map(C.DATA_DIR / "CLEAN_READ_ME_081116.txt"))
    houses = houses or (C.TRAIN_HOUSES + [C.VAL_HOUSE, C.TEST_HOUSE] + C.EXTRA_UNSEEN_HOUSES)
    frames = {h: house_frame(h, f"Appliance{wm[h][0]}") for h in houses}
    spec = SplitSpec(test_houses=[C.TEST_HOUSE] + C.EXTRA_UNSEEN_HOUSES)
    corpora, stats = {}, {}
    for w in windows:
        corpora[w] = build_corpus(frames, assign_splits(frames, spec, w), w)
        stats[w] = standardisation(corpora[w])
    return frames, corpora, stats
