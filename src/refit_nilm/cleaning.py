"""Inspection and cleaning of the RAW REFIT files (Requirement 1).

Pipeline (each step is a separate function so it can be tested and reported on):

1. ``local_epoch_to_utc``   raw ``Unix`` stores UK wall-clock time as if it were UTC; convert it.
2. ``resolve_part_overlap`` Part1 and Part2 overlap in time; keep one source per instant.
3. ``drop_duplicates``      exact duplicate rows, then same-timestamp conflicts (keep last poll).
4. ``mask_impossible``      sentinel / physically impossible readings -> NaN (not 0).
5. ``ffill_event_rows``     Part2 rows only carry sensors that reported; carry values <= 2 min.
6. ``to_minutes``           1-minute mean on a regular UTC grid + reading counts.
7. ``fill_short_gaps``      forward-fill gaps <= 3 min; longer gaps stay NaN and are listed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

IAM_COLS = C.CHANNELS[1:]
SENTINELS = (32766.0, 32767.0, 65535.0, 98301.0)
# A UK single-phase domestic supply is fused at 60-100 A; 100 A x 230 V = 23 kW.
MAX_AGGREGATE_W = 23_000.0
EVENT_FFILL_LIMIT_S = 120


@dataclass
class CleaningReport:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, **info) -> None:
        self.steps.append({"step": step, **info})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# ----------------------------------------------------------------------------------------
# Inspection
# ----------------------------------------------------------------------------------------
def inspect_raw(df: pd.DataFrame, name: str) -> dict:
    """Quality summary of one raw file, in file order (before any change)."""
    u = df["Unix"].to_numpy()
    dt = np.diff(u)
    t = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    time_minus_unix = (t.astype("int64") // 10**9) - df["Unix"]
    iam = df[IAM_COLS]
    return {
        "file": name,
        "rows": len(df),
        "first_unix_as_utc": pd.to_datetime(u.min(), unit="s"),
        "last_unix_as_utc": pd.to_datetime(u.max(), unit="s"),
        "time_parse_failures": int(t.isna().sum()),
        "time_vs_unix_offset_s_max_abs": int(time_minus_unix.abs().max()),
        "backward_steps": int((dt < 0).sum()),
        "duplicate_unix": int(df["Unix"].duplicated().sum()),
        "exact_duplicate_rows": int(df.duplicated(subset=["Unix"] + C.CHANNELS).sum()),
        "median_interval_s": float(np.median(dt[dt > 0])),
        "p99_interval_s": float(np.percentile(dt[dt > 0], 99)),
        "gaps_over_2min": int((dt > 120).sum()),
        "gaps_over_1h": int((dt > 3600).sum()),
        "aggregate_nan_pct": round(100 * df["Aggregate"].isna().mean(), 3),
        "iam_nan_pct": round(100 * iam.isna().to_numpy().mean(), 3),
        "negative_values": int((df[C.CHANNELS] < 0).sum().sum()),
        "aggregate_over_23kW": int((df["Aggregate"] > MAX_AGGREGATE_W).sum()),
        "iam_over_4kW": int((iam > C.APPLIANCE_CLIP_W).sum().sum()),
        "iam_sentinel_values": int(iam.isin(SENTINELS).sum().sum()),
        "rows_iam_sum_gt_aggregate_pct": round(
            100 * float((iam.fillna(0).sum(axis=1) > df["Aggregate"]).mean()), 3
        ),
    }


def clock_change_hours(unix: pd.Series, start: str, end: str) -> pd.DataFrame:
    """Rows logged in the 01:00-02:00 hour of every UK clock-change day in [start, end].

    A file that stores local wall-clock time has no rows in the skipped spring hour and
    roughly double the rows in the repeated autumn hour.
    """
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    out = []
    for y in years:
        # last Sunday of March / October
        for month, kind in ((3, "spring (hour skipped)"), (10, "autumn (hour repeated)")):
            last = pd.Timestamp(year=y, month=month, day=31)
            day = last - pd.Timedelta(days=int((last.weekday() + 1) % 7))
            s = pd.Timestamp(f"{day.date()} 01:00").value // 10**9
            if not (pd.Timestamp(start).value // 10**9 <= s <= pd.Timestamp(end).value // 10**9):
                continue
            n = int(((unix >= s) & (unix < s + 3600)).sum())
            ref = int(((unix >= s - 4 * 3600) & (unix < s - 3600)).sum()) / 3
            out.append({"date": day.date(), "change": kind, "rows_01_02": n, "rows_per_hour_before": round(ref)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------
# Step 1: wall-clock -> UTC
# ----------------------------------------------------------------------------------------
def local_epoch_to_utc(unix_local: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convert epoch seconds that encode UK wall-clock time into true UTC epoch seconds.

    Ambiguous autumn hour: if the file shows a backward time step inside that hour, rows
    before the step are summer time (BST) and rows after it are winter time (GMT). If the
    file was already sorted (no step), the hour cannot be disambiguated and its rows are
    returned as -1 (dropped later and counted in the report).
    Non-existent spring hour: rows there (should be none) are also returned as -1.
    """
    unix_local = np.asarray(unix_local, dtype="int64")
    naive = pd.to_datetime(unix_local, unit="s")
    n = len(naive)
    ambiguous_dst = np.zeros(n, dtype=bool)
    undecidable = np.zeros(n, dtype=bool)
    backward = np.r_[False, np.diff(unix_local) < 0]
    hours = pd.Series(naive).dt
    is_amb_window = (hours.month == 10) & (hours.hour == 1) & (hours.weekday == 6) & (hours.day >= 25)
    for day in pd.unique(pd.Series(naive[is_amb_window.to_numpy()]).dt.date):
        idx = np.flatnonzero(is_amb_window.to_numpy() & (pd.Series(naive).dt.date == day).to_numpy())
        # include the row right after the window: the backward step can land just outside it
        steps = [i for i in range(idx[0], min(idx[-1] + 2, n)) if backward[i]]
        if steps:
            ambiguous_dst[idx[idx < steps[0]]] = True
        else:
            undecidable[idx] = True
    loc = naive.tz_localize(C.LOCAL_TZ, ambiguous=ambiguous_dst, nonexistent="NaT")
    utc = np.where(loc.isna(), -1, loc.tz_convert("UTC").asi8 // 10**9)
    utc[undecidable] = -1
    info = {
        "rows": n,
        "rows_shifted_by_1h": int(((utc - unix_local) == -3600).sum()),
        "rows_dropped_ambiguous_or_nonexistent": int((utc == -1).sum()),
        "backward_steps_before": int(backward.sum()),
    }
    return utc, info


# ----------------------------------------------------------------------------------------
# Steps 2-5
# ----------------------------------------------------------------------------------------
def resolve_part_overlap(p1: pd.DataFrame, p2: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep Part1 inside the overlap window and Part2 after it.

    Both inputs must already have a ``utc`` column. Part1 is the denser, complete-row source
    in the overlap window (one row per poll with all ten channels); Part2 rows there are
    sparse event rows. The choice is checked against the official cleaned file in NB01.
    """
    overlap_start = p2["utc"].min()
    overlap_end = p1["utc"].max()
    keep2 = p2["utc"] > overlap_end
    merged = pd.concat([p1.assign(part=1), p2.loc[keep2].assign(part=2)], ignore_index=True)
    info = {
        "overlap_start_utc": pd.to_datetime(overlap_start, unit="s"),
        "overlap_end_utc": pd.to_datetime(overlap_end, unit="s"),
        "part2_rows_dropped_in_overlap": int((~keep2).sum()),
        "part1_rows_in_overlap": int((p1["utc"] >= overlap_start).sum()),
    }
    return merged, info


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rule: drop exact duplicate rows; for the same instant with different values keep the
    last row in acquisition order (the most recent poll of the sensors)."""
    n0 = len(df)
    d = df.drop_duplicates(subset=["utc"] + C.CHANNELS, keep="first")
    n1 = len(d)
    d = d.drop_duplicates(subset=["utc"], keep="last")
    d = d.sort_values("utc", kind="mergesort").reset_index(drop=True)
    return d, {"exact_duplicates_removed": n0 - n1, "conflicting_same_time_removed": n1 - len(d)}


def mask_impossible(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Physically impossible or sentinel readings become NaN (missing), not 0 (off)."""
    d = df.copy()
    info = {}
    iam = d[IAM_COLS]
    bad_iam = iam.isin(SENTINELS) | (iam > C.APPLIANCE_CLIP_W) | (iam < 0)
    info["iam_cells_masked"] = int(bad_iam.sum().sum())
    d[IAM_COLS] = iam.mask(bad_iam)
    bad_agg = (d["Aggregate"] > MAX_AGGREGATE_W) | (d["Aggregate"] < 0)
    info["aggregate_cells_masked"] = int(bad_agg.sum())
    d.loc[bad_agg, "Aggregate"] = np.nan
    return d, info


def ffill_event_rows(df: pd.DataFrame, limit_s: int = EVENT_FFILL_LIMIT_S) -> tuple[pd.DataFrame, dict]:
    """Carry each channel's last reading forward for at most ``limit_s`` seconds.

    REFIT logged a sensor only when its value changed and Part2 writes NaN for sensors that
    did not report at a poll, so a NaN between two readings usually means "unchanged", not
    "missing". A time limit stops values being carried across real outages.
    """
    d = df.copy()
    t = d["utc"].to_numpy()
    filled = 0
    for col in C.CHANNELS:
        v = d[col].to_numpy()
        has = ~np.isnan(v)
        last_t = pd.Series(np.where(has, t, np.nan)).ffill().to_numpy()
        ff = pd.Series(v).ffill().to_numpy()
        ok = (~has) & (~np.isnan(last_t)) & ((t - last_t) <= limit_s)
        filled += int(ok.sum())
        d[col] = np.where(has, v, np.where(ok, ff, np.nan))
    return d, {"cells_forward_filled": filled, "limit_s": limit_s}


# ----------------------------------------------------------------------------------------
# Steps 6-7
# ----------------------------------------------------------------------------------------
def to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """1-minute mean on a regular UTC grid; ``n_readings`` = readings in the minute."""
    minute = (df["utc"].to_numpy() // 60) * 60
    grp = df[C.CHANNELS].groupby(minute)
    out = grp.mean()
    out["n_readings"] = grp.size()
    full = np.arange(out.index.min(), out.index.max() + 60, 60)
    out = out.reindex(full)
    out["n_readings"] = out["n_readings"].fillna(0).astype("int32")
    out.index = pd.to_datetime(out.index, unit="s", utc=True)
    out.index.name = "utc"
    return out


def gap_table(mask_missing: pd.Series) -> pd.DataFrame:
    """Contiguous runs of missing minutes -> start, end, length."""
    m = mask_missing.to_numpy()
    if not m.any():
        return pd.DataFrame(columns=["start", "end", "minutes"])
    edges = np.diff(np.r_[0, m.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    idx = mask_missing.index
    return pd.DataFrame({"start": idx[starts], "end": idx[ends - 1], "minutes": ends - starts})


def fill_short_gaps(m: pd.DataFrame, max_gap_min: int = C.SHORT_GAP_FILL_MIN) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Forward-fill runs of missing minutes no longer than ``max_gap_min``; flag the rest."""
    d = m.copy()
    missing = d["Aggregate"].isna()
    gaps = gap_table(missing)
    short = gaps[gaps["minutes"] <= max_gap_min]
    fill_mask = pd.Series(False, index=d.index)
    for s, e in zip(short["start"], short["end"]):
        fill_mask.loc[s:e] = True
    filled = d[C.CHANNELS].ffill()
    d.loc[fill_mask, C.CHANNELS] = filled.loc[fill_mask]
    d["imputed"] = fill_mask
    d["long_gap"] = d["Aggregate"].isna()
    long_gaps = gaps[gaps["minutes"] > max_gap_min].reset_index(drop=True)
    info = {
        "short_gaps_filled": len(short),
        "minutes_filled": int(fill_mask.sum()),
        "long_gaps_flagged": len(long_gaps),
        "minutes_in_long_gaps": int(long_gaps["minutes"].sum()) if len(long_gaps) else 0,
    }
    return d, long_gaps, info


def clean_raw_house(p1: pd.DataFrame, p2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Run the full raw cleaning pipeline for a house split into Part1/Part2."""
    rep = CleaningReport()
    parts = []
    for i, p in enumerate((p1, p2), start=1):
        utc, info = local_epoch_to_utc(p["Unix"].to_numpy())
        rep.add(f"Part{i}: wall-clock -> UTC", **info)
        parts.append(p.assign(utc=utc).loc[utc >= 0])
    merged, info = resolve_part_overlap(*parts)
    rep.add("resolve Part1/Part2 overlap", **info)
    d, info = drop_duplicates(merged)
    rep.add("drop duplicates", **info)
    d, info = mask_impossible(d)
    rep.add("mask impossible values", **info)
    d, info = ffill_event_rows(d)
    rep.add("carry event rows forward", **info)
    m = to_minutes(d)
    rep.add("resample to 1 minute", minutes=len(m), minutes_without_readings=int((m["n_readings"] == 0).sum()))
    m, long_gaps, info = fill_short_gaps(m)
    rep.add("fill short gaps / flag long gaps", **info)
    m["issues"] = (m[IAM_COLS].sum(axis=1, min_count=1) > m["Aggregate"]).astype("int8")
    rep.add("flag IAM sum > aggregate", minutes_flagged=int(m["issues"].sum()))
    return m, long_gaps, rep
