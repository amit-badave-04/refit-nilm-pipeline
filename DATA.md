# DATA.md — REFIT data: coverage, quality, cleaning decisions

Every number here is produced by a notebook and saved under `artifacts/`; the notebook and table
are named next to each claim.

## 1. Dataset

**REFIT Electrical Load Measurements** (University of Strathclyde; Murray, Stankovic & Stankovic,
*Scientific Data* 4:160122, 2017). Licence CC BY 4.0.

| | |
|---|---|
| Homes | 20 houses near Loughborough, UK, numbered 1–21 (no House 14) |
| Period | Sep 2013 – Jul 2015 (per house 13–21 months; table `nb02_coverage_quality.csv`) |
| Channels | 1 aggregate (current clamp at the meter) + 9 individual appliance monitors (IAM plugs) per house |
| Quantity / unit | active power, watts; no voltage, current, reactive power or power factor |
| Sampling | logger polled every 6–8 s; a value was stored only when it changed; sensors not synchronised |
| Files used | raw `REFIT_RAW_081116.7z` → `RAW_House1_Part1.csv`, `RAW_House1_Part2.csv` (Requirement 1); cleaned `CLEAN_REFIT_081116.7z` → `CLEAN_House{N}.csv`, all 20 houses (Requirements 2–3); `CLEAN_READ_ME_081116.txt` for the appliance-to-channel map |
| Integrity | SHA-256 of both archives pinned in `src/refit_nilm/config.py` and checked before use |

The portal links in the brief (`pureportal.strath.ac.uk/files/...`) sit behind an interactive
browser check; `scripts/prepare_data.py` downloads the identical files from the equivalent
`pure.strath.ac.uk/ws/portalfiles/portal/...` addresses and verifies the checksums.

**Appliance map.** Parsed from the README (`io.parse_appliance_map`) rather than typed by hand,
and cross-checked against the washing-machine columns used by the reference code of D'Incecco et
al. (2020): all nine agree. 19 houses have a washing machine (WM); House 12 has none; House 4 has
two; Houses 1, 8, 9 and 18 also have a separate washer-dryer.

**Known limitations of REFIT** (dataset paper and README): a network outage in February 2014;
plug monitors cover only 22–55 % of household consumption (dataset paper), so most load is unmetered; houses
3, 11 and 21 have rooftop PV; occupants could move plugs between appliances (documented changes
are listed in the README, e.g. House 13's washing machine was replaced on 25 Mar 2015); IAM
readings above 4 kW are sensor faults.

## 2. Raw House 1: what I found (notebook 01)

| Check | Part 1 (9 Oct 2013 – 1 Oct 2014) | Part 2 (27 Jun 2014 – 10 Jul 2015) |
|---|---|---|
| rows | 4,377,877 | 4,140,886 |
| timestamp parse failures | 0 | 0 |
| backward time steps | 1 (27 Oct 2013 01:59:58 → 01:00:01) | 0 |
| duplicate timestamps | 1,004,404 | 14,755 |
| … of which exact duplicate rows | 884,307 | 393 |
| median / 99th-percentile spacing | 7 s / 16 s | 7 s / 9 s |
| gaps > 2 min / > 1 h | 593 / 14 | 122 / 19 |
| aggregate NaN | 0 % | 4.6 % |
| IAM NaN | 0 % (unavailable sensors written as 0) | 13.7 % (event-style rows) |
| aggregate > 23 kW (impossible for a 100 A supply) | 26 | 51 |
| IAM readings > 4 kW | 3,038 | 911 |
| IAM overflow sentinels (32,767 / 65,535 / 98,301) | 145 | 24 |
| rows where IAMs sum to more than the aggregate | 1.19 % | 0.32 % |

Source: `artifacts/tables/nb01_raw_inspection.csv`, `nb01_extreme_values.csv`.

### The most important finding: the raw clock is UK wall-clock time

The README describes `Unix` as UTC. It is not. In both raw parts the 01:00–02:00 hour of each
spring clock change has **no rows**, and the repeated autumn hour has **about twice** the normal
number (`nb01_clock_change_hours.csv`). The single backward step in Part 1 is the logger's clock
going back at the October 2013 change. A month-by-month correlation against the official cleaned
file confirms a one-hour offset in every summer-time month.

**Decision.** Convert Europe/London wall-clock time to UTC *before* sorting. The repeated autumn
hour is resolved from file order (rows before the backward step are summer time); where a file
is already sorted the hour is ambiguous and its rows are dropped (986 rows in Part 2).
Sorting first would interleave two real hours irreversibly.

### A related problem in the official cleaned release

The cleaned files are true UTC only up to the date each house's raw Part 1 ends (about 1 Oct
2014; House 13: 12 Sep, House 16: 14 Jul 2014). After that they hold wall-clock time again, so
every summer-time reading in 2015 is one hour late (`nb01_official_time_regime.csv`: in Apr–Jul
2015 the official file matches my UTC series only after a one-hour shift, r ≈ 0.99 vs ≈ 0.05).
`io.load_clean_minutes` corrects this per house, using the Part 1 end dates streamed from the
raw archive (`artifacts/tables/raw_part1_end.csv`). Without it, hour-of-day analyses and models
would see 2015 summer laundry an hour late.

## 3. Cleaning pipeline for raw House 1 (`src/refit_nilm/cleaning.py`)

| # | Step | Rule | Effect on House 1 |
|---|---|---|---|
| 1 | Wall-clock → UTC | per part, before sorting; repeated hour resolved by file order | 2.80 M + 2.25 M rows shifted by 1 h; 986 ambiguous rows dropped |
| 2 | Part overlap | keep Part 1 inside the overlap window (27 Jun – 1 Oct 2014), Part 2 after | 945 k Part 2 rows dropped; Part 1 is denser (9.2 vs 6.9 rows/min) and agrees with the official file (r = 0.999 vs 0.777) |
| 3 | Duplicates | drop exact duplicates; for the same instant with different values keep the last poll | 884,252 + 131,753 rows removed |
| 4 | Impossible values | IAM > 4 kW or sentinel, aggregate > 23 kW or < 0 → **NaN** (missing), not 0 | 3,222 IAM + 59 aggregate cells |
| 5 | Event rows | carry each channel's last reading forward for ≤ 120 s (a NaN between readings means "unchanged") | 2.33 M cells filled |
| 6 | Resample | 1-minute **mean** on a regular UTC grid (preserves energy) + reading count | 920,031 minutes; 122,187 without any reading |
| 7 | Short gaps | runs of ≤ 3 missing minutes forward-filled (1-minute REFIT protocol, Petralia et al. 2025) | 446 gaps, 631 minutes |
| 8 | Long gaps | > 3 minutes: left missing and listed, never interpolated | 447 outages, 121,577 minutes (≈ 84 days); longest 41.6 days (24 Jan – 6 Mar 2014) |
| 9 | Sensor mismatch | minutes where IAMs sum above the aggregate flagged `issues = 1`, kept | 7,049 minutes |

Result: 86.8 % of minutes have a valid aggregate. Before/after plots:
`artifacts/figures/nb01_before_after_day.png`, `nb01_before_after_week.png`.

**Why NaN and not 0 for faulty readings.** The official release replaces IAM spikes with 0. For a
disaggregation target, 0 means "the appliance was off" — a label — while NaN means "no label".
Writing faults as 0 silently teaches a model that the appliance was off when the sensor was broken.

**Validation of my pipeline.** Against the official cleaned file (mapped to true UTC), my
independently cleaned House 1 agrees on the aggregate with r = 0.9993 (mean absolute difference
3.0 W) and on the washing machine with r = 0.9992; each channel's total energy agrees within
0.2 kWh (`nb01_validation_vs_official.csv`).

## 4. Cleaned data used for EDA and modelling (notebooks 02–03)

**How outages appear in the cleaned release.** Two forms, both treated as missing:
1. minutes with no row at all (6–25 % of minutes per house);
2. forward-filled **flat lines**: the README says NaNs were forward-filled, so an outage can look
   like a constant aggregate. A real home never holds an identical 1-minute mean for an hour (the
   fridge alone cycles every 20–40 minutes), so runs of ≥ 60 identical minutes are masked. They
   cover up to 11.7 % of House 3 and 7.8 % of Houses 10 and 21 (`nb02_coverage_quality.csv`).
   A day from House 2 that first looked "fully covered" turned out to be one of these.

Zero aggregate readings are essentially absent (≤ 0.4 %), so "aggregate = 0 means missing" is
not a useful rule in this release; the flat-line rule is.

**Per-home quality for modelling** (`nb03_label_quality.csv`): 73–89 % of minutes are usable
targets; the WM is on in 1–9 % of usable minutes; the noise-to-aggregate ratio (unmetered share,
Klemenjak et al. 2020) is 47–80 %, highest in the test house (House 8, 80 %).

**Gap / outlier decisions for modelling.** Aggregate clipped to [0, 10 kW] and WM to [0, 4 kW]
(none of the clipping is active after masking); gaps ≤ 3 minutes forward-filled; longer gaps,
flat lines and `issues` minutes are never targets; input windows with > 10 % missing minutes are
dropped; missing minutes inside an accepted window are filled from the last valid reading for the
input only.

## 5. Remaining concerns

* **Part 1 zeros.** In raw Part 1 an unavailable sensor was written as 0, so "off" and "not
  reporting" cannot be separated for that period.
* **Unsynchronised sensors.** Plug monitors can lead or lag the aggregate by a few readings
  (up to ~7 s); at 1-minute resolution this is minor, but it causes the `issues` minutes.
* **Undocumented plug moves.** A WM that was unplugged without notice reads 0 and looks like a
  home that stopped washing. Per-home cycle frequency (notebook 02) is the only check.
* **Washer-dryers.** Houses 1, 8, 9 and 18 have one; its heating phase looks like a WM's in the
  aggregate, and in House 1 its channel carries over 2,000 impossible readings.
* **Two machines in House 4** (both in use) and a **replaced machine in House 13** make those
  labels unsuitable for training; both are excluded from modelling.
* **PV houses (3, 11, 21).** The aggregate clamp measures magnitude, so generation distorts it;
  excluded from modelling, kept in the EDA (the WM plug itself is unaffected).
* **1-minute averaging** preserves energy but not peaks: a 10 kW, 30-second spike becomes ~5 kW.
