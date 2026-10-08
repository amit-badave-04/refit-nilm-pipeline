# REFIT washing-machine disaggregation

A reproducible pipeline that cleans raw smart-meter data, characterises washing-machine use across
UK homes, and estimates washing-machine power from the whole-home signal alone, evaluated on homes
the model has never seen. Built on the REFIT Electrical Load Measurements dataset (University of
Strathclyde, 20 UK homes, 2013–2015).

| Document | What it contains |
|---|---|
| [`DATA.md`](DATA.md) | dataset coverage, raw-data quality findings, cleaning decisions, remaining concerns |
| [`RESULTS.md`](RESULTS.md) | washing-machine EDA, model results, validation design, failure analysis, 15/30-minute study |
| [`Recommendation.md`](Recommendation.md) | one-page utility-facing summary |
| [`CITATIONS.md`](CITATIONS.md) | the research used and what was adopted from each paper |
| `notebooks/` | the analysis, in order; each notebook has a header, reasons before each step, and a closing summary |

<!-- RESULTS-AT-A-GLANCE -->

## What I built

1. **Raw-data cleaning (House 1, notebook 01).** A tested pipeline that converts the logger's
   wall-clock timestamps to UTC, resolves the overlap between the two raw files, removes
   duplicates by a stated rule, masks impossible readings as missing, resamples to 1 minute,
   fills gaps of up to 3 minutes and flags longer outages. It reproduces the official cleaned
   release to r = 0.999.
2. **Washing-machine EDA (notebook 02).** Coverage and quality for the 20 washing machines in 19
   homes, an explicit cycle rule with sensitivity checks, usage by hour and weekday, and cycle
   duration, power and energy across homes.
3. **Modelling data contract (notebook 03).** The reference REFIT split (train on homes 2, 5, 7, 9,
   15, 16, 17; validate on 18; test on 8) plus five further unseen homes, a seen-home temporal
   test, label-quality masks and a leakage audit.
4. **Models (notebooks 04–05).** Two floors, a LightGBM baseline, Seq2Point, and one targeted
   improvement (a state-gated Seq2Point), each deep model trained with three seeds.
5. **Evaluation (notebook 06).** Scores on six unseen homes and on later data from the training
   homes, actual-vs-predicted plots, a failure analysis that names the appliances behind false
   detections, and a 15/30-minute resolution study.
6. **Inference service (`service/`).** The selected model exported to ONNX (int8, parity-checked)
   behind a FastAPI app with input validation, rate limiting, structured logs, a model card and
   contract tests, packaged for Fly.io.

## Repository layout

```
├── src/refit_nilm/        library used by every notebook (tested)
│   ├── io.py              download + checksum, extraction, README parsing, UTC-correct 1-min loader
│   ├── cleaning.py        raw-data inspection and cleaning steps
│   ├── cycles.py          washing-machine activation rule, threshold checks
│   ├── datasets.py        label masks, splits, leakage-safe window indices, resampling
│   ├── features.py        window features (LightGBM) and usage profile
│   ├── models/seq2point.py  Seq2Point and gated Seq2Point
│   ├── train.py           GPU training loop, early stopping, checkpoints
│   └── metrics.py         MAE, MAE_ON, SAE, NDE, EpD, F1, AUPRC, cycle-level and energy-split metrics
├── notebooks/             01–06 (.py sources and executed .ipynb)
├── scripts/               prepare_data.py, run_notebooks.py, export_onnx.py
├── tests/                 unit tests for the library
├── service/               FastAPI + ONNX inference service, its tests, Dockerfile, fly.toml
├── artifacts/             versioned outputs: figures, tables, metrics JSON (models are rebuilt, not versioned)
├── environment.yml        conda environment
└── requirements-lock.txt  exact package versions used
```

## Setup and run

Tested on Windows 11 with Miniforge and an NVIDIA RTX 5090 laptop GPU; the code also runs on
Linux/macOS and on CPU (training is then slower).

```bash
# 1. environment (conda-forge Python 3.12 + scientific stack + 7-Zip)
mamba env create -f environment.yml
conda activate refit-nilm

# 2. PyTorch with CUDA (Blackwell GPUs need a CUDA >= 12.8 build; on CPU-only machines use
#    --index-url https://download.pytorch.org/whl/cpu)
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install -e . "onnx==1.23.2" onnxscript "onnxruntime==1.30.0" fastapi "uvicorn[standard]" httpx pytest-cov

# 3. Jupyter kernel used by the notebooks
python -m ipykernel install --user --name refit-nilm --display-name refit-nilm

# 4. data: downloads ~1.2 GB (skipped if the archives are already in Data/), verifies SHA-256,
#    extracts and builds the 1-minute cache (~7 GB on disk)
python scripts/prepare_data.py

# 5. tests, then every notebook in order (≈ 1.5 h with a GPU; writes artifacts/)
python -m pytest -q
python scripts/run_notebooks.py
```

`requirements-lock.txt` pins every package version of the environment the results were produced
with. The notebooks are written as percent-format `.py` files (readable diffs) and executed into
`.ipynb` by `scripts/run_notebooks.py`; open either in VS Code or Jupyter.

## Key assumptions and preprocessing decisions

* **Time.** REFIT's raw `Unix` column is UK wall-clock time, not UTC, and the official cleaned
  files revert to wall-clock time after 1 Oct 2014. Everything here is in true UTC; hour-of-day
  uses Europe/London local time (`DATA.md` §2).
* **Resolution.** All channels are resampled to 1-minute means on a regular UTC grid; gaps of up to
  3 minutes are forward-filled, longer gaps are left missing and never used as targets.
* **Missing is not off.** Faulty readings and outages are missing values, not zeros; forward-filled
  flat lines in the cleaned release are detected and treated as outages.
* **Cycle rule.** A washing cycle is ≥ 20 W, with off-gaps < 3 minutes bridged and ≥ 30 minutes
  long (Kelly & Knottenbelt 2015), checked for sensitivity.
* **Homes used for modelling.** Excluded: House 12 (no washing machine), 4 (two machines), 13
  (machine replaced mid-study), 3, 11 and 21 (rooftop PV distorts the aggregate).

## Validation design

* **Primary: unseen homes.** Train on 2, 5, 7, 9, 15, 16, 17; select on 18; test once on 8 (the
  reference REFIT washing-machine split) and on 1, 6, 10, 19, 20.
* **Secondary: seen homes, later period.** The last 20 % of each training home, after a one-day plus
  one-window embargo, measures how much skill is home-specific.
* **No leakage.** Splits are by home; no window crosses a split or home boundary (audited);
  normalisation, window length, loss weight, early stopping and thresholds use training or
  validation data only; inputs are the aggregate only.
* **Metrics.** MAE alone rewards predicting nothing on a load that is off 97 % of the time, so I
  report MAE, MAE while on, total and daily energy error (SAE, EpD), NDE, on/off F1, cycle-level
  F1, and how much true energy is recovered vs added.

## Trade-offs

* **Seq2Point over a Transformer.** Seq2Point is the most replicated model on REFIT washing
  machines and trains in minutes, so I could run three seeds of every variant and a resolution
  study. NILMFormer (KDD 2025) reports lower 1-minute washer errors and is the next model to try.
* **One improvement, isolated.** The gated model changes exactly one thing (an on/off head that
  gates the output), so its effect can be attributed.
* **No NILM framework dependency.** NILMTK is installed from git and nilmtk-contrib pins Python 3.11;
  the components needed here are short and unit-tested, and the protocol follows NILMTK-contrib and
  the papers in `CITATIONS.md`.
* **1-minute input.** It is the resolution the brief asks for and a realistic smart-plug or
  CAD-device rate; notebook 06 quantifies what is lost at 15 and 30 minutes.

## AI tools used

AI coding assistants were used for code drafting, documentation editing and literature lookup.
All analysis decisions, verification of results and interpretation are my own.

## Possible improvements

* **Synthetic-activation augmentation** (Kelly & Knottenbelt 2015; Rafiq et al. 2021): add real
  washing-machine cycles onto other homes' aggregates to multiply training examples and teach the
  model to ignore washer-dryers and showers.
* **NILMFormer-style models** with per-window normalisation, which also helps with voltage
  differences between markets.
* **Fine-tuning on a handful of labelled homes** in a new market (freeze the trunk, retrain the
  head), as D'Incecco et al. show for cross-dataset transfer.
* **Richer electrical inputs** (reactive power, current harmonics) from the meters Flock deploys:
  a washing machine's motor and heater are far easier to separate with reactive power.
* **Probabilistic outputs and monitoring:** calibrate `P(on)`, track input drift per home, and
  trigger re-validation when a home's appliance stock changes.

## Licence and data

Code: MIT (`LICENSE`). Data: REFIT is CC BY 4.0 (Murray, Stankovic & Stankovic 2017); it is
downloaded by `scripts/prepare_data.py` and not redistributed here.
