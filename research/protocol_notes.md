# Protocol notes (extracted from papers and reference code)

## Canonical REFIT washing-machine split (D'Incecco et al. 2020; MingjunZhong/transferNILM and seq2point-nilm code)

- Train houses 2, 5, 7, 9, 15, 16, 17; validation house 18; test house 8. Native 8-s data, no resampling.
- WM appliance column per house in that code: H2→2, H5→3, H7→5, H8→4, H9→3, H15→3, H16→5, H17→4, H18→5.
- Window 599 samples (≈ 80 min at 8 s); the target is the window midpoint.
- Standardisation: aggregate (x − 522)/814; WM (y − 400)/700.
- On-threshold 20 W; max_on_power 3999 W (defined in code, not stated in the paper).
- Seq2Point: Conv(30, k10) → Conv(30, k8) → Conv(40, k6) → Conv(50, k5) → Conv(50, k5), ReLU, same padding → Dense 1024 ReLU → Dense 1.
  The TF2 reimplementation reshapes so the kernels are effectively pointwise in time; a faithful port uses a true Conv1D.
- Adam lr 1e-3, batch 1000, ≤ 50 epochs, early stopping with patience 5.
- Reported (Table IV, test H8): seq2point WM MAE 16.85 W, SAE 2.610 (as printed; implausible next to AFHMM's 0.80), EpD 319.11 Wh, NDE 0.54. AFHMM MAE 48.09 W.

## Other REFIT house-8 WM references (8-s data, context only)

- Langevin et al. 2022: S2P MAE 21.9 W / MAE_ON 344.9 W / F1 61.3%; VAE-NILM 12.0 / 271.7 / 79.6%.
- LAB transformer (Sci. Rep. 2026): F1 0.786, MAE 11.37 W, SAE 9.24.

## 1-minute REFIT protocol (NILMFormer, KDD 2025)

- Mean resample to round minutes; forward-fill gaps ≤ 3 min; clip to [0, 10 000] W; drop windows with remaining NaN; non-overlapping windows; 2 test / 1 validation / rest train (houses not named).
- Washer MAE at window 256: NILMFormer 22.3, Energformer 30.7, STNILM 30.9, BiGRU 34.0, BERT4NILM 34.8, BiLSTM 40.0, FCN 42.0, UNet-NILM 42.4, TSILNet 42.8, DiffNILM 57.7.

## Activation (cycle) rule — Kelly & Knottenbelt 2015, Table 4

- Washing machine: on-threshold 20 W, min-on 1800 s, min-off 160 s, max power 2500 W.
- nilm_metadata defaults for "washing machine": min-on 600 s, min-off 300 s, no threshold (NILMTK then falls back to 10 W).
- Precioso Table 1 at 1-min resolution: λ = 20 W, μ0 (min off) = 3, μ1 (min on) = 30, with units consistent with minutes.

## Metrics

- MAE = mean |ŷ − y|. SAE = |Σŷ − Σy| / Σy. NDE = sqrt(Σ(ŷ − y)² / Σy²).
- MAE_ON = MAE over timesteps with y ≥ 20 W (Langevin). EpD = mean absolute daily energy error.
- G-loss (error metrics) = 100 · (ERR_unseen / ERR_seen − 1) % (Klemenjak 2019).
- NAR = Σ|y_agg − Σ submeters| / Σ y_agg.

## Resolution evidence

- Petralia 2023: WM presence detection at 30 min, macro-F1 0.52–0.61 across classifiers (REFIT best 0.614, InceptionTime); accuracy drops by about 0.1 on average from 1 min to 30 min.
- Zhao et al. 2020: hourly REFIT; CNN with a 7-h window; on/off edges blur.

## UK → India

- iAWE (Delhi, 2013): 230 V nominal, measured 180–260 V; outages up to 9 h/day.
- Resistive heater power ∝ V²: at 180 V ≈ 61% of rated 230 V power.
- Washing-machine ownership (CEEW IRES 2020, via BEE): ~29% urban, ~6% rural.
- D'Incecco 2020 cross-dataset WM MAE: REFIT→REDD 36.8 W without fine-tuning vs 18.0 W after retraining the dense layers of the generic CNN.

## UK washing-machine benchmarks (Household Electricity Survey 2012)

- ~284 cycles/yr (~5.5/week), ~166 kWh/yr → ~0.58 kWh/cycle; 50% of cycles < 0.5 kWh.
