# Washing-machine disaggregation API

A small production-style service around the primary model (three-seed Seq2Point ensemble, 81-minute
window). It takes 1-minute whole-home active power and returns the estimated washing-machine power,
an agreement score, detected cycles and a summary.

* **Model:** `model/model.onnx` (int8 weights, 12.7 MB) with standardisation and ensembling inside
  the graph, so the image needs only onnxruntime + numpy (no PyTorch). `model/model_card.json` holds
  training data, validation and test scores, and limits of use; its SHA-256 is checked at start-up.
* **Parity:** on all of House 8 the int8 model scores NDE 0.740 / cycle F1 0.569 vs 0.737 / 0.564
  for the PyTorch ensemble (`artifacts/metrics/onnx_parity.json`).

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness and loaded model version |
| GET | `/model` | model card |
| POST | `/v1/disaggregate` | 1-minute aggregate (W) → washing-machine power (W), cycles, summary |

Request body:

```json
{"start": "2014-04-15T00:00:00Z", "interval_seconds": 60, "aggregate_w": [312.4, 309.9, null, 2410.0, "..."]}
```

Validation at the boundary (HTTP 422 with a reason): timezone-aware `start`; `interval_seconds`
must be 60; 81 to 20,160 readings (one window to 14 days); values within 0–25,000 W; `null` allowed
for missing readings up to 10 % (the same tolerance the model was trained and evaluated with);
unknown fields rejected. Bodies over 2 MB are refused (413) and
each client is limited to 30 requests a minute (429).

Response fields: `wm_power_w` (one value per input minute), `wm_on_agreement` (share of the three
ensemble members predicting ≥ 20 W; an agreement score, not a calibrated probability), `cycles`
(start, end, duration, energy, peak; same 20 W / 3-minute / 30-minute rule as the analysis),
`summary` (washing and whole-home energy over the request), `warnings` (edge effects, filled gaps).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8080
curl -s -X POST localhost:8080/v1/disaggregate -H "content-type: application/json" --data-binary @examples/house8_2014-04-15.json
```

`examples/house8_2014-04-15.json` is one real day from REFIT House 8 (an unseen test home),
inspected only after the model was frozen; the request, response and plug-monitor truth are saved in
`examples/house8_2014-04-15.result.json`. On that
day the plug monitor recorded two washes (03:39–04:50 and 05:19–07:39 UTC, 3.05 kWh in total). The
service returns four cycles, all inside those two washes: each wash is split where the predicted
low-power tail dips below 20 W, and the predicted energy is 1.04 kWh. Both behaviours match the
evaluation (cycles found reliably, tail energy under-estimated). Bridging longer gaps in predicted
cycles was tested on the validation home only and made cycle detection worse
(`artifacts/tables/service_cycle_postprocessing_val.csv`), so the analysis rule is kept.

## Tests

```bash
python -m pytest            # contract tests: shapes, determinism, validation errors, size limit
```

CI (`.github/workflows/ci.yml`) runs these tests, builds the Docker image and smoke-tests the running
container on every push.

## Deploy to Fly.io

```bash
fly auth login
fly launch --no-deploy --copy-config --name <app-name>   # from this folder; keeps fly.toml
fly deploy
curl https://<app-name>.fly.dev/health
```

`fly.toml` runs one shared-CPU machine with 512 MB in London (`lhr`), stops it when idle and starts
it on the next request, and health-checks `/health`. The image runs as a non-root user.

## Operational notes and limits

* Inference cost is small: a 14-day request is ~20,000 windows; the int8 model handles it in well
  under a second on one shared CPU.
* Logs are one JSON line per request (path, status, latency, body size) and never contain the
  submitted data.
* The rate limiter is in-memory, i.e. per machine; a multi-machine deployment would move it to a
  shared store or the edge.
* Not for: individual billing, 15/30-minute data, homes with PV, or non-UK appliance stock without
  local validation (see `model/model_card.json` and `../Recommendation.md`).
