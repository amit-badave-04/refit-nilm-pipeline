# Washing-machine disaggregation API

**Live:** https://refit-wm-nilm.fly.dev · interactive docs: [https://refit-wm-nilm.fly.dev/docs](https://refit-wm-nilm.fly.dev/docs) · health: [https://refit-wm-nilm.fly.dev/health](https://refit-wm-nilm.fly.dev/health)

A small production-style service around the primary model (three-seed Seq2Point ensemble, 81-minute
window). It takes 1-minute whole-home active power and returns the estimated washing-machine power,
an agreement score, detected cycles and a summary.

* **Model:** `model/model.onnx` (12.9 MB) with standardisation and ensembling inside the graph, so
  the image needs only onnxruntime + numpy (no PyTorch). The dense layers, 99 % of the weights, are
  int8 (dynamic, per-channel). The convolutions stay fp32: on the server CPU (AVX2, no VNNI)
  onnxruntime's int8 convolutions made a day of data take 6.8–8.4 s, against 1.5–1.8 s this way.
  `model/model_card.json` holds training data, validation and test scores, and limits of use; its
  SHA-256 is checked at start-up.
* **Parity:** on all of House 8 the quantised model scores NDE 0.737 / cycle F1 0.566 vs 0.737 / 0.564
  for the PyTorch ensemble (`artifacts/metrics/onnx_parity.json`).

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/` | redirects to the interactive docs (`/docs`), which carry a prefilled real example |
| GET | `/health` | liveness and loaded model version |
| GET | `/model` | model card |
| POST | `/v1/disaggregate` | 1-minute aggregate (W) → washing-machine power (W), cycles, summary |

Request body:

```json
{"start": "2014-04-15T00:00:00Z", "interval_seconds": 60, "aggregate_w": [312.4, 309.9, null, 2410.0, "..."]}
```

Validation at the boundary (HTTP 422 with the reason and field, never echoing the submitted data): timezone-aware `start`; `interval_seconds`
must be 60; 81 to 2,880 readings (one window to 2 days; configurable with `MAX_POINTS`); values within 0–25,000 W; `null` allowed
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
service returns four cycles, two per wash: each wash is split where the predicted low-power tail
dips below 20 W, and the second piece of the first wash runs 14 minutes past its recorded end. The
predicted energy is 1.01 kWh. Both behaviours match the
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
flyctl auth login
flyctl apps create refit-wm-nilm     # from this folder; fly.toml already holds the configuration
flyctl deploy                        # remote build, no local Docker needed
curl https://refit-wm-nilm.fly.dev/health
```

Try the deployed service with the real example:

```bash
curl -s -X POST https://refit-wm-nilm.fly.dev/v1/disaggregate -H "content-type: application/json" --data-binary @examples/house8_2014-04-15.json
```

`fly.toml` runs dedicated-core machines (`performance-1x`, 2 GB) in London (`lhr`). Fly creates two
for availability, stops them when idle and starts one on the next request, and health-checks
`/health`. Each machine takes at most four requests at once; beyond two in flight the proxy prefers,
and if needed starts, the other machine. Machines are billed only while running. The image runs as a
non-root user.

## Operational notes and limits

* Measured latency, one-day request (1,440 windows, three-model ensemble), called from India:
  1.5–1.8 s end to end, of which ~0.95 s is server time. The six-hour example in `/docs` takes
  0.7–0.8 s (~0.23 s server time). The first request after the machines have auto-stopped adds ~5 s
  of start-up.
* A day of data used to take 7–11 s on a shared core. Moving to a dedicated core alone left it at
  5–10 s. Profiling on the machine traced the time to the int8 convolutions, and keeping them in fp32
  (see above) brought it to the figures above.
* Logs are one JSON line per request (path, status, latency, body size) and never contain the
  submitted data.
* The rate limiter is in-memory, i.e. per machine; a multi-machine deployment would move it to a
  shared store or the edge.
* Not for: individual billing, 15/30-minute data, homes with PV, or non-UK appliance stock without
  local validation (see `model/model_card.json` and `../Recommendation.md`).
