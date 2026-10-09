"""Contract tests for the inference API (run against the exported model in service/model)."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def _day(with_wash: bool = True, n: int = 1440) -> list[float]:
    rng = np.random.default_rng(0)
    base = 250 + 30 * rng.standard_normal(n)
    base[::40] += 120  # fridge-like cycling
    if with_wash:
        base[600:615] += 2000  # heating phase
        base[615:680] += 150 + 80 * rng.random(65)  # drum / tail
        base[680:690] += 400  # spin
    return np.clip(base, 0, None).round(1).tolist()


def _req(values, **kw):
    return {"start": "2014-11-11T00:00:00Z", "interval_seconds": 60, "aggregate_w": values, **kw}


def test_health_and_model_card(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    card = client.get("/model").json()
    assert card["window"] > 0 and len(card["sha256"]) == 64
    assert "not_for" in card


def test_disaggregate_shapes_and_summary(client):
    values = _day()
    r = client.post("/v1/disaggregate", json=_req(values))
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["wm_power_w"]) == len(values) == len(body["wm_on_agreement"])
    assert all(p >= 0 for p in body["wm_power_w"])
    assert all(0 <= q <= 1 for q in body["wm_on_agreement"])
    assert body["summary"]["wm_energy_kwh"] >= 0
    assert body["model_version"].count("@") == 1
    assert "x-request-id" in r.headers


def test_deterministic(client):
    a = client.post("/v1/disaggregate", json=_req(_day())).json()["wm_power_w"]
    b = client.post("/v1/disaggregate", json=_req(_day())).json()["wm_power_w"]
    assert a == b


def test_missing_values_are_filled_and_reported(client):
    values = _day()
    for i in range(100, 130):
        values[i] = None
    body = client.post("/v1/disaggregate", json=_req(values)).json()
    assert any("missing" in w for w in body["warnings"])


@pytest.mark.parametrize(
    "payload, fragment",
    [
        (_req([300.0] * 10), "at least"),
        (_req(_day(), interval_seconds=900), "1-minute"),
        (_req([-5.0] + _day()[1:]), "within"),
        (_req([None] * 400 + _day()[400:]), "missing"),
        ({"start": "2014-11-11T00:00:00", "interval_seconds": 60, "aggregate_w": _day()}, "timezone"),
        (_req(_day(), extra_field=1), "extra"),
    ],
)
def test_input_validation(client, payload, fragment):
    r = client.post("/v1/disaggregate", json=payload)
    assert r.status_code == 422
    assert fragment in r.text.lower()


def test_oversized_body_rejected(client):
    r = client.post("/v1/disaggregate", content=b"x" * 10, headers={"content-length": str(10_000_000), "content-type": "application/json"})
    assert r.status_code == 413


def test_model_file_carries_no_build_machine_metadata():
    """The exporter's per-node stack traces embed absolute paths; they must be stripped."""
    from pathlib import Path

    data = (Path(__file__).resolve().parents[1] / "model" / "model.onnx").read_bytes()
    for pattern in (b":\Users", b":/Users", b"/home/", b"site-packages", b"stack_trace"):
        assert pattern not in data, pattern
