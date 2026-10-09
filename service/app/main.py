"""Washing-machine disaggregation API.

POST /v1/disaggregate   1-minute whole-home power -> washing-machine power, on-agreement, cycles
GET  /health            liveness + model loaded
GET  /model             model card (training data, validation/test scores, limits of use)

Operational safeguards: request-size cap, per-client rate limit, input validation at the
boundary, structured logs without payloads, model checksum verified at start-up.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import timedelta

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .inference import Engine, detect_cycles
from .schemas import Cycle, DisaggregationRequest, DisaggregationResponse, Summary

MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", 2_000_000))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", 30))

logger = logging.getLogger("wm-nilm")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class RateLimiter:
    """Sliding one-minute window per client address (in-memory; one instance per machine)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def create_app(engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or Engine()
        logger.info(json.dumps({"event": "startup", "model": app.state.engine.version}))
        yield

    app = FastAPI(
        title="REFIT washing-machine disaggregation",
        version="1.0.0",
        description="Estimates washing-machine power from 1-minute whole-home power (trained on UK REFIT homes).",
        lifespan=lifespan,
    )
    limiter = RateLimiter(RATE_LIMIT_PER_MIN)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """422 with the reason and location only: the submitted readings are never echoed back."""
        errors = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.middleware("http")
    async def guard_and_log(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        size = int(request.headers.get("content-length") or 0)
        if size > MAX_BODY_BYTES:
            return JSONResponse({"detail": f"request body larger than {MAX_BODY_BYTES} bytes"}, status_code=413)
        client = request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")
        if request.url.path.startswith("/v1/") and not limiter.allow(client):
            return JSONResponse({"detail": "rate limit exceeded, try again in a minute"}, status_code=429)
        response = await call_next(request)
        logger.info(json.dumps({
            "event": "request", "id": rid, "path": request.url.path, "status": response.status_code,
            "ms": round(1000 * (time.perf_counter() - t0), 1), "bytes": size,
        }))
        response.headers["x-request-id"] = rid
        return response

    def get_engine(request: Request) -> Engine:
        return request.app.state.engine

    @app.get("/health")
    def health(engine: Engine = Depends(get_engine)) -> dict:
        return {"status": "ok", "model": engine.version}

    @app.get("/model")
    def model_card(engine: Engine = Depends(get_engine)) -> dict:
        return {**engine.card, "version": engine.version}

    @app.post("/v1/disaggregate", response_model=DisaggregationResponse)
    def disaggregate(req: DisaggregationRequest, engine: Engine = Depends(get_engine)) -> DisaggregationResponse:
        n = len(req.aggregate_w)
        if n < engine.window:
            raise HTTPException(422, f"need at least {engine.window} readings (one model window); got {n}")
        pred = engine.predict(req.aggregate_w)
        power = np.round(pred.power_w.astype(float), 1)
        cyc = []
        for s, e in detect_cycles(power):
            seg = power[s : e + 1]
            cyc.append(Cycle(
                start=req.start + timedelta(minutes=int(s)), end=req.start + timedelta(minutes=int(e)),
                duration_min=int(e - s + 1), energy_kwh=round(float(seg.sum()) / 60_000, 3), peak_w=float(seg.max()),
            ))
        agg = np.array([v for v in req.aggregate_w if v is not None], dtype=float)
        wm_kwh = float(power.sum()) / 60_000
        agg_kwh = float(agg.mean() * n) / 60_000 if len(agg) else 0.0
        warnings = [
            f"the first and last {pred.edge_minutes} minutes are predicted with padded context and are less reliable",
        ]
        n_missing = sum(v is None for v in req.aggregate_w)
        if n_missing:
            warnings.append(f"{n_missing} missing readings were filled from neighbouring values")
        return DisaggregationResponse(
            model_version=engine.version, start=req.start, interval_seconds=60,
            wm_power_w=power.tolist(), wm_on_agreement=np.round(pred.p_on.astype(float), 3).tolist(),
            cycles=cyc,
            summary=Summary(
                wm_energy_kwh=round(wm_kwh, 3), aggregate_energy_kwh=round(agg_kwh, 3),
                wm_share_of_aggregate_pct=round(100 * wm_kwh / agg_kwh, 2) if agg_kwh else 0.0, cycles=len(cyc),
            ),
            warnings=warnings,
        )

    return app


app = create_app()
