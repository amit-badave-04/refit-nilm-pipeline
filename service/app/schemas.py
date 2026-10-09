"""Request / response contracts for the disaggregation API (validated at the boundary)."""
from __future__ import annotations

import os
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Requests are capped so a call finishes well within the platform's proxy timeout. Default: 2 days
# of 1-minute data (~23 s on a shared CPU, ~2 s on a dedicated one); raise it with MAX_POINTS.
MAX_POINTS = int(os.getenv("MAX_POINTS", 2_880))
MAX_AGGREGATE_W = 25_000.0   # above a 100 A x 230 V domestic supply: certainly a meter error
MAX_MISSING_FRACTION = 0.10  # same tolerance as training/evaluation windows


class DisaggregationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime = Field(description="Timestamp of the first reading (ISO 8601, timezone required).")
    interval_seconds: int = Field(60, description="Spacing of the readings. Only 60 is supported.")
    aggregate_w: list[float | None] = Field(
        description="Whole-home active power in watts, one value per interval; null = missing.",
        min_length=1,
        max_length=MAX_POINTS,
    )

    @field_validator("start")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("start must include a timezone offset, e.g. 2014-11-11T00:00:00Z")
        return v

    @field_validator("interval_seconds")
    @classmethod
    def _one_minute(cls, v: int) -> int:
        if v != 60:
            raise ValueError("only 1-minute data (interval_seconds = 60) is supported by this model")
        return v

    @field_validator("aggregate_w")
    @classmethod
    def _values(cls, v: list[float | None]) -> list[float | None]:
        bad = [x for x in v if x is not None and not (0.0 <= x <= MAX_AGGREGATE_W)]
        if bad:
            raise ValueError(f"aggregate_w values must be within [0, {MAX_AGGREGATE_W:.0f}] W; got e.g. {bad[0]}")
        return v

    @model_validator(mode="after")
    def _missing(self) -> "DisaggregationRequest":
        n_missing = sum(x is None for x in self.aggregate_w)
        if n_missing / len(self.aggregate_w) > MAX_MISSING_FRACTION:
            raise ValueError(f"more than {MAX_MISSING_FRACTION:.0%} of aggregate_w is missing")
        return self


class Cycle(BaseModel):
    start: datetime
    end: datetime
    duration_min: int
    energy_kwh: float
    peak_w: float


class Summary(BaseModel):
    wm_energy_kwh: float
    aggregate_energy_kwh: float
    wm_share_of_aggregate_pct: float
    cycles: int


class DisaggregationResponse(BaseModel):
    model_version: str
    start: datetime
    interval_seconds: int
    wm_power_w: list[float]
    wm_on_agreement: list[float] = Field(description="Share of ensemble members predicting the machine is on (>= 20 W); an agreement score, not a calibrated probability.")
    cycles: list[Cycle]
    summary: Summary
    warnings: list[str]
