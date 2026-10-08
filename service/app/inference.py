"""ONNX inference engine. Torch-free: the exported graph already contains input
standardisation, the model, the on/off gate and the conversion back to watts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"

# Washing-machine activation rule (Kelly & Knottenbelt 2015), identical to the research code.
ON_THRESHOLD_W = 20.0
MIN_OFF_MIN = 3
MIN_ON_MIN = 30


@dataclass(frozen=True)
class Prediction:
    power_w: np.ndarray
    p_on: np.ndarray
    edge_minutes: int


class Engine:
    def __init__(self, model_dir: Path = MODEL_DIR, batch_size: int = 4096):
        self.card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
        onnx_path = model_dir / self.card["artifact"]
        self.sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        if self.sha256 != self.card["sha256"]:
            raise RuntimeError("model file does not match the checksum in model_card.json")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.window = int(self.card["window"])
        self.batch_size = batch_size
        self.version = f"{self.card['name']}@{self.sha256[:12]}"

    def predict(self, aggregate_w: list[float | None]) -> Prediction:
        x = np.array([np.nan if v is None else v for v in aggregate_w], dtype=np.float32)
        x = _fill(x)
        half = self.window // 2
        padded = np.pad(x, (half, half), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, self.window)
        powers, probs = [], []
        for i in range(0, len(windows), self.batch_size):
            p, q = self.session.run(None, {"aggregate_w": np.ascontiguousarray(windows[i : i + self.batch_size])})
            powers.append(p)
            probs.append(q)
        return Prediction(np.concatenate(powers), np.concatenate(probs), edge_minutes=min(half, len(x)))


def _fill(x: np.ndarray) -> np.ndarray:
    """Forward-fill, then back-fill, missing readings (input only)."""
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    y = x[idx]
    if np.isnan(y[0]):
        first = np.flatnonzero(~np.isnan(y))[0]
        y[:first] = y[first]
    return y


def detect_cycles(power: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (inclusive) indices of washing cycles on a 1-minute power series."""
    on = power >= ON_THRESHOLD_W
    edges = np.diff(np.r_[0, on.astype(np.int8), 0])
    starts, ends = list(np.flatnonzero(edges == 1)), list(np.flatnonzero(edges == -1))
    if not starts:
        return []
    ms, me = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - me[-1] < MIN_OFF_MIN:
            me[-1] = e
        else:
            ms.append(s)
            me.append(e)
    return [(s, e - 1) for s, e in zip(ms, me) if e - s >= MIN_ON_MIN]
