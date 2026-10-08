"""Export a trained checkpoint to a self-contained ONNX graph for the inference service.

The exported graph takes raw 1-minute aggregate windows in watts, shape (batch, W), and returns
(washing-machine power in watts, on-probability). Standardisation, the gate and the conversion
back to watts are inside the graph, so the service needs only onnxruntime and numpy.

The graph is quantised to int8 weights (dynamic quantisation) and checked for parity with the
PyTorch model on real windows before it is written.

Usage: python scripts/export_onnx.py --checkpoint artifacts/models/<run>.pt --name <name>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from refit_nilm import config as C
from refit_nilm import datasets as D
from refit_nilm import train as T

OUT = C.PROJECT_ROOT / "service" / "model"


class Deployable(torch.nn.Module):
    """One or more members (seed ensemble) behind a single watts-in / watts-out interface.

    Output 1: washing-machine power (W), the mean of the members' clipped outputs.
    Output 2: share of members whose output is >= 20 W (an agreement score in [0, 1]; for a
    single plain Seq2Point member it is 0 or 1, not a calibrated probability).
    """

    def __init__(self, models: list[torch.nn.Module], stats: dict):
        super().__init__()
        self.members = torch.nn.ModuleList(models)
        self.register_buffer("agg_mean", torch.tensor(stats["agg_mean"], dtype=torch.float32))
        self.register_buffer("agg_std", torch.tensor(stats["agg_std"], dtype=torch.float32))
        self.register_buffer("wm_std", torch.tensor(stats["wm_std"], dtype=torch.float32))

    def forward(self, aggregate_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (aggregate_w - self.agg_mean) / self.agg_std
        powers = []
        for m in self.members:
            out = m(x)
            p = out["p_on"] * out["power_raw"] if "p_on" in out else out["power"]
            powers.append((p * self.wm_std).clamp_min(0.0))
        stacked = torch.stack(powers, dim=0)
        agree = (stacked >= C.WM_ON_THRESHOLD_W).float().mean(dim=0)
        return stacked.mean(dim=0), agree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, nargs="+", help="one checkpoint, or several for a seed ensemble")
    ap.add_argument("--name", required=True)
    ap.add_argument("--card-extra", default=None, help="JSON file with evaluation results to embed")
    args = ap.parse_args()

    loaded = [T.load_run(Path(c), device="cpu") for c in args.checkpoint]
    cfg, stats = loaded[0][1], loaded[0][2]
    if any(l[2] != stats or l[1].window != cfg.window for l in loaded):
        raise SystemExit("ensemble members must share window and normalisation statistics")
    dep = Deployable([l[0] for l in loaded], stats).eval()
    W = cfg.window
    OUT.mkdir(parents=True, exist_ok=True)
    fp32 = OUT / "model_fp32.onnx"
    example = torch.randn(8, W) * 500 + 500
    torch.onnx.export(
        dep, (example,), str(fp32), input_names=["aggregate_w"], output_names=["wm_power_w", "p_on"],
        dynamic_shapes={"aggregate_w": {0: torch.export.Dim("batch")}}, dynamo=True,
    )
    int8 = OUT / "model.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    # parity on real validation windows
    _, corpora, _ = D.prepare([W], houses=[C.VAL_HOUSE, C.TEST_HOUSE])
    cp = corpora[W]
    rng = np.random.default_rng(0)
    starts = rng.choice(cp.starts["test"], size=4000, replace=False)
    x = np.stack([cp.agg[s : s + W] for s in starts]).astype(np.float32)
    with torch.no_grad():
        ref, _ = dep(torch.from_numpy(x))
    ref = ref.numpy()
    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    got, _ = sess.run(None, {"aggregate_w": x})
    parity = {
        "max_abs_diff_w": float(np.max(np.abs(got - ref))),
        "mean_abs_diff_w": float(np.mean(np.abs(got - ref))),
        "corr": float(np.corrcoef(got, ref)[0, 1]),
        "windows": int(len(x)),
    }
    print("int8 vs torch parity:", parity)
    if parity["corr"] < 0.99:
        raise SystemExit("quantised model deviates too much; keep fp32")
    fp32.unlink()

    card = {
        "name": args.name,
        "artifact": "model.onnx",
        "sha256": hashlib.sha256(int8.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": cfg.model,
        "window": W,
        "input": "1-minute whole-home active power (W), UTC-regular, gaps <= 20 %",
        "output": "washing-machine active power (W) per minute, and the share of ensemble members that predict the machine is on (>= 20 W)",
        "members": [Path(c).stem for c in args.checkpoint],
        "training_data": {
            "dataset": "REFIT Electrical Load Measurements (cleaned), University of Strathclyde, CC BY 4.0",
            "train_houses": C.TRAIN_HOUSES, "validation_house": C.VAL_HOUSE, "test_house": C.TEST_HOUSE,
        },
        "standardisation": stats,
        "training_config": {k: v for k, v in cfg.__dict__.items() if k != "seed"},
        "seeds": [l[1].seed for l in loaded],
        "quantisation": {"type": "dynamic int8 weights", "parity_vs_fp32": parity},
        "intended_use": "Portfolio-level estimates of washing-machine energy and usage timing in UK-like homes.",
        "not_for": [
            "billing or any individually binding decision",
            "data at 15/30-minute resolution (retrain at that resolution)",
            "homes with rooftop PV (aggregate distorted)",
            "markets with different appliance stock (e.g. India) without local validation",
        ],
    }
    if args.card_extra:
        card["evaluation"] = json.loads(Path(args.card_extra).read_text(encoding="utf-8"))
    (OUT / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print("wrote", int8.relative_to(C.PROJECT_ROOT), f"{int8.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
