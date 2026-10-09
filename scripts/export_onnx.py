"""Export a trained checkpoint to a self-contained ONNX graph for the inference service.

The exported graph takes raw 1-minute aggregate windows in watts, shape (batch, W), and returns
(washing-machine power in watts, on-probability). Standardisation, the gate and the conversion
back to watts are inside the graph, so the service needs only onnxruntime and numpy.

The dense layers (99 % of the weights) are quantised to int8 (dynamic quantisation); the
convolutions stay in fp32 because onnxruntime's int8 convolution kernels are several times slower
than fp32 on CPUs without VNNI instructions (measured 6.8-8.4 s vs 1.5-1.8 s per day of data on the
deployment CPU, an AMD EPYC core with AVX2). The graph is checked for parity with the PyTorch model on
real windows before it is written.

Usage: python scripts/export_onnx.py --checkpoint artifacts/models/<run>.pt --name <name>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from refit_nilm import config as C
from refit_nilm import datasets as D
from refit_nilm import train as T

OUT = C.PROJECT_ROOT / "service" / "model"
QUANTISED_OPS = ["MatMul", "Gemm"]  # dense layers only; convolutions stay fp32 (see module docstring)

LEAK_PATTERNS = [rb":\Users", rb":/Users", rb"/home/", rb"site-packages", rb"stack_trace"]


def strip_metadata(proto: "onnx.ModelProto") -> "onnx.ModelProto":
    """Remove exporter metadata (source stack traces, file paths, module names) from every node.

    The PyTorch exporter records, per node, the Python stack trace and module hierarchy it came
    from. They are useful for debugging but embed absolute paths of the machine that exported
    the model, so they must not ship.
    """
    graphs = [proto.graph] + [f for f in proto.functions]
    for g in graphs:
        nodes = g.node
        for n in nodes:
            del n.metadata_props[:]
            n.doc_string = ""
        if hasattr(g, "doc_string"):
            g.doc_string = ""
        if hasattr(g, "metadata_props"):
            del g.metadata_props[:]
    del proto.metadata_props[:]
    proto.doc_string = ""
    return proto


def assert_no_local_paths(path: Path) -> None:
    data = path.read_bytes()
    bad = [pat.decode() for pat in LEAK_PATTERNS if pat in data]
    if bad:
        raise SystemExit(f"{path.name}: exported model still contains {bad}; refusing to write it")

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
    # single-file model without the exporter's intermediate shape annotations (they conflict with
    # onnx shape inference during quantisation)
    proto = onnx.load(str(fp32))
    del proto.graph.value_info[:]
    strip_metadata(proto)
    for extra in OUT.glob("model_fp32.onnx*"):
        extra.unlink()
    onnx.save(proto, str(fp32), save_as_external_data=False)
    int8 = OUT / "model.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=QUANTISED_OPS)
    q = strip_metadata(onnx.load(str(int8)))
    onnx.save(q, str(int8), save_as_external_data=False)
    assert_no_local_paths(int8)

    # parity on real windows from the test house. onnxruntime runs in a separate process: on
    # Windows the PyTorch and onnxruntime wheels ship different OpenMP runtimes.
    frame = D.house_frame(C.TEST_HOUSE, f"Appliance{C.REFERENCE_WM_CHANNEL[C.TEST_HOUSE]}")
    cp = D.build_corpus({C.TEST_HOUSE: frame}, {C.TEST_HOUSE: pd.Series("test", index=frame.index)}, W)
    rng = np.random.default_rng(0)
    starts = rng.choice(cp.starts["test"], size=4000, replace=False)
    x = np.stack([cp.agg[s : s + W] for s in starts]).astype(np.float32)
    with torch.no_grad():
        ref, _ = dep(torch.from_numpy(x))
    tmp = OUT / "_parity.npz"
    np.savez(tmp, x=x, ref=ref.numpy())
    code = (
        "import json, sys, numpy as np, onnxruntime as ort; d = np.load(sys.argv[1]); "
        "s = ort.InferenceSession(sys.argv[2], providers=['CPUExecutionProvider']); "
        "got = s.run(None, {'aggregate_w': d['x']})[0]; ref = d['ref']; "
        "print(json.dumps({'max_abs_diff_w': float(np.max(np.abs(got - ref))), "
        "'mean_abs_diff_w': float(np.mean(np.abs(got - ref))), 'corr': float(np.corrcoef(got, ref)[0, 1]), "
        "'windows': int(len(ref))}))"
    )
    res = subprocess.run([sys.executable, "-c", code, str(tmp), str(int8)], capture_output=True, text=True, check=True)
    parity = json.loads(res.stdout.strip().splitlines()[-1])
    tmp.unlink()
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
        "input": "1-minute whole-home active power (W), UTC-regular, missing readings <= 10 %",
        "output": "washing-machine active power (W) per minute, and the share of ensemble members that predict the machine is on (>= 20 W)",
        "members": [Path(c).stem for c in args.checkpoint],
        "training_data": {
            "dataset": "REFIT Electrical Load Measurements (cleaned), University of Strathclyde, CC BY 4.0",
            "train_houses": C.TRAIN_HOUSES, "validation_house": C.VAL_HOUSE, "test_house": C.TEST_HOUSE,
        },
        "standardisation": stats,
        "training_config": {k: v for k, v in cfg.__dict__.items() if k != "seed"},
        "seeds": [l[1].seed for l in loaded],
        "quantisation": {"type": "dynamic int8 weights (per-channel scales) for the dense layers; convolutions in fp32", "parity_set": "4,000 random windows from House 8", "parity_vs_fp32": parity, "full_house_parity": "artifacts/metrics/onnx_parity.json"},
        "intended_use": "Portfolio-level indications of washing-machine use in UK-like homes: wash frequency and timing, ranking homes by hot-wash frequency, and the direction of month-to-month change. Energy levels run about 20 % low and peak-hour shares high (RESULTS.md, notebook 08), so calibrate against a plug-metered panel before quoting them.",
        "not_for": [
            "billing or any individually binding decision",
            "15/30-minute data as input: models trained on such data had no per-interval skill (RESULTS.md section 6); 1-minute input is required",
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
