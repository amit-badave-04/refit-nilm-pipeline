"""Metric-level parity of the deployed ONNX model against the PyTorch ensemble on a full house.

Step 1 (torch process):  python scripts/check_onnx_parity.py torch  <ckpt> [<ckpt> ...]
Step 2 (onnx process):   python scripts/check_onnx_parity.py onnx
Two processes because the PyTorch and onnxruntime wheels ship different OpenMP runtimes on Windows.
Writes artifacts/metrics/onnx_parity.json.
"""
import json
import sys

import numpy as np
import pandas as pd

from refit_nilm import config as C
from refit_nilm import datasets as D

H = C.TEST_HOUSE
TMP = C.MODEL_DIR / "_parity_house.npz"


def corpus(window: int) -> D.Corpus:
    frame = D.house_frame(H, f"Appliance{C.REFERENCE_WM_CHANNEL[H]}")
    return D.build_corpus({H: frame}, {H: pd.Series("test", index=frame.index)}, window)


if sys.argv[1] == "torch":
    import torch

    from refit_nilm import train as T

    preds = []
    for ck in sys.argv[2:]:
        model, cfg, st = T.load_run(ck, "cuda" if torch.cuda.is_available() else "cpu")
        cp = corpus(cfg.window)
        preds.append(T.predict(model, T.WindowSource(cp, st, model.trunk.conv[1].weight.device.type), cp.starts["test"], st)["power"])
    np.savez(TMP, torch=np.mean(preds, axis=0), window=cfg.window)
    print("torch predictions saved")
else:
    import onnxruntime as ort

    from refit_nilm import metrics

    with np.load(TMP) as f:
        d = {k: f[k] for k in f.files}
    W = int(d["window"])
    cp = corpus(W)
    st = cp.starts["test"]
    sess = ort.InferenceSession(str(C.PROJECT_ROOT / "service" / "model" / "model.onnx"), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(st), 8192):
        x = np.stack([cp.agg[s : s + W] for s in st[i : i + 8192]]).astype(np.float32)
        out.append(sess.run(None, {"aggregate_w": x})[0])
    onnx_pred = np.concatenate(out)
    y = D.to_series(cp, st, cp.wm[st + W // 2], H)
    res = {}
    for name, p in (("pytorch_fp32", d["torch"]), ("onnx_int8", onnx_pred)):
        r = metrics.all_metrics(y, D.to_series(cp, st, p, H))
        res[name] = {k: round(float(r[k]), 4) for k in ["mae_w", "nde", "sae", "epd_wh", "f1", "cycle_f1", "cycle_recall"]}
    res["per_minute"] = {
        "mean_abs_diff_w": float(np.mean(np.abs(onnx_pred - d["torch"]))),
        "p99_abs_diff_w": float(np.percentile(np.abs(onnx_pred - d["torch"]), 99)),
        "max_abs_diff_w": float(np.max(np.abs(onnx_pred - d["torch"]))),
    }
    (C.METRIC_DIR / "onnx_parity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    TMP.unlink()
    print(json.dumps(res, indent=1))
