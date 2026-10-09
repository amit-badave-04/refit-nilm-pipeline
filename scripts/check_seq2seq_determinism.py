"""Train NILMFormer twice with the same seed on the real corpus and compare the weights bit for bit.

Usage: python scripts/check_seq2seq_determinism.py [epochs]   (default 2; needs the REFIT cache and a GPU)
Writes artifacts/metrics/seq2seq_determinism_check.json.
"""
import json
import sys

import torch

from refit_nilm import config as C
from refit_nilm import datasets as D
from refit_nilm import train_seq2seq as S


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, corpora, _ = D.prepare([129])
    cfg = S.Seq2SeqConfig(window=129, seed=42, max_epochs=epochs, patience=epochs + 1)
    runs = [S.fit(corpora[129], cfg, device=device) for _ in range(2)]
    a, b = (r[0].state_dict() for r in runs)
    max_diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    report = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__, "epochs": epochs, "seed": 42,
        "identical_weights": max_diff == 0.0, "max_abs_weight_diff": max_diff,
        "val_nde_by_epoch": [[h["val_nde"] for h in r[1]] for r in runs],
    }
    (C.METRIC_DIR / "seq2seq_determinism_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["identical_weights"]:
        raise SystemExit("training is not bitwise repeatable")


if __name__ == "__main__":
    main()
