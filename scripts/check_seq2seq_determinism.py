"""Retrain notebook 07's NILMFormer seed 42 in a fresh process and compare it bit for bit with the saved run.

Usage: python scripts/check_seq2seq_determinism.py      (needs the REFIT cache and a CUDA GPU)

Uses notebook 07's configuration (published budget, up to 50 epochs with early stopping) and compares
the retrained weights and the validation curve with artifacts/models/m5_nilmformer_l129_s42.pt and
its .json history, which the notebook wrote in another process. Writes
artifacts/metrics/seq2seq_determinism_check.json.
"""
import json

import torch

from refit_nilm import config as C
from refit_nilm import datasets as D
from refit_nilm import train_seq2seq as S

SAVED = C.MODEL_DIR / "m5_nilmformer_l129_s42.pt"


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("this check needs a CUDA GPU: the claim it tests is about GPU training")
    _, corpora, _ = D.prepare([129])
    cfg = S.Seq2SeqConfig(window=129, seed=42)
    model, history, _ = S.fit(corpora[129], cfg, device="cuda")
    saved = torch.load(SAVED, map_location="cuda", weights_only=False)
    saved_history = json.loads(SAVED.with_suffix(".json").read_text())["history"]
    if saved["config"] != cfg.__dict__:
        raise SystemExit(f"saved run used a different configuration: {saved['config']}")
    now = model.state_dict()
    identical = all(torch.equal(now[k], saved["state_dict"][k]) for k in now)
    curve_now, curve_saved = [h["val_nde"] for h in history], [h["val_nde"] for h in saved_history]
    report = {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "seed": 42,
        "epochs_run": len(history), "compared_with": "artifacts/models/m5_nilmformer_l129_s42.pt (written by notebook 07)",
        "identical_weights": identical, "identical_validation_curve": curve_now == curve_saved,
        "val_nde_by_epoch": curve_now,
    }
    (C.METRIC_DIR / "seq2seq_determinism_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not (identical and report["identical_validation_curve"]):
        raise SystemExit("retraining did not reproduce the saved run bit for bit")


if __name__ == "__main__":
    main()
