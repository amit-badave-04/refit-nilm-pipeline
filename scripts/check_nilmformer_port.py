"""Check the NILMFormer port against the original code: identical weights must give identical outputs.

Usage: python scripts/check_nilmformer_port.py

Downloads the four model files of https://github.com/adrienpetralia/NILMFormer at the pinned commit
into a temporary folder, imports them (their unused `xformers` import is stubbed), copies the
original model's weights into src/refit_nilm/models/nilmformer.py's model in parameter order, and
compares, on random input:
* evaluation-mode outputs, at the input scale the model sees in training (load / 10 kW, about
  0.05) and at scale 1;
* training-mode outputs (dropout active, same random seed for both models);
* the gradients of the summed training-mode output with respect to every parameter tensor.
Writes artifacts/metrics/nilmformer_port_check.json and fails if any output difference, or any
gradient difference relative to that parameter tensor's largest gradient, exceeds 1e-5 or is not finite.
"""
import json
import math
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import torch

COMMIT = "e73a975a42fbebed6f9d7e90d75f7f48ae02fed9"
FILES = ["src/nilmformer/model.py", "src/nilmformer/congif.py", "src/nilmformer/layers/transformer.py", "src/nilmformer/layers/embedding.py"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "metrics" / "nilmformer_port_check.json"
TOLERANCE = 1e-5


def fetch(dest: Path) -> None:
    for f in FILES:
        target = dest / f
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/adrienpetralia/NILMFormer/{COMMIT}/{f}"
        with urllib.request.urlopen(url, timeout=30) as r:
            target.write_bytes(r.read())


def _input(scale: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 9, 129, generator=g)
    x[:, 0] = x[:, 0].abs() * scale
    return x


def compare(orig: torch.nn.Module, port: torch.nn.Module) -> dict:
    result = {}
    orig.eval(), port.eval()
    with torch.no_grad():
        for scale in (0.05, 1.0):
            x = _input(scale)
            result[f"eval_max_abs_diff_input_scale_{scale}"] = float((orig(x).squeeze(1) - port(x)["power"]).abs().max())
        eval_out = port(_input(0.05))["power"]
    orig.train(), port.train()
    x = _input(0.05)
    torch.manual_seed(7)
    a = orig(x).squeeze(1)
    torch.manual_seed(7)
    b = port(x)["power"]
    result["train_mode_max_abs_diff"] = float((a - b).detach().abs().max())
    # dropout must actually be active, or the training-mode comparison proves nothing
    result["train_mode_output_differs_from_eval"] = bool((b.detach() - eval_out).abs().max() > 1e-4)
    a.sum().backward()
    b.sum().backward()
    pairs = list(zip(orig.parameters(), port.parameters()))
    # torch.stack(...).max() propagates NaN, unlike Python's max over a generator
    result["gradient_max_abs_diff"] = float(torch.stack([(p.grad - q.grad).abs().max() for p, q in pairs]).max())
    # relative to each tensor's own largest gradient, so small tensors are checked on their own scale
    result["gradient_max_rel_diff_per_tensor"] = float(torch.stack(
        [(p.grad - q.grad).abs().max() / p.grad.abs().max().clamp_min(1e-12) for p, q in pairs]).max())
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fetch(Path(tmp))
        sys.modules["xformers"] = types.ModuleType("xformers")
        sys.modules["xformers.ops"] = types.ModuleType("xformers.ops")
        sys.path.insert(0, tmp)
        sys.path.insert(0, str(ROOT / "src"))
        from src.nilmformer.congif import NILMFormerConfig
        from src.nilmformer.model import NILMFormer as Original

        from refit_nilm.models.nilmformer import NILMFormer as Port

        torch.manual_seed(0)
        orig, port = Original(NILMFormerConfig()), Port()
        so, sp = orig.state_dict(), port.state_dict()
        if [tuple(v.shape) for v in so.values()] != [tuple(v.shape) for v in sp.values()]:
            raise SystemExit("parameter order or shapes differ")
        port.load_state_dict({k: v.clone() for k, v in zip(sp, so.values())})
        diffs = compare(orig, port)
    checked = [v for k, v in diffs.items() if k.startswith(("eval_", "train_mode_max")) or k == "gradient_max_rel_diff_per_tensor"]
    passed = all(math.isfinite(v) and v <= TOLERANCE for v in checked) and diffs["train_mode_output_differs_from_eval"]
    report = {"original_commit": COMMIT, "parameter_tensors": len(so), **diffs, "tolerance": TOLERANCE, "passed": passed}
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("outputs differ")


if __name__ == "__main__":
    main()
