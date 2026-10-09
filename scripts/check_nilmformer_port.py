"""Check the NILMFormer port against the original code: identical weights must give identical outputs.

Usage: python scripts/check_nilmformer_port.py

Downloads the four model files of https://github.com/adrienpetralia/NILMFormer at the pinned commit
into a temporary folder, imports them (their unused `xformers` import is stubbed), copies the
original model's weights into src/refit_nilm/models/nilmformer.py's model in parameter order, and
compares the outputs on random input.
"""
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import torch

COMMIT = "e73a975a42fbebed6f9d7e90d75f7f48ae02fed9"
FILES = ["src/nilmformer/model.py", "src/nilmformer/congif.py", "src/nilmformer/layers/transformer.py", "src/nilmformer/layers/embedding.py"]
ROOT = Path(__file__).resolve().parents[1]


def fetch(dest: Path) -> None:
    for f in FILES:
        target = dest / f
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/adrienpetralia/NILMFormer/{COMMIT}/{f}"
        with urllib.request.urlopen(url, timeout=30) as r:
            target.write_bytes(r.read())


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
        orig, port = Original(NILMFormerConfig()).eval(), Port().eval()
        so, sp = orig.state_dict(), port.state_dict()
        if [tuple(v.shape) for v in so.values()] != [tuple(v.shape) for v in sp.values()]:
            raise SystemExit("parameter order or shapes differ")
        port.load_state_dict({k: v.clone() for k, v in zip(sp, so.values())})
        x = torch.randn(8, 9, 129)
        x[:, 0] = x[:, 0].abs() * 0.05
        with torch.no_grad():
            diff = float((orig(x).squeeze(1) - port(x)["power"]).abs().max())
    print(f"{len(so)} parameter tensors; max |original - port| = {diff:.2e}")
    if diff > 1e-5:
        raise SystemExit("outputs differ")
    print("port is equivalent")


if __name__ == "__main__":
    main()
