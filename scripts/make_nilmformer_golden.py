"""Write tests/data/nilmformer_golden.json: outputs and per-tensor gradient norms of a seeded NILMFormer.

Usage: python scripts/make_nilmformer_golden.py

Run it only when the port is known to be right (scripts/check_nilmformer_port.py passes against the
original code). tests/test_nilmformer.py then pins the whole forward and backward pass to these
values offline, so CI catches a change that the network-dependent port check would not see.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from _golden import GOLDEN, compute  # noqa: E402

if __name__ == "__main__":
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(compute(), indent=1))
    print("wrote", GOLDEN.name)
