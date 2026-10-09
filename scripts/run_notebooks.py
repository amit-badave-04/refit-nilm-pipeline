"""Execute the notebooks in order, in place (the .ipynb files are the source of truth).

Usage: python scripts/run_notebooks.py [01 02 05b ...]   (default: all, in order)

Execution metadata is stripped afterwards, and the run fails if any cell output contains a local
filesystem path (warnings can leak them).
"""
import sys
import time
from pathlib import Path

import nbformat
import papermill as pm

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def notebook(stem: str) -> Path:
    return next(NB_DIR.glob(f"{stem}_*.ipynb"))


def run(stem: str) -> None:
    path = notebook(stem)
    t = time.time()
    pm.execute_notebook(str(path), str(path), kernel_name="refit-nilm", cwd=str(NB_DIR), progress_bar=False)
    out = nbformat.read(path, as_version=4)
    out.metadata.pop("papermill", None)
    for cell in out.cells:
        for key in ("papermill", "execution", "tags"):
            cell.metadata.pop(key, None)
    nbformat.write(out, path)
    leaks = [c for c in out.cells for o in c.get("outputs", []) if "Users" + chr(92) in str(o) or "AppData" in str(o)]
    if leaks:
        raise RuntimeError(f"{path.name}: {len(leaks)} cell output(s) contain a local path; fix the warning source")
    print(f"{path.name}: executed in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    stems = sys.argv[1:] or sorted(p.name.split("_")[0] for p in NB_DIR.glob("[0-9][0-9]*_*.ipynb"))
    for s in stems:
        run(s)
