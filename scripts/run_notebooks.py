"""Convert the percent-format notebook sources to .ipynb and execute them in order.

Usage: python scripts/run_notebooks.py [01 02 ...]   (default: all)
"""
import sys
import time
from pathlib import Path

import jupytext
import nbformat
import papermill as pm

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def run(stem: str) -> None:
    src = next(NB_DIR.glob(f"{stem}_*.py"))
    ipynb = src.with_suffix(".ipynb")
    nb = jupytext.read(src)  # the .py file is the single source of truth
    nb.metadata.pop("jupytext", None)  # no pairing: the .ipynb is a build output
    jupytext.write(nb, ipynb)
    t = time.time()
    pm.execute_notebook(str(ipynb), str(ipynb), kernel_name="refit-nilm", cwd=str(NB_DIR), progress_bar=False)
    out = nbformat.read(ipynb, as_version=4)
    out.metadata.pop("papermill", None)
    for cell in out.cells:
        cell.metadata.pop("papermill", None)
        cell.metadata.pop("execution", None)
        cell.metadata.pop("tags", None)
    nbformat.write(out, ipynb)
    leaks = [c for c in out.cells for o in c.get("outputs", []) if "Users" + chr(92) in str(o) or "AppData" in str(o)]
    if leaks:
        raise RuntimeError(f"{ipynb.name}: {len(leaks)} cell output(s) contain a local path; fix the warning source")
    print(f"{ipynb.name}: executed in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    stems = sys.argv[1:] or sorted(p.name.split("_")[0] for p in NB_DIR.glob("[0-9][0-9]*_*.py"))
    for s in stems:
        run(s)
