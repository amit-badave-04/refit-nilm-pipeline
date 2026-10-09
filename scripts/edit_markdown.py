"""Edit markdown cells of an executed notebook without re-running it (markdown has no outputs).

    python scripts/edit_markdown.py summary <stem> <markdown-file>   replace the closing "## Summary" cell
    from edit_markdown import replace; replace("06", [(old, new), ...])   exact text replacement

Used for prose corrections after a long run; code cells are never touched here (changing code means
re-executing the notebook with scripts/run_notebooks.py).
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def _path(stem: str) -> Path:
    return next((ROOT / "notebooks").glob(f"{stem}_*.ipynb"))


def replace(stem: str, pairs: list[tuple[str, str]]) -> None:
    path = _path(stem)
    nb = nbformat.read(path, as_version=4)
    for old, new in pairs:
        hits = 0
        for cell in nb.cells:
            if cell.cell_type == "markdown" and old in cell.source:
                cell.source = cell.source.replace(old, new)
                hits += 1
        if hits == 0:
            raise SystemExit(f"{path.name}: text not found in markdown cells: {old[:60]!r}")
    nbformat.write(nb, path)
    print("updated", path.name)


def set_summary(stem: str, markdown: str) -> None:
    path = _path(stem)
    nb = nbformat.read(path, as_version=4)
    for cell in reversed(nb.cells):
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("## Summary"):
            cell.source = markdown.strip("\n")
            break
    else:
        raise SystemExit(f"{path.name}: no '## Summary' cell found")
    nbformat.write(nb, path)
    print("updated", path.name)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "summary":
        set_summary(sys.argv[2], Path(sys.argv[3]).read_text(encoding="utf-8"))
    else:
        raise SystemExit(__doc__)
