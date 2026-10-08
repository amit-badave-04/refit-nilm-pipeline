"""Replace text inside markdown cells of a notebook, in both the .py source and the executed
.ipynb, without re-running it (markdown has no outputs).

Usage (from Python): edit_markdown.replace("06", [(old, new), ...])
"""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def replace(stem: str, pairs: list[tuple[str, str]]) -> None:
    src = next((ROOT / "notebooks").glob(f"{stem}_*.py"))
    nb_path = src.with_suffix(".ipynb")
    text = src.read_text(encoding="utf-8")
    nb = nbformat.read(nb_path, as_version=4)
    for old, new in pairs:
        old_c = "\n".join(("# " + l) if l else "#" for l in old.splitlines())
        new_c = "\n".join(("# " + l) if l else "#" for l in new.splitlines())
        if old_c not in text:
            raise SystemExit(f"{src.name}: text not found: {old[:60]!r}")
        text = text.replace(old_c, new_c)
        hits = 0
        for cell in nb.cells:
            if cell.cell_type == "markdown" and old in cell.source:
                cell.source = cell.source.replace(old, new)
                hits += 1
        if hits == 0:
            raise SystemExit(f"{nb_path.name}: text not found in markdown cells: {old[:60]!r}")
    src.write_text(text, encoding="utf-8", newline="\n")
    nbformat.write(nb, nb_path)
    print("updated", src.name, "and", nb_path.name)
