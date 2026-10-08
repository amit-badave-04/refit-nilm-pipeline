"""Replace the closing '## Summary' markdown of a notebook in both the .py source and the
executed .ipynb, without re-running the notebook (markdown cells have no outputs).

Usage: python scripts/set_summary.py <stem> <markdown-file>
The markdown file holds plain markdown (no leading '# ').
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
stem, md_path = sys.argv[1], Path(sys.argv[2])
md = md_path.read_text(encoding="utf-8").strip("\n")

src = next((ROOT / "notebooks").glob(f"{stem}_*.py"))
text = src.read_text(encoding="utf-8")
i = text.index("# ## Summary")
commented = "\n".join(("# " + line) if line else "#" for line in md.splitlines())
src.write_text(text[:i] + commented + "\n", encoding="utf-8", newline="\n")

nb_path = src.with_suffix(".ipynb")
nb = nbformat.read(nb_path, as_version=4)
for cell in reversed(nb.cells):
    if cell.cell_type == "markdown" and cell.source.lstrip().startswith("## Summary"):
        cell.source = md
        break
else:
    raise SystemExit("no Summary cell found")
nbformat.write(nb, nb_path)
print("updated", src.name, "and", nb_path.name)
