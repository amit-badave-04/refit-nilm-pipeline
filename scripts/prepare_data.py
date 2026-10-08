"""Fetch (if needed), verify and extract REFIT, then build the 1-minute Parquet cache.

Usage: python scripts/prepare_data.py [--rebuild]
"""
import argparse
import time

from refit_nilm import config as C
from refit_nilm import io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild Parquet cache")
    args = ap.parse_args()
    clean_7z = io.ensure_archive("clean")
    raw_7z = io.ensure_archive("raw")
    io.ensure_archive("clean_readme")
    if not all(io.clean_csv_path(h).exists() for h in C.HOUSES):
        io.extract(clean_7z, C.CLEAN_CSV_DIR)
    if not all((C.RAW_CSV_DIR / f).exists() for f in io.raw_house_files(1)):
        io.extract(raw_7z, C.RAW_CSV_DIR, io.raw_house_files(1))
    if not io.PART1_END_TABLE.exists():
        C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
        io.raw_part1_last_rows(raw_7z).to_csv(io.PART1_END_TABLE, index=False)
    for h in C.HOUSES:
        t = time.time()
        m = io.load_clean_minutes(h, rebuild=args.rebuild)
        print(f"house {h:2d}: {len(m):,} minutes  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
