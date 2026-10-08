"""Data acquisition, README parsing and loading of REFIT files.

Design notes
------------
* Archives are verified by SHA-256 before use; a file that is already present and matches is
  never downloaded again (idempotent).
* The appliance-to-channel map is parsed from the official README rather than typed by hand,
  then cross-checked against the reference code's washing-machine columns.
* All ordering and resampling uses the ``Unix`` column (UTC). The ``Time`` column is a text
  rendering and is only used for diagnostics.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from . import config as C


# ----------------------------------------------------------------------------------------
# Acquisition
# ----------------------------------------------------------------------------------------
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ensure_archive(key: str, data_dir: Path = C.DATA_DIR) -> Path:
    """Return the local path of a release file, downloading it only if missing or corrupt."""
    spec = C.ARCHIVES[key]
    path = data_dir / spec["filename"]
    if path.exists() and sha256_of(path) == spec["sha256"]:
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    digest = sha256_of(tmp)
    if digest != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{spec['filename']}: checksum mismatch ({digest})")
    tmp.replace(path)
    return path


def find_7z() -> str:
    for name in ("7z", "7zz", "7za"):
        exe = shutil.which(name)
        if exe:
            return exe
    raise FileNotFoundError("7-Zip not found; install the conda-forge '7zip' package")


def extract(archive: Path, out_dir: Path, members: list[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [find_7z(), "x", "-y", "-bso0", "-bsp0", f"-o{out_dir}", str(archive)]
    cmd += members or []
    subprocess.run(cmd, check=True)


def raw_house_files(house: int) -> list[str]:
    """Raw archive member names; naming differs between houses 1-12 and 13-21."""
    stem = f"RAW_House{house}" if house <= 12 else f"RAW_House_{house}"
    return [f"{stem}_Part1.csv", f"{stem}_Part2.csv"]


# ----------------------------------------------------------------------------------------
# README parsing
# ----------------------------------------------------------------------------------------
_HOUSE_RE = re.compile(r"^House\s+(\d+)\s*$")
_ITEM_RE = re.compile(r"^(\d)\.(.*)$")


def parse_appliance_map(readme_path: Path) -> dict[int, dict[int, str]]:
    """Parse the 'APPLIANCE LIST' section into {house: {channel: description}}."""
    text = Path(readme_path).read_text(encoding="utf-8", errors="replace")
    section = text.split("APPLIANCE LIST", 1)[1]
    mapping: dict[int, dict[int, str]] = {}
    house = None
    for line in section.splitlines():
        line = line.strip()
        m = _HOUSE_RE.match(line)
        if m:
            house = int(m.group(1))
            mapping[house] = {}
            continue
        m = _ITEM_RE.match(line)
        if m and house is not None:
            ch = int(m.group(1))
            if ch > 0:
                mapping[house][ch] = m.group(2).strip().rstrip(",").strip()
    return mapping


def _is_washing_machine(desc: str) -> bool:
    return desc.lower().startswith("washing machine")


def _is_washer_dryer(desc: str) -> bool:
    return desc.lower().startswith("washer dryer")


def washing_machine_channels(appliance_map: dict[int, dict[int, str]]) -> dict[int, list[int]]:
    return {h: [c for c, d in chans.items() if _is_washing_machine(d)] for h, chans in appliance_map.items()}


def washer_dryer_channels(appliance_map: dict[int, dict[int, str]]) -> dict[int, list[int]]:
    return {h: [c for c, d in chans.items() if _is_washer_dryer(d)] for h, chans in appliance_map.items()}


def check_against_reference(wm: dict[int, list[int]]) -> list[str]:
    """Return a list of disagreements with the reference code's WM columns (empty = agree)."""
    issues = []
    for house, ch in C.REFERENCE_WM_CHANNEL.items():
        if ch not in wm.get(house, []):
            issues.append(f"House {house}: README WM channels {wm.get(house)} vs reference {ch}")
    return issues


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------
def read_refit_csv(path: Path) -> pd.DataFrame:
    """Read a REFIT CSV (raw or clean) with explicit dtypes; 'NaN' strings become nulls."""
    convert = pacsv.ConvertOptions(
        column_types={"Unix": pa.int64(), "Time": pa.string()},
        null_values=["NaN", "nan", ""],
        strings_can_be_null=True,
    )
    table = pacsv.read_csv(path, convert_options=convert)
    df = table.to_pandas()
    for col in df.columns:
        if col not in ("Time", "Unix"):
            df[col] = df[col].astype("float64")
    return df


def clean_csv_path(house: int) -> Path:
    return C.CLEAN_CSV_DIR / f"CLEAN_House{house}.csv"


# ----------------------------------------------------------------------------------------
# Timestamp regime of the official cleaned files
# ----------------------------------------------------------------------------------------
PART1_END_TABLE = C.TABLE_DIR / "raw_part1_end.csv"


def raw_part1_last_rows(archive: Path | None = None) -> pd.DataFrame:
    """Last row of every house's RAW Part1 file, streamed from the archive (no extraction).

    Used to locate, per house, where the official cleaned file stops holding true UTC.
    """
    archive = archive or (C.DATA_DIR / C.ARCHIVES["raw"]["filename"])
    rows = []
    for h in C.HOUSES:
        member = raw_house_files(h)[0]
        proc = subprocess.run([find_7z(), "e", "-so", str(archive), member], capture_output=True, check=True)
        last = proc.stdout.rstrip(b"\r\n").rsplit(b"\n", 1)[-1].decode().split(",")
        rows.append({"house": h, "last_time_local": last[0], "last_unix_local": int(last[1])})
    return pd.DataFrame(rows)


def clean_utc_switch(house: int) -> int:
    """Local-clock epoch at which the cleaned file switches from true UTC to wall-clock time."""
    tab = pd.read_csv(PART1_END_TABLE)
    return int(tab.loc[tab["house"] == house, "last_unix_local"].iloc[0])


def clean_unix_to_utc(unix: np.ndarray, house: int) -> np.ndarray:
    """True UTC epoch for an official cleaned file.

    Evidence (NB01): rows that came from RAW Part1 were converted to UTC by the publishers;
    rows that came only from RAW Part2 (after Part1 ends) still hold UK wall-clock time, e.g.
    no readings at all in the skipped 01:00-02:00 hour of 29 Mar 2015 and double density in
    the repeated hour of 26 Oct 2014. The repeated hour cannot be disambiguated in a sorted
    file, so it is returned as -1.
    """
    from .cleaning import local_epoch_to_utc  # local import avoids a circular import

    unix = np.asarray(unix, dtype="int64")
    switch = clean_utc_switch(house)
    out = unix.copy()
    after = unix > switch
    if after.any():
        out[after], _ = local_epoch_to_utc(unix[after])
    return out


def resample_clean_to_minutes(df: pd.DataFrame, house: int | None = None) -> pd.DataFrame:
    """Resample an official cleaned file to a regular 1-minute UTC grid.

    * timestamps are first mapped to true UTC (see ``clean_unix_to_utc``) when ``house`` is given;
    * minute = floor(UTC / 60); channel value = mean of the readings in that minute;
    * ``n_readings`` counts readings per minute (0 => no data was logged);
    * ``issues`` = max of the README 'Issues' flag within the minute;
    * minutes with no reading are left as NaN here. Gap filling is a separate, explicit step.
    """
    d = df.copy()
    if house is not None:
        d["Unix"] = clean_unix_to_utc(d["Unix"].to_numpy(), house)
        d = d[d["Unix"] >= 0]
    d = d.sort_values("Unix", kind="mergesort")
    minute = (d["Unix"].to_numpy() // 60) * 60
    cols = C.CHANNELS
    grp = d[cols].groupby(minute)
    out = grp.mean()
    out["n_readings"] = grp.size()
    if "Issues" in d.columns:
        out["issues"] = d["Issues"].groupby(minute).max()
    full = np.arange(out.index.min(), out.index.max() + 60, 60)
    out = out.reindex(full)
    out["n_readings"] = out["n_readings"].fillna(0).astype("int32")
    if "issues" in out:
        out["issues"] = out["issues"].fillna(0).astype("int8")
    out.index = pd.to_datetime(out.index, unit="s", utc=True)
    out.index.name = "utc"
    return out


def minute_parquet_path(house: int) -> Path:
    return C.PARQUET_DIR / f"clean_house{house}_1min.parquet"


def load_clean_minutes(house: int, rebuild: bool = False) -> pd.DataFrame:
    """1-minute version of an official cleaned house file (cached as Parquet)."""
    path = minute_parquet_path(house)
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    df = read_refit_csv(clean_csv_path(house))
    out = resample_clean_to_minutes(df, house=house)
    C.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out
