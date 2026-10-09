from pathlib import Path

import pandas as pd
import pytest

from refit_nilm import config as C
from refit_nilm import io

README_SNIPPET = """
APPLIANCE LIST
The following list shows the appliances ...

House 1
0.Aggregate
1.Fridge, Hotpoint, RLA50P
4.Washer Dryer, Creda, T522VW
5.Washing Machine, Beko, WMC6140

House 4
0.Aggregate,
4.Washing Machine(1), Servis, 6065
5.Washing Machine(2), Zanussi, Z917

House 12
0.Aggregate,
2.???, Unknown, Unknown
"""


@pytest.fixture()
def readme(tmp_path: Path) -> Path:
    p = tmp_path / "readme.txt"
    p.write_text(README_SNIPPET, encoding="utf-8")
    return p


def test_parse_appliance_map(readme):
    m = io.parse_appliance_map(readme)
    assert set(m) == {1, 4, 12}
    assert m[1][5].startswith("Washing Machine")


def test_washing_machine_and_washer_dryer_channels(readme):
    m = io.parse_appliance_map(readme)
    assert io.washing_machine_channels(m) == {1: [5], 4: [4, 5], 12: []}
    assert io.washer_dryer_channels(m)[1] == [4]


def test_raw_member_names_follow_archive_convention():
    assert io.raw_house_files(1) == ["RAW_House1_Part1.csv", "RAW_House1_Part2.csv"]
    assert io.raw_house_files(13)[0] == "RAW_House_13_Part1.csv"


def test_resample_clean_to_minutes_counts_and_flags():
    df = pd.DataFrame({"Unix": [0, 10, 70, 200], "Issues": [0, 1, 0, 0]})
    for col in C.CHANNELS:
        df[col] = [100.0, 200.0, 300.0, 400.0]
    m = io.resample_clean_to_minutes(df)
    assert m["Aggregate"].iloc[0] == 150 and m["n_readings"].iloc[0] == 2
    assert m["issues"].iloc[0] == 1
    assert m["n_readings"].iloc[2] == 0 and pd.isna(m["Aggregate"].iloc[2])


def test_real_readme_matches_reference_code():
    path = C.DATA_DIR / "CLEAN_READ_ME_081116.txt"
    if not path.exists():
        pytest.skip("README not downloaded")
    wm = io.washing_machine_channels(io.parse_appliance_map(path))
    assert io.check_against_reference(wm) == []
    assert wm[12] == []


def test_ensure_archive_falls_back_to_mirror_and_checks_sha(tmp_path, monkeypatch):
    import hashlib

    payload = b"refit-archive-bytes"
    spec = {"filename": "x.7z", "url": "https://portal.example/x.7z", "mirror": "https://mirror.example/x.7z",
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    monkeypatch.setitem(C.ARCHIVES, "test", spec)
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        if "portal" in url:
            raise OSError("HTTP 403 (interactive browser check)")
        dest.write_bytes(payload)

    monkeypatch.setattr(io, "_download", fake_download)
    path = io.ensure_archive("test", tmp_path)
    assert path.read_bytes() == payload and calls == [spec["url"], spec["mirror"]]


def test_ensure_archive_rejects_wrong_checksum(tmp_path, monkeypatch):
    spec = {"filename": "y.7z", "url": "https://portal.example/y.7z", "mirror": None, "sha256": "0" * 64, "bytes": 1}
    monkeypatch.setitem(C.ARCHIVES, "test", spec)
    monkeypatch.setattr(io, "_download", lambda url, dest: dest.write_bytes(b"tampered"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        io.ensure_archive("test", tmp_path)
    assert not (tmp_path / "y.7z").exists()
