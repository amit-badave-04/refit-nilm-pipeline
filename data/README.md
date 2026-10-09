# data/

The REFIT Electrical Load Measurements dataset (University of Strathclyde; Murray, Stankovic &
Stankovic 2017, *Scientific Data* 4:160122, doi:10.1038/sdata.2016.122), licence **CC BY 4.0**.

Only this README, `manifest.json` (file names, sizes, SHA-256, sources) and the dataset's own
`CLEAN_READ_ME_081116.txt` are versioned. The archives (1.2 GB) and everything derived from them
(~7 GB of CSVs, a Parquet cache) are too large for git and are rebuilt reproducibly:

```bash
python scripts/prepare_data.py
```

That command downloads whatever is missing, verifies every file against the SHA-256 in
`manifest.json`, extracts the cleaned houses and raw House 1, and builds the 1-minute cache.

## Sources

| file | size | where it comes from |
|---|---|---|
| `CLEAN_REFIT_081116.7z` | 490 MB | cleaned release, doi:10.15129/9ab14b0e-19ac-4279-938f-27f643078cec |
| `REFIT_RAW_081116.7z` | 666 MB | raw release, doi:10.15129/31da3ece-f902-4e95-a093-e0a9536983c4 |
| `CLEAN_READ_ME_081116.txt` | 12 KB | README of the cleaned release (appliance-to-channel map) |

`scripts/prepare_data.py` first tries the University of Strathclyde portal
(`pure.strath.ac.uk/ws/portalfiles/portal/...`; the `pureportal.strath.ac.uk/files/...` links shown on
the dataset page sit behind an interactive browser check). If that fails, it uses an unmodified
copy attached to this repository's release
[`refit-data-081116`](https://github.com/amit-badave-04/refit-nilm-pipeline/releases/tag/refit-data-081116),
redistributed under CC BY 4.0 with the attribution above. Both routes must match the same
checksums, so the files are byte-identical whichever source is used.

## Layout after `prepare_data.py`

```
data/
├── README.md, manifest.json, CLEAN_READ_ME_081116.txt   versioned
├── CLEAN_REFIT_081116.7z, REFIT_RAW_081116.7z            downloaded, verified
├── extracted/clean/CLEAN_House{N}.csv                    20 houses, ~6.9 GB
├── extracted/raw/RAW_House1_Part{1,2}.csv                raw House 1 only
└── parquet/clean_house{N}_1min.parquet                   1-minute cache, true UTC (see DATA.md)
```

Timestamp handling, cleaning decisions and known problems in both releases are documented in
[`../DATA.md`](../DATA.md).
