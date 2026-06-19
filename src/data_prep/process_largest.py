"""Stream one LargeST yearly file -> hourly flow for a California region
(by Caltrans district set), bounded memory via row-chunked h5py reads.

Usage:  python src/process_largest.py <year> [region] [--delete-raw]
        region in {gba, gla, sd}  (default gba)
Writes: data/ca/<region>_hourly_<year>.npz  (vals: hours x sensors, index, ids)
"""
from __future__ import annotations
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CA = ROOT / "data" / "ca"
META = CA / "ca_meta.csv"
REGIONS = {"gba": [4], "gla": [7, 8, 12], "sd": [11]}   # Bay Area / Greater LA / San Diego
STEPS_PER_HOUR = 12               # 5-min data
CHUNK_HOURS = 2000                # 2000h * 12 * Nsensors * 8B chunk


def region_columns(h5_path: Path, districts):
    """Return (sorted column positions, ordered sensor-id list) for the region,
    matched by sensor ID against the file's own column order."""
    meta = pd.read_csv(META)
    region_ids = set(meta.loc[meta["District"].isin(districts), "ID"].astype(str))
    with h5py.File(h5_path, "r") as f:
        file_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["t/axis0"][:]]
    positions = [i for i, c in enumerate(file_ids) if c in region_ids]
    ids = [file_ids[i] for i in positions]
    return positions, ids


def process_year(h5_path: Path, positions, chunk_hours: int = CHUNK_HOURS):
    cols = np.asarray(positions)
    with h5py.File(h5_path, "r") as f:
        ts_ns = f["t/axis1"][:]
        n_rows = ts_ns.shape[0]
        assert n_rows % STEPS_PER_HOUR == 0, n_rows
        n_hours = n_rows // STEPS_PER_HOUR
        dset = f["t/block0_values"]            # (n_rows, 8600)
        out = np.empty((n_hours, cols.size), dtype=np.float32)
        rows_per_chunk = chunk_hours * STEPS_PER_HOUR
        for r0 in range(0, n_rows, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, n_rows)
            block = dset[r0:r1, :][:, cols]    # rows slice, then region cols
            h = (r1 - r0) // STEPS_PER_HOUR
            block = block.reshape(h, STEPS_PER_HOUR, cols.size)
            # nanmean: an hour is NaN only if all 12 five-min samples are missing
            with np.errstate(invalid="ignore"):
                hourly = np.nanmean(block, axis=1)
            out[r0 // STEPS_PER_HOUR: r1 // STEPS_PER_HOUR] = hourly
            print(f"    rows {r0:>7}-{r1:<7} ({100*r1/n_rows:5.1f}%)", flush=True)
    start = pd.Timestamp(ts_ns[0])
    index = pd.date_range(start=start, periods=n_hours, freq="1h")
    return out, index


def main():
    year = sys.argv[1]
    region = next((a for a in sys.argv[2:] if a in REGIONS), "gba")
    districts = REGIONS[region]
    delete_raw = "--delete-raw" in sys.argv
    h5_path = CA / f"ca_his_raw_{year}.h5"
    print(f"Processing {h5_path.name} ({region.upper()} / districts {districts}) ...")
    positions, ids = region_columns(h5_path, districts)
    print(f"  region sensors: {len(ids)}")
    vals, index = process_year(h5_path, positions)
    out = CA / f"{region}_hourly_{year}.npz"
    np.savez_compressed(out, vals=vals, index=index.values.astype("datetime64[ns]"),
                        ids=np.array(ids))
    print(f"  saved {out.name}: {vals.shape} ({out.stat().st_size/1e6:.1f} MB)")
    if delete_raw:
        h5_path.unlink(missing_ok=True)
        (CA / f"ca_his_raw_{year}.h5.zip").unlink(missing_ok=True)
        print(f"  deleted raw {year} files")


if __name__ == "__main__":
    main()
