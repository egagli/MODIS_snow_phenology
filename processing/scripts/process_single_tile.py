"""
Process a single MODIS tile and write results to the Icechunk store.

Usage:
    python process_single_tile.py --h 10 --v 4

On success: commits data to Icechunk with message "h10v04: processed"
On failure: exits nonzero; no Icechunk commit (store remains clean)
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
import icechunk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modis_snow_phenology import masking
from modis_snow_phenology.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
for noisy in ("azure", "urllib3", "fsspec", "adlfs", "aiohttp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger("process_single_tile")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h", type=int, required=True, dest="h", help="MODIS horizontal tile index (0-35)")
    p.add_argument("--v", type=int, required=True, dest="v", help="MODIS vertical tile index (0-17)")
    return p.parse_args()


def fetch_and_binarize(h: int, v: int, config: Config) -> xr.DataArray:
    """Fetch MOD10A2 for a tile across all water years and cloud-fill."""
    # Fetch from 2013 so we have Oct-Dec of WY_start - 1 for water year alignment
    start_date = f"{config.wy_start - 2}-10-01"
    end_date = f"{config.wy_end}-09-30"

    log.info(f"Fetching MOD10A2 h{h:02d}v{v:02d} ({start_date} to {end_date})")
    raw = masking.get_modis_MOD10A2_max_snow_extent(
        vertical_tile=v,
        horizontal_tile=h,
        start_date=start_date,
        end_date=end_date,
        chunks={"time": -1, "x": 240, "y": 240},
    )
    log.info(f"Raw data shape: {dict(zip(raw.dims, raw.shape))}")

    log.info("Applying cloud filling...")
    binary = masking.binarize_with_cloud_filling(raw)
    return binary


def assign_water_year_coords(da: xr.DataArray, hemisphere: str) -> xr.DataArray:
    """Assign water_year and DOWY coordinates to a time-indexed DataArray."""

    def datetime_to_wy(dt, hemisphere):
        if hemisphere == "northern":
            return dt.year if dt.month >= 10 else dt.year
        else:
            return dt.year + 1 if dt.month >= 4 else dt.year

    def datetime_to_dowy(dt, hemisphere):
        if hemisphere == "northern":
            wy_start = pd.Timestamp(f"{dt.year - 1 if dt.month < 10 else dt.year}-10-01")
        else:
            wy_start = pd.Timestamp(f"{dt.year if dt.month >= 4 else dt.year - 1}-04-01")
        return (dt - wy_start).days + 1

    times = pd.DatetimeIndex(da.time.values)
    water_years = [datetime_to_wy(t, hemisphere) for t in times]
    dowys = [datetime_to_dowy(t, hemisphere) for t in times]

    da = da.assign_coords(
        water_year=("time", water_years),
        DOWY=("time", dowys),
    )
    return da


def compute_snow_metrics(binary: xr.DataArray, config: Config, hemisphere: str) -> xr.Dataset:
    """Run the full snow metrics pipeline and return a Dataset with water_year dimension."""
    binary = assign_water_year_coords(binary, hemisphere)

    log.info("Aligning water year starts...")
    binary_aligned = masking.align_wy_start(binary, hemisphere=hemisphere)

    target_wys = np.arange(config.wy_start, config.wy_end + 1)
    results = []

    for wy in target_wys:
        wy_da = binary_aligned.where(binary_aligned.water_year == wy, drop=True)
        if len(wy_da.time) < 5:
            log.warning(f"WY{wy}: only {len(wy_da.time)} observations, skipping")
            continue

        log.info(f"Computing snow metrics for WY{wy} ({len(wy_da.time)} obs)")
        metrics = masking.get_max_consec_snow_days_SAD_SDD_one_WY(wy_da)
        metrics = metrics.expand_dims(water_year=[wy])
        results.append(metrics)

    if not results:
        raise ValueError("No valid water years found for this tile")

    ds = xr.concat(results, dim="water_year")

    # Pad missing water years with fill value
    fill = np.iinfo(np.int16).min
    ds = ds.reindex(water_year=target_wys, fill_value=fill)

    return ds


def reindex_to_global_grid(ds_tile: xr.Dataset, ds_store: xr.Dataset) -> xr.Dataset:
    """Reindex tile dataset to align with the global Zarr store coordinates."""
    log.info("Reindexing to global coordinate grid...")
    return ds_tile.reindex(
        y=ds_store.y,
        x=ds_store.x,
        method="nearest",
        tolerance=1.0,  # 1 meter tolerance for floating point alignment
    )


def write_to_icechunk(ds_tile: xr.Dataset, config: Config, tile_id: str):
    """Write tile data to Icechunk store and commit."""
    log.info(f"Opening Icechunk repo for writing...")
    repo = config.open_repo()
    session = repo.writable_session("main")
    store = session.store()

    log.info(f"Writing tile data to store (region='auto')...")
    ds_tile.drop_vars("spatial_ref", errors="ignore").to_zarr(
        store,
        region="auto",
        mode="r+",
        zarr_format=3,
    )

    commit_message = f"{tile_id}: processed"
    snapshot_id = session.commit(
        commit_message,
        conflict_solver=icechunk.ConflictDetector(),
    )
    log.info(f"Committed: '{commit_message}' -> snapshot {snapshot_id}")


def main():
    args = parse_args()
    h, v = args.h, args.v
    tile_id = Config.tile_id(h, v)
    hemisphere = Config.hemisphere_for_v(v)

    log.info(f"Processing tile {tile_id} ({hemisphere} hemisphere)")
    start = datetime.now(timezone.utc)

    config = Config()

    # Fetch reference store coordinates (read-only) for reindexing
    log.info("Opening Icechunk store (read-only) to get global coordinates...")
    repo = config.open_repo()
    session_ro = repo.readonly_session("main")
    ds_store = xr.open_zarr(session_ro.store(), zarr_format=3, consolidated=False)

    # Full pipeline
    binary = fetch_and_binarize(h, v, config)
    ds_tile = compute_snow_metrics(binary, config, hemisphere)
    ds_tile = reindex_to_global_grid(ds_tile, ds_store)

    write_to_icechunk(ds_tile, config, tile_id)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log.info(f"Done. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
