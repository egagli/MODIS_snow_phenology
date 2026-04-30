"""
Process a single MODIS tile and write results to the Icechunk store.

Usage:
    python process_single_tile.py --h 10 --v 4
    python process_single_tile.py --h 10 --v 4 --config-file config/config_v1.txt

On success: commits data to Icechunk with message "h10v04: processed"
On failure: exits nonzero; no Icechunk commit (store remains clean)
"""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modis_snow_phenology import processing
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
    p.add_argument("--config-file", default="config/config_v1.txt", help="Path to config file")
    return p.parse_args()


def fetch_and_binarize(h: int, v: int, config: Config) -> xr.DataArray:
    """Fetch MOD10A2 for a tile across all water years and cloud-fill."""
    start_date = f"{config.wy_start - 2}-10-01"
    end_date = f"{config.wy_end}-09-30"

    log.info(f"Fetching MOD10A2 h{h:02d}v{v:02d} ({start_date} to {end_date})")
    raw = processing.get_modis_MOD10A2_max_snow_extent(
        vertical_tile=v,
        horizontal_tile=h,
        start_date=start_date,
        end_date=end_date,
        chunks={"time": -1, "x": 240, "y": 240},
    )
    log.info(f"Raw data shape: {dict(zip(raw.dims, raw.shape))}")

    log.info("Applying cloud filling...")
    binary = processing.binarize_with_cloud_filling(raw)
    return binary


def assign_water_year_coords(da: xr.DataArray, hemisphere: str) -> xr.DataArray:
    """Assign water_year and DOWY coordinates to a time-indexed DataArray."""

    def datetime_to_wy(dt, hemisphere):
        if hemisphere == "northern":
            return dt.year + 1 if dt.month >= 10 else dt.year
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
    binary_aligned = processing.align_wy_start(binary, hemisphere=hemisphere)

    target_wys = np.arange(config.wy_start, config.wy_end + 1)
    results = []

    for wy in target_wys:
        wy_da = binary_aligned.where(binary_aligned.water_year == wy, drop=True)
        if len(wy_da.time) < 5:
            log.warning(f"WY{wy}: only {len(wy_da.time)} observations, skipping")
            continue

        log.info(f"Computing snow metrics for WY{wy} ({len(wy_da.time)} obs)")
        metrics = processing.get_max_consec_snow_days_SAD_SDD_one_WY(wy_da)
        metrics = metrics.expand_dims(water_year=[wy])
        results.append(metrics)

    if not results:
        raise ValueError("No valid water years found for this tile")

    ds = xr.concat(results, dim="water_year")

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
        tolerance=500.0,  # MODIS pixel spacing ~463 m; 500 m tolerance catches float imprecision
    )


def main():
    args = parse_args()
    h, v = args.h, args.v
    config = Config(args.config_file)
    tile_id = Config.tile_id(h, v)
    hemisphere = Config.hemisphere_for_v(v)

    log.info(f"Processing tile {tile_id} ({hemisphere} hemisphere) — config: {config.config_name}")
    start = datetime.now(timezone.utc)

    # Open Icechunk store — credentials from environment
    storage = icechunk.azure_storage(
        account=os.environ["AZURE_STORAGE_ACCOUNT"],
        container=config.azure_container,
        prefix=config.icechunk_prefix,
        sas_token=os.environ["AZURE_STORAGE_SAS_TOKEN"],
    )

    # Read global coordinate grid for reindexing
    log.info("Reading global store coordinates (read-only)...")
    repo_ro = icechunk.Repository.open(storage)
    session_ro = repo_ro.readonly_session("main")
    ds_store = xr.open_zarr(session_ro.store, zarr_format=3, consolidated=False)

    # Full processing pipeline
    binary = fetch_and_binarize(h, v, config)
    ds_tile = compute_snow_metrics(binary, config, hemisphere)
    ds_tile = reindex_to_global_grid(ds_tile, ds_store)

    # Write to Icechunk — open a fresh writable session
    log.info("Opening writable Icechunk session...")
    repo = icechunk.Repository.open(storage)
    session = repo.writable_session("main")

    log.info("Writing tile data (region='auto')...")
    ds_tile.drop_vars("spatial_ref", errors="ignore").to_zarr(
        session.store,
        region="auto",
        mode="r+",
        zarr_format=3,
    )

    commit_message = f"{tile_id}: processed"
    snapshot_id = session.commit(
        commit_message,
        conflict_solver=icechunk.ConflictDetector(),
    )
    log.info(f"Committed: '{commit_message}' -> {snapshot_id}")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log.info(f"Done. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
