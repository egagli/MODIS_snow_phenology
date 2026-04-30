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


def assign_water_year_coords(da: xr.DataArray, hemisphere: str) -> xr.DataArray:
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
    da = da.assign_coords(
        water_year=("time", [datetime_to_wy(t, hemisphere) for t in times]),
        DOWY=("time", [datetime_to_dowy(t, hemisphere) for t in times]),
    )
    return da


def process_water_year(
    h: int, v: int, wy: int, config: Config, hemisphere: str
) -> xr.Dataset | None:
    """
    Fetch, cloud-fill, and compute snow metrics for a single water year.
    Fetches 2 prior water years for cloud-fill warm-up (~92 timesteps, ~1 GB).
    Returns None if there are too few observations.
    """
    if hemisphere == "northern":
        fetch_start = f"{wy - 2}-10-01"
        fetch_end = f"{wy}-09-30"
    else:
        fetch_start = f"{wy - 2}-04-01"
        fetch_end = f"{wy + 1}-03-31"

    log.info(f"WY{wy}: fetching {fetch_start} to {fetch_end}")
    raw = processing.get_modis_MOD10A2_max_snow_extent(
        vertical_tile=v,
        horizontal_tile=h,
        start_date=fetch_start,
        end_date=fetch_end,
        chunks={"time": -1, "x": 2400, "y": 2400},
    )

    binary = processing.binarize_with_cloud_filling(raw)
    binary = assign_water_year_coords(binary, hemisphere)
    binary_aligned = processing.align_wy_start(binary, hemisphere=hemisphere)

    wy_da = binary_aligned.where(binary_aligned.water_year == wy, drop=True)
    if len(wy_da.time) < 5:
        log.warning(f"WY{wy}: only {len(wy_da.time)} observations, skipping")
        return None

    log.info(f"WY{wy}: computing snow metrics ({len(wy_da.time)} obs)")
    metrics = processing.get_max_consec_snow_days_SAD_SDD_one_WY(wy_da)
    return metrics.expand_dims(water_year=[wy])


def main():
    args = parse_args()
    h, v = args.h, args.v
    config = Config(args.config_file)
    tile_id = Config.tile_id(h, v)
    hemisphere = Config.hemisphere_for_v(v)

    log.info(f"Processing tile {tile_id} ({hemisphere} hemisphere) — config: {config.config_name}")
    start = datetime.now(timezone.utc)

    storage = icechunk.azure_storage(
        account=config.azure_storage_account,
        container=config.azure_container,
        prefix=config.icechunk_prefix,
        sas_token=config.azure_storage_sas_token,
    )

    # Read exact store coordinates for this tile's slice.
    # STAC-derived coordinates have float imprecision; we snap to the store's
    # exact values so that region='auto' can match coordinates.
    log.info("Reading store coordinates for tile region...")
    repo_ro = icechunk.Repository.open(storage)
    session_ro = repo_ro.readonly_session("main")
    ds_store = xr.open_zarr(session_ro.store, zarr_format=3, consolidated=False)
    store_y = ds_store.y[v * 2400 : (v + 1) * 2400].values
    store_x = ds_store.x[h * 2400 : (h + 1) * 2400].values

    # Open a single writable session — accumulate all WY writes before committing
    log.info("Opening writable Icechunk session...")
    repo = icechunk.Repository.open(storage)
    session = repo.writable_session("main")

    fill = np.iinfo(np.int16).min
    target_wys = np.arange(config.wy_start, config.wy_end + 1)
    written_wys = []

    for wy in target_wys:
        metrics = process_water_year(h, v, wy, config, hemisphere)
        if metrics is None:
            continue

        # Snap tile coordinates to store's exact values
        if not (np.allclose(store_y, metrics.y.values, atol=1.0) and
                np.allclose(store_x, metrics.x.values, atol=1.0)):
            raise ValueError(
                f"WY{wy}: tile coordinates do not match store grid "
                f"(max y diff: {np.max(np.abs(store_y - metrics.y.values)):.2f} m, "
                f"max x diff: {np.max(np.abs(store_x - metrics.x.values)):.2f} m)"
            )
        metrics = metrics.assign_coords(y=store_y, x=store_x)

        ds_write = metrics.drop_vars("spatial_ref", errors="ignore")
        for var in ds_write.data_vars:
            ds_write[var].attrs.pop("_FillValue", None)
        ds_write = ds_write.chunk({"water_year": 1, "y": 2400, "x": 2400})

        log.info(f"WY{wy}: writing to store...")
        ds_write.to_zarr(session.store, region="auto", mode="r+", zarr_format=3)
        written_wys.append(wy)

    if not written_wys:
        raise ValueError(f"No water years written for tile {tile_id}")

    # Fill any skipped WYs with the fill value (already initialized in store)
    skipped = set(target_wys) - set(written_wys)
    if skipped:
        log.warning(f"Skipped WYs (no data): {sorted(skipped)}")

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
