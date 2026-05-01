"""
Process a single MODIS tile and write results to the Icechunk store.

Usage:
    python process_single_tile.py --h 10 --v 4
    python process_single_tile.py --h 10 --v 4 --config-file config/config_v1.txt

On success: commits data to Icechunk with message "h10v04: processed"
On failure: exits nonzero; no Icechunk commit (store remains clean)
"""

import argparse
import faulthandler
import logging
import signal
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
from modis_snow_phenology.processing import _rss_mb

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
            return dt.year if dt.month >= 4 else dt.year - 1

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
    Fetches 1 prior and 1 following water year so bfill/ffill have full context.
    Returns None if there are too few observations.
    """
    if hemisphere == "northern":
        # NH WY spans Oct(wy-1) – Sep(wy); fetch Oct(wy-2) – Sep(wy+1) for context
        # (wy-2)-10-01 = start of the prior water year WY(wy-1)
        fetch_start = f"{wy - 2}-10-01"
        fetch_end_extended = f"{wy + 1}-09-30"
        fetch_end_fallback = f"{wy}-09-30"
    else:
        # SH WY spans Apr(wy) – Mar(wy+1); fetch Apr(wy-1) – Mar(wy+2) for context
        fetch_start = f"{wy - 1}-04-01"
        fetch_end_extended = f"{wy + 2}-03-31"
        fetch_end_fallback = f"{wy + 1}-03-31"

    log.info(f"WY{wy}: fetching {fetch_start} to {fetch_end_extended} (RSS={_rss_mb()} MB)")
    try:
        raw = processing.get_modis_MOD10A2_max_snow_extent(
            vertical_tile=v,
            horizontal_tile=h,
            start_date=fetch_start,
            end_date=fetch_end_extended,
            chunks={"time": -1, "x": 2400, "y": 2400},
        )
    except ValueError:
        # Extended window (wy+1) likely exceeds archive coverage (MODIS Terra
        # decommissioned Nov 2024); fall back to fetching only through end of
        # the target WY, losing future bfill context for this year only.
        log.warning(
            f"WY{wy}: extended fetch to {fetch_end_extended} returned no data; "
            f"retrying with fallback end {fetch_end_fallback}"
        )
        raw = processing.get_modis_MOD10A2_max_snow_extent(
            vertical_tile=v,
            horizontal_tile=h,
            start_date=fetch_start,
            end_date=fetch_end_fallback,
            chunks={"time": -1, "x": 2400, "y": 2400},
        )
    log.info(f"WY{wy}: raw fetched, shape={raw.shape}, dtype={raw.dtype}, RSS={_rss_mb()} MB")

    # Polar night correction: for Arctic/Antarctic tiles, the sensor records
    # no-snow (25) during winter darkness. Replace those with cloud/fill (255)
    # so that cloud-filling (bfill) handles them instead of treating them as
    # real no-snow observations that would corrupt SDD/SAD.
    if v <= 2 or v >= 15:
        log.info(f"WY{wy}: applying polar night correction (v={v})")
        value25_da = raw.where(lambda x: x == 25).count(dim=["x", "y"])
        value200_da = raw.where(lambda x: x == 200).count(dim=["x", "y"])
        no_decision_and_night_counts = raw.where(lambda x: (x == 1) | (x == 11)).count(dim=["x", "y"])
        land_area_da = value200_da + value25_da
        max_land_pixels = land_area_da.max(dim="time")
        bad_pixel_thresh = int(0.05 * int(max_land_pixels))
        scenes_with_polar_night = no_decision_and_night_counts > bad_pixel_thresh
        scenes_with_polar_night_buffered = (
            scenes_with_polar_night.shift(time=-1).fillna(0)
            | scenes_with_polar_night
            | scenes_with_polar_night.shift(time=1).fillna(0)
        ).astype(int)
        backward_check = scenes_with_polar_night_buffered.rolling(time=4, center=False).sum() >= 4
        forward_check = scenes_with_polar_night_buffered[::-1].rolling(time=4, center=False).sum()[::-1] >= 4
        center_check = scenes_with_polar_night_buffered.rolling(time=4, center=True).sum() >= 4
        scenes_with_polar_night_buffered_filtered = scenes_with_polar_night_buffered.where(
            backward_check | forward_check | center_check, other=0
        ).astype(bool).chunk(dict(time=-1))
        scenes_with_polar_night_buffered_filtered_complete = (
            scenes_with_polar_night_buffered_filtered.where(lambda x: x == 1)
            .interpolate_na(dim="time", method="nearest", max_gap=pd.Timedelta(days=80))
            .where(lambda x: x == 1, other=0)
            .astype(bool)
        )
        raw = raw.where(
            ~((raw == 25) & scenes_with_polar_night_buffered_filtered_complete), other=255
        )

    log.info(f"WY{wy}: binarizing (RSS={_rss_mb()} MB)...")
    binary = processing.binarize_with_cloud_filling(raw)
    del raw  # 794 MB uint8 — not used after binarize; free before align_wy_start
    log.info(f"WY{wy}: binarize done (RSS={_rss_mb()} MB)")
    binary = assign_water_year_coords(binary, hemisphere)
    binary_aligned = processing.align_wy_start(binary, hemisphere=hemisphere)
    del binary  # 794 MB bool — not used after align_wy_start

    wy_da = binary_aligned.where(binary_aligned.water_year == wy, drop=True)
    if len(wy_da.time) < 5:
        log.warning(f"WY{wy}: only {len(wy_da.time)} observations, skipping")
        return None

    log.info(f"WY{wy}: computing snow metrics ({len(wy_da.time)} obs, RSS={_rss_mb()} MB)")
    metrics = processing.get_max_consec_snow_days_SAD_SDD_one_WY(wy_da)
    return metrics.expand_dims(water_year=[wy])


def main():
    # Dump Python+C tracebacks on SIGSEGV/SIGABRT/SIGFPE/SIGBUS (C-level crashes).
    faulthandler.enable()

    # Log and exit cleanly on SIGTERM so the runner's "operation canceled" has context.
    def _sigterm_handler(signum, frame):
        log.error("Received SIGTERM — process is being terminated externally")
        traceback.print_stack(frame)
        sys.exit(1)
    signal.signal(signal.SIGTERM, _sigterm_handler)

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
    snapshot_id = session.commit(commit_message)
    log.info(f"Committed: '{commit_message}' -> {snapshot_id}")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log.info(f"Done. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
