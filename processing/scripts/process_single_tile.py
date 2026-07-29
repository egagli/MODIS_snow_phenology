"""
Process a single MODIS tile and write results to the Icechunk store.

One commit per (tile, water year), each carrying structured metadata
(see modis_snow_phenology.status) — the commit history is the single
source of processing-status truth, mirroring global_snowmelt_runoff_onset.

Usage:
    python process_single_tile.py --h 10 --v 4
    python process_single_tile.py --h 10 --v 4 --water-years 2025
    python process_single_tile.py --h 10 --v 4 \
        --config-file config/config_v1.txt --water-years eligible

--water-years:
    'eligible' (default, alias 'all') — all config years whose season has
        fully elapsed for this tile's hemisphere (+ TRAILING_BUFFER_DAYS);
        the normal mode for both fleet dispatch and manual runs.
    comma list (e.g. '2025' or '2024,2025') — explicit years; ineligible
        ones are skipped with a warning (never processed, never committed).
    NO mode bypasses the eligibility gate — committing a half-elapsed season
        would bake partial data into the store as if it were verified truth.

Per water year:
    data  -> writes the year's slab, commits with status='data' + stats
    empty -> commits an empty snapshot with status='empty' and
             empty_reason='no_granules' | 'insufficient_obs'
    ineligible -> NO commit at all (stays 'missing' so it is dispatched
             automatically once its season closes)
On failure: exits nonzero; no commit for unfinished years (store clean,
    finished years of this run keep their commits).
"""

import argparse
import faulthandler
import logging
import os
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modis_snow_phenology import processing  # noqa: E402
from modis_snow_phenology.config import Config  # noqa: E402
from modis_snow_phenology.processing import _rss_mb  # noqa: E402
from modis_snow_phenology.status import (  # noqa: E402
    EMPTY_INSUFFICIENT_OBS,
    EMPTY_NO_GRANULES,
    MIN_OBS_PER_WY,
    STATUS_DATA,
    STATUS_EMPTY,
    build_commit_message,
    build_commit_metadata,
)

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
    p.add_argument(
        "--h", type=int, required=True, dest="h",
        help="MODIS horizontal tile index (0-35)",
    )
    p.add_argument(
        "--v", type=int, required=True, dest="v",
        help="MODIS vertical tile index (0-17)",
    )
    p.add_argument(
        "--config-file", default="config/config_v1.txt",
        help="Path to config file",
    )
    p.add_argument(
        "--water-years", default="eligible",
        help=(
            "'eligible' (default): all config years passing the hemisphere "
            "eligibility gate; 'all': every config year (deliberate re-runs); "
            "or a comma list like '2025' / '2024,2025' (ineligible years are "
            "skipped with a warning)"
        ),
    )
    return p.parse_args()


def assign_water_year_coords(
    da: xr.DataArray, hemisphere: str
) -> xr.DataArray:
    def datetime_to_wy(dt, hemisphere):
        if hemisphere == "northern":
            return dt.year + 1 if dt.month >= 10 else dt.year
        else:
            return dt.year if dt.month >= 4 else dt.year - 1

    def datetime_to_dowy(dt, hemisphere):
        if hemisphere == "northern":
            yr = dt.year - 1 if dt.month < 10 else dt.year
            wy_start = pd.Timestamp(f"{yr}-10-01")
        else:
            yr = dt.year if dt.month >= 4 else dt.year - 1
            wy_start = pd.Timestamp(f"{yr}-04-01")
        return (dt - wy_start).days + 1

    times = pd.DatetimeIndex(da.time.values)
    da = da.assign_coords(
        water_year=("time", [datetime_to_wy(t, hemisphere) for t in times]),
        DOWY=("time", [datetime_to_dowy(t, hemisphere) for t in times]),
    )
    return da


def process_water_year(
    h: int, v: int, wy: int, config: Config, hemisphere: str
) -> tuple[xr.Dataset | None, int]:
    """
    Fetch, cloud-fill, and compute snow metrics for a single water year.
    Fetches 1 prior and 1 following water year so bfill/ffill have context.

    Returns (metrics_dataset, input_obs), or (None, input_obs) when the
    target WY has fewer than MIN_OBS_PER_WY observations. Raises ValueError
    when the archive has no granules at all for the fetch window (the caller
    records this as an empty year with reason 'no_granules').
    """
    if hemisphere == "northern":
        # NH WY spans Oct(wy-1)–Sep(wy); fetch one extra year each side
        fetch_start = f"{wy - 2}-10-01"
        fetch_end_extended = f"{wy + 1}-09-30"
        fetch_end_fallback = f"{wy}-09-30"
    else:
        # SH WY spans Apr(wy)–Mar(wy+1); fetch one extra year each side
        fetch_start = f"{wy - 1}-04-01"
        fetch_end_extended = f"{wy + 2}-03-31"
        fetch_end_fallback = f"{wy + 1}-03-31"

    log.info(
        "WY%d: fetching %s to %s (RSS=%d MB)",
        wy, fetch_start, fetch_end_extended, _rss_mb(),
    )
    try:
        raw = processing.get_modis_MOD10A2_max_snow_extent(
            vertical_tile=v,
            horizontal_tile=h,
            start_date=fetch_start,
            end_date=fetch_end_extended,
            chunks={"time": -1, "x": 2400, "y": 2400},
        )
    except ValueError:
        # The +1-WY trailing buffer can extend past the archive's current end
        # when processing recent years (MOD10A2 is still produced; verified
        # via CMR 2026-07 — only the Planetary Computer mirror stopped).
        # Fall back to end of target WY only; if THAT also finds nothing,
        # the ValueError propagates and the caller commits 'no_granules'.
        log.warning(
            "WY%d: extended fetch to %s returned no data; "
            "retrying with fallback end %s",
            wy, fetch_end_extended, fetch_end_fallback,
        )
        raw = processing.get_modis_MOD10A2_max_snow_extent(
            vertical_tile=v,
            horizontal_tile=h,
            start_date=fetch_start,
            end_date=fetch_end_fallback,
            chunks={"time": -1, "x": 2400, "y": 2400},
        )
    log.info(
        "WY%d: raw fetched, shape=%s, dtype=%s, RSS=%d MB",
        wy, raw.shape, raw.dtype, _rss_mb(),
    )

    # Polar night correction: for Arctic/Antarctic tiles, the sensor records
    # no-snow (25) during winter darkness. Replace those with cloud/fill (255)
    # so that cloud-filling (bfill) handles them instead of treating them as
    # real no-snow observations that would corrupt SDD/SAD.
    if v <= 2 or v >= 15:
        log.info("WY%d: applying polar night correction (v=%d)", wy, v)
        # Avoid .where(...).count(): promotes (T,2400,2400) uint8→float64
        # (~6 GB) per call. Use numpy boolean sums directly instead.
        _rn = raw.values  # (T, Y, X) uint8, already in memory
        _t = raw.time
        value25_da = xr.DataArray(
            (_rn == 25).sum(axis=(1, 2)), dims="time", coords={"time": _t}
        )
        value200_da = xr.DataArray(
            (_rn == 200).sum(axis=(1, 2)), dims="time", coords={"time": _t}
        )
        no_decision_and_night_counts = xr.DataArray(
            ((_rn == 1) | (_rn == 11)).sum(axis=(1, 2)),
            dims="time", coords={"time": _t},
        )
        del _rn
        land_area_da = value200_da + value25_da
        max_land_pixels = land_area_da.max(dim="time")
        bad_pixel_thresh = int(0.05 * int(max_land_pixels))
        scenes_with_polar_night = (
            no_decision_and_night_counts > bad_pixel_thresh
        )
        scenes_with_polar_night_buffered = (
            scenes_with_polar_night.shift(time=-1).fillna(0)
            | scenes_with_polar_night
            | scenes_with_polar_night.shift(time=1).fillna(0)
        ).astype(int)
        backward_check = (
            scenes_with_polar_night_buffered
            .rolling(time=4, center=False).sum() >= 4
        )
        forward_check = (
            scenes_with_polar_night_buffered[::-1]
            .rolling(time=4, center=False).sum()[::-1] >= 4
        )
        center_check = (
            scenes_with_polar_night_buffered
            .rolling(time=4, center=True).sum() >= 4
        )
        scenes_with_polar_night_buffered_filtered = (
            scenes_with_polar_night_buffered
            .where(backward_check | forward_check | center_check, other=0)
            .astype(bool)
            .chunk(dict(time=-1))
        )
        scenes_with_polar_night_buffered_filtered_complete = (
            scenes_with_polar_night_buffered_filtered
            .where(lambda x: x == 1)
            .interpolate_na(
                dim="time", method="nearest",
                max_gap=pd.Timedelta(days=80),
            )
            .where(lambda x: x == 1, other=0)
            .astype(bool)
        )
        raw = raw.where(
            ~(
                (raw == 25)
                & scenes_with_polar_night_buffered_filtered_complete
            ),
            other=255,
        )

    log.info("WY%d: binarizing (RSS=%d MB)...", wy, _rss_mb())
    binary = processing.binarize_with_cloud_filling(raw)
    # 794 MB uint8 — not used after binarize; free before align_wy_start
    del raw
    log.info("WY%d: binarize done (RSS=%d MB)", wy, _rss_mb())
    binary = assign_water_year_coords(binary, hemisphere)
    binary_aligned = processing.align_wy_start(binary, hemisphere=hemisphere)
    del binary  # 794 MB bool — not used after align_wy_start

    # Use isel — .where(cond, drop=True) on int16 promotes to float64 (8×),
    # causing OOM.
    wy_da = binary_aligned.isel(
        time=(binary_aligned.water_year.values == wy)
    )
    if len(wy_da.time) < MIN_OBS_PER_WY:
        log.warning(
            "WY%d: only %d observations, skipping", wy, len(wy_da.time)
        )
        return None, len(wy_da.time)

    input_obs = len(wy_da.time)
    log.info(
        "WY%d: computing snow metrics (%d obs, RSS=%d MB)",
        wy, input_obs, _rss_mb(),
    )
    metrics = processing.get_max_consec_snow_days_SAD_SDD_one_WY(wy_da)
    return metrics.expand_dims(water_year=[wy]), input_obs


def _write_step_summary(tile_id, wy_results, skipped_ineligible=()):
    """GitHub Actions job summary: one row per committed water year."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [f"## Tile {tile_id}", ""]
    if wy_results:
        lines += [
            "| Water Year | Status | Input obs | Valid pixels | Coverage | Snapshot |",
            "| ---------- | ------ | --------- | ------------ | -------- | -------- |",
        ]
        for wy, r in sorted(wy_results.items()):
            stats = r["stats"] or {}
            status = r["status"] + (f" ({r['reason']})" if r["reason"] else "")
            valid = stats.get("valid_pixels")
            cov = stats.get("coverage")
            valid_str = f"{valid:,}" if valid is not None else "—"
            cov_str = f"{cov:.1f}%" if cov is not None else "—"
            lines.append(
                f"| WY{wy} | {status} | {stats.get('input_obs', '—')} "
                f"| {valid_str} | {cov_str} | `{r['snapshot']}` |"
            )
        lines.append("")
    else:
        lines += ["**No water years committed.**", ""]
    if skipped_ineligible:
        lines.append(
            f"⏳ Ineligible (season not elapsed + buffer), left 'missing': "
            f"{', '.join(f'WY{wy}' for wy in skipped_ineligible)}"
        )
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    # Dump Python+C tracebacks on SIGSEGV/SIGABRT/SIGFPE/SIGBUS.
    faulthandler.enable()

    # Log and exit cleanly on SIGTERM so the runner has context.
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

    log.info("Config:\n%s", config)
    log.info("Processing tile %s (%s hemisphere)", tile_id, hemisphere)
    start = datetime.now(timezone.utc)

    # --- select target water years (hemisphere eligibility gate) ---
    # No mode bypasses the gate: committing a half-elapsed season would bake
    # partial data into the store as if it were verified truth.
    eligible = set(config.eligible_years(hemisphere))
    if args.water_years in ("eligible", "all"):
        requested = list(config.years)
    else:
        requested = sorted({int(t) for t in args.water_years.split(",")})
        outside = [wy for wy in requested if wy not in config.years]
        if outside:
            log.error(
                "Requested water years %s are outside the config range "
                "%d-%d — the store has no slot for them (extend the store "
                "and bump WY_END first)",
                outside, config.WY_START, config.WY_END,
            )
            sys.exit(1)

    target_wys = [wy for wy in requested if wy in eligible]
    skipped_ineligible = [wy for wy in requested if wy not in eligible]
    if skipped_ineligible:
        log.warning(
            "Skipping ineligible water years %s (%s hemisphere season not "
            "elapsed + %dd buffer); they stay 'missing' and will be "
            "dispatched once eligible",
            skipped_ineligible, hemisphere, config.TRAILING_BUFFER_DAYS,
        )
    if not target_wys:
        log.warning(
            "No eligible water years to process for tile %s — exiting "
            "WITHOUT commit (nothing is marked attempted)", tile_id,
        )
        _write_step_summary(tile_id, {}, skipped_ineligible)
        return

    log.info("Target water years: %s", target_wys)

    storage = icechunk.azure_storage(
        account=config.AZURE_STORAGE_ACCOUNT,
        container=config.AZURE_CONTAINER,
        prefix=config.ICECHUNK_PREFIX,
        sas_token=config.AZURE_STORAGE_SAS_TOKEN,
    )
    repo_config = icechunk.RepositoryConfig.default()
    repo_config.storage = icechunk.StorageSettings()
    repo_config.storage.retries = icechunk.StorageRetriesSettings(
        max_tries=20,
        initial_backoff_ms=200,
        max_backoff_ms=60_000,
    )

    # Read exact store coordinates for this tile's slice.
    # STAC-derived coordinates have float imprecision; snap to the store's
    # exact values so that region='auto' can match coordinates.
    log.info("Reading store coordinates for tile region...")
    repo_ro = icechunk.Repository.open(storage, config=repo_config)
    session_ro = repo_ro.readonly_session("main")
    ds_store = xr.open_zarr(
        session_ro.store, zarr_format=3, consolidated=False
    )
    store_y = ds_store.y[v * 2400: (v + 1) * 2400].values
    store_x = ds_store.x[h * 2400: (h + 1) * 2400].values
    store_wys = set(int(wy) for wy in ds_store.water_year.values)
    missing_slots = [wy for wy in target_wys if wy not in store_wys]
    if missing_slots:
        log.error(
            "Store water_year coordinate lacks %s — run "
            "processing/scripts/extend_store_water_years.py first",
            missing_slots,
        )
        sys.exit(1)

    # --- one commit per water year (crash-safe: finished years keep their
    # commits; unfinished years never commit and stay 'missing') ---
    wy_results = {}
    for wy in target_wys:
        wy_start = time.monotonic()
        try:
            metrics, input_obs = process_water_year(h, v, wy, config, hemisphere)
            empty_reason = EMPTY_INSUFFICIENT_OBS if metrics is None else None
        except ValueError as exc:
            log.warning("WY%d: no MOD10A2 granules for fetch window (%s)", wy, exc)
            metrics, input_obs, empty_reason = None, 0, EMPTY_NO_GRANULES

        if metrics is None:
            status = STATUS_EMPTY
            stats = {"input_obs": int(input_obs)}
            ds_write = None
        else:
            # Snap tile coordinates to store's exact values
            y_ok = np.allclose(store_y, metrics.y.values, atol=1.0)
            x_ok = np.allclose(store_x, metrics.x.values, atol=1.0)
            if not (y_ok and x_ok):
                y_diff = np.max(np.abs(store_y - metrics.y.values))
                x_diff = np.max(np.abs(store_x - metrics.x.values))
                raise ValueError(
                    f"WY{wy}: tile coordinates do not match store grid "
                    f"(max y diff: {y_diff:.2f} m, max x diff: {x_diff:.2f} m)"
                )
            metrics = metrics.assign_coords(y=store_y, x=store_x)

            ds_write = metrics.drop_vars("spatial_ref", errors="ignore")
            for var in ds_write.data_vars:
                ds_write[var].attrs.pop("_FillValue", None)
            ds_write = ds_write.chunk({"water_year": 1, "y": 2400, "x": 2400})

            # max_consec_snow_days is int16; fill = -32768 for ocean/no-data,
            # 0 for land pixels that never held snow. np.isnan() is always
            # False for ints, so "valid" = strictly positive.
            mcsd = ds_write["max_consec_snow_days"]
            valid = int(np.sum(mcsd.values > 0))
            coverage = 100.0 * valid / int(mcsd.size) if mcsd.size else 0.0
            status = STATUS_DATA
            stats = {
                "input_obs": int(input_obs),
                "valid_pixels": valid,
                "coverage": round(coverage, 2),
            }

        duration_s = time.monotonic() - wy_start
        metadata = build_commit_metadata(
            h, v, wy, hemisphere, status, config.VERSION,
            empty_reason=empty_reason, stats=stats, duration_s=duration_s,
        )
        message = build_commit_message(
            h, v, wy, status,
            empty_reason=empty_reason, valid_px=stats.get("valid_pixels"),
        )

        # ConflictDetector handles concurrent commits from parallel matrix
        # jobs: each job writes a unique tile region, so there are never real
        # data conflicts and Icechunk rebases automatically.
        while True:
            try:
                repo = icechunk.Repository.open(storage, config=repo_config)
                session = repo.writable_session("main")
                if ds_write is not None:
                    log.info("WY%d: writing...", wy)
                    ds_write.to_zarr(
                        session.store, region="auto", mode="r+", zarr_format=3
                    )
                snapshot_id = session.commit(
                    message,
                    metadata=metadata,
                    rebase_with=icechunk.ConflictDetector(),
                    allow_empty=ds_write is None,
                )
                break
            except Exception as exc:
                delay = random.uniform(3, 10)
                log.warning(
                    "WY%d commit failed (%s: %s); retrying in %.1fs",
                    wy, type(exc).__name__, exc, delay,
                )
                time.sleep(delay)

        log.info("WY%d: committed %s -> %s ('%s')", wy, status, snapshot_id, message)
        wy_results[wy] = {"status": status, "reason": empty_reason,
                          "stats": stats, "snapshot": snapshot_id}
        del metrics, ds_write  # free ~GB-scale arrays before the next year

    _write_step_summary(tile_id, wy_results, skipped_ineligible)

    n_data = sum(1 for r in wy_results.values() if r["status"] == STATUS_DATA)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log.info(
        "Done: %d data / %d empty commit(s), %d ineligible skipped. "
        "Total time: %.1fs",
        n_data, len(wy_results) - n_data, len(skipped_ineligible), elapsed,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
