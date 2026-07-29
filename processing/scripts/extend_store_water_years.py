"""
Extend the Icechunk store's water_year dimension (non-destructively).

Appends new water-year slots up to config.WY_END (or --through-wy) by
resizing every water_year-dimensioned array and rewriting the water_year
coordinate, then committing. This is the prerequisite for processing a new
water year: process_single_tile.py writes with to_zarr(region="auto"),
which can only place years that already exist in the store coordinate.

The append is cheap and safe: shards are (1, 2400, 2400) so an axis-0
append is shard-aligned and metadata-only — no existing chunk is touched,
and the new slots read as fill (-32768) until tiles write them. The commit
carries no status metadata, so status derivation ignores it and every
(tile, new year) simply shows up as 'missing' work once eligible.

Usage:
    python processing/scripts/extend_store_water_years.py \
        --config-file config/config_with_secrets_v1.txt
    python processing/scripts/extend_store_water_years.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modis_snow_phenology.config import Config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extend_store_water_years")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config-file", default="config/config_v1.txt",
        help="Path to config file",
    )
    p.add_argument(
        "--through-wy", type=int, default=None,
        help="Extend the water_year coordinate through this year "
             "(default: config.WY_END)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be appended without writing or committing",
    )
    return p.parse_args()


def main():
    args = parse_args()
    config = Config(args.config_file)
    through_wy = args.through_wy if args.through_wy is not None else config.WY_END

    repo = config.open_icechunk_repo()
    session = repo.writable_session("main")
    group = zarr.open_group(session.store, mode="r+")

    current = [int(wy) for wy in group["water_year"][:]]
    log.info("Store water_year: %d..%d (%d slots)", current[0], current[-1], len(current))
    if current != list(range(current[0], current[-1] + 1)):
        raise SystemExit(f"store water_year is not contiguous: {current}")
    if through_wy < current[-1]:
        raise SystemExit(
            f"--through-wy {through_wy} < store max {current[-1]}: "
            "shrinking the water_year dimension is not supported"
        )
    new_years = list(range(current[-1] + 1, through_wy + 1))
    if not new_years:
        log.info("Nothing to do: store already extends through WY%d", through_wy)
        return

    n_old, n_new = len(current), len(current) + len(new_years)
    wy_arrays = [
        name for name, arr in group.arrays()
        if name != "water_year"
        and (arr.metadata.dimension_names or [None])[0] == "water_year"
    ]
    log.info("Appending %s to arrays %s", new_years, sorted(wy_arrays))
    if args.dry_run:
        log.info("Dry run: no changes made")
        return

    for name in sorted(wy_arrays):
        arr = group[name]
        arr.resize((n_new, *arr.shape[1:]))
    wy_arr = group["water_year"]
    wy_arr.resize((n_new,))
    wy_arr[n_old:] = np.array(new_years, dtype=wy_arr.dtype)

    snapshot_id = session.commit(
        f"Extend water_year through WY{through_wy} "
        f"(appended {', '.join(f'WY{wy}' for wy in new_years)})"
    )
    log.info("Committed -> %s", snapshot_id)

    # Verify: coordinate is as expected and a sample of the new slab reads
    # as fill (nothing has written it yet).
    ds = xr.open_zarr(
        repo.readonly_session("main").store, zarr_format=3, consolidated=False
    )
    expected = list(range(current[0], through_wy + 1))
    got = [int(wy) for wy in ds.water_year.values]
    assert got == expected, f"water_year mismatch after extend: {got}"
    sample = ds[sorted(wy_arrays)[0]].sel(water_year=new_years[0])
    sample_vals = sample.isel(y=slice(0, 64), x=slice(0, 64)).values
    assert (sample_vals == -32768).all(), "new water-year slab is not all-fill"
    log.info(
        "Verified: water_year = %d..%d, new slab reads as fill. "
        "New (tile, year) work will appear via get_tiles_for_batch once eligible.",
        expected[0], expected[-1],
    )


if __name__ == "__main__":
    main()
