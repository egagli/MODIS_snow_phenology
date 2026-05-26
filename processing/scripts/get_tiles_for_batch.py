"""
Get unprocessed land tiles for GitHub Actions matrix processing.

Queries Icechunk commit history to determine which tiles have already been
processed, then returns the remaining land tiles as a JSON matrix or count.

Two modes:
    Legacy (single-level matrix):
        --output json    Print bare tile array: [{"h": H, "v": V}, ...]
        --output count   Print tile count

    Batching (two-level matrix, handles >256 tiles):
        --list-batches   Print batch index JSON: {"batch_index": [0, 1, ...]}
        --batch-index N  Print tile JSON for batch N:
                         {"tile": [{"h": H, "v": V}, ...]}

Usage:
    python get_tiles_for_batch.py --which-tiles unprocessed --output json
    python get_tiles_for_batch.py --which-tiles unprocessed --output count
    python get_tiles_for_batch.py --config-file config/config_v1.txt \
        --which-tiles all --output json
    python get_tiles_for_batch.py --list-batches
    python get_tiles_for_batch.py --batch-index 0
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modis_snow_phenology.config import (  # noqa: E402
    Config,
    get_processed_tiles_from_icechunk,
)

BATCH_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config-file",
        default="config/config_v1.txt",
        help="Path to config file",
    )
    p.add_argument(
        "--which-tiles",
        default="unprocessed",
        choices=["unprocessed", "all"],
        help=(
            "'unprocessed': land tiles with no Icechunk commit yet "
            "(includes previously failed); "
            "'all': all land tiles (re-process everything)"
        ),
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--output",
        choices=["json", "count"],
        help=(
            "Legacy output format: 'json' for bare GH Actions matrix, "
            "'count' for tile count"
        ),
    )
    mode.add_argument(
        "--list-batches",
        action="store_true",
        help='Print batch index list as JSON: {"batch_index": [0, 1, ...]}',
    )
    mode.add_argument(
        "--batch-index",
        type=int,
        metavar="N",
        help='Print tile list for batch N: {"tile": [{"h": H, "v": V}, ...]}',
    )
    return p.parse_args()


def main():
    args = parse_args()
    config = Config(args.config_file)

    land_gdf = config.get_land_tiles()

    if args.which_tiles == "unprocessed":
        repo = config.open_icechunk_repo()
        processed = get_processed_tiles_from_icechunk(repo)
        gdf = land_gdf[~land_gdf["tile"].isin(processed)].copy()
    else:
        gdf = land_gdf

    tiles = [{"h": int(row.h), "v": int(row.v)} for _, row in gdf.iterrows()]
    total = len(tiles)

    # --- Legacy modes ---
    if args.output == "count":
        print(total)
        return

    if args.output == "json":
        print(json.dumps(tiles))
        return

    # --- Batching modes ---
    num_batches = math.ceil(total / BATCH_SIZE) if tiles else 0
    print(
        f"{total} tiles remaining, "
        f"{num_batches} batch(es) of up to {BATCH_SIZE}",
        file=sys.stderr,
    )

    if args.list_batches:
        print(json.dumps({"batch_index": list(range(num_batches))}))
        return

    if args.batch_index is not None:
        start = args.batch_index * BATCH_SIZE
        batch = tiles[start : start + BATCH_SIZE]
        print(json.dumps({"tile": batch}))
        return

    # Default (no mode flag): print bare JSON array (same as --output json)
    print(json.dumps(tiles))


if __name__ == "__main__":
    main()
