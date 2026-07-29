"""
Get (tile, water-years) work items for GitHub Actions matrix processing.

Queries Icechunk commit history via status.get_remaining_work to determine
which (tile, water year) pairs are still missing — i.e. flagged for
processing, hemisphere-eligible, and without a commit (data or verified-
empty) — then emits them as a JSON matrix or count. Each matrix item carries
the tile indices plus the comma-joined water years for that tile:

    {"h": 10, "v": 4, "water_years": "2025"}

which process_batch.yml forwards to process_single_tile.py --water-years.
Ineligible water years (season not fully elapsed for the tile's hemisphere,
plus TRAILING_BUFFER_DAYS) are simply absent from the work list; they appear
automatically once their eligibility date passes.

Two output modes:
    Legacy (single-level matrix):
        --output json    Print bare work array: [{"h", "v", "water_years"}, ...]
        --output count   Print work-item (tile) count

    Batching (two-level matrix, handles >256 tiles):
        --list-batches   Print batch index JSON: {"batch_index": [0, 1, ...]}
        --batch-index N  Print work JSON for batch N:
                         {"tile": [{"h", "v", "water_years"}, ...]}

--which-tiles:
    missing (default) — tiles with >= 1 eligible-but-uncommitted water year;
        each item lists only those years. Re-runnable: already-committed
        years (data or empty) are never re-dispatched.
    all — every to_process tile, listing all its eligible years (full
        re-run; newer commits simply supersede older ones in the status
        derivation).

Usage:
    python get_tiles_for_batch.py --which-tiles missing --output json
    python get_tiles_for_batch.py --which-tiles missing --output count
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

from modis_snow_phenology.config import Config  # noqa: E402
from modis_snow_phenology.status import get_remaining_work  # noqa: E402

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
        default="missing",
        choices=["missing", "all"],
        help=(
            "'missing': tiles with eligible water years that have no commit "
            "yet (skips committed years — data and verified-empty alike); "
            "'all': all to_process tiles with all their eligible years "
            "(re-runs everything)"
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
        help='Print work list for batch N: {"tile": [{"h","v","water_years"}, ...]}',
    )
    return p.parse_args()


def main():
    args = parse_args()
    config = Config(args.config_file)

    if args.which_tiles == "missing":
        repo = config.open_icechunk_repo()
        work = get_remaining_work(config, repo=repo)
    else:
        # All to_process tiles x all eligible years, regardless of Icechunk
        # history (useful for re-running everything from scratch).
        work = []
        for row in config.get_process_tiles().itertuples():
            wys = config.eligible_years(Config.hemisphere_for_v(int(row.v)))
            if wys:
                work.append({"h": int(row.h), "v": int(row.v), "water_years": wys})

    items = [
        {
            "h": item["h"],
            "v": item["v"],
            "water_years": ",".join(str(wy) for wy in item["water_years"]),
        }
        for item in work
    ]
    total = len(items)

    # --- Legacy modes ---
    if args.output == "count":
        print(total)
        return

    if args.output == "json":
        print(json.dumps(items))
        return

    # --- Batching modes ---
    num_batches = math.ceil(total / BATCH_SIZE) if items else 0
    n_years = sum(len(i["water_years"].split(",")) for i in items if i["water_years"])
    print(
        f"{total} tiles ({n_years} tile-years) remaining, "
        f"{num_batches} batch(es) of up to {BATCH_SIZE}",
        file=sys.stderr,
    )

    if args.list_batches:
        print(json.dumps({"batch_index": list(range(num_batches))}))
        return

    if args.batch_index is not None:
        start = args.batch_index * BATCH_SIZE
        batch = items[start: start + BATCH_SIZE]
        print(json.dumps({"tile": batch}))
        return

    # Default (no mode flag): print bare JSON array (same as --output json)
    print(json.dumps(items))


if __name__ == "__main__":
    main()
