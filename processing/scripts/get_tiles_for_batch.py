"""
Get tiles from tile_processing_status.geojson for GitHub Actions matrix processing.

Usage:
    python get_tiles_for_batch.py --which-tiles unprocessed --output json
    python get_tiles_for_batch.py --which-tiles unprocessed_and_failed --output count
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modis_snow_phenology.config import Config

VALID_STATUSES = {
    "unprocessed": ["unprocessed"],
    "failed": ["failed"],
    "unprocessed_and_failed": ["unprocessed", "failed"],
    "all": ["unprocessed", "failed", "processed"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--which-tiles",
        default="unprocessed_and_failed",
        choices=list(VALID_STATUSES),
        help="Which tiles to include",
    )
    p.add_argument(
        "--output",
        default="json",
        choices=["json", "count"],
        help="Output format: 'json' for GH Actions matrix, 'count' for tile count",
    )
    return p.parse_args()


def main():
    args = parse_args()
    config = Config()

    statuses = VALID_STATUSES[args.which_tiles]
    gdf = config.get_tiles_by_status(statuses)

    if args.output == "count":
        print(len(gdf))
        return

    # JSON matrix for GitHub Actions: list of {"h": int, "v": int}
    tiles = [{"h": int(row.h), "v": int(row.v)} for _, row in gdf.iterrows()]
    print(json.dumps(tiles))


if __name__ == "__main__":
    main()
