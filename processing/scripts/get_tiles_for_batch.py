"""
Get unprocessed land tiles for GitHub Actions matrix processing.

Queries Icechunk commit history to determine which tiles have already been
processed, then returns the remaining land tiles as a JSON matrix or count.

Usage:
    python get_tiles_for_batch.py --which-tiles unprocessed --output json
    python get_tiles_for_batch.py --which-tiles unprocessed --output count
    python get_tiles_for_batch.py --config-file config/config_v1.txt --which-tiles all --output json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modis_snow_phenology.config import Config, get_processed_tiles_from_icechunk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="config/config_v1.txt", help="Path to config file")
    p.add_argument(
        "--which-tiles",
        default="unprocessed",
        choices=["unprocessed", "all"],
        help=(
            "'unprocessed': land tiles with no Icechunk commit yet (includes previously failed); "
            "'all': all land tiles (re-process everything)"
        ),
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
    config = Config(args.config_file)

    land_gdf = config.get_land_tiles()

    if args.which_tiles == "unprocessed":
        repo = config.open_icechunk_repo()
        processed = get_processed_tiles_from_icechunk(repo)
        gdf = land_gdf[
            ~land_gdf["tile"].isin(processed)
        ].copy()
    else:
        gdf = land_gdf

    if args.output == "count":
        print(len(gdf))
        return

    tiles = [{"h": int(row.h), "v": int(row.v)} for _, row in gdf.iterrows()]
    print(json.dumps(tiles))


if __name__ == "__main__":
    main()
