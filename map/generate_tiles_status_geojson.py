"""Generate map/public/tiles-status.geojson from the Icechunk commit history.

Run locally:
    pixi run -e ci python map/generate_tiles_status_geojson.py
    pixi run -e ci python map/generate_tiles_status_geojson.py --config config/config_with_secrets_v1.txt

Run in CI: see .github/workflows/deploy-map.yml — this script is called
automatically before the Next.js build so the map always reflects the
latest processing status.

Requires AZURE_STORAGE_SAS_TOKEN in the environment (resolved via config_v1.txt
ENV placeholder), or use config_with_secrets_v1.txt for local runs.
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent
OUTPUT = Path(__file__).parent / "public" / "tiles-status.geojson"

sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Generate tiles-status.geojson for the web map.")
    parser.add_argument(
        "--config",
        default="config/config_v1.txt",
        help="Config file to use (default: config/config_v1.txt)",
    )
    args = parser.parse_args()

    from modis_snow_phenology.config import Config, get_processing_status_gdf

    config = Config(args.config)
    repo = config.open_icechunk_repo()
    gdf = get_processing_status_gdf(repo, config.TILE_STATUS_PATH)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT, driver="GeoJSON")
    print(f"Written {len(gdf)} tiles → {OUTPUT}")


if __name__ == "__main__":
    main()
