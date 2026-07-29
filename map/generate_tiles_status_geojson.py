"""Generate map/public/tiles-status.geojson from the Icechunk commit history.

Run locally:
    pixi run -e ci python map/generate_tiles_status_geojson.py
    pixi run -e ci python map/generate_tiles_status_geojson.py \
        --config config/config_with_secrets_v1.txt

Run in CI: see .github/workflows/deploy-map.yml — this script is called
automatically before the Next.js build so the map always reflects the
latest processing status.

Requires AZURE_STORAGE_SAS_TOKEN in the environment (resolved via config_v1.txt
ENV placeholder), or use config_with_secrets_v1.txt for local runs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).parent.parent
OUTPUT = Path(__file__).parent / "public" / "tiles-status.geojson"

sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Generate tiles-status.geojson for the web map."
    )
    parser.add_argument(
        "--config",
        default="config/config_v1.txt",
        help="Config file to use (default: config/config_v1.txt)",
    )
    args = parser.parse_args()

    from modis_snow_phenology.config import Config
    from modis_snow_phenology.status import get_tile_status_gdf

    config = Config(args.config)
    repo = config.open_icechunk_repo()
    gdf = get_tile_status_gdf(config, repo=repo)

    # Clip tile geometries to the valid sinusoidal domain before reprojecting.
    # The MODIS tile grid is slightly larger than the sinusoidal ellipse: the
    # bottom row (v=17) extends past the south pole and corner points on h=0
    # and h=35 extend outside the valid domain at high latitudes.  pyproj
    # returns inf for out-of-domain points, corrupting the whole polygon.
    # We approximate the valid domain as the "peanut" ellipse:
    #   |x| ≤ R·π·cos(y/R)   (the sinusoidal equal-area boundary)
    _R = 6_371_007.181
    _lat = np.linspace(-np.pi / 2, np.pi / 2, 5_000)
    _y = _R * _lat
    _x = _R * np.pi * np.cos(_lat)
    _sinu_domain = Polygon(
        list(zip(_x, _y)) + list(zip(-_x[::-1], _y[::-1]))
    )
    gdf.geometry = gdf.geometry.intersection(_sinu_domain)

    # Densify edges in sinusoidal space so that the curved tile boundaries in
    # WGS84 are represented accurately after reprojection.
    # 50 km segments ≈ 22 intermediate points per 1 111 km tile edge.
    gdf.geometry = gdf.geometry.segmentize(50_000)
    gdf = gdf.to_crs("EPSG:4326")

    # Drop verbose per-year scene-count columns (num_scenes_2000…2026).
    # The map only needs total_num_MOD10A2_scenes for a summary figure.
    drop_cols = [c for c in gdf.columns if c.startswith("num_scenes_")]
    gdf = gdf.drop(columns=drop_cols, errors="ignore")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT, driver="GeoJSON")
    print(f"Written {len(gdf)} tiles → {OUTPUT}")


if __name__ == "__main__":
    main()
