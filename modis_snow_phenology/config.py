"""
Configuration loader for the MODIS snow phenology processing pipeline.

Reads an INI-style config file (e.g. config/config_v1.txt) and exposes
values as attributes. Icechunk setup is left to the calling scripts.
"""

import configparser
from pathlib import Path

import geopandas as gpd

REPO_ROOT = Path(__file__).parent.parent


class Config:
    def __init__(self, config_file: str = "config/config_v1.txt"):
        self._path = REPO_ROOT / config_file
        parser = configparser.ConfigParser()
        parser.read(self._path)
        v = parser["VALUES"]

        self.config_name: str = parser["METADATA"]["config_name"]
        self.version: str = parser["METADATA"]["version"]

        self.azure_container: str = v["azure_container"]
        self.icechunk_prefix: str = v["icechunk_prefix"]

        self.tile_status_path: Path = REPO_ROOT / v["tile_status_path"]

        self.wy_start: int = int(v["wy_start"])
        self.wy_end: int = int(v["wy_end"])

        self.shard_shape: tuple[int, ...] = tuple(int(x) for x in v["shard_shape"].split(","))
        self.inner_chunk_shape: tuple[int, ...] = tuple(int(x) for x in v["inner_chunk_shape"].split(","))

    def load_tile_status(self) -> gpd.GeoDataFrame:
        return gpd.read_file(self.tile_status_path)

    def get_tiles_by_status(self, statuses: list[str]) -> gpd.GeoDataFrame:
        gdf = self.load_tile_status()
        return gdf[gdf["processing_status"].isin(statuses)].copy()

    @staticmethod
    def tile_id(h: int, v: int) -> str:
        return f"h{h:02d}v{v:02d}"

    @staticmethod
    def parse_tile_id(tile_id: str) -> tuple[int, int]:
        return int(tile_id[1:3]), int(tile_id[4:6])

    @staticmethod
    def hemisphere_for_v(v: int) -> str:
        """Tiles with v >= 9 are in the southern hemisphere."""
        return "southern" if v >= 9 else "northern"
