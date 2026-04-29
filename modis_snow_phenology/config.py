"""
Configuration for the MODIS snow phenology processing pipeline.
"""

import os
from pathlib import Path

import geopandas as gpd
import icechunk


REPO_ROOT = Path(__file__).parent.parent

AZURE_CONTAINER = "snowmelt"
ICECHUNK_PREFIX = "snow-phenology/global_modis_snow_phenology_v1"

TILE_STATUS_PATH = REPO_ROOT / "processing" / "tile_data" / "tile_processing_status.geojson"

# MODIS water years
WY_START = 2015
WY_END = 2024

# Southern hemisphere tiles (v >= 9 are mostly SH, but v index used to detect hemisphere)
# Tiles with v >= 9 are in the southern hemisphere
SH_V_THRESHOLD = 9


class Config:
    def __init__(self):
        self.account_name = os.environ.get("AZURE_STORAGE_ACCOUNT", "uwcryo")
        self.sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN", "")
        self.tile_status_path = TILE_STATUS_PATH
        self.wy_start = WY_START
        self.wy_end = WY_END

    def get_storage(self) -> icechunk.Storage:
        return icechunk.Storage.new_azure_blob(
            container=AZURE_CONTAINER,
            prefix=ICECHUNK_PREFIX,
            account_name=self.account_name,
            sas_token=self.sas_token,
        )

    def open_repo(self) -> icechunk.Repository:
        return icechunk.Repository.open(self.get_storage())

    def create_repo(self) -> icechunk.Repository:
        return icechunk.Repository.create(self.get_storage())

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
        return "southern" if v >= SH_V_THRESHOLD else "northern"
