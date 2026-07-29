"""
Configuration loader for the MODIS snow phenology processing pipeline.

Reads a flat key=value config file (e.g. config/config_v1.txt). Values set to
"ENV" are resolved from the environment variable of the same name. For local
development, copy config_v1.txt to config_with_secrets_v1.txt, fill in the
credentials, and pass that path instead (it is gitignored).

Usage:
    config = Config("config/config_v1.txt")              # CI: env vars
    config = Config("config/config_with_secrets_v1.txt") # local: literal
"""

import os
from pathlib import Path

import geopandas as gpd
import icechunk

REPO_ROOT = Path(__file__).parent.parent

# MODIS sinusoidal projection — matches the source tile grid shapefile.
# GeoJSON does not preserve CRS, so every read of tile_list.geojson must
# explicitly set this.
MODIS_SINUSOIDAL_CRS = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"


class Config:
    def __init__(self, config_file: str = "config/config_v1.txt"):
        self._path = REPO_ROOT / config_file
        raw = {}
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if val == "ENV":
                    try:
                        raw[key] = os.environ[key]
                    except KeyError:
                        raise ValueError(
                            f"Config field '{key}' is set to ENV but "
                            f"${key} is not set in the environment"
                        )
                else:
                    raw[key] = val

        self.CONFIG_NAME: str = raw["CONFIG_NAME"]
        self.VERSION: str = raw["VERSION"]

        self.AZURE_STORAGE_ACCOUNT: str = raw["AZURE_STORAGE_ACCOUNT"]
        self.AZURE_STORAGE_SAS_TOKEN: str = raw["AZURE_STORAGE_SAS_TOKEN"]
        self.AZURE_CONTAINER: str = raw["AZURE_CONTAINER"]
        self.ICECHUNK_PREFIX: str = raw["ICECHUNK_PREFIX"]
        self.MULTISCALE_PREFIX: str = raw["MULTISCALE_PREFIX"]

        self.TILE_LIST_PATH: Path = REPO_ROOT / raw["TILE_LIST_PATH"]

        self.WY_START: int = int(raw["WY_START"])
        self.WY_END: int = int(raw["WY_END"])
        # Days past a hemisphere's season end before a water year may be
        # processed (bfill context + MOD10A2 compositing/NSIDC latency).
        self.TRAILING_BUFFER_DAYS: int = int(raw.get("TRAILING_BUFFER_DAYS", "90"))

        self.SHARD_SHAPE: tuple[int, ...] = tuple(
            int(x) for x in raw["SHARD_SHAPE"].split(",")
        )
        self.INNER_CHUNK_SHAPE: tuple[int, ...] = tuple(
            int(x) for x in raw["INNER_CHUNK_SHAPE"].split(",")
        )

    def __str__(self) -> str:
        sas = self.AZURE_STORAGE_SAS_TOKEN
        masked_sas = (sas[:8] + "...") if len(sas) > 8 else "***"
        return (
            f"Config({self.CONFIG_NAME} v{self.VERSION})\n"
            f"  AZURE_STORAGE_ACCOUNT  : {self.AZURE_STORAGE_ACCOUNT}\n"
            f"  AZURE_CONTAINER        : {self.AZURE_CONTAINER}\n"
            f"  AZURE_STORAGE_SAS_TOKEN: {masked_sas}\n"
            f"  ICECHUNK_PREFIX        : {self.ICECHUNK_PREFIX}\n"
            f"  MULTISCALE_PREFIX      : {self.MULTISCALE_PREFIX}\n"
            f"  TILE_LIST_PATH         : {self.TILE_LIST_PATH}\n"
            f"  WY_START / WY_END      : {self.WY_START} / {self.WY_END}\n"
            f"  TRAILING_BUFFER_DAYS   : {self.TRAILING_BUFFER_DAYS}\n"
            f"  SHARD_SHAPE            : {self.SHARD_SHAPE}\n"
            f"  INNER_CHUNK_SHAPE      : {self.INNER_CHUNK_SHAPE}"
        )

    @property
    def years(self) -> list[int]:
        """All water years covered by this config, inclusive."""
        return list(range(self.WY_START, self.WY_END + 1))

    def eligible_years(self, hemisphere: str, today=None) -> list[int]:
        """Config years whose season has fully elapsed for ``hemisphere``
        (plus TRAILING_BUFFER_DAYS). See status.wy_eligible for the rule."""
        from modis_snow_phenology.status import wy_eligible

        return [
            wy for wy in self.years
            if wy_eligible(wy, hemisphere, today=today,
                           trailing_buffer_days=self.TRAILING_BUFFER_DAYS)
        ]

    @property
    def multiscale_zarr_url(self) -> str:
        """Full Azure blob URL for the multiscale Zarr pyramid."""
        return (
            f"https://{self.AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
            f"/{self.AZURE_CONTAINER}/{self.MULTISCALE_PREFIX}"
        )

    def open_icechunk_repo(
        self, config: "icechunk.RepositoryConfig | None" = None
    ):
        """Open the Icechunk repository for this config's Azure storage."""
        storage = icechunk.azure_storage(
            account=self.AZURE_STORAGE_ACCOUNT,
            container=self.AZURE_CONTAINER,
            prefix=self.ICECHUNK_PREFIX,
            sas_token=self.AZURE_STORAGE_SAS_TOKEN,
        )
        return icechunk.Repository.open(storage, config=config)

    def load_tile_list(self) -> gpd.GeoDataFrame:
        """Load the static tile registry from tile_list.geojson."""
        return gpd.read_file(self.TILE_LIST_PATH).set_crs(MODIS_SINUSOIDAL_CRS, allow_override=True)

    def get_process_tiles(self) -> gpd.GeoDataFrame:
        """Return tiles flagged for processing (to_process == True)."""
        gdf = self.load_tile_list()
        return gdf[gdf["to_process"].astype(bool)].copy()

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
