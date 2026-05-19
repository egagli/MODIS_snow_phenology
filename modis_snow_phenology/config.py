"""
Configuration loader for the MODIS snow phenology processing pipeline.

Reads a flat key=value config file (e.g. config/config_v1.txt). Values set to
"ENV" are resolved from the environment variable of the same name. For local
development, copy config_v1.txt to config_with_secrets_v1.txt, fill in the
credentials, and pass that path instead (it is gitignored).

Usage:
    config = Config("config/config_v1.txt")               # CI: credentials from env vars
    config = Config("config/config_with_secrets_v1.txt")  # local: literal credentials
"""

import os
import re
from pathlib import Path

import geopandas as gpd
import icechunk

REPO_ROOT = Path(__file__).parent.parent

# Matches new-format commit: "h10v04: processed. Stats: [...]"
_NEW_MSG_RE = re.compile(r"^(h\d{2}v\d{2}): processed\. Stats:")
# Matches legacy commit: "h10v04: processed"
_OLD_MSG_RE = re.compile(r"^(h\d{2}v\d{2}): processed$")


def get_processed_tiles_from_icechunk(repo) -> set[str]:
    """Return the set of tile_ids already committed to the Icechunk store.

    Walks commit history on the main branch and matches both the new rich
    format (``h10v04: processed. Stats: ...``) and the legacy simple format
    (``h10v04: processed``). Returns a set of tile_id strings like ``"h10v04"``.
    """
    done = set()
    for commit in repo.ancestry(branch="main"):
        m = _NEW_MSG_RE.match(commit.message) or _OLD_MSG_RE.match(commit.message)
        if m:
            done.add(m.group(1))
    return done


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
                            f"Config field '{key}' is set to ENV but ${key} is not set in the environment"
                        )
                else:
                    raw[key] = val

        self.CONFIG_NAME: str = raw["CONFIG_NAME"]
        self.VERSION: str = raw["VERSION"]

        self.AZURE_STORAGE_ACCOUNT: str = raw["AZURE_STORAGE_ACCOUNT"]
        self.AZURE_STORAGE_SAS_TOKEN: str = raw["AZURE_STORAGE_SAS_TOKEN"]
        self.AZURE_CONTAINER: str = raw["AZURE_CONTAINER"]
        self.ICECHUNK_PREFIX: str = raw["ICECHUNK_PREFIX"]

        self.TILE_STATUS_PATH: Path = REPO_ROOT / raw["TILE_STATUS_PATH"]

        self.WY_START: int = int(raw["WY_START"])
        self.WY_END: int = int(raw["WY_END"])

        self.SHARD_SHAPE: tuple[int, ...] = tuple(int(x) for x in raw["SHARD_SHAPE"].split(","))
        self.INNER_CHUNK_SHAPE: tuple[int, ...] = tuple(int(x) for x in raw["INNER_CHUNK_SHAPE"].split(","))

    def open_icechunk_repo(self, config: "icechunk.RepositoryConfig | None" = None):
        """Open the Icechunk repository for this config's Azure storage location."""
        storage = icechunk.azure_storage(
            account=self.AZURE_STORAGE_ACCOUNT,
            container=self.AZURE_CONTAINER,
            prefix=self.ICECHUNK_PREFIX,
            sas_token=self.AZURE_STORAGE_SAS_TOKEN,
        )
        return icechunk.Repository.open(storage, config=config)

    def load_tile_status(self) -> gpd.GeoDataFrame:
        return gpd.read_file(self.TILE_STATUS_PATH)

    def get_land_tiles(self) -> gpd.GeoDataFrame:
        """Return all tiles that intersect land (used as the candidate processing pool)."""
        gdf = self.load_tile_status()
        return gdf[gdf["land"]].copy()

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
