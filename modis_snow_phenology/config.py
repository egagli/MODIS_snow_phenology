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
import re
from pathlib import Path

import geopandas as gpd
import icechunk

REPO_ROOT = Path(__file__).parent.parent

# Special note values embedded in commit messages.
SPECIAL_NOTE_NONE = "None"
SPECIAL_NOTE_NODATA = (
    "No input data found for any year, therefore no output written to tile."
)

# MODIS sinusoidal projection — matches the source tile grid shapefile.
# GeoJSON does not preserve CRS, so every read of tile_list.geojson must
# explicitly set this.
MODIS_SINUSOIDAL_CRS = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"

# New-format: "Tile(h=10, v=4) processed. Stats: [...] Special note: ..."
_NEW_MSG_RE = re.compile(
    r"Tile\(h=(\d+), v=(\d+)\) processed\."
    r" Stats: \[([^\]]*)\] Special note: (.+)"
)
# Per-WY stat entry inside the Stats [...] list
_STAT_RE = re.compile(
    r"WY(\d{4}): input_obs=(\d+), valid_pixels=(\d+), coverage=([\d.]+)%"
)
# Legacy format: "h10v04: processed..." (old commit messages, backward compat)
_OLD_MSG_RE = re.compile(r"^(h\d{2}v\d{2}): processed")


def get_processing_status_gdf(
    repo, tile_list_path: Path, years: list[int]
) -> gpd.GeoDataFrame:
    """Return all tiles with runtime processing status and per-year stats.

    Reads ``tile_list.geojson`` (whose ``to_process`` boolean column marks
    tiles that should be processed) and merges with the Icechunk commit
    history to assign a runtime ``processing_status`` to each tile.

    ``processing_status`` values:
        ``"processed"``   — tile flagged for processing; data written to store
        ``"nodata"``      — tile flagged for processing; no MODIS input found
        ``"unprocessed"`` — tile flagged for processing; no commit yet
        ``"skip"``        — tile not flagged for processing (e.g. ocean)

    ``processing_notes`` — the special note from the commit message, or
        ``None`` for skipped / unprocessed tiles and tiles with no note.

    Per-year columns ``{year}_valid_pixels`` and ``{year}_input_obs`` (int or
    NaN) are added for each year in ``years``. Legacy commits (old
    ``h10v04`` format) are treated as ``"processed"`` with NaN stats.

    Column order: tile, h, v, processing_status, processing_notes, land,
    to_process, tile_notes, then per-year stats, then any remaining columns.
    """
    tile_gdf = gpd.read_file(tile_list_path).set_crs(MODIS_SINUSOIDAL_CRS).copy()

    # Walk commit history once; build a dict keyed by (h, v) int tuples.
    commit_info: dict[tuple[int, int], dict] = {}
    for commit in repo.ancestry(branch="main"):
        msg = commit.message
        m = _NEW_MSG_RE.match(msg)
        if m:
            h, v = int(m.group(1)), int(m.group(2))
            special_note = m.group(4).strip()
            year_stats = {
                int(sm.group(1)): {
                    "input_obs": int(sm.group(2)),
                    "valid_pixels": int(sm.group(3)),
                    "coverage": float(sm.group(4)),
                }
                for sm in _STAT_RE.finditer(m.group(3))
            }
            commit_info[(h, v)] = {
                "special_note": special_note,
                "year_stats": year_stats,
            }
            continue
        m = _OLD_MSG_RE.match(msg)
        if m:
            tile_id = m.group(1)  # e.g. "h10v04"
            h, v = int(tile_id[1:3]), int(tile_id[4:6])
            # Legacy commit: treat as processed, no per-year stats available.
            commit_info.setdefault(
                (h, v), {"special_note": SPECIAL_NOTE_NONE, "year_stats": {}}
            )

    def _processing_status(row):
        if not bool(row["to_process"]):
            return "skip"
        key = (int(row["h"]), int(row["v"]))
        if key not in commit_info:
            return "unprocessed"
        if commit_info[key]["special_note"] == SPECIAL_NOTE_NODATA:
            return "nodata"
        return "processed"

    def _processing_notes(row):
        if not bool(row["to_process"]):
            return None
        key = (int(row["h"]), int(row["v"]))
        info = commit_info.get(key)
        if info is None:
            return None
        note = info["special_note"]
        return None if note == SPECIAL_NOTE_NONE else note

    tile_gdf["processing_status"] = tile_gdf.apply(_processing_status, axis=1)
    tile_gdf["processing_notes"] = tile_gdf.apply(_processing_notes, axis=1)

    for yr in years:

        def _valid_pixels(row, yr=yr):
            key = (int(row["h"]), int(row["v"]))
            info = commit_info.get(key)
            if info is None:
                return float("nan")
            return info["year_stats"].get(yr, {}).get(
                "valid_pixels", float("nan")
            )

        def _input_obs(row, yr=yr):
            key = (int(row["h"]), int(row["v"]))
            info = commit_info.get(key)
            if info is None:
                return float("nan")
            return info["year_stats"].get(yr, {}).get(
                "input_obs", float("nan")
            )

        tile_gdf[f"{yr}_valid_pixels"] = tile_gdf.apply(_valid_pixels, axis=1)
        tile_gdf[f"{yr}_input_obs"] = tile_gdf.apply(_input_obs, axis=1)

    # Reorder: fixed cols → per-year stats → remaining → geometry last.
    fixed = [
        "tile", "h", "v",
        "processing_status", "processing_notes",
        "land", "to_process", "tile_notes",
    ]
    yr_cols = [
        col
        for yr in years
        for col in (f"{yr}_valid_pixels", f"{yr}_input_obs")
    ]
    present_fixed = [c for c in fixed if c in tile_gdf.columns]
    present_yr = [c for c in yr_cols if c in tile_gdf.columns]
    remaining = [
        c for c in tile_gdf.columns
        if c not in present_fixed + present_yr + ["geometry"]
    ]
    tile_gdf = tile_gdf[present_fixed + present_yr + remaining + ["geometry"]]

    return tile_gdf


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
            f"  SHARD_SHAPE            : {self.SHARD_SHAPE}\n"
            f"  INNER_CHUNK_SHAPE      : {self.INNER_CHUNK_SHAPE}"
        )

    @property
    def years(self) -> list[int]:
        """All water years covered by this config, inclusive."""
        return list(range(self.WY_START, self.WY_END + 1))

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
        return gpd.read_file(self.TILE_LIST_PATH).set_crs(MODIS_SINUSOIDAL_CRS)

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
