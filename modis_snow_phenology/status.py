"""
Processing status derived from Icechunk commit history.

The pipeline records everything it does as structured metadata on Icechunk
commits (one commit per tile x water year, mirroring the
global_snowmelt_runoff_onset repo). This module is the single reader of that
record: it walks ``repo.ancestry()`` once and derives which tile x water
years hold data, which are verified-empty, and which are still missing
(never attempted, or attempted and failed -- failures never commit, so
absence == not done).

Commit metadata schema (see build_commit_metadata):

    {
      "schema": 1,
      "kind": "tile_year",
      "tile": [h, v],
      "water_year": 2025,
      "hemisphere": "northern" | "southern",
      "status": "data" | "empty",
      "empty_reason": "no_granules" | "insufficient_obs",   # empty only
      "stats": {"input_obs": 46, "valid_pixels": 123456, "coverage": 2.1},
      "config_version": "1",
      "duration_s": 123.4,
    }

Hemisphere-aware water-year eligibility also lives here: a water year may
only be processed once its season has fully elapsed for the tile's
hemisphere (northern WY N: Oct 1 N-1 .. Sep 30 N; southern WY N:
Apr 1 N .. Mar 31 N+1) plus a trailing buffer so the bidirectional cloud
filling has post-season context (MOD10A2 is an 8-day composite and the
pipeline bfills across the WY end; ~90 days keeps that context healthy --
the +-1 WY fetch window degrades gracefully when the trailing year is
incomplete). The dispatch layer (get_remaining_work) and the processor
(process_single_tile.py --water-years) both apply the same gate, so a
half-elapsed season can never be committed as verified-empty.
"""

import datetime
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd

COMMIT_SCHEMA_VERSION = 1

KIND_TILE_YEAR = "tile_year"

STATUS_DATA = "data"
STATUS_EMPTY = "empty"

EMPTY_NO_GRANULES = "no_granules"          # earthaccess found no MOD10A2 input
EMPTY_INSUFFICIENT_OBS = "insufficient_obs"  # < MIN_OBS observations in the WY

# Minimum MOD10A2 observations inside the target WY for metrics to be computed
# (mirrors the guard in processing.align_wy_start / process_single_tile).
MIN_OBS_PER_WY = 5

DEFAULT_TRAILING_BUFFER_DAYS = 90


# ---------------------------------------------------------------------------
# Hemisphere-aware water-year eligibility
# ---------------------------------------------------------------------------

def season_end(wy: int, hemisphere: str) -> datetime.date:
    """Last calendar day of water year ``wy`` for a hemisphere.

    Northern WY N runs Oct 1 (N-1) .. Sep 30 (N); southern WY N runs
    Apr 1 (N) .. Mar 31 (N+1).
    """
    if hemisphere == "northern":
        return datetime.date(wy, 9, 30)
    if hemisphere == "southern":
        return datetime.date(wy + 1, 3, 31)
    raise ValueError(f"unknown hemisphere: {hemisphere!r}")


def wy_eligible(
    wy: int,
    hemisphere: str,
    today: Optional[datetime.date] = None,
    trailing_buffer_days: int = DEFAULT_TRAILING_BUFFER_DAYS,
) -> bool:
    """True once ``wy`` has fully elapsed for ``hemisphere`` plus the buffer.

    The buffer covers MOD10A2 8-day compositing + NSIDC latency and gives the
    bidirectional cloud filling post-season context. Processing an ineligible
    year would bake a half-observed season into the store as if it were real.
    """
    today = today or datetime.date.today()
    return today >= season_end(wy, hemisphere) + datetime.timedelta(
        days=trailing_buffer_days
    )


# ---------------------------------------------------------------------------
# Commit metadata (writer side)
# ---------------------------------------------------------------------------

def build_commit_metadata(
    h: int,
    v: int,
    water_year: int,
    hemisphere: str,
    status: str,
    config_version: str,
    empty_reason: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    duration_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build the structured metadata dictionary attached to every pipeline commit.

    Machine-readable counterpart to the human-readable commit message; all
    status derivation in this module parses these dictionaries (never the
    message strings).
    """
    metadata: Dict[str, Any] = {
        "schema": COMMIT_SCHEMA_VERSION,
        "kind": KIND_TILE_YEAR,
        "tile": [int(h), int(v)],
        "water_year": int(water_year),
        "hemisphere": hemisphere,
        "status": status,
        "config_version": config_version,
    }
    if empty_reason is not None:
        metadata["empty_reason"] = empty_reason
    if stats is not None:
        metadata["stats"] = stats
    if duration_s is not None:
        metadata["duration_s"] = round(float(duration_s), 1)
    return metadata


def build_commit_message(
    h: int,
    v: int,
    water_year: int,
    status: str,
    empty_reason: Optional[str] = None,
    valid_px: Optional[int] = None,
) -> str:
    """Human-readable commit message (for `icechunk log`-style inspection only)."""
    prefix = f"tile(h={h},v={v}) WY{water_year}"
    if status == STATUS_EMPTY:
        return f"{prefix}: empty ({empty_reason})"
    if valid_px is not None:
        return f"{prefix}: {valid_px:,} valid px"
    return f"{prefix}: processed"


# ---------------------------------------------------------------------------
# Commit records (reader side)
# ---------------------------------------------------------------------------

def get_commit_records(
    repo, branch: str = "main", as_of_snapshot: Optional[str] = None
) -> pd.DataFrame:
    """
    Walk the branch ancestry once and return one row per pipeline commit.

    Non-pipeline commits (store init, water_year extension, ...) are skipped
    by requiring the metadata schema keys. Ancestry iterates newest -> oldest;
    the returned frame preserves that order in the 'ancestry_index' column
    (0 = newest), so "first seen" == newest for any (h, v, wy).

    Args:
        as_of_snapshot: derive status as of this snapshot instead of the
            branch tip. Used to pin one consistent work list across all
            batches of a fleet run.

    Returns:
        DataFrame with columns: ancestry_index, snapshot_id, written_at, h, v,
        water_year, hemisphere, status, empty_reason, config_version,
        stats (dict), duration_s
    """
    if as_of_snapshot:
        ancestry = repo.ancestry(snapshot_id=as_of_snapshot)
    else:
        ancestry = repo.ancestry(branch=branch)
    records = []
    for index, snap in enumerate(ancestry):
        meta = snap.metadata or {}
        if "schema" not in meta or meta.get("kind") != KIND_TILE_YEAR:
            continue
        h, v = meta["tile"]
        records.append({
            "ancestry_index": index,
            "snapshot_id": snap.id,
            "written_at": snap.written_at,
            "h": int(h),
            "v": int(v),
            "water_year": int(meta["water_year"]),
            "hemisphere": meta.get("hemisphere"),
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "config_version": meta.get("config_version"),
            "stats": meta.get("stats"),
            "duration_s": meta.get("duration_s"),
        })
    columns = ["ancestry_index", "snapshot_id", "written_at", "h", "v",
               "water_year", "hemisphere", "status", "empty_reason",
               "config_version", "stats", "duration_s"]
    return pd.DataFrame.from_records(records, columns=columns)


# ---------------------------------------------------------------------------
# Tile status / remaining work
# ---------------------------------------------------------------------------

def _newest_per_tile_year(commits_df: pd.DataFrame) -> Dict[tuple, Any]:
    """Newest commit record per (h, v, water_year); ancestry order is newest-first."""
    newest = {}
    for record in commits_df.itertuples():
        key = (record.h, record.v, int(record.water_year))
        if key not in newest:  # first seen == newest
            newest[key] = record
    return newest


def get_tile_status_gdf(
    config,
    repo=None,
    branch: str = "main",
    as_of_snapshot: Optional[str] = None,
    today: Optional[datetime.date] = None,
) -> gpd.GeoDataFrame:
    """Per-tile processing status: the tile registry joined with commit history.

    For each registry tile (``tile_list.geojson``; ``to_process`` marks tiles
    that should be processed) derives:

    - ``hemisphere`` — from the MODIS v index (v >= 9 -> southern)
    - per-year ``{yr}_status`` — 'data', 'empty', 'missing' (eligible, no
      commit yet), or 'ineligible' (season not fully elapsed + buffer)
    - per-year ``{yr}_valid_pixels`` / ``{yr}_input_obs`` — from commit stats
      (NaN where absent); consumed by the web map
    - ``missing_wys`` — comma-joined eligible-but-uncommitted years
    - ``processing_status``:
        ``"skip"``        — tile not flagged for processing (e.g. ocean)
        ``"unprocessed"`` — no commit yet for any eligible year
        ``"partial"``     — some eligible years committed, some missing
        ``"processed"``   — every eligible year committed, >= 1 has data
        ``"nodata"``      — every eligible year committed, all verified-empty

    Column order: tile, h, v, hemisphere, processing_status, missing_wys,
    land, to_process, tile_notes, per-year stats, remaining, geometry.
    """
    if repo is None:
        repo = config.open_icechunk_repo()
    commits_df = get_commit_records(repo, branch=branch, as_of_snapshot=as_of_snapshot)
    newest = _newest_per_tile_year(commits_df)

    from modis_snow_phenology.config import Config  # late import, no cycle at module load

    buffer_days = getattr(config, "TRAILING_BUFFER_DAYS", DEFAULT_TRAILING_BUFFER_DAYS)
    years = [int(wy) for wy in config.years]
    tile_gdf = config.load_tile_list().copy()
    tile_gdf["hemisphere"] = tile_gdf["v"].astype(int).map(Config.hemisphere_for_v)

    eligible_by_hemi = {
        hemi: [wy for wy in years
               if wy_eligible(wy, hemi, today=today, trailing_buffer_days=buffer_days)]
        for hemi in ("northern", "southern")
    }

    yr_status_cols: Dict[int, list] = {wy: [] for wy in years}
    valid_px_cols: Dict[int, list] = {wy: [] for wy in years}
    input_obs_cols: Dict[int, list] = {wy: [] for wy in years}
    processing_status = []
    missing_wys_col = []

    for row in tile_gdf.itertuples():
        h, v = int(row.h), int(row.v)
        eligible = eligible_by_hemi[row.hemisphere]
        statuses = {}
        missing = []
        for wy in years:
            record = newest.get((h, v, wy))
            stats = (record.stats or {}) if record is not None else {}
            valid_px_cols[wy].append(stats.get("valid_pixels", float("nan")))
            input_obs_cols[wy].append(stats.get("input_obs", float("nan")))
            if record is not None:
                statuses[wy] = record.status
            elif wy in eligible:
                statuses[wy] = "missing"
                missing.append(wy)
            else:
                statuses[wy] = "ineligible"
            yr_status_cols[wy].append(statuses[wy])

        if not bool(row.to_process):
            processing_status.append("skip")
        else:
            committed = [s for wy, s in statuses.items()
                         if s in (STATUS_DATA, STATUS_EMPTY)]
            if not committed:
                processing_status.append("unprocessed")
            elif missing:
                processing_status.append("partial")
            elif STATUS_DATA in committed:
                processing_status.append("processed")
            else:
                processing_status.append("nodata")
        missing_wys_col.append(",".join(str(wy) for wy in missing))

    tile_gdf["processing_status"] = processing_status
    tile_gdf["missing_wys"] = missing_wys_col
    for wy in years:
        tile_gdf[f"{wy}_status"] = yr_status_cols[wy]
        tile_gdf[f"{wy}_valid_pixels"] = valid_px_cols[wy]
        tile_gdf[f"{wy}_input_obs"] = input_obs_cols[wy]

    # Reorder: fixed cols -> per-year stats -> remaining -> geometry last.
    fixed = ["tile", "h", "v", "hemisphere", "processing_status", "missing_wys",
             "land", "to_process", "tile_notes"]
    yr_cols = [col for wy in years
               for col in (f"{wy}_status", f"{wy}_valid_pixels", f"{wy}_input_obs")]
    present_fixed = [c for c in fixed if c in tile_gdf.columns]
    present_yr = [c for c in yr_cols if c in tile_gdf.columns]
    remaining = [c for c in tile_gdf.columns
                 if c not in present_fixed + present_yr + ["geometry"]]
    return tile_gdf[present_fixed + present_yr + remaining + ["geometry"]]


def get_remaining_work(
    config,
    repo=None,
    branch: str = "main",
    as_of_snapshot: Optional[str] = None,
    today: Optional[datetime.date] = None,
) -> List[Dict[str, Any]]:
    """
    Derive the outstanding work list from commit history.

    Returns:
        List of {"h", "v", "water_years": [...]} dicts in registry order, one
        per to_process tile that still has eligible-but-uncommitted water
        years. Ineligible years (season not elapsed for the tile's
        hemisphere) are simply absent — they reappear here automatically once
        their eligibility date passes.
    """
    status_gdf = get_tile_status_gdf(
        config, repo=repo, branch=branch, as_of_snapshot=as_of_snapshot, today=today
    )
    work = []
    for row in status_gdf.itertuples():
        if not bool(row.to_process) or not row.missing_wys:
            continue
        work.append({
            "h": int(row.h),
            "v": int(row.v),
            "water_years": [int(wy) for wy in row.missing_wys.split(",")],
        })
    return work
