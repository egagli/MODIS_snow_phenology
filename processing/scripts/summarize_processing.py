"""
Update tile_processing_status.geojson by querying:
  1. Icechunk commit history  -> mark committed tiles as "processed"
  2. GitHub Actions job logs  -> mark failed jobs as "failed" with error excerpts

Usage (called by GitHub Actions summarize-results job):
    python summarize_processing.py \
        --run-id 12345678 \
        --github-token $GITHUB_TOKEN \
        --repo egagli/MODIS_snow_phenology

Usage (standalone, without GH Actions context — only updates from Icechunk history):
    python summarize_processing.py
"""

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import icechunk
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modis_snow_phenology.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("summarize_processing")

PROCESSED_PATTERN = re.compile(r"^h(\d{2})v(\d{2}): processed$")
JOB_NAME_PATTERN = re.compile(r"process-tile-h(\d+)-v(\d+)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="config/config_v1.txt", help="Path to config file")
    p.add_argument("--run-id", default=None, help="GitHub Actions run ID")
    p.add_argument("--github-token", default=None, help="GitHub token for API access")
    p.add_argument("--repo", default=None, help="GitHub repo (owner/repo)")
    return p.parse_args()


def get_processed_tiles_from_icechunk(config: Config) -> dict[str, str]:
    """
    Query Icechunk commit history and return a dict of tile_id -> ISO timestamp
    for all tiles with a 'processed' commit.
    """
    log.info("Querying Icechunk commit history...")
    storage = icechunk.azure_storage(
        account=config.azure_storage_account,
        container=config.azure_container,
        prefix=config.icechunk_prefix,
        sas_token=config.azure_storage_sas_token,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    processed = {}
    for snap in repo.ancestry(branch="main"):
        m = PROCESSED_PATTERN.match(snap.message)
        if m:
            tile_id = f"h{m[1]}v{m[2]}"
            if tile_id not in processed:  # ancestry is newest-first; keep most recent
                ts = getattr(snap, "committed_at", None)
                if ts is None:
                    ts = datetime.now(timezone.utc).isoformat()
                elif hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                processed[tile_id] = str(ts)

    log.info(f"Found {len(processed)} processed tiles in Icechunk history")
    return processed


def get_failed_jobs_from_github(run_id: str, token: str, repo: str) -> dict[str, str]:
    """
    Query GitHub Actions API for failed tile jobs in a given run.
    Returns dict of tile_id -> error_excerpt.
    """
    if not all([run_id, token, repo]):
        log.info("No GitHub API credentials provided — skipping failure detection")
        return {}

    log.info(f"Querying GitHub Actions API for run {run_id}...")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base_url = f"https://api.github.com/repos/{repo}"

    failed_tiles = {}
    page = 1
    while True:
        resp = requests.get(
            f"{base_url}/actions/runs/{run_id}/jobs",
            headers=headers,
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        jobs = resp.json().get("jobs", [])
        if not jobs:
            break

        for job in jobs:
            if job.get("conclusion") != "failure":
                continue
            m = JOB_NAME_PATTERN.search(job.get("name", ""))
            if not m:
                continue
            tile_id = f"h{int(m[1]):02d}v{int(m[2]):02d}"
            error_excerpt = _fetch_job_log_excerpt(job["id"], headers, base_url)
            failed_tiles[tile_id] = error_excerpt
            log.info(f"  Failed: {tile_id} — {error_excerpt[:80]}")

        if len(jobs) < 100:
            break
        page += 1

    log.info(f"Found {len(failed_tiles)} failed tile jobs")
    return failed_tiles


def _fetch_job_log_excerpt(job_id: int, headers: dict, base_url: str) -> str:
    """Download job logs and return the last meaningful error line."""
    try:
        resp = requests.get(
            f"{base_url}/actions/jobs/{job_id}/logs",
            headers=headers,
            timeout=30,
            allow_redirects=True,
        )
        if resp.status_code == 200:
            lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
            for line in reversed(lines):
                if len(line) > 10 and not line.startswith("20"):
                    return line[-300:]
    except Exception as e:
        log.warning(f"Could not fetch logs for job {job_id}: {e}")
    return "Error details unavailable (see GH Actions logs)"


def update_geojson(
    config: Config,
    processed: dict[str, str],
    failed: dict[str, str],
) -> gpd.GeoDataFrame:
    gdf = config.load_tile_status()

    for tile_id, timestamp in processed.items():
        mask = gdf["tile"] == tile_id
        if mask.any() and gdf.loc[mask, "processing_status"].values[0] != "processed":
            gdf.loc[mask, "processing_status"] = "processed"
            gdf.loc[mask, "notes"] = f"Processed at {timestamp}"
            log.info(f"  {tile_id}: -> processed")

    for tile_id, error in failed.items():
        mask = gdf["tile"] == tile_id
        if mask.any() and gdf.loc[mask, "processing_status"].values[0] not in ("processed", "skip"):
            gdf.loc[mask, "processing_status"] = "failed"
            gdf.loc[mask, "notes"] = error
            log.info(f"  {tile_id}: -> failed")

    return gdf


def main():
    args = parse_args()
    config = Config(args.config_file)

    processed = get_processed_tiles_from_icechunk(config)
    failed = get_failed_jobs_from_github(args.run_id, args.github_token, args.repo)

    gdf = update_geojson(config, processed, failed)

    counts = gdf["processing_status"].value_counts()
    log.info("Tile status summary:")
    for status, count in counts.items():
        log.info(f"  {status}: {count}")

    gdf.to_file(config.tile_status_path, driver="GeoJSON")
    log.info(f"Saved updated GeoJSON to {config.tile_status_path}")


if __name__ == "__main__":
    main()
