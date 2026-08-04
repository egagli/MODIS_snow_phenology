# MODIS Snow Phenology

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21783366.svg)](https://doi.org/10.5281/zenodo.21783366)
[![Repository DOI](https://zenodo.org/badge/1224935546.svg)](https://doi.org/10.5281/zenodo.21783174)

Two Zenodo records: the first badge is the **dataset** (the Zarr archive described below), the second is the **archived source repository** (GitHub release `v1.0`). Cite the dataset for the data, the repository for the code.

Global snow phenology dataset derived from MODIS MOD10A2 8-day maximum snow extent (water years 2015–2025, extendable — see "Adding new water years" below).

Three variables per pixel per water year:

- **SAD_DOWY** — Snow Appearance Date (day of water year)
- **SDD_DOWY** — Snow Disappearance Date (day of water year; defined as first day with no snow)
- **max_consec_snow_days** — Length of the longest continuous snow period (days)

Cloud filling follows [Wrzesien et al. 2019](https://doi.org/10.1029/2019WR025350). No-data pixels use fill value `int16` minimum (`-32768`).

## Data product

The dataset is published on Zenodo as a single Zarr v3 archive (3.6 GB), exported by [`notebooks/download_dataset.ipynb`](notebooks/download_dataset.ipynb). See [`zenodo/zenodo_metadata.md`](zenodo/zenodo_metadata.md) for the record metadata.

> Gagliano, E. (2026). *Global MODIS snow phenology: snow appearance date, snow disappearance date, and maximum consecutive snow days, water years 2015–2025* (1.0.0) [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.21783366>

The archive is stored uncompressed, so it reads without unzipping:

```python
import xarray as xr
import zarr

store = zarr.storage.ZipStore("modis_snow_phenology_v1.zarr.zip", mode="r")
ds = xr.open_zarr(store, zarr_format=3, consolidated=False, decode_coords="all")
```

This supersedes [zenodo.15692530](https://doi.org/10.5281/zenodo.15692530) (water years 2015–2024), produced by the predecessor repo [MODIS_seasonal_snow_mask](https://github.com/egagli/MODIS_seasonal_snow_mask).

## Dataset

| Property | Value |
| --- | --- |
| Source | MODIS MOD10A2.061 (8-day maximum snow extent) via NASA Earthdata |
| Spatial resolution | 500 m |
| Grid | MODIS sinusoidal, 86 400 x 43 200 pixels (global) |
| Temporal coverage | Water years 2015–2025 (hemisphere-aware: northern WY N = Oct 1 N−1 … Sep 30 N; southern WY N = Apr 1 N … Mar 31 N+1) |
| Data format | `int16`, fill = `-32768` |
| Storage | Icechunk on Azure Blob Storage (`uwcryo` / `snowmelt` container) |
| Zarr format | Zarr v3 + ShardingCodec (shard: `1x2400x2400`, chunk: `1x600x600`) |
| CRS | MODIS sinusoidal (`+proj=sinu +R=6371007.181`) |

## Architecture

| Component | Technology |
| --- | --- |
| Package management | [pixi](https://pixi.sh) |
| Storage | Icechunk on Azure Blob Storage |
| Processing | GitHub Actions (tile-parallel matrix) |
| Web map | Next.js static export deployed to GitHub Pages |

## Processing Pipeline

Each MODIS tile (`h<HH>v<VV>`, 2400×2400 pixels, ~1111 km × 926 km) is processed independently. Processing is implemented in `processing/scripts/process_single_tile.py` using library functions from `modis_snow_phenology/processing.py`.

### Step 1 — Fetch MOD10A2 HDF4 granules

`get_modis_MOD10A2_max_snow_extent()` authenticates to NASA Earthdata via `earthaccess` (using `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` environment variables) and downloads all granules covering a date window spanning one water year plus one extra year on each side (for cloud-filling continuity across water year boundaries). Granules are downloaded sequentially into a temporary directory, opened with `rasterio` directly (not rioxarray, to avoid GDAL segfault issues with 100+ HDF4 files), and stacked into an `(time, y, x)` DataArray.

> **Why NSIDC/earthaccess and not Planetary Computer:** Processing previously fetched MOD10A2 from Microsoft Planetary Computer via STAC, but the **PC mirror stopped updating in mid-2025**. MODIS Terra was *not* decommissioned — it is still observing and MOD10A2 is still produced (verified against NASA CMR in July 2026: granules through 2026-07-12, full coverage of both hemispheres' WY2025 windows). Only the mirror died, so the pipeline fetches from the authoritative NSIDC archive via `earthaccess` (which queries CMR directly). The old PC implementation is preserved as `_get_modis_MOD10A2_max_snow_extent_planetary_computer` for reference. Before extending to a new water year, sanity-check that granules are still flowing:
>
> ```bash
> curl -s "https://cmr.earthdata.nasa.gov/search/granules.json?short_name=MOD10A2&page_size=1&sort_key=-start_date" \
>   | python3 -c "import sys,json; print(json.load(sys.stdin)['feed']['entry'][0]['time_start'])"
> ```
>
> (Terra's orbital drift is the eventual trigger for a VIIRS-based successor, but that is a data-quality watch item, not a current availability problem.)

### Step 2 — Polar night correction (Arctic/Antarctic tiles only)

For tiles in the extreme polar rows (`v ≤ 2` or `v ≥ 15`), the sensor records no-decision (`1`) or nighttime (`11`) values during winter darkness. These are wrongly interpreted as no-snow, which corrupts SAD/SDD calculations. The correction:

1. Counts per-scene pixels with no-decision/night values and land pixels to identify scenes dominated by polar night.
2. Buffers identified scenes by ±1 step and applies a rolling 4-scene minimum to require a sustained polar night period (rejects isolated noisy scenes).
3. Fills gaps between identified periods via nearest-neighbor interpolation (up to 80 days).
4. Replaces `no-snow (25)` pixels in identified polar-night scenes with `cloud (255)` so the subsequent cloud-filling step handles them correctly.

### Step 3 — Cloud filling (Wrzesien et al. 2019)

`binarize_with_cloud_filling()` converts MOD10A2 raw values to a binary snow mask while filling cloud gaps:

1. Remap darkness (`11`), fill (`255`), and no-decision (`1`) codes to cloud (`50`) in-place (stays `uint8` to avoid 8× memory expansion).
2. Forward-fill: for each time step, carry the last non-cloud value forward over any cloud pixels.
3. Backward-fill: carry the next non-cloud value backward over any cloud pixels.
4. A pixel is classified as **snow** only where both the forward-filled and backward-filled arrays agree on snow (`200`).

This means clouds bracketed on both sides by snow are filled as snow; clouds with no snow on one or both sides become no-snow. The entire multi-year time series is processed as a unit (not per water year) to preserve continuity across year boundaries.

### Step 4 — Water year coordinate assignment

Each time step is assigned a `water_year` and `DOWY` (day of water year) coordinate:

- **Northern hemisphere** (`v ≤ 8`): water year starts Oct 1. WY 2020 spans Oct 1, 2019 – Sep 30, 2020.
- **Southern hemisphere** (`v ≥ 9`): water year starts Apr 1. WY 2020 spans Apr 1, 2020 – Mar 31, 2021.

### Step 5 — Water year start alignment

`align_wy_start()` prepends a synthetic observation to each water year: the last observation from the preceding water year is duplicated at DOWY=1 (Oct 1 for NH, Apr 1 for SH). This ensures the snow-disappearance-date algorithm can detect snow that was already present at the start of a new water year. Water years with fewer than 5 observations are discarded.

### Step 6 — Snow metrics per water year

`get_max_consec_snow_days_SAD_SDD_one_WY()` applies a numba-JIT-compiled scan (`get_longest_consec_stretch_vectorized`) pixel-by-pixel along the time axis to find the longest consecutive run of snow days. The indices are then mapped back to DOWY values via a substitution dictionary. Leap years are detected by checking whether Feb 29 falls within the water year's actual date range, so SDD can reach DOWY 367 in leap years.

Output per water year:

- `SAD_DOWY`: DOWY of the first day of the longest snow period
- `SDD_DOWY`: DOWY of the first day without snow after the longest period
- `max_consec_snow_days`: `SDD_DOWY − SAD_DOWY`

Fill value (`-32768`) is used for ocean, pixels with no snow, and pixels with insufficient observations.

### Step 7 — Write to Icechunk store

Each water year is written and committed **individually** via `xr.Dataset.to_zarr(..., region='auto', mode='r+')` — one commit per (tile, water year), mirroring the [global_snowmelt_runoff_onset](https://github.com/egagli/global_snowmelt_runoff_onset) pipeline. Tile coordinates are snapped to the store's exact grid coordinates to prevent float-precision mismatches. Concurrent parallel tile jobs are handled via `icechunk.ConflictDetector` with randomized retry backoff — since each job writes to a disjoint tile region, there are never real data conflicts and Icechunk rebases automatically. Per-year commits make runs crash-safe: finished years keep their commits, unfinished years never commit and stay `missing`.

Every commit carries **structured metadata** (see `modis_snow_phenology/status.py`) — the machine-readable record that all status derivation parses (never the message strings):

```json
{"schema": 1, "kind": "tile_year", "tile": [10, 4], "water_year": 2025,
 "hemisphere": "northern", "status": "data", "config_version": "v1",
 "stats": {"input_obs": 46, "valid_pixels": 1234567, "coverage": 21.4},
 "duration_s": 312.7}
```

Water years with no usable input commit an **empty snapshot** with `status: "empty"` and an `empty_reason` (`no_granules` — the archive has nothing for the fetch window; `insufficient_obs` — fewer than 5 observations inside the target WY). Empty commits mark the year as attempted so it is never re-dispatched; absence of a commit means "not done".

## Getting Started

### Install environment

```bash
pixi install
pixi run notebook   # launches JupyterLab in notebooks/
```

### Initialize (once)

Run `notebooks/01_initialize.ipynb`:

1. Creates `processing/tile_list.geojson` — static registry of all 648 MODIS land tiles in sinusoidal CRS with `h`, `v`, `land`, `to_process`, and `tile_notes` columns. Edit `to_process` to `False` to skip tiles (e.g. ocean-only tiles already excluded automatically).
2. Creates the Icechunk store on Azure and writes the empty global Zarr arrays (requires `AZURE_STORAGE_SAS_TOKEN` in the environment).

### Process tiles

Trigger **Process All Tiles** in GitHub Actions:

- Queries Icechunk commit history for missing **(tile, water year)** pairs — eligible years without a commit (data or verified-empty)
- Hemisphere-aware eligibility: a water year is only dispatched once its season has fully elapsed for the tile's hemisphere plus `TRAILING_BUFFER_DAYS` (northern tiles pick up a new WY ~6 months before southern ones)
- Fans out up to 256 tiles per batch; batches run sequentially, tiles within a batch run in parallel; each job processes only its missing years
- Re-entrant: committed years (data and empty alike) are automatically skipped

To reprocess everything: trigger with `which_tiles = all` (all to_process tiles × all their eligible years — newer commits supersede older ones in the status derivation).

To process a single tile manually: trigger **Process Single Tile** with `h`, `v`, and optionally `water_years` (default `eligible`; or a comma list like `2025`). No mode can process an ineligible year — the processor vetoes them so a half-elapsed season is never committed.

### Monitor progress

Processing status is derived at runtime from the Icechunk commit history:

```python
import sys
sys.path.insert(0, "/path/to/MODIS_snow_phenology")
from modis_snow_phenology.config import Config
from modis_snow_phenology.status import get_tile_status_gdf, get_remaining_work

config = Config("config/config_with_secrets_v1.txt")
gdf = get_tile_status_gdf(config)
gdf["processing_status"].value_counts()
# processed / partial / nodata / unprocessed / skip
# plus per-year columns: {yr}_status ('data'/'empty'/'missing'/'ineligible'),
# {yr}_valid_pixels, {yr}_input_obs, and 'missing_wys' per tile

get_remaining_work(config)   # [{"h": 10, "v": 4, "water_years": [2025]}, ...]
```

### Read the dataset

```python
import icechunk
import xarray as xr
from modis_snow_phenology.config import Config

config = Config("config/config_with_secrets_v1.txt")
repo = config.open_icechunk_repo()
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, zarr_format=3, consolidated=False)
# Dimensions: (water_year: 11, y: 43200, x: 86400)
# Data vars:  SAD_DOWY, SDD_DOWY, max_consec_snow_days
```

### Adding new water years (hemisphere-aware)

A water year closes at different times in each hemisphere (northern WY N:
Sep 30 N; southern WY N: Mar 31 N+1), so one hemisphere is ready to process
~6 months before the other. The pipeline handles the stagger automatically:

1. **Reminder** — the **Water Year Watch** workflow (monthly cron) opens a
   GitHub issue the moment a (water year, hemisphere) passes its season end +
   `TRAILING_BUFFER_DAYS`, with the exact checklist. One issue per
   hemisphere-year, ever (deduplicated by title, open or closed).
2. **Extend the store** (once per new water year; cheap, metadata-only):

   ```bash
   # after bumping WY_END in the config
   pixi run python processing/scripts/extend_store_water_years.py \
       --config-file config/config_with_secrets_v1.txt
   ```

3. **Dispatch** — trigger **Process All Tiles** (`which_tiles = missing`).
   Only tiles whose hemisphere-season has closed pick up the new year; the
   other hemisphere's tiles follow automatically on a later dispatch, with no
   manual coordination. The eligibility gate exists in both the dispatcher
   (`status.get_remaining_work`) and the processor (`--water-years` veto), so
   a half-elapsed season can never be committed.
4. **Downstream** — rebuild the multiscale pyramid
   (`map/create_zarr_multiscales.ipynb`), regenerate the map status geojson,
   add the year to `WATER_YEARS` in `map/lib/store.ts`, and let the
   snowmelt-runoff repo's own watcher pick it up from there.

Why the buffer: the cloud filling is bidirectional (ffill **and** bfill), so
end-of-season cloudy pixels need post-season observations to resolve; the
fetch window already spans ±1 water year and degrades gracefully, and ~90
days of trailing data keeps the bfill context healthy without waiting a full
extra year.

For local development, copy `config/config_v1.txt` to `config/config_with_secrets_v1.txt` and replace the `ENV` placeholders with real credentials (that file is gitignored).

## Web Map

A Next.js static map (`map/`) is deployed to GitHub Pages via `deploy-map.yml` on every push to `main` that touches `map/`, `modis_snow_phenology/`, or `processing/tile_list.geojson`.

The map renders the dataset as a slippy map using [zarr-layer](https://github.com/egagli/zarr-layer) (egagli fork with MODIS sinusoidal worldFraction fix). GPU-side reprojection means the store stays in native MODIS sinusoidal — no Web Mercator reproject needed.

### Multiscale pyramid

Before deploying the map, build the multiscale Zarr pyramid using `map/create_zarr_multiscales.ipynb`:

1. Opens the Icechunk store and loads the full dataset.
2. Uses `topozarr` to create a 6-level pyramid (2× coarsening per level, level 0 = native 86 400 × 43 200, level 5 = ~1350 × 2700).
3. Encodes all levels as `int16` with `_FillValue = -32768`.
4. Writes to plain Zarr v3 on Azure (not Icechunk — no versioning needed for a derived product) via `obstore.store.AzureStore` + `zarr.storage.ObjectStore`.
5. Sets `Cache-Control: public, max-age=31536000` on all blobs for browser/CDN caching.

The `deploy-map.yml` workflow:

1. Runs `map/generate_tiles_status_geojson.py` to query Icechunk and write `map/public/tiles-status.geojson` showing processing status per tile.
2. Reads `MULTISCALE_PREFIX` from the config file and injects `NEXT_PUBLIC_ZARR_URL` into the Next.js build.
3. Builds the Next.js static export and deploys it to GitHub Pages.

## Repository Structure

```text
MODIS_snow_phenology/
├── modis_snow_phenology/           Python package
│   ├── processing.py               Core algorithms (fetch, polar night, cloud-fill, metrics)
│   └── config.py                   Config loader + Icechunk helpers + tile status
├── notebooks/
│   ├── 01_initialize.ipynb         Tile list creation + Icechunk store initialization
│   ├── 02_compare_with_v1.ipynb    Comparison notebook for validating reprocessed data
│   └── download_dataset.ipynb      Export the store to a plain Zarr v3 .zip for Zenodo
├── dataset/                        Exported Zarr store + .zip staged for Zenodo (gitignored)
├── zenodo/
│   └── zenodo_metadata.md          Record metadata + upload checklist for publication
├── processing/
│   ├── scripts/
│   │   ├── process_single_tile.py  Tile processor (fetch → cloud-fill → metrics → commit)
│   │   └── get_tiles_for_batch.py  GH Actions matrix generator (single batch or batch-of-batches)
│   └── tile_list.geojson           Static tile registry (648 tiles, sinusoidal CRS)
├── config/
│   ├── config_v1.txt               Config template (credentials use ENV placeholder)
│   └── config_with_secrets_v1.txt  Local config with real credentials (gitignored)
├── map/
│   ├── create_zarr_multiscales.ipynb        Build multiscale pyramid for web map
│   ├── generate_tiles_status_geojson.py     Generate tiles-status.geojson from Icechunk history
│   ├── components/                          React components (map, sidebar, floating cards)
│   ├── pages/                               Next.js pages
│   └── pixi.toml                            Separate pixi env for map Python tooling
├── .github/workflows/
│   ├── process_all_tiles.yml       Manually triggered batch processor (batch-of-batches)
│   ├── process_batch.yml           Reusable workflow: fan out one batch of ≤256 tile jobs
│   ├── process_single_tile.yml     Manually triggered single-tile processor
│   └── deploy-map.yml              Build + deploy Next.js map to GitHub Pages
├── CITATION.cff                    Citation metadata
├── pixi.toml                       Environment definition
└── pyproject.toml                  Package metadata (hatchling build)
```

## Secrets Required (GitHub Actions)

| Secret | Description |
| --- | --- |
| `AZURE_STORAGE_SAS_TOKEN` | SAS token with read/write access to the `snowmelt` container on `uwcryo` |
| `EARTHDATA_USERNAME` | NASA Earthdata login username (for MOD10A2 HDF4 download via earthaccess) |
| `EARTHDATA_PASSWORD` | NASA Earthdata login password |

`AZURE_STORAGE_ACCOUNT` and `AZURE_CONTAINER` are not secrets — they are hardcoded in `config/config_v1.txt` (`uwcryo` and `snowmelt`).
