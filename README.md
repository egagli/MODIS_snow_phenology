# MODIS Snow Phenology

Global snow phenology dataset derived from MODIS MOD10A2 8-day maximum snow extent (water years 2015–2024).

Three variables per pixel per water year:

- **SAD_DOWY** — Snow Appearance Date (day of water year)
- **SDD_DOWY** — Snow Disappearance Date (day of water year; defined as first day with no snow)
- **max_consec_snow_days** — Length of the longest continuous snow period (days)

Cloud filling follows [Wrzesien et al. 2019](https://doi.org/10.1029/2018WR023453). No-data pixels use fill value `int16` minimum (`-32768`).

## Dataset

| Property | Value |
| --- | --- |
| Source | MODIS MOD10A2.061 (8-day maximum snow extent) via NASA Earthdata |
| Spatial resolution | 500 m |
| Grid | MODIS sinusoidal, 86 400 x 43 200 pixels (global) |
| Temporal coverage | Water years 2015–2024 (10 years) |
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

> **Note:** Processing previously fetched data from Microsoft Planetary Computer via STAC. PC stopped archiving MOD10A2 in mid-2025 when MODIS Terra was decommissioned (Nov 2024). The old implementation is preserved as `_get_modis_MOD10A2_max_snow_extent_planetary_computer` for reference.

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

All water years for the tile are written together in a single Icechunk commit via `xr.Dataset.to_zarr(..., region='auto', mode='r+')`. Tile coordinates are snapped to the store's exact grid coordinates to prevent float-precision mismatches. Concurrent parallel tile jobs are handled via `icechunk.ConflictDetector` with randomized retry backoff — since each job writes to a disjoint tile region, there are never real data conflicts and Icechunk rebases automatically.

The commit message encodes per-water-year statistics:

```text
Tile(h=10, v=4) processed. Stats: [(WY2015: input_obs=46, valid_pixels=1234567, coverage=21.4%), ...] Special note: None
```

Tiles with no MODIS input for any water year commit an empty snapshot with `Special note: No input data found...`.

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

- Queries Icechunk commit history to find unprocessed tiles
- Fans out up to 256 tiles per batch; batches run sequentially, tiles within a batch run in parallel
- Re-entrant: already-processed and nodata tiles are automatically skipped

To reprocess everything: trigger with `which_tiles = all` (modify `process_all_tiles.yml`).

To process a single tile manually: trigger **Process Single Tile** with `h` and `v` inputs.

### Monitor progress

Processing status is derived at runtime from the Icechunk commit history:

```python
import icechunk
import xarray as xr
import sys
sys.path.insert(0, "/path/to/MODIS_snow_phenology")
from modis_snow_phenology.config import Config, get_processing_status_gdf

config = Config("config/config_with_secrets_v1.txt")
repo = config.open_icechunk_repo()
gdf = get_processing_status_gdf(repo, config.TILE_LIST_PATH, config.years)
gdf["processing_status"].value_counts()
# processed / nodata / unprocessed / skip
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
# Dimensions: (water_year: 10, y: 43200, x: 86400)
# Data vars:  SAD_DOWY, SDD_DOWY, max_consec_snow_days
```

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
│   └── 02_compare_with_v1.ipynb    Comparison notebook for validating reprocessed data
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
