# MODIS Snow Phenology

Global snow phenology dataset derived from MODIS MOD10A2 8-day maximum snow extent (2015–2024).

Three variables per pixel per water year:
- **SAD_DOWY** — Snow Appearance Date (day of water year)
- **SDD_DOWY** — Snow Disappearance Date (day of water year)
- **max_consec_snow_days** — Length of longest continuous snow period (days)

Cloud filling follows [Wrzesien et al. 2019](https://doi.org/10.1029/2018WR023453).

## Architecture

| Component | Technology |
|-----------|-----------|
| Storage | Icechunk on Azure Blob Storage |
| Format | Zarr v3 + ShardingCodec (shard: 1×2400×2400, inner chunk: 1×600×600) |
| CRS | MODIS sinusoidal (`spatial_ref` + `grid_mapping`, GeoZarr convention) |
| Package management | [pixi](https://pixi.sh) |
| Processing | GitHub Actions (tile-parallel matrix) |

## Getting Started

### Install environment

```bash
pixi install
pixi run notebook   # launches JupyterLab
```

### Initialize (once)

Run `notebooks/01_initialize.ipynb`:
1. **Phase 1**: Creates `processing/tile_data/tile_processing_status.geojson` (648 tiles, WGS84 geometries)
2. **Phase 2**: Creates the Icechunk store on Azure (requires `AZURE_STORAGE_ACCOUNT` + `AZURE_STORAGE_SAS_TOKEN`)

Edit the GeoJSON to mark any tiles as `"skip"` before batch processing.

### Process tiles

Trigger the **Process Batch** workflow in GitHub Actions:
- `which_tiles`: `unprocessed_and_failed` (default) | `unprocessed` | `failed` | `all`

Up to 243 land tiles run in parallel. After all tiles finish, a `summarize-results` job updates `tile_processing_status.geojson` from Icechunk commit history and GitHub Actions job logs.

### Monitor progress

The tile status is always visible in `processing/tile_data/tile_processing_status.geojson`.
Load it with geopandas to inspect or plot:

```python
import geopandas as gpd
gdf = gpd.read_file("processing/tile_data/tile_processing_status.geojson")
gdf["processing_status"].value_counts()
```

### Read the dataset

```python
import icechunk
import xarray as xr
from modis_snow_phenology.config import Config

config = Config()
repo = config.open_repo()
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store(), zarr_format=3, consolidated=False)
```

## Repository Structure

```
MODIS_snow_phenology/
├── modis_snow_phenology/       Python package
│   ├── masking.py              Snow algorithm (MOD10A2 fetch, cloud fill, metrics)
│   └── config.py               Config (Azure creds, Icechunk helpers, tile status)
├── notebooks/
│   └── 01_initialize.ipynb     Tile selection + Icechunk store creation
├── processing/
│   ├── scripts/
│   │   ├── process_single_tile.py    Core tile processor
│   │   ├── get_tiles_for_batch.py    GH Actions matrix generator
│   │   └── summarize_processing.py   Post-batch status updater
│   └── tile_data/
│       └── tile_processing_status.geojson
└── .github/workflows/
    ├── process_single_tile.yml   Manual single-tile processing
    ├── process_batch.yml         Parallel batch processing + summarize
    └── summarize_results.yml     Standalone status update
```

## Secrets Required (GitHub Actions)

| Secret | Description |
|--------|-------------|
| `AZURE_STORAGE_ACCOUNT` | Azure storage account name (e.g. `uwcryo`) |
| `AZURE_STORAGE_SAS_TOKEN` | SAS token with read/write access to the `snowmelt` container |
