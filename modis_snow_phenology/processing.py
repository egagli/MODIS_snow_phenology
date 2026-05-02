"""
Library of functions to create MODIS seasonal snow masks, snow appearance date, and snow disappearance date.

Author: Eric Gagliano (egagli@uw.edu)
Created: 04/2024
"""

import re
import tempfile
import time
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import rasterio
import pystac_client
import planetary_computer
import odc.stac
import xarray as xr
import rioxarray as rxr  # noqa: F401 — registers .rio accessor
#import numba


def _rss_mb() -> int:
    """Return current resident set size in MB (Linux only, no extra deps)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def get_modis_MOD10A2_max_snow_extent(
    vertical_tile, horizontal_tile, start_date, end_date,
    chunks={"time": -1, "x": 240, "y": 240},
):
    """Fetch MOD10A2 Maximum_Snow_Extent via earthaccess (NASA Earthdata HDF4 files).

    Replaced Planetary Computer STAC + odc.stac; see
    _get_modis_MOD10A2_max_snow_extent_planetary_computer for the old implementation.
    Reason: PC stopped archiving MOD10A2 ~June 2025 (MODIS Terra decommissioned Nov 2024).

    To revert to Planetary Computer:
      1. Rename this function to _get_modis_MOD10A2_max_snow_extent_earthaccess
      2. Rename _get_modis_MOD10A2_max_snow_extent_planetary_computer back to
         get_modis_MOD10A2_max_snow_extent
      3. Remove EARTHDATA_USERNAME/EARTHDATA_PASSWORD from workflow env blocks
         (AZURE_STORAGE_SAS_TOKEN is still present for Icechunk — unchanged)
      4. pystac-client, planetary-computer, odc-stac remain in pixi.toml — no lock regen needed
    """
    import logging as _logging
    import time as _time
    _log_auth = _logging.getLogger(__name__)

    # Login with username/password. Retry to handle transient ENETUNREACH on
    # some GitHub Actions runners (IPv6 routing failures to urs.earthdata.nasa.gov).
    for _attempt in range(5):
        try:
            earthaccess.login(strategy="environment")
            break
        except Exception as _exc:
            if _attempt == 4:
                raise
            _log_auth.warning(
                f"earthaccess login attempt {_attempt + 1} failed ({_exc}), "
                f"retrying in {2 ** _attempt}s..."
            )
            _time.sleep(2 ** _attempt)

    _auth_session = earthaccess.get_requests_https_session()

    tile_id = f"h{horizontal_tile:02d}v{vertical_tile:02d}"
    results = earthaccess.search_data(
        short_name="MOD10A2",
        temporal=(start_date, end_date),
        granule_name=f"MOD10A2.A*.{tile_id}.*",
    )

    if len(results) == 0:
        raise ValueError(
            f"No earthaccess granules found for {tile_id} {start_date}–{end_date}"
        )

    _log_auth.info(f"earthaccess: building authenticated HTTPS session (RSS={_rss_mb()} MB)...")
    _log = _log_auth
    session = _auth_session

    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = tmpdir_obj.name
    try:
        _log.info(f"earthaccess: downloading {len(results)} granules (RSS={_rss_mb()} MB)...")
        files = _download_granules_sequential(results, tmpdir, session)
        _log.info(f"earthaccess: opening stack (RSS={_rss_mb()} MB)...")
        da = _open_mod10a2_stack(files)
        _log.info(f"earthaccess: stack ready, shape={da.shape}, RSS={_rss_mb()} MB")
    finally:
        _log.info(f"earthaccess: cleaning up tmpdir (RSS={_rss_mb()} MB)...")
        tmpdir_obj.cleanup()
        _log.info(f"earthaccess: tmpdir cleaned up (RSS={_rss_mb()} MB)")

    return da


def _download_granules_sequential(granules, local_path, session):
    """Download earthaccess granules one at a time using a plain requests session.

    Replaces earthaccess.download() which uses pqdm internally. pqdm triggers
    Python's multiprocessing.resource_tracker even with threads=1, causing a
    semaphore leak and "operation canceled" crash on Python 3.14 + GitHub Actions.
    """
    import logging as _logging
    import time as _time
    _log = _logging.getLogger(__name__)

    local_path = Path(local_path)
    files = []
    for i, granule in enumerate(granules, 1):
        url = granule.data_links()[0]
        filename = local_path / url.split("/")[-1]
        _log.info(f"  [{i}/{len(granules)}] GET {filename.name}")
        for attempt in range(5):
            if attempt:
                _time.sleep(2 ** attempt)  # 2, 4, 8, 16 s
            try:
                # timeout=(connect_s, read_s): fail fast on hung connections.
                with session.get(url, stream=True, timeout=(30, 300)) as r:
                    r.raise_for_status()
                    with open(filename, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                break  # success
            except Exception as exc:
                if attempt == 4:
                    raise
                _log.warning(f"  [{i}/{len(granules)}] attempt {attempt + 1} failed ({exc}), retrying...")
                filename.unlink(missing_ok=True)
        files.append(str(filename))
    return files


def _open_mod10a2_stack(files):
    sorted_files = sorted(f for f in files if f.endswith(".hdf"))
    dates = [pd.Timestamp(_parse_modis_date(Path(f))) for f in sorted_files]

    # Use rasterio directly (not rioxarray) so each file is opened and closed
    # via a proper context manager — rioxarray's lazy DataArray cleanup caused
    # intermittent GDAL segfaults when reading 100+ HDF4 files sequentially.
    hdf4_path = lambda f: f'HDF4_EOS:EOS_GRID:"{f}":MOD_Grid_Snow_500m:Maximum_Snow_Extent'

    # Read first file to determine shape, dtype, and cell-center coordinates.
    with rasterio.open(hdf4_path(sorted_files[0])) as src:
        ny, nx = src.height, src.width
        t = src.transform
        x_coords = t.c + t.a * (np.arange(nx) + 0.5)
        y_coords = t.f + t.e * (np.arange(ny) + 0.5)
        data = np.empty((len(sorted_files), ny, nx), dtype=src.dtypes[0])
        data[0] = src.read(1)

    for i, f in enumerate(sorted_files[1:], start=1):
        with rasterio.open(hdf4_path(f)) as src:
            data[i] = src.read(1)

    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": dates, "y": y_coords, "x": x_coords},
    ).sortby("time")


def _parse_modis_date(filepath: Path):
    m = re.search(r"\.A(\d{4})(\d{3})\.", filepath.name)
    year, doy = int(m.group(1)), int(m.group(2))
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy - 1)


# DEPRECATED — kept for reference and easy revert.
# Stopped working ~June 2025: Planetary Computer no longer archives MOD10A2
# (MODIS Terra decommissioned Nov 2024). PC STAC returns 0 items for queries
# past the archive cutoff, causing odc.stac.load to fail with
# "Failed to auto-guess CRS/resolution."
def _get_modis_MOD10A2_max_snow_extent_planetary_computer(
    vertical_tile, horizontal_tile, start_date, end_date, chunks={"time": -1, "x": 240, "y": 240},
    max_retries=5,
):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=[f"modis-10A2-061"],
        datetime=(start_date, end_date),
        query={
            "modis:vertical-tile": {"eq": vertical_tile},
            "modis:horizontal-tile": {"eq": horizontal_tile},
        },
    )

    for attempt in range(1, max_retries + 1):
        try:
            items = search.item_collection()
            break
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"STAC search attempt {attempt}/{max_retries} failed ({e}); retrying in {wait}s...")
            time.sleep(wait)

    if len(items) == 0:
        raise ValueError(
            f"No STAC items found for h{horizontal_tile:02d}v{vertical_tile:02d} "
            f"{start_date} – {end_date}. Collection may not cover this date range."
        )

    modis_snow = odc.stac.load(
        items=items, bands="Maximum_Snow_Extent", chunks=chunks
    )["Maximum_Snow_Extent"]

    return modis_snow


def get_modis_MOD10A2_full_grid():
    start_date="2020-09-22"
    end_date="2020-09-22"

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=[f"modis-10A2-061"],
        datetime=(start_date, end_date),
    )

    load_params = {
        "items": search.item_collection(),
        "bands": "Maximum_Snow_Extent",
        "chunks": {},
    }

    modis_grid = odc.stac.load(**load_params).to_dataarray(dim='Maximum_Snow_Extent') #["Maximum_Snow_Extent"]

    return modis_grid

fill_value = np.iinfo(np.int16).min


def binarize_with_cloud_filling(da):
    """
    Binarize the MODIS DataArray with cloud filling.

    This function implements a cloud filling approach similar to the one described in Wrzesien et al. 2019.
    It assumes that if two snow-covered MOD10A2 observations bracket one or more cloudy
    MOD10A2 observations, the cloudy period is likely snow covered, too. Therefore, only
    the first and last 8-day MOD10A2 observations need to be snow covered. If there is one
    snowy MOD10A2 observation, five cloudy MOD10A2 periods, and one snowy observation,
    the 56-day period is classified as snow covered.

    This function should be run on the entire time series (not per water year groupby group) for continuity between water years. For example, let's say Dec 25 snow, Jan 2nd clouds, Jan 10 snow. If we groupby water year first, Jan 2nd would not be correctly identified as snow.

    Parameters:
    da (xarray.DataArray): The input MODIS MOD10A2 8 day DataArray.

    Returns:
    xarray.DataArray: The binarized DataArray (0: no snow, 1: snow), where 1 can be either
    snow or cloud(s) bracketed by snow.
    """
    SNOW_VALUE = 200
    CLOUD_VALUE = 50
    NO_SNOW_VALUE = 25
    DARKNESS_VALUE = 11
    NO_DECISION_VALUE = 1
    FILL_VALUE = 255

    # Work directly in uint8 numpy to avoid float64 promotion.
    # xarray's da.where(cond) with no `other` fills masked positions with NaN,
    # converting uint8 → float64 (8×). For a 138×2400×2400 tile that's ~6 GB
    # per temporary array; two of them (ffill + bfill) exceed GH Actions RAM.
    vals = da.values.copy()  # (T, H, W) uint8

    # Remap all "uncertain" codes to CLOUD_VALUE in-place (stays uint8)
    uncertain = (vals == DARKNESS_VALUE) | (vals == FILL_VALUE) | (vals == NO_DECISION_VALUE)
    vals[uncertain] = CLOUD_VALUE
    del uncertain

    cloud = vals == CLOUD_VALUE  # (T, H, W) bool

    # Forward fill: carry last non-cloud value forward over cloud gaps
    ff = vals.copy()
    for t in range(1, ff.shape[0]):
        np.copyto(ff[t], ff[t - 1], where=cloud[t])

    # Backward fill: carry next non-cloud value backward over cloud gaps
    bf = vals.copy()
    for t in range(bf.shape[0] - 2, -1, -1):
        np.copyto(bf[t], bf[t + 1], where=cloud[t])

    del cloud, vals

    effective_snow = (ff == SNOW_VALUE) & (bf == SNOW_VALUE)
    del ff, bf

    result = da.copy(data=effective_snow)
    if da.rio.crs is not None:
        result = result.rio.write_crs(da.rio.crs)
    return result


def get_longest_consec_stretch(arr):
    """
    Finds the longest consecutive stretch of snow days in a given array.

    This function iterates over the input array and finds the longest stretch of
    consecutive days where the value is True (indicating snow). It returns the start
    and end indices (end+1) of this stretch, as well as its length.

    Parameters:
    arr (list or array-like): The input array. Each element should be a boolean
    indicating whether there is snow on that day (True) or not (False).

    Returns:
    tuple: A tuple containing three elements:
        - The start index of the longest consecutive stretch of snow days.
        - The end index (+1) of the longest consecutive stretch of snow days.
        - The length of the longest consecutive stretch of snow days.
    """
    max_len = 0
    max_start = 0
    max_end = 0
    current_start = None
    for i, val in enumerate(arr):
        if val:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                length = i - current_start
                if length >= max_len:
                    max_len = length
                    max_start = current_start
                    max_end = i
                current_start = None
    if current_start is not None:
        length = len(arr) - current_start
        if length > max_len: # purposefully changed from >= to > to avoid including the last day as a SAD (avoid case where SAD=365,SDD=366,max_consec=1)
            max_len = length
            max_start = current_start
            max_end = len(arr) # used to be max_end = len(arr) -1, changed to include the last day in the stretch

    if max_len == 0:
        return fill_value, fill_value, fill_value
    return max_start, max_end, max_len

from numba import jit

@jit(nopython=True)
def get_longest_consec_stretch_vectorized(arr):
    """
    Optimized version using numba for speed.
    """
    n = len(arr)
    if n == 0:
        return fill_value, fill_value, fill_value
    
    max_len = 0
    max_start = 0
    max_end = 0
    current_start = -1
    
    for i in range(n):
        if arr[i]:  # Snow day
            if current_start == -1:
                current_start = i
        else:  # No snow day
            if current_start != -1:
                length = i - current_start
                if length >= max_len:
                    max_len = length
                    max_start = current_start
                    max_end = i
                current_start = -1
    
    # Handle case where snow period extends to end
    if current_start != -1:
        length = n - current_start
        if length > max_len:
            max_len = length
            max_start = current_start
            max_end = n
    
    if max_len == 0:
        return fill_value, fill_value, fill_value
    
    return max_start, max_end, max_len

def map_DOWY_values(value, substitution_dict):
    """
    Maps the input values based on a predefined substitution dictionary.

    This function uses the 'np.vectorize' function to apply the 'get' method of the
    'substitution_dict' dictionary to the input values. The 'get' method returns the
    value for each key in the dictionary. If a key is not found in the dictionary,
    it returns None.

    Parameters:
    value (array-like): The input values to be mapped.

    Returns:
    numpy.ndarray: An array with the mapped values.
    """
    # return np.vectorize(substitution_dict.get)(value)
    return np.vectorize(
        lambda x: substitution_dict.get(x, fill_value), otypes=[np.int16]
    )(value)

def align_wy_start(da,hemisphere='northern'):
    """This function should operate on da and duplicate the last observation of each previous water year and relabel it as the start of the new water year, and then sort everything"""

    # Get unique water years
    water_years = da.water_year.values

    values, counts = np.unique(water_years, return_counts=True)

    # only inclue water years that have at least three observations
    valid_water_years = values[counts >= 5]
    
    # Create a new DataArray to hold the modified data
    new_data = []
    
    wy_vals = da.water_year.values  # 1-D array of water year per time step

    for wy in np.unique(valid_water_years):
        # Use isel+numpy mask — da.where(cond, drop=True) on a bool array promotes
        # bool → float64 (8×) before dropping masked time steps, causing OOM.
        prev_mask = wy_vals == (wy - 1)
        if not prev_mask.any():
            print(f"Warning: No last observation of water year {wy-1}. This will affect calculation of water year {wy}, as the earliest possible snow appearance date will be DOWY 7 or 8. Skipping.")
            continue
        last_obs = da.isel(time=prev_mask).isel(time=-1)

        # Create a new observation for the start of the next water year
        new_obs = last_obs.copy()

        if hemisphere == 'northern':
            first_date_of_water_year = pd.to_datetime(f"{wy-1}-10-01")
        if hemisphere == 'southern':
            first_date_of_water_year = pd.to_datetime(f"{wy}-04-01")

        new_obs['time'] = first_date_of_water_year  # Set to October 1st of the next water year
        new_obs['water_year'] = wy
        new_obs['DOWY'] = 1

        # Append the original and new observations
        new_data.append(da.isel(time=(wy_vals == wy)))
        new_data.append(new_obs)

    # Concatenate all observations into a single DataArray
    combined_da = xr.concat(new_data, dim='time')
    new_data.clear()  # free per-WY slices immediately — they're now in combined_da

    # Sort by time
    combined_da = combined_da.sortby('time')

    # Filter to valid water years — use isel to avoid bool→float64 promotion
    keep_mask = np.isin(combined_da.water_year.values, valid_water_years)
    combined_da = combined_da.isel(time=keep_mask)
    combined_da = combined_da.astype(np.int16)

    
    return combined_da


def get_max_consec_snow_days_SAD_SDD_one_WY(effective_snow_da):
    """
    Calculates the maximum consecutive snow days, snow appearance day (SAD), and snow disappearance day (SDD) per water year.

    This function applies the 'get_longest_consec_stretch' function along the time dimension of the input DataArray to count
    consecutive snow days. It then maps the start and end days of the longest stretch of snow days to the 'DOWY' (Day of Water Year)
    coordinate of the input DataArray. The function returns a Dataset with three variables: 'SAD_DOWY', 'SDD_DOWY', and
    'max_consec_snow_days'.

    Parameters:
    effective_snow_da (xarray.DataArray): The input DataArray with effective snow data.

    Returns:
    xarray.Dataset: A Dataset with the following variables:
        - 'SAD_DOWY': The snow appearance day (SAD) for each water year, represented as a DOWY.
        - 'SDD_DOWY': The snow disappearance day (SDD) for each water year, represented as a DOWY. We say SDD is the first day with NO snow.
        - 'max_consec_snow_days': The maximum number of consecutive snow days for each water year.
    """

    # Apply function along the time dimension using the effective snow data to count consecutive snow days
    results = xr.apply_ufunc(
        get_longest_consec_stretch_vectorized,
        effective_snow_da,
        input_core_dims=[["time"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={'allow_rechunk':True},
        output_dtypes=[np.int16, np.int16, np.int16],
    )

    substitution_dict = {
        index: value for index, value in enumerate(effective_snow_da.DOWY.values)
    }

    # add entry to substitution_dict for if at end of water year. SDD should be set to 366 for non-leap years and 367 for leap years.

    if effective_snow_da.time.dt.is_leap_year.any():
        last_dowy = 367
    else:
        last_dowy = 366

    substitution_dict[len(effective_snow_da.DOWY)] = last_dowy

    snow_start_DOWY = xr.apply_ufunc(
        map_DOWY_values,
        results[0],
        kwargs={"substitution_dict": substitution_dict},
        vectorize=True,
        dask="parallelized",
    )

    snow_end_DOWY = xr.apply_ufunc(
        map_DOWY_values,
        results[1],
        kwargs={"substitution_dict": substitution_dict},
        vectorize=True,
        dask="parallelized",
    )

    # if snow appearance date is last date

    snow_mask = xr.Dataset(
        {
            "SAD_DOWY": snow_start_DOWY,
            "SDD_DOWY": snow_end_DOWY,
            "max_consec_snow_days": snow_end_DOWY - snow_start_DOWY,
        }
    )

    snow_mask["max_consec_snow_days"] = snow_mask["max_consec_snow_days"].where(
        snow_mask["max_consec_snow_days"] > 0, fill_value
    )

    for var in snow_mask:
        snow_mask[var].rio.write_nodata(fill_value, encoded=False, inplace=True)

    return snow_mask


