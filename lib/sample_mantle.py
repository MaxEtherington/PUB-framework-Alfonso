"""Functions to sample mantle data from G-ADOPT
output grids and join to point data.
"""
import numpy as np
import pandas as pd
import xarray as xr
import gplately as gplt

from numpy.typing import ArrayLike
from .misc import _PathLike

# Non-dimensionalisation offset used by G-ADOPT: surface radius in Earth radii.
_GADOPT_SURFACE_R = 2.208

DEFAULT_DEPTHS: tuple[int, ...] = (np.arange(100, 2900, 100))  # km
MANTLE_FIELDS: tuple[str, ...] = (
    "FullTemperature_CG",
    "Pressure",
    "Radial_Velocity",
    "Temperature_CG",
    "Temperature_Deviation_CG",
    "Velocity_x",
    "Velocity_y",
    "Velocity_z",
    "Viscosity_CG",
)


def _to_km(depths: ArrayLike) -> np.ndarray:
    """Convert G-ADOPT nondimensionalised depths to km below surface."""
    return (_GADOPT_SURFACE_R - np.asarray(depths, dtype=float)) * gplt.EARTH_RADIUS


def _to_nondim(depths_km: ArrayLike) -> np.ndarray:
    """Convert depths in km below surface to G-ADOPT nondimensionalised depths."""
    return _GADOPT_SURFACE_R - (np.asarray(depths_km, dtype=float) / gplt.EARTH_RADIUS)


def _match_longitude_convention(ds: xr.Dataset, lons: ArrayLike) -> np.ndarray:
    """Map longitudes to match the dataset longitude convention."""
    lons_arr = np.asarray(lons, dtype=float)

    # Use finite values only to detect whether the dataset uses 0..360 or -180..180.
    ds_lons = np.asarray(ds["lon"].values, dtype=float)
    ds_lons = ds_lons[np.isfinite(ds_lons)]
    if ds_lons.size == 0:
        return lons_arr

    ds_min = float(np.min(ds_lons))
    ds_max = float(np.max(ds_lons))

    # Common G-ADOPT convention: [0, 360].
    if ds_min >= 0.0 and ds_max > 180.0:
        return np.mod(lons_arr, 360.0)

    # Common geospatial convention: [-180, 180].
    if ds_min < 0.0 and ds_max <= 180.0:
        return ((lons_arr + 180.0) % 360.0) - 180.0
    
    if ds_min < 0.0 and ds_max > 180.0:
        raise ValueError(f"Cannot determine dataset longitude convention from range [{ds_min:.2f}, {ds_max:.2f}].")

    return lons_arr


def _sample_mantle(
    ds: xr.Dataset,
    var: str,
    lons: ArrayLike,
    lats: ArrayLike,
    times: ArrayLike,
    depths: ArrayLike,
    method: str = 'linear',
) -> np.ndarray:
    """Sample a mantle variable at a series of (lon, lat, time) points.

    `lons`, `lats`, and `times` are point-wise (index i of each corresponds to
    the same point). `depths` is orthogonal — every depth is sampled at every
    point.

    Parameters
    ----------
    ds : xr.Dataset
        Time-indexed mantle dataset.
    var : str
        Variable name to sample (e.g. 'Temperature_Deviation_CG').
    lons, lats, times : array-like, shape (n_points,)
        Coordinates of each sample point.
    depths : scalar or array-like
        Single depth → shape (n_points,).
        Array → shape (n_points, n_depths).
    method : {'linear', 'nearest'}, optional
        Interpolation method. Default 'linear'.

    Returns
    -------
    np.ndarray, shape (n_points,) or (n_points, n_depths)
    """
    lons = _match_longitude_convention(ds=ds, lons=lons)

    result = ds[var].interp(
        lon  =xr.DataArray(lons, dims='points'),
        lat  =xr.DataArray(np.asarray(lats,   dtype=float), dims='points'),
        time =xr.DataArray(np.asarray(times,  dtype=float), dims='points'),
        depth=np.asarray(depths, dtype=float),
        method=method,
    ).values

    if np.isnan(np.sum(result)):
        raise ValueError("NaN values found in sampled mantle data. Check that all points are within the dataset bounds.")
    
    return result


def extract_basic_mantle_features(
    mantle_dir: _PathLike,
    points: pd.DataFrame,
    depths_km: ArrayLike = DEFAULT_DEPTHS,
    mantle_fields: ArrayLike = MANTLE_FIELDS,
) -> pd.DataFrame:
    """Extract basic mantle features at a series of labelled points.

    Parameters
    ----------
    mantle_dir : path-like
        Directory containing G-ADOPT output files.
    points : DataFrame
        Must contain columns 'lon', 'lat', and 'age (Ma)'.
    depths_km : array-like, optional
        Depths in km below surface. Default is `DEFAULT_DEPTHS` (100, 200, ..., 2800 km).

    Returns
    -------
    DataFrame
        Copy of `points` with additional columns for each sampled mantle
        variable and depth (e.g. 'Temperature_Deviation_CG_100km').
    """
    depths_km = list(depths_km)
    depths_nondim = _to_nondim(depths_km)
    out = points.copy()
    new_cols = {}

    with xr.open_mfdataset(
        sorted(mantle_dir.glob("*.nc")),
        combine='nested',
        concat_dim='time',
    ) as ds:
        for var in mantle_fields:
            sampled = _sample_mantle(
                ds=ds,
                var=var,
                lons=points['lon'],
                lats=points['lat'],
                times=points['age (Ma)'],
                depths=depths_nondim,
            )
            for i, depth_km in enumerate(depths_km):
                new_cols[f"{var}_{int(depth_km)}km"] = sampled[:, i]

    # Join all new columns at once to avoid DataFrame fragmentation
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out

def calculate_velocity_magnitude(ds: xr.Dataset) -> xr.DataArray:
    """Calculate velocity magnitude from velocity components."""
    vx = ds['Velocity_x']
    vy = ds['Velocity_y']
    vz = ds['Velocity_z']
    return np.sqrt(vx**2 + vy**2 + vz**2)

def calculate_tangential_velocity_magnitude(ds: xr.Dataset) -> xr.DataArray:
    """Calculate tangential velocity magnitude from velocity components."""
    vx = ds['Velocity_x']
    vy = ds['Velocity_y']
    vz = ds['Velocity_z']
    vr = ds['Radial_Velocity']
    return np.sqrt(vx**2 + vy**2 + vz**2 - vr**2) # u_tangential = sqrt(u^2 - u_radial^2)

def calculate_radial_tangential_ratio(ds: xr.Dataset) -> xr.DataArray:
    """Calculate radial-to-tangential velocity ratio."""
    vr = ds['Radial_Velocity']
    vt = calculate_tangential_velocity_magnitude(ds)
    return np.abs(vr) / vt