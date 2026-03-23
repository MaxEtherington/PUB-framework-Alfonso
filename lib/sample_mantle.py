"""Functions to sample mantle data from G-ADOPT
output grids and join to point data.
"""
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import xarray as xr
import gplately as gplt

from .misc import _PathLike

# Non-dimensionalisation offset used by G-ADOPT: surface radius in Earth radii.
_GADOPT_SURFACE_R = 2.208

DEFAULT_DEPTHS: tuple[int, ...] = (100, 200, 300, 400, 500, 600)  # km
MANTLE_FIELDS: tuple[str, ...] = ('Temperature_Deviation_CG',)


def _to_km(depths: ArrayLike) -> np.ndarray:
    """Convert G-ADOPT nondimensionalised depths to km below surface."""
    return (_GADOPT_SURFACE_R - np.asarray(depths, dtype=float)) * gplt.EARTH_RADIUS


def _to_nondim(depths_km: ArrayLike) -> np.ndarray:
    """Convert depths in km below surface to G-ADOPT nondimensionalised depths."""
    return _GADOPT_SURFACE_R - (np.asarray(depths_km, dtype=float) / gplt.EARTH_RADIUS)


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
    result = ds[var].interp(
        time =xr.DataArray(np.asarray(times,  dtype=float), dims='points'),
        lon  =xr.DataArray(np.asarray(lons,   dtype=float), dims='points'),
        lat  =xr.DataArray(np.asarray(lats,   dtype=float), dims='points'),
        depth=np.asarray(depths, dtype=float),
        method=method,
    ).values

    if np.ndim(depths) > 0:
        return result.T  # (n_depths, n_points) → (n_points, n_depths)
    return result.flatten()


def extract_basic_mantle_features(
    mantle_dir: _PathLike,
    points: pd.DataFrame,
    depths_km: ArrayLike = DEFAULT_DEPTHS,
) -> pd.DataFrame:
    """Extract basic mantle features at a series of labelled points.

    Parameters
    ----------
    mantle_dir : path-like
        Directory containing G-ADOPT output files.
    points : DataFrame
        Must contain columns 'lon', 'lat', and 'age (Ma)'.
    depths_km : array-like, optional
        Depths in km below surface. Default: (100, 200, 300, 400, 500, 600) km.

    Returns
    -------
    DataFrame
        Copy of `points` with additional columns for each sampled mantle
        variable and depth (e.g. 'Temperature_Deviation_CG_100km').
    """
    depths_km = list(depths_km)
    depths_nondim = _to_nondim(depths_km)
    out = points.copy()

    with xr.open_mfdataset(
        sorted(mantle_dir.glob("*.nc")),
        combine='nested',
        concat_dim='time',
    ) as ds:
        for var in MANTLE_FIELDS:
            sampled = _sample_mantle(
                ds=ds,
                var=var,
                lons=points['lon'],
                lats=points['lat'],
                times=points['age (Ma)'],
                depths=depths_nondim,
            )
            for i, depth_km in enumerate(depths_km):
                out[f"{var}_{int(depth_km)}km"] = sampled[:, i]

    return out