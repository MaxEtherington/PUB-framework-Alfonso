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
_MANTLE_THICKNESS = 2891.0  # km

DEFAULT_DEPTHS: np.ndarray = np.arange(100, 2900, 100)  # km
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
    return (_GADOPT_SURFACE_R - np.asarray(depths, dtype=float)) * _MANTLE_THICKNESS


def _to_nondim(depths_km: ArrayLike) -> np.ndarray:
    """Convert depths in km below surface to G-ADOPT nondimensionalised depths."""
    return _GADOPT_SURFACE_R - (np.asarray(depths_km, dtype=float) / _MANTLE_THICKNESS)


def _detect_lon_convention(lon_values: ArrayLike) -> str | None:
    """Detect longitude convention: '0-360', '-180-180', or None if unknown."""
    vals = np.asarray(lon_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    lon_min = float(np.min(vals))
    lon_max = float(np.max(vals))

    if lon_min >= 0.0 and lon_max > 180.0:
        return "0-360"
    if lon_min < 0.0 and lon_max <= 180.0:
        return "-180-180"
    if lon_min < 0.0 and lon_max > 180.0:
        raise ValueError(
            f"Cannot determine dataset longitude convention from range [{lon_min:.2f}, {lon_max:.2f}]."
        )
    return None


def _normalize_lons(lons: ArrayLike, convention: str | None) -> np.ndarray:
    """Normalize longitudes to a target convention."""
    lons_arr = np.asarray(lons, dtype=float)
    if convention == "0-360":
        return np.mod(lons_arr, 360.0)
    if convention == "-180-180":
        return ((lons_arr + 180.0) % 360.0) - 180.0
    return lons_arr


def _match_longitude_convention(ds: xr.Dataset, lons: ArrayLike) -> np.ndarray:
    """Map longitudes to match the dataset longitude convention."""
    convention = _detect_lon_convention(ds["lon"].values)
    return _normalize_lons(lons, convention)


def _interp_cyclic_lon(
    da: xr.DataArray,
    lons: ArrayLike,
    lats: ArrayLike,
    times: ArrayLike,
    depths: ArrayLike,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate with periodic longitude so seam points use 359<->0 neighbours.

    Assumes `lons` are already normalized to the dataset longitude convention.
    """
    lons_arr = np.asarray(lons, dtype=float)
    lats_arr = np.asarray(lats, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    depths_arr = np.asarray(depths, dtype=float)

    lon_vals = np.asarray(da["lon"].values, dtype=float)
    lon_vals = lon_vals[np.isfinite(lon_vals)]
    if lon_vals.size == 0:
        raise ValueError("Longitude coordinate is empty.")

    # Add wrapped slices on both sides of the longitude axis to interpolate
    # across the seam (e.g. between 359 and 0 degrees).
    left = da.isel(lon=-1).assign_coords(lon=da["lon"].isel(lon=-1) - 360.0)
    right = da.isel(lon=0).assign_coords(lon=da["lon"].isel(lon=0) + 360.0)
    da_cyclic = xr.concat([left, da, right], dim="lon").sortby("lon")

    return da_cyclic.interp(
        lon=xr.DataArray(lons_arr, dims="points"),
        lat=xr.DataArray(lats_arr, dims="points"),
        time=xr.DataArray(times_arr, dims="points"),
        depth=depths_arr,
        method=method,
    ).values


def _sample_mantle(
    ds: xr.Dataset,
    var: str,
    lons: ArrayLike,
    lats: ArrayLike,
    times: ArrayLike,
    depths: ArrayLike,
    method: str = "linear",
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

    result = _interp_cyclic_lon(
        da=ds[var],
        lons=lons,
        lats=lats,
        times=times,
        depths=depths,
        method=method,
    )

    if np.isnan(result).any():
        raise ValueError("NaN values found in sampled mantle data. Check that all points are within the dataset bounds.")
    
    return result


def extract_basic_mantle_features(
    points: pd.DataFrame,
    mantle_dir: _PathLike,
    depths_km: ArrayLike = DEFAULT_DEPTHS,
    output_fields: ArrayLike = MANTLE_FIELDS,
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
        for var in output_fields:
            sampled = _sample_mantle(
                ds=ds,
                var=var,
                lons=points["lon"],
                lats=points["lat"],
                times=points["age (Ma)"],
                depths=depths_nondim,
            )
            for i, depth_km in enumerate(depths_km):
                new_cols[f"{var}_{int(depth_km)}km"] = sampled[:, i]

    # Join all new columns at once to avoid DataFrame fragmentation.
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out


def _calculate_velocity_magnitude(ds: xr.Dataset) -> xr.DataArray:
    """Calculate velocity magnitude from velocity components."""
    vx = ds["Velocity_x"]
    vy = ds["Velocity_y"]
    vz = ds["Velocity_z"]
    return np.sqrt(vx**2 + vy**2 + vz**2)


def _calculate_tangential_velocity(ds: xr.Dataset) -> xr.DataArray:
    """Calculate tangential velocity magnitude from velocity components."""
    vx = ds["Velocity_x"]
    vy = ds["Velocity_y"]
    vz = ds["Velocity_z"]
    vr = ds["Radial_Velocity"]
    # Clip tiny negative values from floating-point error before sqrt.
    return np.sqrt(np.maximum(vx**2 + vy**2 + vz**2 - vr**2, 0.0))


def _calculate_radial_tangential_ratio(ds: xr.Dataset) -> xr.DataArray:
    """Calculate radial-to-tangential velocity ratio."""
    vr = ds["Radial_Velocity"]
    vt = _calculate_tangential_velocity(ds)
    return np.abs(vr) / vt

def calculate_derived_mantle_velocity_features(ds: xr.Dataset) -> xr.Dataset:
    """Calculate derived mantle features and add to dataset."""
    ds = ds.copy()
    ds["Velocity_Magnitude"] = _calculate_velocity_magnitude(ds)
    ds["Tangential_Velocity"] = _calculate_tangential_velocity(ds)
    ds["Radial_Tangential_Ratio"] = _calculate_radial_tangential_ratio(ds)
    return ds