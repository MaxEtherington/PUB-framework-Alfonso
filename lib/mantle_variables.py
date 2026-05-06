"""Functions to sample mantle data from G-ADOPT
output grids and join to point data.
"""
from dataclasses import dataclass
from typing import Callable, Any
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import pint_xarray # noqa: F401
from scipy.interpolate import RegularGridInterpolator

from numpy.typing import ArrayLike

# Non-dimensionalisation offset used by G-ADOPT: surface radius in Earth radii.
_GADOPT_SURFACE_R = 2.208
_MANTLE_THICKNESS = 2891.0  # km

DEFAULT_DEPTHS: np.ndarray = np.arange(100, 500, 100)  # km (upper mantle)
BASE_MANTLE_VARS: list[str] = [
    "FullTemperature_CG",
    "Pressure",
    "Radial_Velocity",
    "Temperature_CG",
    "Temperature_Deviation_CG",
    "Velocity_x",
    "Velocity_y",
    "Velocity_z",
    "Viscosity_CG",
    "East_Velocity",
    "North_Velocity",
]

# ==================
# Feature registry
# ==================

@dataclass
class MantleVariable:
    """A registered mantle variable with a computation function."""
    name: str
    compute: Callable[..., xr.DataArray]


class MantleVariableRegistry:
    def __init__(self):
        self._variables: dict[str, MantleVariable] = {}
        self._transforms: dict[str, Callable[..., Any]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str):
        """Decorator to register a variable computation function."""
        def decorator(fn):
            self._variables[name] = MantleVariable(name=name, compute=fn)
            return fn
        return decorator
    
    def register_base(self, *names: str):
        """Register one or more base variables that are loaded directly from the dataset."""
        for name in names:
            self._variables[name] = MantleVariable(
                name=name,
                compute=lambda ds, n=name: ds[n],
            )

    def register_transform(self, name: str):
        """Register a named transformation (e.g. depth averaging)."""
        def decorator(fn):
            self._transforms[name] = fn
            return fn
        return decorator

    def register_derived(self, name: str, transform: str, **kwargs):
        """Register a new named variable by applying a transform to an existing one."""
        if transform not in self._transforms:
            raise KeyError(f"Unknown transform: '{transform}'. Available: {list(self._transforms)}")
        self._variables[name] = MantleVariable(
            name=name,
            compute=lambda ds: self._transforms[transform](ds, **kwargs),
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, name: str, ds: xr.Dataset) -> xr.DataArray:
        if name not in self._variables:
            raise KeyError(f"Unknown variable: '{name}'. Available: {list(self._variables)}")
        
        if name in ds.data_vars:
            return ds[name]
        
        result = self._variables[name].compute(ds)
        ds[name] = result  # Cache in dataset for future retrievals
        return result.rename(name) if isinstance(result, xr.DataArray) else result

    @property
    def available(self) -> list[str]:
        return list(self._variables)


variables = MantleVariableRegistry()

variables.register_base(*BASE_MANTLE_VARS)


# ==================
# Helper functions
# ==================


def _to_km(depths: ArrayLike) -> np.ndarray:
    """Convert G-ADOPT nondimensionalised depths to km below surface."""
    return (_GADOPT_SURFACE_R - np.asarray(depths, dtype=float)) * _MANTLE_THICKNESS


def _to_nondim(depths_km: ArrayLike) -> np.ndarray:
    """Convert depths in km below surface to G-ADOPT nondimensionalised depths."""
    return _GADOPT_SURFACE_R - (np.asarray(depths_km, dtype=float) / _MANTLE_THICKNESS)


# ==================
# Derived variables
# ==================

@variables.register("Cell_Volume")
def _cell_volume(ds: xr.Dataset) -> xr.DataArray:
    """Calculate spherical cell volumes from center coordinates."""

    def centers_to_edges(centers: xr.DataArray, dim: str) -> xr.DataArray:
        """Convert evenly-spaced cell centers to edges."""
        spacing = centers.diff(dim).mean().item()
        interior = (centers.values[:-1] + centers.values[1:]) / 2
        edges = np.concatenate([[centers.values[0] - spacing/2],
                                interior,
                                [centers.values[-1] + spacing/2]])
        return xr.DataArray(edges, dims=[dim], attrs=centers.attrs).sortby(dim)
    
    r      = centers_to_edges(ds.depth, 'depth')
    theta  = centers_to_edges(np.deg2rad(90 - ds.lat), 'lat')
    phi    = centers_to_edges(np.deg2rad(ds.lon), 'lon')

    da = (r**3).diff('depth') / 3 * (-np.cos(theta)).diff('lat') * phi.diff('lon')
    
    depth_units = ds.depth.attrs["units"]
    da.attrs = {
        "units": f"{depth_units}**3",
        "long_name": "cell volume",
    }
    
    return da.rename("Cell_Volume")


@variables.register("Speed")
def _speed(ds: xr.Dataset) -> xr.DataArray:
    """Calculate flow speed from velocity components."""
    ds = ds.pint.quantify()
    
    vx = ds["Velocity_x"]
    vy = ds["Velocity_y"]
    vz = ds["Velocity_z"]
    da = np.sqrt(vx**2 + vy**2 + vz**2)
    
    da.attrs = {"long_name": "speed"}
    da = da.pint.dequantify()
    return da.rename("Speed")


@variables.register("Tangential_Speed")
def _tangential_speed(ds: xr.Dataset) -> xr.DataArray:
    """Calculate tangential speed from velocity components."""
    ds = ds.pint.quantify()
    ve = ds["East_Velocity"]
    vn = ds["North_Velocity"]
    
    da = np.sqrt(ve**2 + vn**2)
    
    da.attrs = {"long_name": "tangential speed"}
    da = da.pint.dequantify()
    return da.rename("Tangential_Speed")


@variables.register("Radial_Tangential_Ratio")
def _radial_tangential_ratio(ds: xr.Dataset) -> xr.DataArray:
    """Calculate radial-to-tangential velocity ratio."""
    vr = ds["Radial_Velocity"]
    vt = _tangential_speed(ds)
    
    for v in [vr, vt]:
        v = v.pint.quantify()
    da = np.abs(vr) / vt
    
    da.attrs = {"long_name": "radial-to-tangential velocity ratio"}
    da = da.pint.dequantify()
    return da.rename("Radial_Tangential_Ratio")


@variables.register("LAB_Depth")
def _LAB_depth(ds: xr.Dataset) -> xr.DataArray:
    if "Lithosphere_Indicator" not in ds:
        raise ValueError("Dataset must contain 'Lithosphere_Indicator' variable to calculate LAB depth.")
       
    da = xr.where(ds["Lithosphere_Indicator"] > 0.5, ds["depth"], 0, keep_attrs=True).max(dim="depth")
    
    da.attrs = {"long_name": "depth to lithosphere-asthenosphere boundary"}
    return da.rename("LAB_Depth")


@variables.register("Slab_Depth") #TODO
def _slab_depth(ds: xr.Dataset) -> xr.DataArray:
    pass



# ==================
# Transforms
# ==================

@variables.register_transform("average_over_depth") #TODO
def _average_over_depth(
    ds: xr.Dataset,
    var_name: str,
    depth_range: tuple[float, float] | None = None,
) -> xr.DataArray:
    """Average a variable over a depth range, defaulting to full extent of the mantle."""
    da = variables.get(var_name, ds)
    if "depth" not in da.dims:
        raise ValueError(f"'{var_name}' has no 'depth' dimension.")

    depth_range = depth_range or (float(da.depth.min()), float(da.depth.max()))
    d0, d1 = depth_range

    da = da.sel(depth=(da.depth >= d0) & (da.depth <= d1)).mean(dim="depth")
    return da.rename(f"{var_name}_avg_{int(d0)}-{int(d1)}km")


@variables.register_transform("average_over_rolling_time")
def _average_over_rolling_time(
    ds: xr.Dataset,
    var_name: str,
    window_size: int,
) -> xr.DataArray:
    """Calculate a rolling average of a variable over time."""
    da = variables.get(var_name, ds)
    if "time" not in da.dims:
        raise ValueError(f"'{var_name}' has no 'time' dimension.")
    
    da.sortby("time", ascending=False)  # Ensure time (Ma) decreases so rolling average looks backward
    da = da.rolling(time=window_size, center=True, min_periods=1).mean()
    return da.rename(f"{var_name}_rolling_{window_size}Ma")


# ==================
# Transformed variables
# ==================

for t, b in [(0, 400), (100, 400)]:
    av_name = f"Temperature_Deviation_avg_{t}-{b}km"
    variables.register_derived(
        name=av_name,
        transform="average_over_depth",
        var_name="Temperature_Deviation_CG",
        depth_range=(t, b),
    )
    for window in [30, 50]:
        variables.register_derived(
            name=f"{av_name}_rolling_{window}Ma",
            transform="average_over_rolling_time",
            var_name=av_name,
            window_size=window,
        )


# ==================
# Sampling utilities
# ==================


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


def _match_longitude_convention(da: xr.DataArray, lons: ArrayLike) -> np.ndarray:
    """Map longitudes to match the dataset longitude convention."""
    convention = _detect_lon_convention(da["lon"].values)
    return _normalize_lons(lons, convention)


def _wrap_longitude_seam(
    da: xr.DataArray,
) -> np.ndarray:
    """Make longitude cyclic so seam points interpolate using 359<->0 neighbours."""

    lon_vals = np.asarray(da["lon"].values, dtype=float)
    lon_vals = lon_vals[np.isfinite(lon_vals)]
    if lon_vals.size == 0:
        raise ValueError("Longitude coordinate is empty.")

    # Add wrapped slices on both sides of the longitude axis to interpolate
    # across the seam (e.g. between 359 and 0 degrees).
    left = da.isel(lon=-1).assign_coords(lon=da["lon"].isel(lon=-1) - 360.0)
    right = da.isel(lon=0).assign_coords(lon=da["lon"].isel(lon=0) + 360.0)
    da_cyclic = xr.concat([left, da, right], dim="lon").sortby("lon")
    
    return da_cyclic


def _do_interp_cyclic_lon(
    da: xr.DataArray,
    lons: ArrayLike,
    lats: ArrayLike,
    times: ArrayLike,
    depths: ArrayLike,
    broadcast_depth: bool = False,
    method: str = "linear",
) -> xr.DataArray:
    """
    Interpolate with antimeridian longitude seam wrapping; depth is broadcast
    (treated as orthogonal) when requested. Uses RegularGridInterpolator for
    joint ND interpolation, avoiding the sequential-axis artefacts of DA.interp.
    """
    if lons is not None:
        lons = _match_longitude_convention(da=da, lons=lons)
        da = _wrap_longitude_seam(da)

    # Active dims in da's own axis order — drives both RGI grid and query columns
    active_dims = [d for d in da.dims if d in {"lon", "lat", "depth", "time"}]

    rgi = RegularGridInterpolator(
        points=tuple(da[d].values.astype(float) for d in active_dims),
        values=da.values,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    # Numeric query arrays, keyed by dim name
    point_coords: dict[str, np.ndarray] = {}
    if "lon"   in active_dims: point_coords["lon"]   = np.asarray(lons,   dtype=float)
    if "lat"   in active_dims: point_coords["lat"]   = np.asarray(lats,   dtype=float)
    if "time"  in active_dims: point_coords["time"]  = np.asarray(times,  dtype=float)
    if "depth" in active_dims: point_coords["depth"] = np.asarray(depths, dtype=float)

    if broadcast_depth and "depth" in active_dims:
        # Cartesian product: repeat each point coord n_depths times,
        # tile depth array n_points times → shape (n_points * n_depths, n_dims)
        n_points = len(next(v for k, v in point_coords.items() if k != "depth"))
        n_depths = len(point_coords["depth"])
        query = np.column_stack([
            np.tile(point_coords[d], n_points) if d == "depth"
            else np.repeat(point_coords[d], n_depths)
            for d in active_dims
        ])
        values = rgi(query).reshape(n_points, n_depths)
        result = xr.DataArray(
            values,
            dims=("points", "depth"),
            coords={"depth": point_coords["depth"]},
        )
    else:
        query = np.column_stack([point_coords[d] for d in active_dims])
        values = rgi(query)
        result = xr.DataArray(values, dims="points")

    if np.any(np.isnan(values)):
        warnings.warn(
            f"NaN values found in sampled mantle data ({da.name}). "
            "Check that all points are within the dataset bounds."
        )

    return result.rename(da.name)


def sample_mantle_var(
    da: xr.DataArray,
    lons: ArrayLike = None,
    lats: ArrayLike = None,
    times: ArrayLike = None,
    depths: ArrayLike = None,
    method: str = "linear",
) -> pd.DataFrame:
    """Sample/interpolate mantle data at fixed (time, depth, lat, lon) coordinates."""
    # Check if available coordinates match those of the data array; if not, raise an error
    coord_map = {"lon": lons, "lat": lats, "time": times, "depth": depths}
    for coord in da.dims:
        if coord_map.get(coord) is None:
            raise ValueError(
                f"Data array '{da.name}' has dimension '{coord}' but no corresponding coordinates were provided."
            )
    
    result = pd.DataFrame(_do_interp_cyclic_lon(
        da=da,
        lons=lons,
        lats=lats,
        times=times,
        depths=depths,
        broadcast_depth=False,
        method=method,
    ).to_pandas())
    
    result.columns = [f"{da.name.lower()} ({da.attrs.get("units", "unitless")})"]
    
    return result


def sample_mantle_var_depths(
    da: xr.DataArray,
    lons: ArrayLike = None,
    lats: ArrayLike = None,
    times: ArrayLike = None,
    depths: ArrayLike = DEFAULT_DEPTHS,
    method: str = "linear",
) -> pd.DataFrame:
    """Sample/interpolate mantle data at (time, lat, lon) coordinates, broadcasting depth."""
    # Check if available coordinates match those of the data array; if not, raise an error
    coord_map = {"lon": lons, "lat": lats, "time": times, "depth": depths}
    for coord in da.dims:
        if coord_map.get(coord) is None:
            raise ValueError(
                f"Data array '{da.name}' has dimension '{coord}' but no corresponding coordinates were provided."
            )
    
    result = pd.DataFrame(_do_interp_cyclic_lon(
        da=da,
        lons=lons,
        lats=lats,
        times=times,
        depths=depths,
        broadcast_depth=True,
        method=method,
    ).to_pandas())
    
    if "depth" not in da.dims:
        raise ValueError(f"Data array '{da.name}' has no 'depth' dimension, cannot sample over depths.")
    result.columns = [
        f"{da.name.lower()}_{depth:.0f}km ({da.attrs.get('units', 'unitless')})"
        for depth in depths
    ]
    
    return result


def sample_LAB_depths(
    ds: xr.Dataset,
    da: xr.DataArray,
    times: ArrayLike=None,
    lats: ArrayLike=None,
    lons: ArrayLike=None,
    offset_km: float=0.0,
) -> pd.DataFrame:
    """
    Sample mantle variable at the depth of the lithosphere-asthenosphere boundary.
    
    `offset_km` allows sampling at a fixed depth below the LAB (positive) or above it (negative).
    """

    coords = {
        "lons": lons,
        "lats": lats,
        "times": times,
    }
    
    lab_depths = np.asarray(sample_mantle_var(
        da=variables.get("LAB_Depth", ds),
        **coords,
        method="linear",
    )).flatten()  # Flatten to 1D array to use as depth coordinate
    
    result = sample_mantle_var(
        da=da,
        **coords,
        depths=lab_depths + offset_km,
        method="nearest",
    )
    
    result.columns = [
        f"{da.name.lower()}_LAB_{offset_km:+.0f}km ({da.attrs.get('units', 'unitless')})"
        if offset_km != 0 else
        f"{da.name.lower()}_LAB ({da.attrs.get('units', 'unitless')})"
        ]
    
    return result


def calculate_lambdas(
    da: xr.DataArray,
    times: ArrayLike=None,
    lats: ArrayLike=None,
    lons: ArrayLike=None,
    depth_range: tuple[float, float] | None = None,
    n_lambdas: int=5,
) -> pd.DataFrame:
    """Calculate the first `n_lambdas` polynomial regression coefficients for the given data and depth range."""
    
    if "depth" not in da.dims:
        raise ValueError(f"Data array '{da.name}' has no 'depth' dimension, cannot calculate lambdas.")
    
    d_min, d_max = depth_range or (float(da.depth.min()), float(da.depth.max()))
    
    # Select depth range for polynomial fitting using explicit bounds (works with descending coordinates)
    depths = np.asarray(da.sel(depth=(da.depth >= d_min) & (da.depth <= d_max)).depth.values, dtype=float)
    
    if depths.size == 0:
        raise ValueError(
            f"No depths found in range [{d_min}, {d_max}] km. "
            f"Dataset depth range: [{float(da.depth.min()):.1f}, {float(da.depth.max()):.1f}] km"
        )
    
    # Sample with broadcast_depth=True to get (n_points, n_depths) matrix
    sampled = _do_interp_cyclic_lon(
        da=da,
        times=times,
        lats=lats,
        lons=lons,
        depths=depths,
        broadcast_depth=True,
        method="linear",
    )  # shape (n_points, n_depths)
    
    depth_profiles = sampled.to_numpy().T  # shape (n_depths, n_points)
    
    # Fit a polynomial of degree n_lambdas-1 to each depth profile and return the coefficients as features
    lambda_coeffs = np.polyfit(depths, depth_profiles, deg=n_lambdas-1) # shape (n_lambdas, n_points)
    
    lambda_cols = [f"{da.name.lower()}_lambda_{i} ({da.attrs.get('units', 'unitless')})" for i in reversed(range(n_lambdas))]
    result = pd.DataFrame(lambda_coeffs.T, columns=lambda_cols)  # shape (n_points, n_lambdas)

    return result